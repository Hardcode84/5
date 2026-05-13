# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Microbenchmark surface for `CompiledKernel`.

`hc.compile(kernel, bench=True)` flips on `-hc-emit-bench-wrapper`,
which mints an `i64 wrapper_bench` sibling per host wrapper that calls
`hc_rt_launch_kernel_repeat` instead of `hc_rt_launch_kernel`. The
C-side runtime entry does the inner N-launch loop + `hipStreamSynchronize`
under one `CLOCK_MONOTONIC` bracket and hands back the elapsed
nanoseconds. This module wraps that contract for Python callers:

* `make_bench_invoker(cache, module, kernel_name, bench_wrapper_name)`
  builds a ctypes thunk into the bench wrapper and returns a small
  closure `(stream, *args, n_inner) -> int` returning ns.
* `BenchResult` is the value type the public `CompiledKernel.bench`
  method returns: `samples_ns` plus on-demand numpy aggregates.

Outer-sample orchestration lives on `CompiledKernel.bench` itself; the
factory here only owns the ctypes thunk + the per-sample contract.
Pure-Python loop for the outer iterations is fine — the inner loop is
in C and dominates the wall clock for any kernel worth benchmarking.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._invoke import InvokerCache, ensure_engine, kernel_arg_count

__all__ = ["BenchResult", "make_bench_invoker"]


def make_bench_invoker(
    cache: InvokerCache,
    module: Any,
    kernel_name: str,
    bench_wrapper_name: str,
) -> Callable[..., int]:
    """Build a callable that runs one outer sample against `<name>_bench`.

    Reuses the cache's `ExecutionEngine` so the JIT compiles the
    module exactly once per `CompiledKernel`, no matter how the user
    interleaves `.invoke()` / `.bench()` calls. The returned closure's
    signature mirrors `make_invoker`'s — `(*args, stream=None, n_inner=N)` —
    except it returns the `uint64` elapsed-ns the runtime measured for
    the (n_inner launches + trailing sync) bracket. The outer M-sample
    loop and stats aggregation live on the caller (`CompiledKernel.bench`),
    so this layer stays single-sample.
    """
    engine, handle = ensure_engine(cache, module)
    # Both wrappers carry the same user-arg arity; `kernel_arg_count`
    # reads it off the regular wrapper, which is always present in the
    # post-pipeline module (the bench pass mints a sibling, never
    # replaces the original).
    num_args = kernel_arg_count(module, kernel_name)
    func_ptr = engine.lookup(handle, bench_wrapper_name)
    if not func_ptr:
        raise RuntimeError(
            f"hc.bench: lookup of bench wrapper '@{bench_wrapper_name}' "
            "returned a null address (the JIT loaded the module but the "
            "symbol is not visible — check that `-hc-emit-bench-wrapper` "
            "ran and `symbol-dce` did not drop the clone)"
        )

    # `uint64 (void* stream, PyObject* arg0, ..., size_t n_inner)` —
    # leading `c_void_p` is the HIP stream, the middle `py_object` slots
    # are passed by borrowed reference (a `py_object` _is_ a borrowed
    # `PyObject *`), and the trailing `c_size_t` carries the inner loop
    # count the C-side `hc_rt_launch_kernel_repeat` consumes. The
    # `restype = c_uint64` is the elapsed-ns the runtime measured under
    # `CLOCK_MONOTONIC`.
    func_type = ctypes.CFUNCTYPE(
        ctypes.c_uint64,
        ctypes.c_void_p,
        *([ctypes.py_object] * num_args),
        ctypes.c_size_t,
    )
    cfunc = func_type(func_ptr)

    def bench(*args: Any, stream: int | None = None, n_inner: int) -> int:
        # Pin the engine for the same reason `make_invoker` does — ctypes
        # cfunc references aren't tracked as strong references against
        # the engine, so without this the JIT'd code could be reclaimed
        # mid-loop on a low-refcount path.
        _ = engine
        if len(args) != num_args:
            raise TypeError(
                f"hc.bench: kernel '{kernel_name}' takes {num_args} "
                f"argument(s), got {len(args)}"
            )
        if n_inner <= 0:
            raise ValueError(
                f"hc.bench: n_inner must be a positive int, got {n_inner!r}"
            )
        elapsed = cfunc(
            stream,
            *(ctypes.py_object(arg) for arg in args),
            n_inner,
        )
        return int(elapsed)

    bench.__name__ = f"bench_{kernel_name}"
    bench.__qualname__ = bench.__name__
    return bench


@dataclass(frozen=True)
class BenchResult:
    """Outcome of `CompiledKernel.bench(...)` — raw samples + stats.

    `samples_ns[i]` is the wall-clock nanoseconds the C-side runtime
    measured for outer-sample `i`'s inner loop of `n_inner` kernel
    launches plus the trailing `hipStreamSynchronize`. Each statistic
    is computed on-demand from `samples_ns`; the array is the source
    of truth and external callers can hang their own numpy reductions
    off it without going through the canned properties.

    `per_launch_*` numbers are the per-sample value divided by
    `n_inner`, i.e. the average launch+sync amortized cost from one
    outer iteration. They are the conventional headline number for
    sub-µs kernels.
    """

    samples_ns: np.ndarray
    n_inner: int
    kernel_name: str
    # Frozen-but-mutable side-channel for the formatted-table cache.
    # Computing the table is a microsecond, but `__repr__` may be hit
    # in REPL completion / debugger loops where a tiny per-access cost
    # adds up — and `__post_init__` writes go through `object.__setattr__`
    # to dodge the frozen check.
    _summary_cache: dict[str, str] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        # Pin the dtype defensively: stat formulas below assume integer
        # nanoseconds and `float64` math derived from it. A caller that
        # constructs a `BenchResult` from Python ints would land us at
        # the wrong dtype otherwise.
        if not isinstance(self.samples_ns, np.ndarray):
            raise TypeError(
                "BenchResult.samples_ns must be an np.ndarray, got "
                f"{type(self.samples_ns).__name__}"
            )
        if self.samples_ns.dtype != np.int64:
            raise TypeError(
                f"BenchResult.samples_ns must be int64, got {self.samples_ns.dtype}"
            )
        if self.samples_ns.ndim != 1:
            raise ValueError(
                "BenchResult.samples_ns must be 1-D (m_outer,), got shape "
                f"{self.samples_ns.shape}"
            )
        if self.samples_ns.size == 0:
            raise ValueError("BenchResult.samples_ns is empty")
        if self.n_inner <= 0:
            raise ValueError(
                f"BenchResult.n_inner must be positive, got {self.n_inner}"
            )

    @property
    def m_outer(self) -> int:
        return int(self.samples_ns.size)

    @property
    def mean_ns(self) -> float:
        return float(np.mean(self.samples_ns))

    @property
    def median_ns(self) -> float:
        return float(np.median(self.samples_ns))

    @property
    def std_ns(self) -> float:
        # Population std (ddof=0). Sample variance with ddof=1 collapses
        # at m_outer=1 (which we don't reject — single-sample bench is
        # legal, just uninformative), so ddof=0 keeps the property
        # total. The reported number is "spread of the samples we have";
        # callers wanting Bessel-corrected SE-of-mean can derive it.
        return float(np.std(self.samples_ns))

    @property
    def min_ns(self) -> int:
        return int(np.min(self.samples_ns))

    @property
    def max_ns(self) -> int:
        return int(np.max(self.samples_ns))

    @property
    def p25_ns(self) -> float:
        return float(np.percentile(self.samples_ns, 25))

    @property
    def p75_ns(self) -> float:
        return float(np.percentile(self.samples_ns, 75))

    @property
    def per_launch_median_ns(self) -> float:
        return self.median_ns / self.n_inner

    @property
    def per_launch_mean_ns(self) -> float:
        return self.mean_ns / self.n_inner

    def __repr__(self) -> str:
        # One-line shape: enough to identify the result in a REPL /
        # debugger frame, but not so wide it wraps. Per-launch median
        # is the headline number for sub-µs kernels.
        return (
            f"BenchResult(kernel={self.kernel_name!r}, "
            f"n_outer={self.m_outer}, n_inner={self.n_inner}, "
            f"per_launch_median={self.per_launch_median_ns:.1f}ns)"
        )

    def summary(self) -> str:
        """Multi-line text table with both per-launch and per-sample stats."""
        cached = self._summary_cache.get("text")
        if cached is not None:
            return cached
        text = _format_summary(self)
        self._summary_cache["text"] = text
        return text


def _format_summary(result: BenchResult) -> str:
    # Twin-column layout: left column is the per-launch amortized cost
    # (the headline for sub-µs kernels), right column is the raw outer
    # sample (visible only when n_inner > 1 — otherwise they're equal
    # by definition and the second column is noise). Column widths are
    # hand-picked at 22 chars so the table fits in an 80-col terminal
    # with header padding to spare.
    n_inner = result.n_inner
    per_launch = result.samples_ns.astype(np.float64) / n_inner

    rows: Sequence[tuple[str, float, float]] = [
        ("median", float(np.median(per_launch)), result.median_ns),
        ("mean", float(np.mean(per_launch)), result.mean_ns),
        ("std", float(np.std(per_launch)), result.std_ns),
        ("min", float(np.min(per_launch)), float(result.min_ns)),
        ("max", float(np.max(per_launch)), float(result.max_ns)),
        ("p25", float(np.percentile(per_launch, 25)), result.p25_ns),
        ("p75", float(np.percentile(per_launch, 75)), result.p75_ns),
    ]

    header = (
        f"hc.bench({result.kernel_name})\n"
        f"  m_outer={result.m_outer}  n_inner={n_inner}\n"
        f"  {'stat':<8}  {'per-launch (ns)':>20}  {'outer-sample (ns)':>20}\n"
        f"  {'-' * 8}  {'-' * 20}  {'-' * 20}"
    )
    body = "\n".join(
        f"  {label:<8}  {per_launch_val:>20.2f}  {sample_val:>20.2f}"
        for label, per_launch_val, sample_val in rows
    )
    return f"{header}\n{body}"
