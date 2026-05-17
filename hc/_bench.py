# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Microbenchmark surface for `CompiledKernel`.

`bench=True` flips on `-hc-emit-bench-wrapper`: a `<wrapper>_bench`
sibling calls `hc_rt_launch_kernel_repeat`, which loops N launches +
`hipStreamSynchronize` under one `CLOCK_MONOTONIC` bracket and returns
elapsed nanoseconds.

* `make_bench_invoker(...)` -> ctypes thunk closure
  `(stream, *args, n_inner) -> ns`.
* `BenchResult` carries raw `samples_ns` plus on-demand stats.

Outer-sample orchestration is on `CompiledKernel.bench`. Pure-Python
outer loop is fine -- the inner loop is in C.
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
    """One-outer-sample callable into `<name>_bench`.

    Shares the cache's `ExecutionEngine` with `.invoke()`. Signature:
    `(*args, stream=None, n_inner=N) -> elapsed_ns`. Outer M-sample
    loop and stats live in `CompiledKernel.bench`.
    """
    engine, handle = ensure_engine(cache, module)
    # Bench wrapper inherits the regular wrapper's user-arg arity;
    # bench pass adds a sibling, never replaces.
    num_args = kernel_arg_count(module, kernel_name)
    func_ptr = engine.lookup(handle, bench_wrapper_name)
    if not func_ptr:
        raise RuntimeError(
            f"hc.bench: lookup of bench wrapper '@{bench_wrapper_name}' "
            "returned a null address (the JIT loaded the module but the "
            "symbol is not visible -- check that `-hc-emit-bench-wrapper` "
            "ran and `symbol-dce` did not drop the clone)"
        )

    # Signature: `uint64 (void* stream, PyObject* arg0..., size_t n_inner)`.
    # `py_object` slots are borrowed `PyObject*`; `c_size_t` -> n_inner;
    # `c_uint64` -> elapsed `CLOCK_MONOTONIC` ns.
    func_type = ctypes.CFUNCTYPE(
        ctypes.c_uint64,
        ctypes.c_void_p,
        *([ctypes.py_object] * num_args),
        ctypes.c_size_t,
    )
    cfunc = func_type(func_ptr)

    def bench(*args: Any, stream: int | None = None, n_inner: int) -> int:
        # Pin engine: ctypes cfunc isn't a strong ref; without this the
        # JIT'd code could be reclaimed mid-loop on a low-refcount path.
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
    """Raw samples + on-demand stats from `CompiledKernel.bench(...)`.

    `samples_ns[i]` = C-side wall-clock ns for sample i's inner loop +
    `hipStreamSynchronize`. `per_launch_*` = per-sample / `n_inner`,
    the headline number for sub-us kernels.
    """

    samples_ns: np.ndarray
    n_inner: int
    kernel_name: str
    # Mutable side-channel for the summary-table cache.
    _summary_cache: dict[str, str] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        # Stat formulas assume int64 ns / float64 math; reject other dtypes.
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
        # ddof=0 stays total at m_outer=1; sample bench is legal.
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
        # One-liner; per-launch median is the headline number.
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
    # Twin columns: per-launch amortized cost (left), raw outer sample
    # (right; equal when n_inner=1). 22-char columns fit 80 cols.
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
