# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public `hc.compile` entry point.

Runs the Python frontend, drives an MLIR transform schedule from
`hc_front` to the `hc` dialect, then the GPU lowering chain to a
self-contained LLVM IR module (host wrapper + embedded HSACO blob).
Default schedule: `hc/schedules/front_to_hc.mlir`; override via
`schedule=` (file path or inline MLIR text).

The returned handle is callable: lazy `ExecutionEngine` + ctypes
dispatch into the host wrapper, each arg forwarded as `PyObject *`.

Pipeline failure: `hc_ir = None`, diagnostics in `pipeline_diagnostics`;
no compile-time exception. Invoking such a handle raises `RuntimeError`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._bench import BenchResult, make_bench_invoker
from ._invoke import InvokerCache, make_invoker
from ._pipeline import ScheduleSource
from .core import KernelMetadata

# MLIR imports stay lazy inside function bodies so simulator-only callers
# never load the native bindings.

__all__ = ["BenchResult", "CompiledKernel", "ScheduleSource", "compile"]


@dataclass(frozen=True)
class CompiledKernel:
    """Handle returned by `hc.compile`; see the module docstring for status."""

    kernel: Any
    bindings: Mapping[str, int]
    front_ir: Any
    front_ir_text: str
    front_ir_symbols: tuple[str, ...] = field(default=())
    # `None` on pipeline failure; inspect `pipeline_diagnostics`.
    hc_ir: Any | None = field(default=None)
    hc_ir_text: str | None = field(default=None)
    pipeline_diagnostics: tuple[str, ...] = field(default=())
    # Echo of the caller's `target=`. Actual recipe selection happened
    # inside the pipeline.
    target: str | None = field(default=None)
    # `<kernel>_bench` symbol when `bench=True`; `None` refuses
    # `.bench()` with a message pointing back at `bench=True`.
    bench_wrapper_name: str | None = field(default=None)
    # Mutable side-channel for the lazy JIT so the dataclass stays frozen.
    # Excluded from compare/repr -- invocation state doesn't define identity.
    _invoker_cache: InvokerCache = field(
        default_factory=InvokerCache, compare=False, repr=False
    )

    def invoke(self, *args: Any, stream: int | None = None) -> None:
        """Dispatch the JIT'd host wrapper.

        Lazy-builds an `ExecutionEngine` on first call, caches the
        invoker on the handle. Buffer slots accept tensor-like objects
        with `data_ptr()` / `size(i)` / `stride(i)` (e.g. `torch.Tensor`);
        scalars accept plain `int`/`float`. The host wrapper unpacks
        each slot inside JIT'd code.

        `stream=None` is HIP default-stream semantics; an `int` is a
        raw stream-handle address (PyTorch:
        `torch.cuda.current_stream().cuda_stream`).

        Raises `RuntimeError` on pipeline failure or missing runtime
        `.so`s. Helper-side errors currently unwind via C++ across the
        C boundary (process abort); migration to a sentinel-return
        contract is on the runtime backlog.
        """
        if self.hc_ir is None:
            diagnostics = (
                "\n  ".join(self.pipeline_diagnostics)
                if self.pipeline_diagnostics
                else "(no diagnostics captured)"
            )
            raise RuntimeError(
                "hc.compile: pipeline failed to lower this kernel; "
                "cannot invoke. Diagnostics:\n  " + diagnostics
            )
        if self._invoker_cache.invoker is None:
            kernel_name = self._resolve_kernel_name(surface="hc.invoke")
            self._invoker_cache.invoker = make_invoker(
                self._invoker_cache, self.hc_ir, kernel_name
            )
        self._invoker_cache.invoker(*args, stream=stream)

    def bench(
        self,
        args: tuple[Any, ...] | list[Any],
        *,
        n_inner: int,
        m_outer: int,
        warmup: int = 2,
        stream: int | None = None,
    ) -> BenchResult:
        """Run bench wrapper m_outer x n_inner times; return stats.

        `warmup` untimed outer iters, then `m_outer` timed outer
        samples. Each outer sample dispatches
        `hc_rt_launch_kernel_repeat` for `n_inner` launches +
        `hipStreamSynchronize` and returns C-side `CLOCK_MONOTONIC`
        nanoseconds. Timing never crosses the language boundary.

        Caveats:
        * Inputs reused across every launch -- cache-hot latency for
          small / L2-fitting kernels. Rotate in warmup if it matters.
        * One number per outer sample. Sweep `n_inner` to separate
          submit vs sync (large -> submit; 1 -> submit + sync per launch).

        Raises `RuntimeError` if the handle was not built with
        `bench=True`.
        """
        self._require_bench_ready()
        _validate_bench_counts(n_inner=n_inner, m_outer=m_outer, warmup=warmup)
        kernel_name = self._resolve_kernel_name(surface="hc.bench")
        bench_call = self._ensure_bench_invoker(kernel_name)

        # Lazy: simulator-only callers must not load native deps.
        import numpy as np

        # Burn in `hc_rt_load_kernel`'s per-callsite cache slot before
        # timing, even when caller passes warmup=0.
        for _ in range(warmup):
            bench_call(*args, stream=stream, n_inner=n_inner)
        samples = np.empty(m_outer, dtype=np.int64)
        for i in range(m_outer):
            samples[i] = bench_call(*args, stream=stream, n_inner=n_inner)
        return BenchResult(
            samples_ns=samples,
            n_inner=n_inner,
            kernel_name=kernel_name,
        )

    def _require_bench_ready(self) -> None:
        if self.hc_ir is None:
            diagnostics = (
                "\n  ".join(self.pipeline_diagnostics)
                if self.pipeline_diagnostics
                else "(no diagnostics captured)"
            )
            raise RuntimeError(
                "hc.compile: pipeline failed to lower this kernel; "
                "cannot bench. Diagnostics:\n  " + diagnostics
            )
        if self.bench_wrapper_name is None:
            raise RuntimeError(
                "hc.bench: this CompiledKernel was built without "
                "bench=True; the bench wrapper symbol does not exist in "
                "the JIT'd module. Re-call hc.compile(..., bench=True) "
                "and retry."
            )

    def _ensure_bench_invoker(self, kernel_name: str) -> Callable[..., Any]:
        if self._invoker_cache.bench_invoker is None:
            assert self.bench_wrapper_name is not None  # asserted in caller
            self._invoker_cache.bench_invoker = make_bench_invoker(
                self._invoker_cache,
                self.hc_ir,
                kernel_name,
                self.bench_wrapper_name,
            )
        return self._invoker_cache.bench_invoker

    def _resolve_kernel_name(self, *, surface: str) -> str:
        kernel_name = getattr(self.kernel, "__name__", None)
        if not isinstance(kernel_name, str) or not kernel_name:
            raise RuntimeError(
                f"{surface}: kernel has no __name__; cannot resolve "
                "the host wrapper symbol"
            )
        return kernel_name

    def __call__(self, *args: Any, stream: int | None = None, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                "hc.compile: invoking a CompiledKernel with arbitrary kwargs "
                "is not supported; only `stream=` is accepted, all other "
                "arguments must be positional"
            )
        return self.invoke(*args, stream=stream)

    def __repr__(self) -> str:
        name = getattr(self.kernel, "__name__", "<kernel>")
        joined = ", ".join(f"{k}={v}" for k, v in sorted(self.bindings.items()))
        stage = "hc" if self.hc_ir_text is not None else "hc_front"
        target = "" if self.target is None else f", target={self.target!r}"
        return f"CompiledKernel({name}, {{{joined}}}, stage={stage}{target})"


def _snapshot_and_clone_front_ir(front_module: Any, context: Any) -> tuple[str, Any]:
    """Pin pre-pipeline IR text; return a mutable sibling for the pipeline.

    No cheap in-memory module clone in the MLIR Python bindings, so
    round-trip through text. Parse+print loses some debug info / exotic
    attrs; bit-exact lineage callers must compare `front_ir_text`, not
    module objects.
    """
    from .mlir import ir as _ir

    front_ir_text = str(front_module)
    pipeline_module = _ir.Module.parse(front_ir_text, context=context)
    return front_ir_text, pipeline_module


def _bench_wrapper_symbol(kernel_fn: Any, hc_module: Any, *, bench: bool) -> str | None:
    """`<kernel>_bench` when `bench=True`, else `None`. No IR walk needed."""
    if not bench or hc_module is None:
        return None
    kernel_name = getattr(kernel_fn, "__name__", None)
    if not kernel_name:
        return None
    return f"{kernel_name}_bench"


def _validate_bench_counts(*, n_inner: int, m_outer: int, warmup: int) -> None:
    """Reject non-int / non-positive counts before any JIT lookup."""
    if not isinstance(n_inner, int) or n_inner <= 0:
        raise ValueError(f"n_inner must be a positive int, got {n_inner!r}")
    if not isinstance(m_outer, int) or m_outer <= 0:
        raise ValueError(f"m_outer must be a positive int, got {m_outer!r}")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative int, got {warmup!r}")


def compile(
    kernel_fn: Any,
    symbols: Mapping[Any, int] | None = None,
    *,
    schedule: ScheduleSource = None,
    target: str | None = None,
    bench: bool = False,
) -> CompiledKernel:
    """Run the frontend + hc_front->hc pipeline on a kernel.

    `kernel_fn`: a `@kernel`-decorated function.

    `symbols`: name -> int bindings (`Symbol` or `str` keys). Keys must
    match the kernel's `literals=` whitelist if declared; missing
    entries are legal (partial specialization). `$`-prefixed keys
    (`$WGS<axis>`, `$WS<axis>`, `$WV0`, `$GSZ0`) are launch-context
    overrides -- they bypass the whitelist and override the values the
    front-to-hc handshake would seed from integer-literal `group_shape`
    / `work_shape` / `subgroup_size`. Use them to pin those dims for
    the native lowering when the decorator left them symbolic.

    `schedule`: overrides `hc/schedules/front_to_hc.mlir`. `Path`: read
    from disk. `str`: inline MLIR text. Must define `@__transform_main`.

    `target`: substituted into the schedule's `__HC_TARGET__`
    placeholder, feeds `hc-interpret-intrinsic-recipes`'s `target=`.
    `None` leaves it empty -- every recipe fires regardless of
    `hc.target` (correct while each intrinsic registers at most one
    recipe per compile). A user `schedule` without the placeholder
    silently ignores `target`.

    `bench=True`: splice `-hc-emit-bench-wrapper`, mint `<name>_bench`,
    enable `.bench(...)`. Default `False` leaves the JIT'd module
    byte-identical and `.bench(...)` raises.

    `hc-specialize-literals` folds bindings into IR (see
    `doc/schedules.md`). `front_ir` is the pre-specialization snapshot;
    `hc_ir` is fully specialized.
    """

    metadata = getattr(kernel_fn, "__hc_kernel__", None)
    if not isinstance(metadata, KernelMetadata):
        raise TypeError(
            f"hc.compile expects a @kernel-decorated function, got {kernel_fn!r}"
        )

    if symbols is None:
        symbols = {}
    elif not isinstance(symbols, Mapping):
        raise TypeError(f"symbols must be a Mapping, got {type(symbols).__name__}")

    bindings = _normalise_bindings(symbols, metadata)

    # Lazy: simulator-only callers must not load native MLIR bindings.
    from ._pipeline import prepared_context, run_front_to_hc
    from ._resolve import resolve_front_ir

    context = prepared_context()
    resolved = resolve_front_ir(kernel_fn, context=context)
    front_module = resolved.module
    _stamp_literal_bindings(front_module, bindings, context)
    front_ir_text, pipeline_module = _snapshot_and_clone_front_ir(front_module, context)
    result = run_front_to_hc(
        pipeline_module, schedule=schedule, target=target, bench=bench
    )
    bench_wrapper_name = _bench_wrapper_symbol(kernel_fn, result.module, bench=bench)
    # Only decorated top-levels reach the public handle; inline helpers
    # are consumed by `-hc-front-inline` and stay internal.
    return CompiledKernel(
        kernel=kernel_fn,
        bindings=bindings,
        front_ir=front_module,
        front_ir_text=front_ir_text,
        front_ir_symbols=resolved.exported_symbol_names,
        hc_ir=result.module,
        hc_ir_text=result.module_text,
        pipeline_diagnostics=result.diagnostics,
        target=target,
        bench_wrapper_name=bench_wrapper_name,
    )


def _stamp_literal_bindings(
    module: Any,
    bindings: Mapping[str, int],
    context: Any,
) -> None:
    """Attach `literal_bindings = {name = i64}` to every `hc_front.kernel`.

    Stamped pre-snapshot so the snapshot reproduces under `hc-opt`, and
    `convert-hc-front-to-hc` can carry the dict to `hc.kernel` for
    `hc-specialize-literals`. Empty `bindings` is a no-op. Anchor is
    `hc_front.kernel` (not `func` / `intrinsic`): only kernels are
    specialization roots; helpers see substitutions via their kernel
    caller.
    """
    if not bindings:
        return
    from .mlir import ir

    with context, ir.Location.unknown():
        i64 = ir.IntegerType.get_signless(64, context=context)
        entries = {
            str(name): ir.IntegerAttr.get(i64, int(value))
            for name, value in bindings.items()
        }
        dict_attr = ir.DictAttr.get(entries, context=context)
        for op in module.body.operations:
            if op.operation.name != "hc_front.kernel":
                continue
            op.operation.attributes["literal_bindings"] = dict_attr


def _normalise_bindings(
    symbols: Mapping[Any, int],
    metadata: KernelMetadata,
) -> dict[str, int]:
    allowed = {_symbol_name(s) for s in metadata.literals}
    out: dict[str, int] = {}
    for key, value in symbols.items():
        name = _symbol_name(key)
        # `$`-prefixed entries (`$WGS<axis>`, `$WS<axis>`, `$WV0`,
        # `$GSZ0`) bypass the whitelist by design -- launch-context
        # overrides, not kernel-declared specialization points. Mirrors
        # `validateBindingsAgainstLiterals` in
        # `HCSpecializeLiteralsPass.cpp`. Empty `literals` means no
        # whitelist declared; accept any name.
        if allowed and name not in allowed and not name.startswith("$"):
            raise ValueError(
                f"'{name}' is not a declared literal symbol; "
                f"kernel declares {sorted(allowed)}"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"literal symbol '{name}' must bind to an int, "
                f"got {type(value).__name__}"
            )
        # Symbol and its string form collapse to one key; reject
        # conflicting duplicates rather than last-write-wins.
        if name in out and out[name] != value:
            raise ValueError(
                f"literal symbol '{name}' bound twice with conflicting "
                f"values ({out[name]!r} and {value!r})"
            )
        out[name] = value
    return out


def _symbol_name(obj: Any) -> str:
    # `str` first so path-like `.name` ducks can't pose as symbol keys.
    if isinstance(obj, str):
        return obj
    # Lazy: `hc.symbols` stays optional for simulator-only callers.
    from .symbols import Symbol

    if isinstance(obj, Symbol):
        return obj.name
    raise TypeError(f"cannot interpret {obj!r} as a symbol name")


def normalise_bindings(
    symbols: Mapping[Any, int],
    metadata: KernelMetadata,
) -> dict[str, int]:
    return _normalise_bindings(symbols, metadata)


def symbol_name(obj: Any) -> str:
    return _symbol_name(obj)
