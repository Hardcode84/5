# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public `hc.compile` entry point.

`hc.compile` runs the Python frontend, drives an MLIR transform-dialect
schedule over the resulting `hc_front` module to lower it into the `hc`
dialect, and runs the GPU lowering chain to produce a self-contained
LLVM IR module with the host wrapper plus an embedded HSACO blob. The
default schedule lives at `hc/schedules/front_to_hc.mlir`; callers may
override with their own file path or inline MLIR text via `schedule=`.

The returned handle is callable: invoking it with positional args lazily
spins up an MLIR `ExecutionEngine` (loading the runtime helpers and HIP
shim shared libraries), looks up the host wrapper, and dispatches via
ctypes. Each argument is forwarded as a `PyObject *` and unpacked
inside JIT'd code.

On pipeline failure, the handle carries `hc_ir = None` and the captured
diagnostics in `pipeline_diagnostics`; no exception is raised at compile
time so callers can still inspect `front_ir_text` for debugging.
Attempting to invoke such a handle raises `RuntimeError` with the
captured diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._bench import BenchResult, make_bench_invoker
from ._invoke import InvokerCache, make_invoker
from ._pipeline import ScheduleSource
from .core import KernelMetadata

# `_pipeline` at the module top-level is cheap: it only touches stdlib at
# import time. The MLIR-heavy imports (`hc.mlir.ir`, dialects, PassManager)
# stay lazy inside function bodies, so simulator-only callers that never
# invoke `hc.compile` don't load the native bindings.

__all__ = ["BenchResult", "CompiledKernel", "ScheduleSource", "compile"]


@dataclass(frozen=True)
class CompiledKernel:
    """Handle returned by `hc.compile`; see the module docstring for status."""

    kernel: Any
    bindings: Mapping[str, int]
    front_ir: Any
    front_ir_text: str
    front_ir_symbols: tuple[str, ...] = field(default=())
    # `hc_ir` mirrors the Python-side module handle after the transform
    # schedule completes. `hc_ir` / `hc_ir_text` are both `None` when the
    # pipeline fails — callers should treat that as "frontend ran but
    # lowering didn't" and inspect `pipeline_diagnostics` for the why.
    hc_ir: Any | None = field(default=None)
    hc_ir_text: str | None = field(default=None)
    pipeline_diagnostics: tuple[str, ...] = field(default=())
    # Echo of the `target=` argument the caller passed (or `None` for
    # "any target"). Useful for downstream stages and debugging — the
    # actual recipe selection happened inside the pipeline.
    target: str | None = field(default=None)
    # Symbol name of the bench wrapper if `hc.compile(bench=True)` was
    # used, else `None`. `-hc-emit-bench-wrapper` mints `<kernel>_bench`
    # next to the regular `<kernel>` wrapper; recording the name here
    # gives `.bench()` a direct ctypes-lookup target without re-parsing
    # the IR. `None` is the trigger to refuse `bench()` cleanly with a
    # message pointing the user back at `bench=True`.
    bench_wrapper_name: str | None = field(default=None)
    # Lazy JIT cache. Lives in a mutable side-channel so the dataclass
    # can stay frozen while the engine and cfunc materialize on first
    # invoke. Excluded from compare/repr so two handles compiled from
    # the same kernel still compare equal regardless of whether one of
    # them has been invoked.
    _invoker_cache: InvokerCache = field(
        default_factory=InvokerCache, compare=False, repr=False
    )

    def invoke(self, *args: Any, stream: int | None = None) -> None:
        """Dispatch the JIT'd host wrapper, calling `hc_rt_helpers` inside.

        Lazy-builds an `hc.execution_engine.ExecutionEngine` (with the
        `_mlir_ciface_hc_get_*` and `hc_rt_*` symbols resolved out of
        the bundled shared libraries and registered via `set_symbol_map`)
        the first time it's called, and caches the resulting invoker on
        the handle so subsequent calls reuse the same JIT'd code. Each
        positional argument is a Python object: tensor-like objects
        (anything with `data_ptr()` / `size(i)` / `stride(i)`, e.g.
        `torch.Tensor`) for buffer slots, plain `int`/`float` for
        scalar slots. The host wrapper unpacks each slot inside JIT'd
        code, so the Python-side call is just a ctypes thunk.

        `stream=` is the leading argument the host wrapper threads
        through to `hc_rt_load_kernel` / `hc_rt_launch_kernel`; pass
        `None` (the default) for HIP's default-stream semantics, or an
        integer stream handle for explicit ordering. PyTorch users
        usually want `torch.cuda.current_stream().cuda_stream`.

        Raises `RuntimeError` if the pipeline failed (the handle has no
        `hc_ir`) or if either runtime shared library is not present in
        the install. Helper-side errors (missing `data_ptr()`, wrong
        type, etc.) currently unwind via a C++ exception across the C
        boundary — this is undefined behavior and tends to manifest as
        a process abort; replacement with a sentinel-return + PyErr
        contract is on the runtime-helpers backlog.
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
        """Run the bench wrapper m_outer x n_inner times, return stats.

        Driver shape: do `warmup` untimed outer iterations to prime the
        kernel cache / JIT path, then collect `m_outer` timed outer
        samples. Each outer sample drops into JIT'd code once, dispatches
        `hc_rt_launch_kernel_repeat` for an inner loop of `n_inner`
        launches plus one `hipStreamSynchronize`, and returns the
        wall-clock nanoseconds the C-side measured under
        `CLOCK_MONOTONIC`. No `perf_counter_ns` bracketing in Python —
        the per-sample window never crosses the language boundary.

        Caveats baked into the contract:
        * Inputs are reused across every `m_outer * n_inner` launch.
          Small / L2-fitting kernels report cache-hot latency. Rotate
          inputs in the warmup phase or wait for the cache-cold mode
          slice if it matters for your kernel.
        * One time number per outer sample. The submit-vs-sync split is
          derivable by sweeping `n_inner` (large `n_inner` → submit
          path; `n_inner=1` → submit + sync per launch).

        Raises `RuntimeError` if this handle was not compiled with
        `bench=True` (the bench wrapper would not exist in the JIT'd
        module).
        """
        self._require_bench_ready()
        _validate_bench_counts(n_inner=n_inner, m_outer=m_outer, warmup=warmup)
        kernel_name = self._resolve_kernel_name(surface="hc.bench")
        bench_call = self._ensure_bench_invoker(kernel_name)

        # Lazy numpy import so the bench surface respects the same
        # "simulator-only callers don't load native deps" boundary
        # `_compile` itself maintains for the resolver / pipeline.
        import numpy as np

        # Burn-in: the JIT'd module's first call also lazily fills the
        # per-callsite `_handle` cache slot via `hc_rt_load_kernel`.
        # That single-flight memoization is a one-time cost we want to
        # exclude from the timing sample even if the caller passes
        # warmup=0.
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
    """Pin the pre-pipeline IR text and produce a sibling module to mutate.

    The MLIR Python bindings don't expose a cheap in-memory module clone,
    so we round-trip through text: `str(front_module)` is the snapshot
    the public handle keeps, and a fresh `Module.parse` gives the
    pipeline its own mutable copy. Parse + print does not round-trip
    every piece of metadata (some debug info, some exotic attributes);
    callers needing bit-exact lineage should compare `front_ir_text`
    rather than the module objects.
    """
    from .mlir import ir as _ir

    front_ir_text = str(front_module)
    pipeline_module = _ir.Module.parse(front_ir_text, context=context)
    return front_ir_text, pipeline_module


def _bench_wrapper_symbol(kernel_fn: Any, hc_module: Any, *, bench: bool) -> str | None:
    """Derive the bench wrapper symbol name without re-parsing the IR.

    `-hc-emit-bench-wrapper` appends `_bench` to the host wrapper's
    name, and the host wrapper inherits the kernel function's
    `__name__`. Computing the derived symbol here lets `.bench()` jump
    straight to a JIT lookup without an IR walk.
    """
    if not bench or hc_module is None:
        return None
    kernel_name = getattr(kernel_fn, "__name__", None)
    if not kernel_name:
        return None
    return f"{kernel_name}_bench"


def _validate_bench_counts(*, n_inner: int, m_outer: int, warmup: int) -> None:
    """Surface the bench API's int contract before any JIT lookup happens.

    Splitting the validation out of `CompiledKernel.bench` keeps the
    method body close to the lizard CCN threshold; the checks themselves
    are mechanical so collapsing them here costs nothing readability-
    wise. Each error names the failing arg so a misuse points at a
    single line in the caller.
    """
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
    """Run the current compilation pipeline (frontend + hc_front -> hc) on a kernel.

    `kernel_fn` must be a `@kernel`-decorated function. `symbols` maps
    literal symbol names (`Symbol` instances or plain strings) to
    integer bindings. Keys must match the kernel's declared `literals=`
    set; a kernel that did not declare a whitelist accepts any key
    (later stages will tighten this). Missing entries are allowed —
    partial specialization is legal and later pipeline stages refine
    what remains symbolic.

    `schedule` overrides the default `hc/schedules/front_to_hc.mlir`
    transform-dialect schedule: a `pathlib.Path` is read from disk, a
    `str` is treated as inline MLIR text. The schedule must define a
    `@__transform_main` named sequence.

    `target` selects which intrinsic lowering recipe the schedule's
    `hc-interpret-intrinsic-recipes` step applies. The string is
    substituted into the schedule's `__HC_TARGET__` placeholder before
    the pass runs, so it ends up in the pass's `target=` option and the
    interpreter only fires named sequences whose `hc.target` matches.
    Pass `None` (the default) to leave the placeholder empty — the pass
    then runs every recipe regardless of `hc.target`, which is the
    right behaviour while each intrinsic registers at most one recipe
    per compile. A user-provided `schedule` that drops the placeholder
    silently ignores `target`; the override owns its own pass
    invocations.

    `bench=True` splices `-hc-emit-bench-wrapper` into the GPU lowering
    chain so each host wrapper grows a sibling `<name>_bench` that calls
    `hc_rt_launch_kernel_repeat`. The returned handle's `.bench(...)`
    method becomes callable; default `False` leaves the JIT'd module
    byte-identical to today and `.bench(...)` raises with a pointer
    back here.

    Bindings are folded into the IR by `hc-specialize-literals` (see
    `doc/schedules.md`). `front_ir` is the pre-specialization snapshot
    with the binding dict attached; `hc_ir` is fully specialized.
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

    # Lazy imports: the resolver and pipeline pull in the native MLIR
    # bindings, which simulator-only callers should not have to install.
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
    # Only decorated top-levels are surfaced on the public handle;
    # undecorated inline helpers are an implementation detail of the
    # `hc_front` pipeline (they're consumed by `-hc-front-inline`
    # before any downstream stage sees them) so exposing them here
    # would commit the compiler to a shape users would then depend on.
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

    Stamped before the front-IR snapshot so the snapshot is reproducible
    (re-run `hc-opt` against it and the same specialized `hc_ir` falls
    out), and so `convert-hc-front-to-hc` can carry the dict over to the
    corresponding `hc.kernel` for `hc-specialize-literals` to consume.
    Empty `bindings` is a no-op — leaves the IR byte-identical to the
    unspecialized path. `hc_front.kernel` is the right anchor (not
    `hc_front.func` / `hc_front.intrinsic`): only kernels are
    specialization roots, so a helper that happens to reference `K`
    sees the substitution through its kernel caller, not through its
    own metadata.
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
        # Empty `literals` on the decorator means the kernel declared no
        # whitelist; pass the binding through rather than rejecting it.
        if allowed and name not in allowed:
            raise ValueError(
                f"'{name}' is not a declared literal symbol; "
                f"kernel declares {sorted(allowed)}"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"literal symbol '{name}' must bind to an int, "
                f"got {type(value).__name__}"
            )
        # A Symbol and its string form collapse to the same key; refuse
        # conflicting duplicates rather than silently last-write-wins.
        if name in out and out[name] != value:
            raise ValueError(
                f"literal symbol '{name}' bound twice with conflicting "
                f"values ({out[name]!r} and {value!r})"
            )
        out[name] = value
    return out


def _symbol_name(obj: Any) -> str:
    # Plain strings resolve to themselves first so a path-like object
    # (anything with a `.name` attribute) cannot be mistaken for a symbol
    # key. Only real `Symbol` instances — not arbitrary duck-typed objects
    # — are accepted via `.name`.
    if isinstance(obj, str):
        return obj
    # Lazy import so `hc._compile` stays light for simulator-only callers;
    # `hc.symbols` is deliberately lazy-loaded in `hc/__init__.py`.
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
