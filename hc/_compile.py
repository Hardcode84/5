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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._invoke import InvokerCache, make_invoker
from ._pipeline import ScheduleSource
from .core import KernelMetadata

# `_pipeline` at the module top-level is cheap: it only touches stdlib at
# import time. The MLIR-heavy imports (`hc.mlir.ir`, dialects, PassManager)
# stay lazy inside function bodies, so simulator-only callers that never
# invoke `hc.compile` don't load the native bindings.

__all__ = ["CompiledKernel", "ScheduleSource", "compile"]


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
    # Lazy JIT cache. Lives in a mutable side-channel so the dataclass
    # can stay frozen while the engine and cfunc materialize on first
    # invoke. Excluded from compare/repr so two handles compiled from
    # the same kernel still compare equal regardless of whether one of
    # them has been invoked.
    _invoker_cache: InvokerCache = field(
        default_factory=InvokerCache, compare=False, repr=False
    )

    def invoke(self, *args: Any) -> None:
        """Dispatch the JIT'd host wrapper, calling `hc_rt_helpers` inside.

        Lazy-builds an MLIR `ExecutionEngine` (loading
        `libhc_rt_helpers.so` + `libhc_hip_runtime.so` as shared libs so
        their `_mlir_ciface_hc_get_*` and `hc_rt_*` symbols resolve via
        the JIT's process-wide search) the first time it's called, and
        caches the resulting invoker on the handle so subsequent calls
        reuse the same JIT'd code. Each argument is a Python object:
        tensor-like objects (anything with `data_ptr()` / `size(i)` /
        `stride(i)`, e.g. `torch.Tensor`) for buffer slots, plain
        `int`/`float` for scalar slots. The host wrapper unpacks each
        slot inside JIT'd code, so the Python-side call is just a
        ctypes thunk.

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
            kernel_name = getattr(self.kernel, "__name__", None)
            if not kernel_name:
                raise RuntimeError(
                    "hc.invoke: kernel has no __name__; cannot resolve "
                    "the host wrapper symbol"
                )
            self._invoker_cache.invoker = make_invoker(self.hc_ir, kernel_name)
        self._invoker_cache.invoker(*args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                "hc.compile: invoking a CompiledKernel with kwargs is not "
                "supported yet; pass arguments positionally"
            )
        return self.invoke(*args)

    def __repr__(self) -> str:
        name = getattr(self.kernel, "__name__", "<kernel>")
        joined = ", ".join(f"{k}={v}" for k, v in sorted(self.bindings.items()))
        stage = "hc" if self.hc_ir_text is not None else "hc_front"
        target = "" if self.target is None else f", target={self.target!r}"
        return f"CompiledKernel({name}, {{{joined}}}, stage={stage}{target})"


def compile(
    kernel_fn: Any,
    symbols: Mapping[Any, int] | None = None,
    *,
    schedule: ScheduleSource = None,
    target: str | None = None,
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

    Bindings are stored on the returned handle but the current pipeline
    does not substitute them into the emitted IR; `front_ir`/`hc_ir`
    stay symbolic until specialization lands.
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
    # Snapshot the frontend IR text before the pipeline rewrites ops in
    # place. The `front_ir` handle is kept as-is by re-parsing into a
    # sibling clone below; without this, `front_ir_text` and `hc_ir_text`
    # would end up identical after a successful pipeline run.
    front_ir_text = str(front_module)

    # Round-trip through text is our "clone" primitive: the MLIR Python
    # bindings don't expose a cheap in-memory module clone, and we need
    # two handles to the same IR — one pinned as the pre-pipeline
    # snapshot, one handed to the driver to be mutated. Parse + print
    # does not round-trip every piece of metadata (some debug info, some
    # exotic attributes); any caller that needs bit-exact lineage should
    # keep their own copy of `front_ir_text` rather than comparing
    # `front_ir` and `hc_ir` module objects.
    from .mlir import ir as _ir

    pipeline_module = _ir.Module.parse(front_ir_text, context=context)
    result = run_front_to_hc(pipeline_module, schedule=schedule, target=target)
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
    )


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
