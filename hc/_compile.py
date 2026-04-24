# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public `hc.compile` entry point.

`hc.compile` runs the Python frontend and then drives an MLIR
transform-dialect schedule over the resulting `hc_front` module,
lowering it into the `hc` dialect. The default schedule lives at
`hc/schedules/front_to_hc.mlir`; callers may override with their own
file path or inline MLIR text via `schedule=`.

Invoking the returned handle still raises `NotImplementedError` until
specialization and launch stages land. `symbols=` bindings are recorded
on the handle for later stages; the frontend itself emits purely
symbolic IR either way.

On pipeline failure, the handle carries `hc_ir = None` and the captured
diagnostics in `pipeline_diagnostics`; no exception is raised so
callers can still inspect `front_ir_text` for debugging.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .core import KernelMetadata

ScheduleSource = Any  # re-exported shape; the real alias lives in ._pipeline


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "hc.compile only runs the Python frontend + hc_front -> hc "
            "lowering today; the specialization and launch stages are not "
            "implemented yet."
        )

    def __repr__(self) -> str:
        name = getattr(self.kernel, "__name__", "<kernel>")
        joined = ", ".join(f"{k}={v}" for k, v in sorted(self.bindings.items()))
        stage = "hc" if self.hc_ir_text is not None else "hc_front"
        return f"CompiledKernel({name}, {{{joined}}}, stage={stage})"


def compile(
    kernel_fn: Any,
    symbols: Mapping[Any, int] | None = None,
    *,
    schedule: ScheduleSource = None,
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

    from .mlir import ir as _ir

    pipeline_module = _ir.Module.parse(front_ir_text, context=context)
    result = run_front_to_hc(pipeline_module, schedule=schedule)
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
