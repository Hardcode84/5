# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public `hc.compile` entry point.

Only the Python frontend stage exists today; `hc.compile` runs that stage
and hands back a `CompiledKernel` handle carrying the emitted `hc_front`
module. Invoking the handle raises `NotImplementedError` until the rest of
the pipeline (`hc_front -> hc` lowering, specialization, launch) is wired
up. `symbols=` bindings are recorded on the handle for later stages; the
frontend itself emits purely symbolic IR either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .core import KernelMetadata


@dataclass(frozen=True)
class CompiledKernel:
    """Handle returned by `hc.compile`; see the module docstring for status."""

    kernel: Any
    bindings: Mapping[str, int]
    front_ir: Any
    front_ir_text: str
    front_ir_symbols: tuple[str, ...] = field(default=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "hc.compile only runs the Python frontend today; the "
            "hc_front -> hc lowering, specialization, and launch stages "
            "are not implemented yet."
        )

    def __repr__(self) -> str:
        name = getattr(self.kernel, "__name__", "<kernel>")
        joined = ", ".join(f"{k}={v}" for k, v in sorted(self.bindings.items()))
        return f"CompiledKernel({name}, {{{joined}}})"


def compile(
    kernel_fn: Any,
    symbols: Mapping[Any, int] | None = None,
) -> CompiledKernel:
    """Run the current compilation pipeline (frontend only) on a kernel.

    `kernel_fn` must be a `@kernel`-decorated function. `symbols` maps
    literal symbol names (`Symbol` instances or plain strings) to integer
    bindings. Keys must match the kernel's declared `literals=` set; a
    kernel that did not declare a whitelist accepts any key (later stages
    will tighten this). Missing entries are allowed — partial specialization
    is legal and later pipeline stages refine what remains symbolic.

    Bindings are stored on the returned handle but the current frontend-only
    stage does not substitute them into the emitted IR; `front_ir` stays
    symbolic until specialization lands.
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

    # Lazy import: the resolver pulls in the native MLIR bindings, which
    # simulator-only callers should not have to install.
    from ._resolve import resolve_front_ir

    resolved = resolve_front_ir(kernel_fn)
    module = resolved.module
    return CompiledKernel(
        kernel=kernel_fn,
        bindings=bindings,
        front_ir=module,
        front_ir_text=str(module),
        front_ir_symbols=resolved.symbol_names,
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
