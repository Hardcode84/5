# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thin compile entry point for the `hc` DSL.

Only the Python frontend stage exists today; this module exists mostly to
claim the public API surface and make the shape of things visible. A
`CompiledKernel` handle carries the emitted `hc_front` module and refuses
to launch until the rest of the pipeline (hc_front -> hc, specialization,
launch) is wired up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hc.core import KernelMetadata


@dataclass(frozen=True)
class CompiledKernel:
    """Handle to a kernel that has been run through the compilation pipeline.

    Invoking the handle raises `NotImplementedError` until the rest of the
    pipeline catches up. The `front_ir` / `front_ir_text` fields are there
    so callers can already feed the frontend output into tooling (lit
    runners, canonicalizers, visualisers) without waiting for launch.
    """

    kernel: Any
    bindings: Mapping[str, int]
    front_ir: Any
    front_ir_text: str

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
    literal symbol names (either `Symbol` instances or plain strings) to
    integer bindings; unknown names on kernels that declared a `literals=`
    set are rejected. Missing entries are allowed — partial specialization
    is legal and later pipeline stages refine what is left.
    """

    metadata = getattr(kernel_fn, "__hc_kernel__", None)
    if not isinstance(metadata, KernelMetadata):
        raise TypeError(
            f"hc.compile expects a @kernel-decorated function, got {kernel_fn!r}"
        )

    bindings = _normalise_bindings(symbols or {}, metadata)

    # Import lazily: the frontend pulls in the native MLIR bindings, which
    # simulator-only callers should not have to install.
    from hc._frontend import lower_function_to_front_ir

    module = lower_function_to_front_ir(kernel_fn)
    return CompiledKernel(
        kernel=kernel_fn,
        bindings=bindings,
        front_ir=module,
        front_ir_text=str(module),
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
        out[name] = value
    return out


def _symbol_name(obj: Any) -> str:
    # `Symbol` instances carry a `.name` str; tolerate bare strings so
    # `hc.compile(k, {"M": 1024})` works without importing `sym`.
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(obj, str):
        return obj
    raise TypeError(f"cannot interpret {obj!r} as a symbol name")
