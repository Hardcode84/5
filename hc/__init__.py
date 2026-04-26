# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ._compile import CompiledKernel, compile
from .core import (
    Buffer,
    BufferSpec,
    CurrentGroup,
    IdxTypeSpec,
    IndexMap,
    Result,
    Scope,
    SubGroup,
    TensorTypeSpec,
    UndefTypeSpec,
    VectorTypeSpec,
    WorkGroup,
    WorkItem,
    as_layout,
    idx_type,
    index_map,
    kernel,
    tensor_type,
    undef_type,
    vector_type,
)

if TYPE_CHECKING:
    from . import simulator, symbols
    from .symbols import Symbol

__all__ = [
    "Buffer",
    "BufferSpec",
    "CompiledKernel",
    "CurrentGroup",
    "IdxTypeSpec",
    "IndexMap",
    "Result",
    "Scope",
    "SubGroup",
    "Symbol",
    "TensorTypeSpec",
    "UndefTypeSpec",
    "VectorTypeSpec",
    "WorkGroup",
    "WorkItem",
    "as_layout",
    "compile",
    "idx_type",
    "index_map",
    "kernel",
    "simulator",
    "sym",
    "symbols",
    "tensor_type",
    "undef_type",
    "vector_type",
]

_SYMBOLS_MODULE: ModuleType | None = None
_SIMULATOR_MODULE: ModuleType | None = None


def _load_symbols_module() -> ModuleType:
    global _SYMBOLS_MODULE
    if _SYMBOLS_MODULE is None:
        _SYMBOLS_MODULE = importlib.import_module(".symbols", __name__)
    return _SYMBOLS_MODULE


def _load_simulator_module() -> ModuleType:
    global _SIMULATOR_MODULE
    if _SIMULATOR_MODULE is None:
        _SIMULATOR_MODULE = importlib.import_module(".simulator", __name__)
    return _SIMULATOR_MODULE


def __getattr__(name: str) -> Any:
    if name in {"Symbol", "sym", "symbols"}:
        _symbols = _load_symbols_module()

        if name == "symbols":
            return _symbols
        return getattr(_symbols, name)
    if name == "simulator":
        return _load_simulator_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
