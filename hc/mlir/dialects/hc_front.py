# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from ..._mlir_loader import load_hc_mlir


def _public_names(module: ModuleType) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return [str(name) for name in exported]
    return [name for name in dir(module) if not name.startswith("_")]


load_hc_mlir()
_HC_FRONT_MODULE = importlib.import_module("hc_mlir.dialects.hc_front")
__all__ = _public_names(_HC_FRONT_MODULE)

for _exported_name in __all__:
    globals()[_exported_name] = getattr(_HC_FRONT_MODULE, _exported_name)


def __getattr__(name: str) -> Any:
    try:
        return getattr(_HC_FRONT_MODULE, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
