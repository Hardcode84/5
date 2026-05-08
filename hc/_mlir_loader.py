# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

from ._native_paths import package_mlir_python_package_dir

_LOCK = threading.Lock()
_PACKAGE_NAME = "hc_mlir"
_SOURCE_BOOTSTRAP_COMMAND = "python -m build_tools.hc_native_tools"


def load_hc_mlir() -> ModuleType:
    package_root = _package_root()
    module = _loaded_hc_mlir(package_root)
    if module is not None:
        return module

    with _LOCK:
        module = _loaded_hc_mlir(package_root)
        if module is not None:
            return module

        _require_local_build(package_root)
        _prepend_package_root(package_root)
        importlib.invalidate_caches()
        module = importlib.import_module(_PACKAGE_NAME)
        _validate_loaded_module(module, package_root)
        return module


def _package_root() -> Path:
    override = os.environ.get("HC_MLIR_PYTHON_PACKAGE_DIR")
    if override:
        return Path(override).resolve()
    package_root = package_mlir_python_package_dir()
    if _is_built_package_root(package_root):
        return package_root
    return _expected_package_root()


def _expected_package_root() -> Path:
    try:
        from build_tools import hc_native_tools, llvm_toolchain
    except ModuleNotFoundError as exc:
        raise ImportError(
            "hc.mlir requires the managed hc_front Python bindings from a source "
            "checkout.\n"
            "Set HC_MLIR_PYTHON_PACKAGE_DIR to a built package root or run "
            f"`{_SOURCE_BOOTSTRAP_COMMAND}` from the checkout root."
        ) from exc

    llvm_install_root = llvm_toolchain.llvm_toolchain_layout(
        llvm_toolchain.load_llvm_lock()
    ).install_root
    return hc_native_tools.hc_native_tools_layout(
        llvm_install_root
    ).mlir_python_package_dir


def _loaded_hc_mlir(package_root: Path) -> ModuleType | None:
    module = sys.modules.get(_PACKAGE_NAME)
    if module is None:
        return None

    _validate_loaded_module(module, package_root)
    _require_local_build(package_root)
    return module


def _validate_loaded_module(module: ModuleType, package_root: Path) -> None:
    module_paths = _module_paths(module)
    if module_paths and all(
        module_path.is_relative_to(package_root.resolve())
        for module_path in module_paths
    ):
        return
    location = (
        "<unknown>"
        if not module_paths
        else ", ".join(str(module_path) for module_path in module_paths)
    )
    raise ImportError(
        "hc.mlir found a pre-imported `hc_mlir` outside the managed package "
        f"directory: {location}. Import `hc.mlir` before `hc_mlir`."
    )


def _module_paths(module: ModuleType) -> list[Path]:
    module_file = getattr(module, "__file__", None)
    if module_file:
        return [Path(module_file).resolve()]

    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return []
    return [Path(entry).resolve() for entry in module_path]


def _require_local_build(package_root: Path) -> None:
    if _is_built_package_root(package_root):
        return
    raise _missing_build_error(package_root)


def _is_built_package_root(package_root: Path) -> bool:
    package_dir = package_root / _PACKAGE_NAME
    return package_root.is_dir() and (package_dir / "ir.py").exists()


def _prepend_package_root(package_root: Path) -> None:
    path_entry = str(package_root)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)


def _missing_build_error(package_root: Path) -> ImportError:
    if "HC_MLIR_PYTHON_PACKAGE_DIR" in os.environ:
        return ImportError(
            "HC_MLIR_PYTHON_PACKAGE_DIR does not point to a built hc_mlir package "
            "root.\n"
            f"expected: {package_root / _PACKAGE_NAME / 'ir.py'}"
        )
    return ImportError(
        "Packaged hc_front Python bindings are missing.\n"
        "Reinstall hc so the native MLIR artifacts are copied into the "
        "package, or set HC_MLIR_PYTHON_PACKAGE_DIR to a built package root.\n"
        f"Source checkouts can also run `{_SOURCE_BOOTSTRAP_COMMAND}`.\n"
        f"expected: {package_root / _PACKAGE_NAME / 'ir.py'}"
    )
