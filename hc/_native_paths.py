# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_NATIVE_DIR = "_native"
_MLIR_PACKAGE_RELATIVE = Path("python_packages") / "hc_front"


def package_native_root() -> Path:
    return Path(__file__).resolve().parent / _PACKAGE_NATIVE_DIR


def package_mlir_python_package_dir() -> Path:
    return package_native_root() / _MLIR_PACKAGE_RELATIVE


def hc_opt_path() -> Path:
    override = os.environ.get("HC_OPT_PATH")
    if override:
        return Path(override).resolve()
    return package_native_root() / "bin" / "hc-opt"


def package_native_lib_dir() -> Path:
    """Directory where bundled runtime shared libraries live in the wheel."""
    return package_native_root() / "lib"


def runtime_helpers_lib_path() -> Path:
    """libhc_rt_helpers.so — provides the `_mlir_ciface_hc_get_*` family.

    The helpers are loaded by `ctypes` and their symbol addresses are
    handed to the JIT execution engine, which resolves them when the host
    wrapper calls in. ``HC_RT_HELPERS_PATH`` overrides for source-tree
    development against a freshly-built install dir without going through
    a full wheel install.
    """
    override = os.environ.get("HC_RT_HELPERS_PATH")
    if override:
        return Path(override).resolve()
    return package_native_lib_dir() / "libhc_rt_helpers.so"
