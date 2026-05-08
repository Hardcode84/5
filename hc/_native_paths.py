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
