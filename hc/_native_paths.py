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
    """Bundled runtime .so dir in the wheel."""
    return package_native_root() / "lib"


def runtime_helpers_lib_path() -> Path:
    """libhc_rt_helpers.so -- provides `_mlir_ciface_hc_get_*`.

    ctypes-loaded; addresses fed to the JIT. `HC_RT_HELPERS_PATH`
    overrides for source-tree dev.
    """
    override = os.environ.get("HC_RT_HELPERS_PATH")
    if override:
        return Path(override).resolve()
    return package_native_lib_dir() / "libhc_rt_helpers.so"


def hip_runtime_lib_path() -> Path:
    """libhc_hip_runtime.so -- `hc_rt_init / load_kernel / launch_kernel`.

    No build-time ROCm dep; `hc_rt_init` dlopens `libamdhip64.so`
    lazily.
    """
    override = os.environ.get("HC_RT_HIP_RUNTIME_PATH")
    if override:
        return Path(override).resolve()
    return package_native_lib_dir() / "libhc_hip_runtime.so"


def lld_path() -> Path:
    """ld.lld -- used by `hc-lower-gpu-to-binary` to link HSACO.

    Bundled at `hc/_native/bin/ld.lld`. `HC_LLD` overrides.
    """
    override = os.environ.get("HC_LLD")
    if override:
        return Path(override).resolve()
    return package_native_root() / "bin" / "ld.lld"
