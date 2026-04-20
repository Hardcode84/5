# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import threading
from pathlib import Path

from setuptools import build_meta as _build_meta

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.ixsimpl_toolchain import ensure_ixsimpl_built
from build_tools.llvm_toolchain import (
    ensure_llvm_toolchain,
    export_toolchain_environment,
)

_BOOTSTRAP_LOCK = threading.Lock()
_IXSIMPL_BOOTSTRAPPED = False
_LLVM_BOOTSTRAPPED = False
_HC_NATIVE_BOOTSTRAPPED = False
_LLVM_INSTALL_ROOT: Path | None = None
_HC_NATIVE_INSTALL_ROOT: Path | None = None


def _ensure_build_dependencies_bootstrapped() -> None:
    global _IXSIMPL_BOOTSTRAPPED
    global _LLVM_BOOTSTRAPPED
    global _HC_NATIVE_BOOTSTRAPPED
    global _LLVM_INSTALL_ROOT
    global _HC_NATIVE_INSTALL_ROOT
    need_llvm = os.environ.get("HC_SKIP_LLVM_BOOTSTRAP") != "1"
    if _IXSIMPL_BOOTSTRAPPED and not need_llvm:
        return
    with _BOOTSTRAP_LOCK:
        if not _IXSIMPL_BOOTSTRAPPED:
            ensure_ixsimpl_built()
            _IXSIMPL_BOOTSTRAPPED = True
        if need_llvm and not _LLVM_BOOTSTRAPPED:
            _LLVM_INSTALL_ROOT = ensure_llvm_toolchain()
            _LLVM_BOOTSTRAPPED = True
        if need_llvm:
            if _LLVM_INSTALL_ROOT is None:
                raise RuntimeError("LLVM bootstrap completed without an install root")
            export_toolchain_environment(_LLVM_INSTALL_ROOT, os.environ)
            if not _HC_NATIVE_BOOTSTRAPPED:
                _HC_NATIVE_INSTALL_ROOT = ensure_hc_native_tools_built(
                    _LLVM_INSTALL_ROOT
                )
                _HC_NATIVE_BOOTSTRAPPED = True
            if _HC_NATIVE_INSTALL_ROOT is None:
                raise RuntimeError(
                    "hc native bootstrap completed without an install root"
                )
            export_hc_native_environment(_HC_NATIVE_INSTALL_ROOT, os.environ)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_build_dependencies_bootstrapped()
    return _build_meta.build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_build_dependencies_bootstrapped()
    return _build_meta.build_editable(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


# Source archives and metadata discovery stay side-effect free. Only the wheel
# and editable build hooks need the managed native dependencies bootstrapped.
def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
) -> str:
    return _build_meta.build_sdist(
        sdist_directory,
        config_settings=config_settings,
    )


def get_requires_for_build_wheel(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return _build_meta.get_requires_for_build_wheel(config_settings=config_settings)


def get_requires_for_build_editable(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return _build_meta.get_requires_for_build_editable(config_settings=config_settings)


def get_requires_for_build_sdist(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return _build_meta.get_requires_for_build_sdist(config_settings=config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
) -> str:
    return _build_meta.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=config_settings,
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
) -> str:
    return _build_meta.prepare_metadata_for_build_editable(
        metadata_directory,
        config_settings=config_settings,
    )
