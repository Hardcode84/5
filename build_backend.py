# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import threading

from setuptools import build_meta as _build_meta

from build_tools.ixsimpl_toolchain import ensure_ixsimpl_built
from build_tools.llvm_toolchain import (
    ensure_llvm_toolchain,
    export_toolchain_environment,
)

_BOOTSTRAP_LOCK = threading.Lock()
_IXSIMPL_BOOTSTRAPPED = False
_LLVM_BOOTSTRAPPED = False


def _ensure_build_dependencies_bootstrapped() -> None:
    global _IXSIMPL_BOOTSTRAPPED
    global _LLVM_BOOTSTRAPPED
    need_llvm = os.environ.get("HC_SKIP_LLVM_BOOTSTRAP") != "1"
    if _IXSIMPL_BOOTSTRAPPED and (_LLVM_BOOTSTRAPPED or not need_llvm):
        return
    with _BOOTSTRAP_LOCK:
        if not _IXSIMPL_BOOTSTRAPPED:
            ensure_ixsimpl_built()
            _IXSIMPL_BOOTSTRAPPED = True
        if need_llvm and not _LLVM_BOOTSTRAPPED:
            install_root = ensure_llvm_toolchain()
            export_toolchain_environment(install_root, os.environ)
            _LLVM_BOOTSTRAPPED = True


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
