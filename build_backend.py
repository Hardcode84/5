# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

from setuptools import build_meta as _build_meta

from build_tools.llvm_toolchain import (
    ensure_llvm_toolchain,
    export_toolchain_environment,
)

_BOOTSTRAPPED = False


def _ensure_toolchain_bootstrapped() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED or os.environ.get("HC_SKIP_LLVM_BOOTSTRAP") == "1":
        return
    install_root = ensure_llvm_toolchain()
    export_toolchain_environment(install_root, os.environ)
    _BOOTSTRAPPED = True


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_toolchain_bootstrapped()
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
    _ensure_toolchain_bootstrapped()
    return _build_meta.build_editable(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


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
