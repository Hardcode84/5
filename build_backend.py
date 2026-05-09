# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
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
_PROJECT_ROOT = Path(__file__).resolve().parent
_PACKAGE_NATIVE_ROOT = _PROJECT_ROOT / "hc" / "_native"
_PACKAGE_NATIVE_SUBTREES = ("bin", "lib", "python_packages")


def _ensure_build_dependencies_bootstrapped() -> Path | None:
    global _IXSIMPL_BOOTSTRAPPED
    global _LLVM_BOOTSTRAPPED
    global _HC_NATIVE_BOOTSTRAPPED
    global _LLVM_INSTALL_ROOT
    global _HC_NATIVE_INSTALL_ROOT
    need_llvm = os.environ.get("HC_SKIP_LLVM_BOOTSTRAP") != "1"
    if _IXSIMPL_BOOTSTRAPPED and not need_llvm:
        return None
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
                    _LLVM_INSTALL_ROOT,
                    package_build=True,
                )
                _HC_NATIVE_BOOTSTRAPPED = True
            if _HC_NATIVE_INSTALL_ROOT is None:
                raise RuntimeError(
                    "hc native bootstrap completed without an install root"
                )
            export_hc_native_environment(_HC_NATIVE_INSTALL_ROOT, os.environ)
            return _HC_NATIVE_INSTALL_ROOT
        return None


def _install_package_native_artifacts(
    native_install_root: Path, llvm_install_root: Path | None = None
) -> None:
    _validate_native_install(native_install_root)
    if _PACKAGE_NATIVE_ROOT.exists():
        shutil.rmtree(_PACKAGE_NATIVE_ROOT)
    _PACKAGE_NATIVE_ROOT.mkdir(parents=True)
    for name in _PACKAGE_NATIVE_SUBTREES:
        source = native_install_root / name
        if source.exists():
            shutil.copytree(
                source,
                _PACKAGE_NATIVE_ROOT / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    if llvm_install_root is not None:
        _stage_lld_into_native_bin(llvm_install_root)
    _write_native_manifest(native_install_root)


def _stage_lld_into_native_bin(llvm_install_root: Path) -> None:
    # Copy the pinned `ld.lld` from the LLVM toolchain install into
    # `hc/_native/bin/ld.lld` so the runtime can resolve it via the
    # bundled-resource path returned by `_native_paths.lld_path`,
    # without needing the `HC_LLD` env var. The toolchain ships
    # `ld.lld` as a symlink to `lld`; dereference the symlink so wheel
    # installs work without preserving the link.
    src = llvm_install_root / "bin" / "ld.lld"
    if not src.exists():
        raise RuntimeError(f"hc llvm toolchain is missing ld.lld; expected at {src}")
    dest_bin = _PACKAGE_NATIVE_ROOT / "bin"
    dest_bin.mkdir(parents=True, exist_ok=True)
    dest = dest_bin / "ld.lld"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(src.resolve(), dest)


def _validate_native_install(native_install_root: Path) -> None:
    hc_opt = native_install_root / "bin" / "hc-opt"
    hc_mlir = native_install_root / "python_packages" / "hc_front" / "hc_mlir"
    missing = [str(path) for path in (hc_opt, hc_mlir / "ir.py") if not path.exists()]
    if missing:
        raise RuntimeError(
            "hc native install is incomplete; missing:\n" + "\n".join(missing)
        )


def _write_native_manifest(native_install_root: Path) -> None:
    manifest = {
        "source": str(native_install_root),
        "subtrees": list(_PACKAGE_NATIVE_SUBTREES),
    }
    (_PACKAGE_NATIVE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    native_install_root = _ensure_build_dependencies_bootstrapped()
    if native_install_root is not None:
        _install_package_native_artifacts(native_install_root, _LLVM_INSTALL_ROOT)
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
    native_install_root = _ensure_build_dependencies_bootstrapped()
    if native_install_root is not None:
        _install_package_native_artifacts(native_install_root, _LLVM_INSTALL_ROOT)
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
