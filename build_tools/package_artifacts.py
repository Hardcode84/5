# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stage the hc-native cmake install into `hc/_native/`.

Copies cmake install subtrees into the package and drops `ld.lld` next
to `hc-opt` so `hc-lower-gpu-to-binary` finds it via `hc._native_paths`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PACKAGE_NATIVE_SUBTREES = ("bin", "lib", "python_packages")


def install_package_native_artifacts(
    native_install_root: Path,
    llvm_install_root: Path | None = None,
    *,
    package_native_root: Path,
) -> None:
    """Refresh `package_native_root` from `native_install_root`.

    Rebuilt from scratch each call -- prior toolchain keys would shadow
    the current install. `llvm_install_root=None` skips lld staging
    (sdist / metadata-only paths).
    """
    _validate_native_install(native_install_root)
    if package_native_root.exists():
        shutil.rmtree(package_native_root)
    package_native_root.mkdir(parents=True)
    for name in PACKAGE_NATIVE_SUBTREES:
        source = native_install_root / name
        if source.exists():
            shutil.copytree(
                source,
                package_native_root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    if llvm_install_root is not None:
        _stage_lld_into_native_bin(llvm_install_root, package_native_root)
    _write_native_manifest(native_install_root, package_native_root)


def _stage_lld_into_native_bin(
    llvm_install_root: Path, package_native_root: Path
) -> None:
    # `ld.lld` is a symlink to `lld`; dereference so wheels don't carry the link.
    src = llvm_install_root / "bin" / "ld.lld"
    if not src.exists():
        raise RuntimeError(f"hc llvm toolchain is missing ld.lld; expected at {src}")
    dest_bin = package_native_root / "bin"
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


def _write_native_manifest(
    native_install_root: Path, package_native_root: Path
) -> None:
    manifest = {
        "source": str(native_install_root),
        "subtrees": list(PACKAGE_NATIVE_SUBTREES),
    }
    (package_native_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
