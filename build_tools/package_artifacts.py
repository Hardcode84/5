# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stage the hc-native cmake install into the package-relative `hc/_native/`
tree the runtime actually loads from.

The cmake install lives under `.hc/native/install/<toolchain-key>/`
(driven by `build_tools.hc_native_tools`), but `hc._native_paths`
resolves `hc-opt`, the bundled `ld.lld`, and the MLIR Python bindings
relative to the installed `hc` package itself — `hc/_native/bin/`,
`hc/_native/lib/`, `hc/_native/python_packages/`. This module bridges
the two: copy the install tree's subtrees into the package, then
stage `ld.lld` next to `hc-opt` so `hc-lower-gpu-to-binary` can
resolve it via the bundled-resource path without needing `HC_LLD`.

Two callers exercise the staging:

  * `build_backend.build_wheel` / `build_editable` — the PEP 517
    hooks that pip drives during `pip install -e .` / `pip wheel`.
  * `python -m build_tools.hc_native_tools` — the README-documented
    explicit source-tree bootstrap; left to a cmake-only driver
    historically, dev environments ended up with `hc/_native/`
    half-populated (cmake install bits but no lld) and the WMMA
    pipeline failed at the linker stage. Routing the same staging
    through both call sites keeps the source-tree command
    idempotent and self-contained.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Subtrees mirrored verbatim from the cmake install into the package
# tree. `bin/` carries `hc-opt`, `lib/` ships the runtime helpers,
# `python_packages/` holds the generated MLIR Python bindings. Anything
# else the cmake install drops (e.g. cmake config files) is irrelevant
# at runtime and stays in the install root.
PACKAGE_NATIVE_SUBTREES = ("bin", "lib", "python_packages")


def install_package_native_artifacts(
    native_install_root: Path,
    llvm_install_root: Path | None = None,
    *,
    package_native_root: Path,
) -> None:
    """Refresh `package_native_root` from `native_install_root`.

    The package tree is rebuilt from scratch each call so a stale layer
    from a previous toolchain key can't shadow the current install. The
    manifest file written at the end records the source install root —
    `hc/_native/manifest.json` is the cheap "which toolchain produced
    this?" answer for diagnostics.

    `llvm_install_root` is optional: callers that don't bootstrap LLVM
    (sdist, metadata-only) skip the lld staging step rather than
    tripping its own validation. The `hc-lower-gpu-to-binary` pass
    surfaces an actionable diagnostic when the bundled lld is missing,
    so a partially-staged tree is debuggable rather than mysterious.
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
    # Copy the pinned `ld.lld` from the LLVM toolchain install into
    # `hc/_native/bin/ld.lld` so the runtime can resolve it via the
    # bundled-resource path returned by `_native_paths.lld_path`,
    # without needing the `HC_LLD` env var. The toolchain ships
    # `ld.lld` as a symlink to `lld`; dereference the symlink so wheel
    # installs work without preserving the link.
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
