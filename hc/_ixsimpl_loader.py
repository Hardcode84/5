# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

_LOCK = threading.Lock()
_SENTINEL_NAME = ".hc-ixsimpl-root"
_SOURCE_BOOTSTRAP_COMMAND = "python -m build_tools.ixsimpl_toolchain"


def load_ixsimpl() -> ModuleType:
    vendor_root = _vendor_root()
    module = _loaded_ixsimpl(vendor_root)
    if module is not None:
        return module

    with _LOCK:
        module = _loaded_ixsimpl(vendor_root)
        if module is not None:
            return module

        _require_local_build(vendor_root)
        _prepend_vendor_root(vendor_root)
        importlib.invalidate_caches()
        module = importlib.import_module("ixsimpl")
        _validate_loaded_module(module, vendor_root)
        return module


def ensure_ixsimpl_built() -> Path:
    try:
        from build_tools.ixsimpl_toolchain import (
            ensure_ixsimpl_built as _ensure_ixsimpl_built,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Explicit ixsimpl bootstrap is only available from a source "
            "checkout.\n"
            "Reinstall hc through the package build step instead."
        ) from exc

    return _ensure_ixsimpl_built()


def _require_local_build(vendor_root: Path) -> None:
    _validate_vendor_root(vendor_root)
    state, stamp = _installed_stamp(vendor_root)
    if state == "missing":
        raise _missing_build_error()
    if state == "invalid":
        raise _invalid_build_error()
    submodule_root = _submodule_root()
    if _submodule_available(submodule_root) and stamp != _expected_stamp(
        submodule_root
    ):
        raise _stale_build_error()


def _loaded_ixsimpl(vendor_root: Path) -> ModuleType | None:
    module = sys.modules.get("ixsimpl")
    if module is None:
        return None

    _validate_loaded_module(module, vendor_root)
    _require_local_build(vendor_root)
    return module


def _validate_loaded_module(module: ModuleType, vendor_root: Path) -> None:
    module_path = _module_path(module)
    if module_path is not None and module_path.is_relative_to(vendor_root.resolve()):
        return
    location = "<unknown>" if module_path is None else str(module_path)
    raise ImportError(
        "hc.symbols found a pre-imported `ixsimpl` outside the managed vendor "
        f"directory: {location}. Import `hc.symbols` before `ixsimpl`."
    )


def _module_path(module: ModuleType) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    return Path(module_file).resolve()


def _validate_vendor_root(vendor_root: Path) -> None:
    if not _vendor_root_overridden() or not vendor_root.exists():
        return
    if not vendor_root.is_dir():
        raise ImportError("HC_IXSIMPL_VENDOR_DIR must point to a directory")
    if _sentinel_path(vendor_root).exists():
        return
    if any(vendor_root.iterdir()):
        raise ImportError(
            "HC_IXSIMPL_VENDOR_DIR must be empty or already managed by hc"
        )


def _installed_stamp(vendor_root: Path) -> tuple[str, dict[str, str] | None]:
    stamp_path = _stamp_path(vendor_root)
    if not stamp_path.exists():
        return "missing", None
    try:
        raw = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None
    if not isinstance(raw, dict):
        return "invalid", None
    stamp = {str(key): str(value) for key, value in raw.items()}
    if not {"python", "source_rev"} <= stamp.keys():
        return "invalid", None
    return "valid", stamp


def _expected_stamp(submodule_root: Path) -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "source_rev": _source_revision(submodule_root),
    }


def _source_revision(submodule_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(submodule_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        return result.stdout.strip()

    source_file = submodule_root / "bindings" / "python" / "ixsimpl" / "__init__.py"
    return str(source_file.stat().st_mtime_ns)


def _prepend_vendor_root(vendor_root: Path) -> None:
    path_entry = str(vendor_root)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _submodule_root() -> Path:
    return _project_root() / "third_party" / "ixsimpl"


def _submodule_available(submodule_root: Path) -> bool:
    return (submodule_root / "pyproject.toml").exists()


def _vendor_root() -> Path:
    override = os.environ.get("HC_IXSIMPL_VENDOR_DIR")
    if override:
        return Path(override).resolve()
    return _project_root() / ".hc" / "vendor" / "ixsimpl"


def _vendor_root_overridden() -> bool:
    return "HC_IXSIMPL_VENDOR_DIR" in os.environ


def _stamp_path(vendor_root: Path) -> Path:
    return vendor_root / ".hc-ixsimpl-stamp.json"


def _sentinel_path(vendor_root: Path) -> Path:
    return vendor_root / _SENTINEL_NAME


def _missing_build_error() -> ImportError:
    if _submodule_available(_submodule_root()):
        return ImportError(
            "Vendored ixsimpl backend is not built.\n"
            "Install hc through the package build step. If you are working from "
            f"a source checkout, run `{_SOURCE_BOOTSTRAP_COMMAND}`."
        )
    return ImportError(
        "hc.symbols requires the ixsimpl submodule and a built vendored copy.\n"
        "Run `git submodule update --init --recursive` first, then reinstall hc "
        f"or, from a source checkout, run `{_SOURCE_BOOTSTRAP_COMMAND}`."
    )


def _invalid_build_error() -> ImportError:
    if _submodule_available(_submodule_root()):
        return ImportError(
            "Vendored ixsimpl backend metadata is invalid.\n"
            "Remove the vendored copy and rebuild it through the package build "
            f"step, or run `{_SOURCE_BOOTSTRAP_COMMAND}` from a source checkout."
        )
    return ImportError(
        "hc.symbols requires the ixsimpl submodule and a valid built vendored "
        "copy.\n"
        "Run `git submodule update --init --recursive` first, then reinstall hc "
        f"or, from a source checkout, run `{_SOURCE_BOOTSTRAP_COMMAND}`."
    )


def _stale_build_error() -> ImportError:
    return ImportError(
        "Vendored ixsimpl backend is stale for the current checkout.\n"
        "Reinstall hc through the package build step or, from a source "
        f"checkout, run `{_SOURCE_BOOTSTRAP_COMMAND}`, then restart the "
        "interpreter."
    )


def main() -> int:
    print(ensure_ixsimpl_built())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
