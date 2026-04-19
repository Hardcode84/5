# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None

fcntl: Any = _fcntl

_LOCK = threading.Lock()
_SENTINEL_NAME = ".hc-ixsimpl-root"


def load_ixsimpl() -> ModuleType:
    vendor_root = _vendor_root()
    module = _loaded_ixsimpl(vendor_root)
    if module is not None:
        return module

    with _LOCK:
        module = _loaded_ixsimpl(vendor_root)
        if module is not None:
            return module

        _ensure_local_build(vendor_root)
        _prepend_vendor_root(vendor_root)
        importlib.invalidate_caches()
        module = importlib.import_module("ixsimpl")
        _validate_loaded_module(module, vendor_root)
        return module


def ensure_ixsimpl_built() -> Path:
    vendor_root = _vendor_root()
    with _LOCK:
        _ensure_local_build(vendor_root)
    return vendor_root


def _ensure_local_build(vendor_root: Path) -> None:
    submodule_root = _submodule_root()
    if not (submodule_root / "pyproject.toml").exists():
        raise ImportError(
            "hc.symbols requires the ixsimpl submodule. "
            "Run `git submodule update --init --recursive` first."
        )

    with _build_lock(vendor_root):
        _validate_vendor_root(vendor_root)
        if _installed_stamp(vendor_root) == _expected_stamp(submodule_root):
            return

        _build_local_ixsimpl(submodule_root, vendor_root)


def _build_local_ixsimpl(submodule_root: Path, vendor_root: Path) -> None:
    build_root = _temporary_build_root(vendor_root)
    command = _install_command(submodule_root, build_root)
    result = _run_install(command)
    if result.returncode != 0:
        shutil.rmtree(build_root, ignore_errors=True)
        raise _build_error(command, result)

    _write_metadata(build_root, submodule_root)
    _replace_vendor_root(build_root, vendor_root)


def _loaded_ixsimpl(vendor_root: Path) -> ModuleType | None:
    module = sys.modules.get("ixsimpl")
    if module is None:
        return None

    _validate_loaded_module(module, vendor_root)
    submodule_root = _submodule_root()
    if (submodule_root / "pyproject.toml").exists():
        if _installed_stamp(vendor_root) != _expected_stamp(submodule_root):
            raise ImportError(
                "vendored ixsimpl is stale for the current checkout. "
                "Rebuild it and restart the interpreter."
            )
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


@contextmanager
def _build_lock(vendor_root: Path) -> Iterator[None]:
    lock_path = _lock_path(vendor_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _installed_stamp(vendor_root: Path) -> dict[str, str] | None:
    stamp_path = _stamp_path(vendor_root)
    if not stamp_path.exists():
        return None
    try:
        raw = json.loads(stamp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): str(value) for key, value in raw.items()}


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


def _temporary_build_root(vendor_root: Path) -> Path:
    vendor_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ixsimpl-", dir=vendor_root.parent))


def _run_install(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=_project_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def _replace_vendor_root(build_root: Path, vendor_root: Path) -> None:
    shutil.rmtree(vendor_root, ignore_errors=True)
    build_root.rename(vendor_root)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _submodule_root() -> Path:
    return _project_root() / "third_party" / "ixsimpl"


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


def _lock_path(vendor_root: Path) -> Path:
    return vendor_root.parent / f"{vendor_root.name}.lock"


def _install_command(submodule_root: Path, vendor_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "--target",
        str(vendor_root),
        str(submodule_root),
    ]


def _build_error(
    command: list[str], result: subprocess.CompletedProcess[str]
) -> ImportError:
    return ImportError(
        "failed to build local ixsimpl dependency.\n"
        f"command: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _write_metadata(vendor_root: Path, submodule_root: Path) -> None:
    _sentinel_path(vendor_root).write_text("hc ixsimpl vendor root\n", encoding="utf-8")
    _write_stamp(vendor_root, submodule_root)


def _write_stamp(vendor_root: Path, submodule_root: Path) -> None:
    _stamp_path(vendor_root).write_text(
        json.dumps(_expected_stamp(submodule_root), sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    print(ensure_ixsimpl_built())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
