# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None

fcntl: Any = _fcntl

_LOCK = threading.Lock()
_SENTINEL_NAME = ".hc-ixsimpl-root"
_STAMP_NAME = ".hc-ixsimpl-stamp.json"


@dataclass(frozen=True)
class IxSimplToolchainLayout:
    project_root: Path
    submodule_root: Path
    vendor_root: Path
    lock_path: Path


def ensure_ixsimpl_built(*, project_root: Path | None = None) -> Path:
    layout = ixsimpl_toolchain_layout(project_root=project_root)
    with _LOCK:
        _ensure_ixsimpl_built(layout)
    return layout.vendor_root


def ixsimpl_toolchain_layout(
    *, project_root: Path | None = None
) -> IxSimplToolchainLayout:
    root = _project_root(project_root)
    vendor_root = _vendor_root(root)
    return IxSimplToolchainLayout(
        project_root=root,
        submodule_root=root / "third_party" / "ixsimpl",
        vendor_root=vendor_root,
        lock_path=vendor_root.parent / f"{vendor_root.name}.lock",
    )


def expected_ixsimpl_stamp(submodule_root: Path) -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "source_rev": _source_revision(submodule_root),
    }


def installed_ixsimpl_stamp(vendor_root: Path) -> dict[str, str] | None:
    stamp_path = vendor_root / _STAMP_NAME
    if not stamp_path.exists():
        return None
    try:
        raw = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): str(value) for key, value in raw.items()}


def main() -> int:
    print(ensure_ixsimpl_built())
    return 0


def _ensure_ixsimpl_built(layout: IxSimplToolchainLayout) -> None:
    if not (layout.submodule_root / "pyproject.toml").exists():
        raise ImportError(
            "hc.symbols requires the ixsimpl submodule. "
            "Run `git submodule update --init --recursive` first."
        )
    with _build_lock(layout.lock_path):
        _validate_vendor_root(layout.vendor_root)
        if installed_ixsimpl_stamp(layout.vendor_root) == expected_ixsimpl_stamp(
            layout.submodule_root
        ):
            return
        _build_local_ixsimpl(layout)


def _build_local_ixsimpl(layout: IxSimplToolchainLayout) -> None:
    build_root = _temporary_build_root(layout.vendor_root)
    command = _install_command(layout.submodule_root, build_root)
    result = _run_install(command, layout.project_root)
    if result.returncode != 0:
        shutil.rmtree(build_root, ignore_errors=True)
        raise _build_error(command, result)
    _write_vendor_metadata(build_root, layout.submodule_root)
    _replace_vendor_root(build_root, layout.vendor_root)


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
def _build_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _temporary_build_root(vendor_root: Path) -> Path:
    vendor_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ixsimpl-", dir=vendor_root.parent))


def _run_install(
    command: list[str],
    project_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _replace_vendor_root(build_root: Path, vendor_root: Path) -> None:
    shutil.rmtree(vendor_root, ignore_errors=True)
    build_root.rename(vendor_root)


def _project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parent.parent


def _vendor_root(project_root: Path) -> Path:
    override = os.environ.get("HC_IXSIMPL_VENDOR_DIR")
    if override:
        return Path(override).resolve()
    return project_root / ".hc" / "vendor" / "ixsimpl"


def _vendor_root_overridden() -> bool:
    return "HC_IXSIMPL_VENDOR_DIR" in os.environ


def _sentinel_path(vendor_root: Path) -> Path:
    return vendor_root / _SENTINEL_NAME


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
    command: list[str],
    result: subprocess.CompletedProcess[str],
) -> ImportError:
    return ImportError(
        "Failed to build local ixsimpl dependency.\n"
        f"command: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _write_vendor_metadata(vendor_root: Path, submodule_root: Path) -> None:
    _sentinel_path(vendor_root).write_text("hc ixsimpl vendor root\n", encoding="utf-8")
    (vendor_root / _STAMP_NAME).write_text(
        json.dumps(expected_ixsimpl_stamp(submodule_root), sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
