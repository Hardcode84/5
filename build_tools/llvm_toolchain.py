# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
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
_LOCK_FILE_DIR = "locks"
_SENTINEL_NAME = ".hc-llvm-toolchain-root"
_STAMP_NAME = ".hc-llvm-toolchain-stamp.json"


@dataclass(frozen=True)
class LlvmLock:
    repo: str
    revision: str
    projects: tuple[str, ...]
    targets: tuple[str, ...]
    build_type: str
    build_shared_libs: bool
    enable_assertions: bool
    enable_rtti: bool
    install_utils: bool
    enable_python_bindings: bool


@dataclass(frozen=True)
class LlvmToolchainLayout:
    root: Path
    source_root: Path
    build_root: Path
    install_root: Path
    staging_root: Path
    lock_path: Path

    @property
    def llvm_cmake_dir(self) -> Path:
        return self.install_root / "lib" / "cmake" / "llvm"

    @property
    def mlir_cmake_dir(self) -> Path:
        return self.install_root / "lib" / "cmake" / "mlir"


def ensure_llvm_toolchain() -> Path:
    lock = load_llvm_lock()
    layout = llvm_toolchain_layout(lock)
    with _LOCK:
        _ensure_llvm_toolchain(lock, layout)
    return layout.install_root


def export_toolchain_environment(
    install_root: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    exported = os.environ.copy() if env is None else env
    exported["HC_LLVM_INSTALL_DIR"] = str(install_root)
    exported["LLVM_DIR"] = str(install_root / "lib" / "cmake" / "llvm")
    exported["MLIR_DIR"] = str(install_root / "lib" / "cmake" / "mlir")
    return exported


def load_llvm_lock(path: Path | None = None) -> LlvmLock:
    lock_path = _lock_path(path)
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    return LlvmLock(
        repo=_require_string(raw, "repo"),
        revision=_require_string(raw, "revision"),
        projects=_require_string_list(raw, "projects"),
        targets=_require_string_list(raw, "targets"),
        build_type=_require_string(raw, "build_type"),
        build_shared_libs=_require_bool(raw, "build_shared_libs"),
        enable_assertions=_require_bool(raw, "enable_assertions"),
        enable_rtti=_require_bool(raw, "enable_rtti"),
        install_utils=_require_bool(raw, "install_utils"),
        enable_python_bindings=_require_bool(raw, "enable_python_bindings"),
    )


def llvm_toolchain_layout(
    lock: LlvmLock,
    *,
    project_root: Path | None = None,
) -> LlvmToolchainLayout:
    root = _cache_root(project_root)
    key = toolchain_key(lock)
    source_root = root / "src" / f"llvm-project-{lock.revision[:12]}"
    build_root = root / "build" / key
    install_root = root / "install" / key
    staging_root = root / "staging" / key
    lock_path = root / _LOCK_FILE_DIR / f"{key}.lock"
    return LlvmToolchainLayout(
        root=root,
        source_root=source_root,
        build_root=build_root,
        install_root=install_root,
        staging_root=staging_root,
        lock_path=lock_path,
    )


def toolchain_key(lock: LlvmLock) -> str:
    parts = [
        lock.revision[:12],
        lock.build_type.lower(),
        "shared" if lock.build_shared_libs else "static",
        "assert" if lock.enable_assertions else "noassert",
        "rtti" if lock.enable_rtti else "nortti",
        "py" if lock.enable_python_bindings else "nopy",
        _slug_join(lock.projects),
        _slug_join(lock.targets),
    ]
    return "-".join(parts)


def expected_toolchain_stamp(lock: LlvmLock) -> dict[str, object]:
    return {
        "repo": lock.repo,
        "revision": lock.revision,
        "projects": list(lock.projects),
        "targets": list(lock.targets),
        "build_type": lock.build_type,
        "build_shared_libs": lock.build_shared_libs,
        "enable_assertions": lock.enable_assertions,
        "enable_rtti": lock.enable_rtti,
        "install_utils": lock.install_utils,
        "enable_python_bindings": lock.enable_python_bindings,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def toolchain_is_current(lock: LlvmLock, layout: LlvmToolchainLayout) -> bool:
    if _force_rebuild_requested():
        return False
    return _toolchain_root_is_current(lock, layout.install_root)


def main() -> int:
    print(ensure_llvm_toolchain())
    return 0


def _ensure_llvm_toolchain(lock: LlvmLock, layout: LlvmToolchainLayout) -> None:
    with _build_lock(layout.lock_path):
        if toolchain_is_current(lock, layout):
            return
        if _staging_toolchain_is_current(lock, layout):
            _replace_install_root(layout)
            return
        _ensure_source_checkout(lock, layout.source_root)
        _build_llvm_toolchain(lock, layout)


def _build_llvm_toolchain(lock: LlvmLock, layout: LlvmToolchainLayout) -> None:
    shutil.rmtree(layout.staging_root, ignore_errors=True)
    shutil.rmtree(layout.build_root, ignore_errors=True)
    layout.staging_root.mkdir(parents=True, exist_ok=True)
    layout.build_root.mkdir(parents=True, exist_ok=True)
    _configure_llvm_build(lock, layout)
    _install_llvm_build(layout)
    _write_toolchain_metadata(lock, layout.staging_root)
    _replace_install_root(layout)


def _configure_llvm_build(lock: LlvmLock, layout: LlvmToolchainLayout) -> None:
    args = _cmake_generator_args()
    args.extend(_llvm_cmake_args(lock, layout.staging_root))
    _run_cmake(
        [str(layout.source_root / "llvm"), *args],
        cwd=layout.build_root,
    )


def _install_llvm_build(layout: LlvmToolchainLayout) -> None:
    _run_cmake(["--build", ".", "--target", "install"], cwd=layout.build_root)


def _write_toolchain_metadata(lock: LlvmLock, install_root: Path) -> None:
    _sentinel_path(install_root).write_text(
        "hc llvm toolchain root\n",
        encoding="utf-8",
    )
    _stamp_path(install_root).write_text(
        json.dumps(expected_toolchain_stamp(lock), sort_keys=True),
        encoding="utf-8",
    )


def _replace_install_root(layout: LlvmToolchainLayout) -> None:
    if not layout.staging_root.exists():
        raise RuntimeError(
            "LLVM bootstrap staging directory is missing after install.\n"
            f"expected: {layout.staging_root}"
        )
    layout.install_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(layout.install_root, ignore_errors=True)
    layout.staging_root.rename(layout.install_root)


def _ensure_source_checkout(lock: LlvmLock, source_root: Path) -> None:
    if not source_root.exists():
        _clone_source_checkout(lock, source_root)
    _validate_source_checkout(lock, source_root)
    if _checked_out_revision(source_root) == lock.revision:
        return
    _run_git(["fetch", "--depth", "1", "origin", lock.revision], cwd=source_root)
    _run_git(["checkout", "--force", lock.revision], cwd=source_root)


def _clone_source_checkout(lock: LlvmLock, source_root: Path) -> None:
    source_root.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            lock.repo,
            str(source_root),
        ],
        cwd=source_root.parent,
    )


def _validate_source_checkout(lock: LlvmLock, source_root: Path) -> None:
    if not (source_root / ".git").exists():
        raise RuntimeError(
            f"cached LLVM checkout is not a git repository: {source_root}"
        )
    remote = _git_stdout(["remote", "get-url", "origin"], cwd=source_root)
    if remote == lock.repo:
        return
    raise RuntimeError(
        "cached LLVM checkout points at the wrong remote.\n"
        f"expected: {lock.repo}\n"
        f"actual: {remote}\n"
        f"remove {source_root} and retry."
    )


def _checked_out_revision(source_root: Path) -> str | None:
    try:
        return _git_stdout(["rev-parse", "HEAD"], cwd=source_root)
    except RuntimeError:
        return None


def _git_stdout(args: list[str], *, cwd: Path) -> str:
    result = _run_command(["git", *args], cwd=cwd)
    if result.returncode == 0:
        return result.stdout.strip()
    raise _command_error("git", ["git", *args], cwd, result)


def _run_git(args: list[str], *, cwd: Path) -> None:
    command = ["git", *args]
    result = _run_command(command, cwd=cwd)
    if result.returncode == 0:
        return
    raise _command_error("git", command, cwd, result)


def _run_cmake(args: list[str], *, cwd: Path) -> None:
    command = [_cmake_executable(), *args]
    result = _run_command(command, cwd=cwd)
    if result.returncode == 0:
        return
    raise _command_error("cmake", command, cwd, result)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _command_error(
    kind: str,
    command: list[str],
    cwd: Path,
    result: subprocess.CompletedProcess[str],
) -> RuntimeError:
    return RuntimeError(
        f"failed to run {kind} command.\n"
        f"cwd: {cwd}\n"
        f"command: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _cmake_executable() -> str:
    path = _scripts_or_path_executable("cmake")
    if path is not None:
        return path
    try:
        import cmake
    except Exception:  # pragma: no cover - fallback only
        cmake = None
    if cmake is not None:
        return str(Path(cmake.CMAKE_BIN_DIR) / "cmake")
    raise RuntimeError("cmake is required to bootstrap the pinned LLVM toolchain")


def _cmake_generator_args() -> list[str]:
    ninja_path = _ninja_executable()
    if ninja_path is None:
        return []
    return ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_path}"]


def _ninja_executable() -> str | None:
    path = _scripts_or_path_executable("ninja")
    if path is not None:
        return path
    try:
        import ninja
    except Exception:  # pragma: no cover - fallback only
        ninja = None
    if ninja is not None:
        return str(Path(ninja.BIN_DIR) / "ninja")
    return None


def _scripts_or_path_executable(name: str) -> str | None:
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        path = shutil.which(name, path=scripts_dir)
        if path is not None:
            return path
    return shutil.which(name)


def _llvm_cmake_args(lock: LlvmLock, install_root: Path) -> list[str]:
    return [
        f"-DCMAKE_INSTALL_PREFIX={install_root}",
        f"-DCMAKE_BUILD_TYPE={lock.build_type}",
        f"-DLLVM_ENABLE_PROJECTS={';'.join(lock.projects)}",
        f"-DLLVM_TARGETS_TO_BUILD={';'.join(lock.targets)}",
        f"-DBUILD_SHARED_LIBS={_cmake_bool(lock.build_shared_libs)}",
        f"-DLLVM_ENABLE_ASSERTIONS={_cmake_bool(lock.enable_assertions)}",
        f"-DLLVM_ENABLE_RTTI={_cmake_bool(lock.enable_rtti)}",
        f"-DLLVM_INSTALL_UTILS={_cmake_bool(lock.install_utils)}",
        "-DLLVM_ENABLE_ZSTD=OFF",
        "-DLLVM_ENABLE_TERMINFO=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        f"-DMLIR_ENABLE_BINDINGS_PYTHON={_cmake_bool(lock.enable_python_bindings)}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]


def _cmake_bool(value: bool) -> str:
    return "ON" if value else "OFF"


def _lock_path(path: Path | None) -> Path:
    if path is not None:
        return path
    return _project_root() / "third_party" / "llvm.lock.json"


def _cache_root(project_root: Path | None) -> Path:
    override = os.environ.get("HC_LLVM_CACHE_DIR")
    if override:
        return Path(override).resolve()
    base = _project_root() if project_root is None else project_root.resolve()
    return base / ".hc" / "toolchains" / "llvm"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _config_files_exist(layout: LlvmToolchainLayout) -> bool:
    return _config_files_exist_under(layout.install_root)


def _config_files_exist_under(root: Path) -> bool:
    return _llvm_config_path(root).exists() and _mlir_config_path(root).exists()


def _llvm_config_path(root: Path) -> Path:
    return root / "lib" / "cmake" / "llvm" / "LLVMConfig.cmake"


def _mlir_config_path(root: Path) -> Path:
    return root / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake"


def _installed_stamp(install_root: Path) -> dict[str, object] | None:
    stamp_path = _stamp_path(install_root)
    if not stamp_path.exists():
        return None
    try:
        raw = json.loads(stamp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _toolchain_root_is_current(lock: LlvmLock, root: Path) -> bool:
    if _installed_stamp(root) != expected_toolchain_stamp(lock):
        return False
    if not _sentinel_path(root).exists():
        return False
    return _config_files_exist_under(root)


def _staging_toolchain_is_current(lock: LlvmLock, layout: LlvmToolchainLayout) -> bool:
    if _force_rebuild_requested():
        return False
    return _toolchain_root_is_current(lock, layout.staging_root)


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


def _force_rebuild_requested() -> bool:
    return os.environ.get("HC_LLVM_FORCE_REBUILD") == "1"


def _sentinel_path(install_root: Path) -> Path:
    return install_root / _SENTINEL_NAME


def _stamp_path(install_root: Path) -> Path:
    return install_root / _STAMP_NAME


def _require_string(raw: object, key: str) -> str:
    if isinstance(raw, dict):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    raise RuntimeError(f"llvm lock file field {key!r} must be a non-empty string")


def _require_bool(raw: object, key: str) -> bool:
    if isinstance(raw, dict):
        value = raw.get(key)
        if isinstance(value, bool):
            return value
    raise RuntimeError(f"llvm lock file field {key!r} must be a boolean")


def _require_string_list(raw: object, key: str) -> tuple[str, ...]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"llvm lock file field {key!r} must be a string list")
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"llvm lock file field {key!r} must be a string list")
    if all(isinstance(item, str) and item for item in value):
        return tuple(value)
    raise RuntimeError(f"llvm lock file field {key!r} must be a string list")


def _slug_join(values: tuple[str, ...]) -> str:
    return "_".join(_slug(value) for value in values)


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
