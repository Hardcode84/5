# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from build_tools import llvm_toolchain
from build_tools.llvm_toolchain import ensure_llvm_toolchain

_LOCK = threading.Lock()
_LOCK_FILE_DIR = "locks"


@dataclass(frozen=True)
class HcNativeToolsLayout:
    project_root: Path
    root: Path
    build_root: Path
    install_root: Path
    lock_path: Path

    @property
    def hc_opt_path(self) -> Path:
        return self.install_root / "bin" / "hc-opt"


def ensure_hc_native_tools_built(
    llvm_install_root: Path | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    llvm_root = (
        ensure_llvm_toolchain() if llvm_install_root is None else llvm_install_root
    )
    layout = hc_native_tools_layout(llvm_root, project_root=project_root)
    with _LOCK:
        _ensure_hc_native_tools_built(layout, llvm_root)
    return layout.install_root


def export_hc_native_environment(
    install_root: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    exported = os.environ.copy() if env is None else env
    exported["HC_NATIVE_INSTALL_DIR"] = str(install_root)
    exported["HC_OPT_PATH"] = str(install_root / "bin" / "hc-opt")
    return exported


def hc_native_tools_layout(
    llvm_install_root: Path,
    *,
    project_root: Path | None = None,
) -> HcNativeToolsLayout:
    root = _project_root(project_root) / ".hc" / "native"
    key = llvm_install_root.name
    return HcNativeToolsLayout(
        project_root=_project_root(project_root),
        root=root,
        build_root=root / "build" / key,
        install_root=root / "install" / key,
        lock_path=root / _LOCK_FILE_DIR / f"{key}.lock",
    )


def main() -> int:
    print(ensure_hc_native_tools_built())
    return 0


def _ensure_hc_native_tools_built(
    layout: HcNativeToolsLayout,
    llvm_install_root: Path,
) -> None:
    with _build_lock(layout.lock_path):
        _configure_hc_native_tools(layout, llvm_install_root)
        _install_hc_native_tools(layout)
        if layout.hc_opt_path.exists():
            return
        raise RuntimeError(
            "hc native tools install completed without producing hc-opt.\n"
            f"expected: {layout.hc_opt_path}"
        )


def _configure_hc_native_tools(
    layout: HcNativeToolsLayout,
    llvm_install_root: Path,
) -> None:
    layout.build_root.mkdir(parents=True, exist_ok=True)
    _run_cmake(
        [
            "-S",
            str(layout.project_root),
            "-B",
            str(layout.build_root),
            *llvm_toolchain._cmake_generator_args(),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={layout.install_root}",
            f"-DLLVM_DIR={llvm_install_root / 'lib' / 'cmake' / 'llvm'}",
            f"-DMLIR_DIR={llvm_install_root / 'lib' / 'cmake' / 'mlir'}",
        ],
        cwd=layout.project_root,
    )


def _install_hc_native_tools(layout: HcNativeToolsLayout) -> None:
    _run_cmake(
        ["--build", str(layout.build_root), "--target", "install"],
        cwd=layout.project_root,
    )


@contextmanager
def _build_lock(lock_path: Path) -> Iterator[None]:
    with llvm_toolchain._build_lock(lock_path):
        yield


def _run_cmake(args: list[str], *, cwd: Path) -> None:
    command = [llvm_toolchain._cmake_executable(), *args]
    result = llvm_toolchain._run_command(command, cwd=cwd)
    if result.returncode == 0:
        return
    raise llvm_toolchain._command_error("cmake", command, cwd, result)


def _project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    raise SystemExit(main())
