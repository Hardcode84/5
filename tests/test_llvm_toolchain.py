# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from build_tools import llvm_toolchain
from build_tools.llvm_toolchain import (
    LlvmLock,
    export_toolchain_environment,
    llvm_toolchain_layout,
    load_llvm_lock,
    toolchain_is_current,
    toolchain_key,
)

# White-box bootstrap tests intentionally exercise private file-state helpers
# that are not part of the runtime-facing API.


def _sample_lock(
    *, revision: str = "0123456789abcdef0123456789abcdef01234567"
) -> LlvmLock:
    return LlvmLock(
        repo="https://example.com/llvm-project.git",
        revision=revision,
        projects=("mlir",),
        targets=("host", "AMDGPU"),
        build_type="Release",
        build_shared_libs=False,
        enable_assertions=True,
        enable_rtti=True,
        install_utils=True,
        enable_python_bindings=False,
    )


def _write_fake_toolchain(root: Path, lock: LlvmLock) -> None:
    (root / "lib" / "cmake" / "llvm").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "cmake" / "mlir").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "cmake" / "llvm" / "LLVMConfig.cmake").write_text(
        "",
        encoding="utf-8",
    )
    (root / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake").write_text(
        "",
        encoding="utf-8",
    )
    llvm_toolchain._write_toolchain_metadata(lock, root)


def _assert_path_missing(path: Path, message: str) -> None:
    if path.exists():
        raise AssertionError(message)


def _write_installed_marker(layout: llvm_toolchain.LlvmToolchainLayout) -> None:
    (layout.staging_root / "installed.txt").write_text("ok\n", encoding="utf-8")


def test_load_llvm_lock_reads_repo_pin() -> None:
    lock = load_llvm_lock()

    assert lock.repo == "https://github.com/llvm/llvm-project.git"
    assert len(lock.revision) == 40
    assert lock.projects == ("mlir",)
    assert lock.targets == ("host", "AMDGPU")


def test_toolchain_layout_uses_project_local_cache_dirs(tmp_path: Path) -> None:
    layout = llvm_toolchain_layout(_sample_lock(), project_root=tmp_path)

    assert layout.root == tmp_path / ".hc" / "toolchains" / "llvm"
    assert layout.source_root == layout.root / "src" / "llvm-project-0123456789ab"
    assert layout.build_root.parent == layout.root / "build"
    assert layout.install_root.parent == layout.root / "install"


def test_toolchain_key_tracks_build_configuration() -> None:
    base = _sample_lock()
    debug = LlvmLock(**{**base.__dict__, "build_type": "Debug"})

    assert toolchain_key(base) != toolchain_key(debug)


def test_toolchain_is_current_requires_stamp_and_cmake_configs(tmp_path: Path) -> None:
    lock = _sample_lock()
    layout = llvm_toolchain_layout(lock, project_root=tmp_path)

    assert not toolchain_is_current(lock, layout)

    layout.llvm_cmake_dir.mkdir(parents=True, exist_ok=True)
    layout.mlir_cmake_dir.mkdir(parents=True, exist_ok=True)
    (layout.llvm_cmake_dir / "LLVMConfig.cmake").write_text("", encoding="utf-8")
    (layout.mlir_cmake_dir / "MLIRConfig.cmake").write_text("", encoding="utf-8")
    llvm_toolchain._write_toolchain_metadata(lock, layout.install_root)

    assert toolchain_is_current(lock, layout)


def test_export_toolchain_environment_sets_mlir_config_dirs(tmp_path: Path) -> None:
    install_root = tmp_path / "install"
    env = export_toolchain_environment(install_root, {})

    assert env["HC_LLVM_INSTALL_DIR"] == str(install_root)
    assert env["LLVM_DIR"] == str(install_root / "lib" / "cmake" / "llvm")
    assert env["MLIR_DIR"] == str(install_root / "lib" / "cmake" / "mlir")


def test_replace_install_root_creates_missing_install_parent(tmp_path: Path) -> None:
    lock = _sample_lock()
    layout = llvm_toolchain_layout(lock, project_root=tmp_path)
    marker = layout.staging_root / "marker.txt"

    layout.staging_root.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")

    llvm_toolchain._replace_install_root(layout)

    assert not layout.staging_root.exists()
    assert (layout.install_root / "marker.txt").read_text(encoding="utf-8") == "ok\n"


def test_build_llvm_toolchain_clears_stale_build_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lock = _sample_lock()
    layout = llvm_toolchain_layout(lock, project_root=tmp_path)
    stale_marker = layout.build_root / "stale.txt"

    layout.build_root.mkdir(parents=True, exist_ok=True)
    stale_marker.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(
        llvm_toolchain,
        "_configure_llvm_build",
        lambda lock, layout: _assert_path_missing(
            stale_marker,
            "stale build root reused",
        ),
    )
    monkeypatch.setattr(llvm_toolchain, "_install_llvm_build", _write_installed_marker)

    llvm_toolchain._build_llvm_toolchain(lock, layout)

    assert layout.build_root.exists()
    assert not stale_marker.exists()
    assert (layout.install_root / "installed.txt").read_text(encoding="utf-8") == "ok\n"


def test_ensure_llvm_toolchain_promotes_current_staging_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lock = _sample_lock()
    layout = llvm_toolchain_layout(lock, project_root=tmp_path)
    _write_fake_toolchain(layout.staging_root, lock)

    monkeypatch.setattr(
        llvm_toolchain,
        "_ensure_source_checkout",
        lambda lock, source_root: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    monkeypatch.setattr(
        llvm_toolchain,
        "_build_llvm_toolchain",
        lambda lock, layout: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    llvm_toolchain._ensure_llvm_toolchain(lock, layout)

    assert layout.install_root.exists()
    assert not layout.staging_root.exists()


def test_scripts_or_path_executable_prefers_python_scripts_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    cmake_path = scripts_dir / "cmake"
    cmake_path.write_text("", encoding="utf-8")

    def fake_which(name: str, path: str | None = None) -> str | None:
        if name == "cmake" and path == str(scripts_dir):
            return str(cmake_path)
        if name == "cmake" and path is None:
            return "/usr/bin/cmake"
        return None

    monkeypatch.setattr(
        llvm_toolchain.sysconfig, "get_path", lambda key: str(scripts_dir)
    )
    monkeypatch.setattr(llvm_toolchain.shutil, "which", fake_which)

    assert llvm_toolchain._scripts_or_path_executable("cmake") == str(cmake_path)
