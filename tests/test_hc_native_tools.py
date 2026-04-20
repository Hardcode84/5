# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from build_tools import hc_native_tools
from build_tools.hc_native_tools import (
    export_hc_native_environment,
    hc_native_tools_layout,
)


def _sample_llvm_install_root(tmp_path: Path) -> Path:
    return tmp_path / "llvm-install" / "toolchain-key"


def test_hc_native_tools_layout_uses_project_local_cache(tmp_path: Path) -> None:
    llvm_install_root = _sample_llvm_install_root(tmp_path)
    layout = hc_native_tools_layout(llvm_install_root, project_root=tmp_path)

    assert layout.project_root == tmp_path
    assert layout.root == tmp_path / ".hc" / "native"
    assert layout.build_root == layout.root / "build" / llvm_install_root.name
    assert layout.install_root == layout.root / "install" / llvm_install_root.name
    assert layout.hc_opt_path == layout.install_root / "bin" / "hc-opt"
    assert (
        layout.mlir_python_package_dir
        == layout.install_root / "python_packages" / "hc_front"
    )


def test_export_hc_native_environment_sets_tool_paths(tmp_path: Path) -> None:
    install_root = tmp_path / "native-install"
    env = export_hc_native_environment(install_root, {})

    assert env["HC_NATIVE_INSTALL_DIR"] == str(install_root)
    assert env["HC_OPT_PATH"] == str(install_root / "bin" / "hc-opt")
    assert env["HC_MLIR_PYTHON_PACKAGE_DIR"] == str(
        install_root / "python_packages" / "hc_front"
    )


def test_ensure_hc_native_tools_runs_cmake_and_installs_hc_opt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    llvm_install_root = _sample_llvm_install_root(tmp_path)
    layout = hc_native_tools_layout(llvm_install_root, project_root=tmp_path)
    calls: list[tuple[list[str], Path]] = []

    def fake_run_cmake(args: list[str], *, cwd: Path) -> None:
        calls.append((args, cwd))
        if args[:3] != ["--build", str(layout.build_root), "--target"]:
            return
        layout.hc_opt_path.parent.mkdir(parents=True, exist_ok=True)
        layout.hc_opt_path.write_text("hc-opt\n", encoding="utf-8")

    monkeypatch.setattr(hc_native_tools, "_run_cmake", fake_run_cmake)

    install_root = hc_native_tools.ensure_hc_native_tools_built(
        llvm_install_root,
        project_root=tmp_path,
    )

    assert install_root == layout.install_root
    assert calls[0][1] == tmp_path
    assert calls[0][0][:4] == ["-S", str(tmp_path), "-B", str(layout.build_root)]
    assert calls[1] == (
        ["--build", str(layout.build_root), "--target", "install"],
        tmp_path,
    )
    assert layout.hc_opt_path.read_text(encoding="utf-8") == "hc-opt\n"
