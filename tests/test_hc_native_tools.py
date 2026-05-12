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


def test_hc_native_tools_layout_uses_separate_package_build_cache(
    tmp_path: Path,
) -> None:
    llvm_install_root = _sample_llvm_install_root(tmp_path)
    layout = hc_native_tools_layout(
        llvm_install_root,
        project_root=tmp_path,
        package_build=True,
    )

    assert layout.build_root == layout.root / "package-build" / llvm_install_root.name
    assert layout.install_root == layout.root / "install" / llvm_install_root.name


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


def test_main_stages_package_native_artifacts(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """`python -m build_tools.hc_native_tools` populates `hc/_native/`.

    The CLI entry exists for source-tree devs who don't want to go
    through `pip install -e .` just to refresh the cmake install.
    Before this test landed, the entry stopped after the cmake install
    and left the package-relative `hc/_native/` tree to whatever a
    previous pip run had left behind — most often a half-populated
    state missing `ld.lld`, which surfaced five passes downstream as
    a confusing `hc-lower-gpu-to-binary` linker failure. The CLI must
    now invoke the same staging the wheel/editable build hooks use,
    so the source-tree workflow is self-contained.
    """
    llvm_install_root = tmp_path / "llvm-install" / "toolchain-key"
    native_install_root = tmp_path / "native-install"
    project_root = tmp_path / "project"
    # Stage lld into the fake llvm install so the staging step has
    # something to copy. The on-disk layout matches what the real
    # llvm bootstrap produces.
    (llvm_install_root / "bin").mkdir(parents=True)
    (llvm_install_root / "bin" / "ld.lld").write_text("lld\n", encoding="utf-8")
    # Fake out the heavy bootstrap calls — we don't need a real LLVM
    # toolchain or cmake invocation to exercise the staging surface.
    monkeypatch.setattr(
        hc_native_tools, "ensure_llvm_toolchain", lambda: llvm_install_root
    )
    monkeypatch.setattr(
        hc_native_tools,
        "ensure_hc_native_tools_built",
        lambda llvm_root: (
            _build_fake_native_install(native_install_root) or native_install_root
        ),
    )
    # `_project_root` is called inside `main` without args; redirect
    # it at the per-test fixture root so the staging lands somewhere
    # we can inspect without touching the real `hc/_native/`.
    monkeypatch.setattr(
        hc_native_tools, "_project_root", lambda *args, **kw: project_root
    )

    rc = hc_native_tools.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == str(native_install_root)
    package_native_root = project_root / "hc" / "_native"
    assert (package_native_root / "bin" / "hc-opt").read_text(
        encoding="utf-8"
    ) == "hc-opt\n"
    assert (package_native_root / "bin" / "ld.lld").read_text(
        encoding="utf-8"
    ) == "lld\n"
    assert (package_native_root / "manifest.json").exists()


def _build_fake_native_install(native_install_root: Path) -> None:
    """Lay out a minimal cmake-install tree for the staging step.

    Mirrors the bits `_validate_native_install` requires: `bin/hc-opt`
    and `python_packages/hc_front/hc_mlir/ir.py`. Everything else is
    optional from the staging side.
    """
    (native_install_root / "bin").mkdir(parents=True)
    (native_install_root / "bin" / "hc-opt").write_text("hc-opt\n", encoding="utf-8")
    hc_mlir = native_install_root / "python_packages" / "hc_front" / "hc_mlir"
    hc_mlir.mkdir(parents=True)
    (hc_mlir / "ir.py").write_text("IR = True\n", encoding="utf-8")
