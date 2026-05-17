# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_backend


def _reset_bootstrap_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_backend, "_IXSIMPL_BOOTSTRAPPED", False)
    monkeypatch.setattr(build_backend, "_LLVM_BOOTSTRAPPED", False)
    monkeypatch.setattr(build_backend, "_HC_NATIVE_BOOTSTRAPPED", False)
    monkeypatch.setattr(build_backend, "_LLVM_INSTALL_ROOT", None)
    monkeypatch.setattr(build_backend, "_HC_NATIVE_INSTALL_ROOT", None)


def _unexpected_bootstrap() -> Path:
    raise AssertionError("unexpected")


def _record_ixsimpl_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[object],
) -> None:
    monkeypatch.setattr(
        build_backend,
        "ensure_ixsimpl_built",
        lambda: calls.append("ixsimpl"),
    )


def _record_llvm_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[object],
    install_root: Path,
) -> None:
    monkeypatch.setattr(
        build_backend,
        "ensure_llvm_toolchain",
        lambda: install_root,
    )
    monkeypatch.setattr(
        build_backend,
        "export_toolchain_environment",
        lambda root, env: calls.append(("llvm", root)),
    )


def _record_hc_native_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[object],
    install_root: Path,
) -> None:
    monkeypatch.setattr(
        build_backend,
        "ensure_hc_native_tools_built",
        lambda llvm_install_root, *, package_build=False: calls.append(
            ("native-build", llvm_install_root, package_build)
        )
        or install_root,
    )
    monkeypatch.setattr(
        build_backend,
        "export_hc_native_environment",
        lambda root, env: calls.append(("native-env", root)),
    )


def _record_package_native_install(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[object],
) -> None:
    monkeypatch.setattr(
        build_backend,
        "_install_package_native_artifacts",
        lambda root, llvm_root=None: calls.append(("native-package", root, llvm_root)),
    )


def _record_directory_build_hook(
    monkeypatch: pytest.MonkeyPatch,
    hook_name: str,
    calls: list[object],
    event: str,
    result: str,
) -> None:
    monkeypatch.setattr(
        build_backend._build_meta,
        hook_name,
        lambda wheel_directory, config_settings=None, metadata_directory=None: (
            calls.append((event, wheel_directory)) or result
        ),
    )


def _record_build_hook(
    monkeypatch: pytest.MonkeyPatch,
    hook_name: str,
    calls: list[object],
    event: str,
    result: str,
) -> None:
    monkeypatch.setattr(
        build_backend._build_meta,
        hook_name,
        lambda wheel_directory, config_settings=None, metadata_directory=None: (
            calls.append(event) or result
        ),
    )


def test_build_wheel_bootstraps_ixsimpl_llvm_and_native_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    llvm_install_root = tmp_path / "llvm-install"
    native_install_root = tmp_path / "native-install"
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.delenv("HC_SKIP_LLVM_BOOTSTRAP", raising=False)
    _record_ixsimpl_bootstrap(monkeypatch, calls)
    _record_llvm_bootstrap(monkeypatch, calls, llvm_install_root)
    _record_hc_native_bootstrap(monkeypatch, calls, native_install_root)
    _record_package_native_install(monkeypatch, calls)
    _record_directory_build_hook(
        monkeypatch,
        "build_wheel",
        calls,
        "wheel",
        "hc.whl",
    )
    result = build_backend.build_wheel(str(tmp_path))
    assert result == "hc.whl"
    assert calls == [
        "ixsimpl",
        ("llvm", llvm_install_root),
        ("native-build", llvm_install_root, True),
        ("native-env", native_install_root),
        ("native-package", native_install_root, llvm_install_root),
        ("wheel", str(tmp_path)),
    ]


def test_build_editable_skip_llvm_still_bootstraps_ixsimpl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setenv("HC_SKIP_LLVM_BOOTSTRAP", "1")
    _record_ixsimpl_bootstrap(monkeypatch, calls)
    monkeypatch.setattr(
        build_backend,
        "ensure_llvm_toolchain",
        _unexpected_bootstrap,
    )
    monkeypatch.setattr(
        build_backend,
        "ensure_hc_native_tools_built",
        _unexpected_bootstrap,
    )
    _record_directory_build_hook(
        monkeypatch,
        "build_editable",
        calls,
        "editable",
        "hc-editable.whl",
    )
    result = build_backend.build_editable(str(tmp_path))
    assert result == "hc-editable.whl"
    assert calls == ["ixsimpl", ("editable", str(tmp_path))]


def test_build_backend_bootstraps_llvm_after_prior_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    llvm_install_root = tmp_path / "llvm-install"
    native_install_root = tmp_path / "native-install"
    _reset_bootstrap_state(monkeypatch)
    _record_ixsimpl_bootstrap(monkeypatch, calls)
    _record_llvm_bootstrap(monkeypatch, calls, llvm_install_root)
    _record_hc_native_bootstrap(monkeypatch, calls, native_install_root)
    _record_package_native_install(monkeypatch, calls)
    _record_build_hook(
        monkeypatch, "build_editable", calls, "editable", "hc-editable.whl"
    )
    _record_build_hook(monkeypatch, "build_wheel", calls, "wheel", "hc.whl")
    monkeypatch.setenv("HC_SKIP_LLVM_BOOTSTRAP", "1")
    assert build_backend.build_editable(str(tmp_path)) == "hc-editable.whl"
    monkeypatch.delenv("HC_SKIP_LLVM_BOOTSTRAP", raising=False)
    assert build_backend.build_wheel(str(tmp_path)) == "hc.whl"
    assert calls == [
        "ixsimpl",
        "editable",
        ("llvm", llvm_install_root),
        ("native-build", llvm_install_root, True),
        ("native-env", native_install_root),
        ("native-package", native_install_root, llvm_install_root),
        "wheel",
    ]


def test_install_package_native_artifacts_copies_runtime_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_install_root = tmp_path / "native-install"
    llvm_install_root = tmp_path / "llvm-install"
    package_native_root = tmp_path / "package" / "_native"
    (native_install_root / "bin").mkdir(parents=True)
    (native_install_root / "bin" / "hc-opt").write_text("hc-opt\n", encoding="utf-8")
    hc_mlir = native_install_root / "python_packages" / "hc_front" / "hc_mlir"
    hc_mlir.mkdir(parents=True)
    (hc_mlir / "ir.py").write_text("IR = True\n", encoding="utf-8")
    (hc_mlir / "__pycache__").mkdir()
    (hc_mlir / "__pycache__" / "ir.pyc").write_bytes(b"stale")
    (native_install_root / "lib").mkdir()
    (native_install_root / "lib" / "libHC.a").write_text("archive\n", encoding="utf-8")
    (llvm_install_root / "bin").mkdir(parents=True)
    (llvm_install_root / "bin" / "ld.lld").write_text("lld\n", encoding="utf-8")

    monkeypatch.setattr(build_backend, "_PACKAGE_NATIVE_ROOT", package_native_root)

    build_backend._install_package_native_artifacts(
        native_install_root, llvm_install_root
    )

    assert (package_native_root / "bin" / "hc-opt").read_text(
        encoding="utf-8"
    ) == "hc-opt\n"
    assert (
        package_native_root / "python_packages" / "hc_front" / "hc_mlir" / "ir.py"
    ).read_text(encoding="utf-8") == "IR = True\n"
    assert not (
        package_native_root / "python_packages" / "hc_front" / "hc_mlir" / "__pycache__"
    ).exists()
    assert (package_native_root / "lib" / "libHC.a").read_text(
        encoding="utf-8"
    ) == "archive\n"
    # `ld.lld` lands in `_native/bin/` for `_native_paths.lld_path`.
    assert (package_native_root / "bin" / "ld.lld").read_text(
        encoding="utf-8"
    ) == "lld\n"
    manifest = json.loads(
        (package_native_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"] == str(native_install_root)


def test_install_package_native_artifacts_skips_lld_when_no_llvm_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # No LLVM root (sdist / metadata-only) -> skip lld stage, don't fail.
    native_install_root = tmp_path / "native-install"
    package_native_root = tmp_path / "package" / "_native"
    (native_install_root / "bin").mkdir(parents=True)
    (native_install_root / "bin" / "hc-opt").write_text("hc-opt\n", encoding="utf-8")
    hc_mlir = native_install_root / "python_packages" / "hc_front" / "hc_mlir"
    hc_mlir.mkdir(parents=True)
    (hc_mlir / "ir.py").write_text("IR = True\n", encoding="utf-8")

    monkeypatch.setattr(build_backend, "_PACKAGE_NATIVE_ROOT", package_native_root)

    build_backend._install_package_native_artifacts(native_install_root, None)

    assert (package_native_root / "bin" / "hc-opt").exists()
    assert not (package_native_root / "bin" / "ld.lld").exists()


def test_build_sdist_skips_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setattr(build_backend, "ensure_ixsimpl_built", _unexpected_bootstrap)
    monkeypatch.setattr(
        build_backend,
        "ensure_llvm_toolchain",
        _unexpected_bootstrap,
    )
    monkeypatch.setattr(
        build_backend,
        "ensure_hc_native_tools_built",
        _unexpected_bootstrap,
    )
    monkeypatch.setattr(
        build_backend._build_meta,
        "build_sdist",
        lambda sdist_directory, config_settings=None: "hc.tar.gz",
    )

    assert build_backend.build_sdist(str(tmp_path)) == "hc.tar.gz"


def test_prepare_metadata_for_build_wheel_skips_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setattr(build_backend, "ensure_ixsimpl_built", _unexpected_bootstrap)
    monkeypatch.setattr(
        build_backend,
        "ensure_llvm_toolchain",
        _unexpected_bootstrap,
    )
    monkeypatch.setattr(
        build_backend,
        "ensure_hc_native_tools_built",
        _unexpected_bootstrap,
    )
    monkeypatch.setattr(
        build_backend._build_meta,
        "prepare_metadata_for_build_wheel",
        lambda metadata_directory, config_settings=None: "hc.dist-info",
    )

    assert (
        build_backend.prepare_metadata_for_build_wheel(str(tmp_path)) == "hc.dist-info"
    )
