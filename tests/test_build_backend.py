# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

import build_backend


def _reset_bootstrap_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_backend, "_IXSIMPL_BOOTSTRAPPED", False)
    monkeypatch.setattr(build_backend, "_LLVM_BOOTSTRAPPED", False)


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


def test_build_wheel_bootstraps_ixsimpl_and_llvm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    install_root = tmp_path / "llvm-install"

    _reset_bootstrap_state(monkeypatch)
    monkeypatch.delenv("HC_SKIP_LLVM_BOOTSTRAP", raising=False)
    _record_ixsimpl_bootstrap(monkeypatch, calls)
    _record_llvm_bootstrap(monkeypatch, calls, install_root)
    _record_directory_build_hook(
        monkeypatch,
        "build_wheel",
        calls,
        "wheel",
        "hc.whl",
    )

    result = build_backend.build_wheel(str(tmp_path))

    assert result == "hc.whl"
    assert calls == ["ixsimpl", ("llvm", install_root), ("wheel", str(tmp_path))]


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
    install_root = tmp_path / "llvm-install"

    _reset_bootstrap_state(monkeypatch)
    _record_ixsimpl_bootstrap(monkeypatch, calls)
    _record_llvm_bootstrap(monkeypatch, calls, install_root)
    _record_build_hook(
        monkeypatch, "build_editable", calls, "editable", "hc-editable.whl"
    )
    _record_build_hook(monkeypatch, "build_wheel", calls, "wheel", "hc.whl")

    monkeypatch.setenv("HC_SKIP_LLVM_BOOTSTRAP", "1")
    assert build_backend.build_editable(str(tmp_path)) == "hc-editable.whl"

    monkeypatch.delenv("HC_SKIP_LLVM_BOOTSTRAP", raising=False)
    assert build_backend.build_wheel(str(tmp_path)) == "hc.whl"

    assert calls == ["ixsimpl", "editable", ("llvm", install_root), "wheel"]


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
        build_backend._build_meta,
        "prepare_metadata_for_build_wheel",
        lambda metadata_directory, config_settings=None: "hc.dist-info",
    )

    assert (
        build_backend.prepare_metadata_for_build_wheel(str(tmp_path)) == "hc.dist-info"
    )
