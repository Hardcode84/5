# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(
    env: dict[str, str],
    script: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def _write_fake_hc_mlir_package(package_root: Path, tag: str) -> None:
    package_dir = package_root / "hc_mlir"
    dialects_dir = package_dir / "dialects"
    dialects_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        'from . import ir\n__all__ = ["ir"]\n',
        encoding="utf-8",
    )
    (package_dir / "ir.py").write_text(
        textwrap.dedent(f"""
            __all__ = ["Context", "IR_TAG"]

            IR_TAG = {tag!r}

            class Context:
                LABEL = {tag!r}
            """),
        encoding="utf-8",
    )
    (dialects_dir / "__init__.py").write_text(
        'from . import hc_front\n__all__ = ["hc_front"]\n',
        encoding="utf-8",
    )
    (dialects_dir / "hc_front.py").write_text(
        textwrap.dedent(f"""
            __all__ = ["KernelOp", "register_dialects"]

            class KernelOp:
                LABEL = {tag!r}

            def register_dialects(*args, **kwargs):
                return {tag!r}
            """),
        encoding="utf-8",
    )


def test_hc_mlir_requires_built_package_root(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HC_MLIR_PYTHON_PACKAGE_DIR"] = str(tmp_path / "missing")

    result = _run_python(
        env,
        """
        from hc.mlir import ir  # noqa: F401
        """,
        check=False,
    )

    assert result.returncode != 0
    assert "HC_MLIR_PYTHON_PACKAGE_DIR does not point to a built hc_mlir package" in (
        result.stderr
    )


def test_hc_mlir_loads_bindings_from_managed_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "python_packages"
    env = os.environ.copy()
    env["HC_MLIR_PYTHON_PACKAGE_DIR"] = str(package_root)
    _write_fake_hc_mlir_package(package_root, "managed")

    result = _run_python(
        env,
        """
        import json

        from hc.mlir import ir
        from hc.mlir.dialects import hc_front

        print(
            json.dumps(
                {
                    "context": ir.Context.LABEL,
                    "kernel": hc_front.KernelOp.LABEL,
                    "register": hc_front.register_dialects(),
                }
            )
        )
        """,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "context": "managed",
        "kernel": "managed",
        "register": "managed",
    }


def test_hc_mlir_rejects_preimported_package_from_elsewhere(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected"
    unexpected_root = tmp_path / "unexpected"
    env = os.environ.copy()
    env["HC_MLIR_PYTHON_PACKAGE_DIR"] = str(expected_root)
    _write_fake_hc_mlir_package(expected_root, "expected")
    _write_fake_hc_mlir_package(unexpected_root, "unexpected")

    result = _run_python(
        env,
        f"""
        import sys

        sys.path.insert(0, {str(unexpected_root)!r})
        import hc_mlir  # noqa: F401
        from hc.mlir import ir  # noqa: F401
        """,
        check=False,
    )

    assert result.returncode != 0
    assert "pre-imported `hc_mlir` outside the managed package directory" in (
        result.stderr
    )
