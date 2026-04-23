# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.llvm_toolchain import ensure_llvm_toolchain

REPO_ROOT = Path(__file__).resolve().parents[1]
_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)
_PYTHON_BINDINGS_TIMEOUT_SECONDS = 60.0
_HC_FRONT_MODULE_SOURCE = textwrap.dedent("""
    module {
      "hc_front.kernel"() <{name = "example"}> ({
        %value = "hc_front.constant"() <{value = 1 : i64}> : () -> !hc_front.value
        "hc_front.return"(%value) : (!hc_front.value) -> ()
      }) : () -> ()
    }
    """)


@lru_cache(maxsize=1)
def _python_bindings_env() -> dict[str, str]:
    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    return export_hc_native_environment(native_install_root, os.environ.copy())


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=REPO_ROOT,
            env=_python_bindings_env(),
            capture_output=True,
            text=True,
            timeout=_PYTHON_BINDINGS_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "hc_front Python bindings smoke test timed out.\n"
            f"timeout: {_PYTHON_BINDINGS_TIMEOUT_SECONDS:.0f}s\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            "hc_front Python bindings smoke test failed.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_hc_front_python_bindings_parse_ir_and_expose_front_types() -> None:
    result = _run_python(f"""
        import json

        from hc.mlir import ir
        from hc.mlir.dialects import hc_front

        context = ir.Context()
        hc_front.register_dialects(context)
        module = ir.Module.parse({json.dumps(_HC_FRONT_MODULE_SOURCE)}, context=context)
        value_type = hc_front.ValueType.get(context)
        typeexpr_type = hc_front.TypeExprType.get(context)

        print(
            json.dumps(
                {{
                    "module": str(module),
                    "value_type": str(value_type),
                    "typeexpr_type": str(typeexpr_type),
                    "kernel_op": issubclass(hc_front.KernelOp, ir.OpView),
                    "return_op": issubclass(hc_front.ReturnOp, ir.OpView),
                }}
            )
        )
        """)

    payload = json.loads(result.stdout)
    assert "hc_front.kernel" in payload["module"]
    assert "hc_front.return" in payload["module"]
    assert "!hc_front.value" in payload["module"]
    assert payload["value_type"] == "!hc_front.value"
    assert payload["typeexpr_type"] == "!hc_front.typeexpr"
    assert payload["kernel_op"] is True
    assert payload["return_op"] is True
