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
# The hc_front smoke test already builds the native package once per session
# via `@lru_cache`; this suite hangs off the same knob so both sets skip
# together when the env opts out.
# The env knob is named `HC_SKIP_HC_FRONT_DIALECT_TESTS` across the suite;
# keeping the mark symbol aligned with its sibling files makes grep for
# "which tests get gated by that env" honest.
_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc dialect smoke tests disabled by env",
)
_PYTHON_BINDINGS_TIMEOUT_SECONDS = 60.0

# A fully-lowered `hc` module — no `hc_front` ops — exercises the hc dialect
# registration path without relying on the conversion pass. `hc.kernel` +
# `hc.return` is the smallest shape the verifier accepts; trying to make
# this smaller starts getting pedantic about typed/untyped values.
_HC_MODULE_SOURCE = textwrap.dedent("""
    module {
      hc.kernel @example {
        hc.return
      }
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
            "hc dialect Python bindings smoke test timed out.\n"
            f"timeout: {_PYTHON_BINDINGS_TIMEOUT_SECONDS:.0f}s\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            "hc dialect Python bindings smoke test failed.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_hc_dialect_registers_and_parses_lowered_ir() -> None:
    result = _run_python(f"""
        import json

        from hc.mlir import ir
        from hc.mlir.dialects import hc, hc_front

        context = ir.Context()
        hc_front.register_dialects(context)
        hc.register_dialects(context)
        module = ir.Module.parse({json.dumps(_HC_MODULE_SOURCE)}, context=context)
        print(json.dumps({{"module": str(module)}}))
        """)

    payload = json.loads(result.stdout)
    assert "hc.kernel" in payload["module"]
    assert "hc.return" in payload["module"]


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_pass_registry_exposes_canonical_front_to_hc_pipeline() -> None:
    # PassManager.parse raises if any pass name in the pipeline string is
    # unknown, so a successful parse is proof that both our hc-specific
    # passes and upstream stock passes made it into the registry. Covers
    # the `hc.compile` driver's minimum surface.
    result = _run_python("""
        import json

        from hc.mlir import ir, passmanager
        from hc.mlir.dialects import hc, hc_front

        hc.register_passes()
        context = ir.Context()
        hc_front.register_dialects(context)
        hc.register_dialects(context)

        canonical = (
            "builtin.module("
            "hc-front-fold-region-defs,"
            "hc-front-inline,"
            "convert-hc-front-to-hc,"
            "hc-promote-names,"
            "hc-infer-types,"
            "hc-verify-static-shapes)"
        )
        transform_driver = "builtin.module(transform-interpreter)"

        passmanager.PassManager.parse(canonical, context=context)
        passmanager.PassManager.parse(transform_driver, context=context)
        print(json.dumps({"ok": True}))
        """)

    payload = json.loads(result.stdout)
    assert payload["ok"] is True


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_register_passes_is_idempotent_across_contexts() -> None:
    # The CAPI shim wraps registration in a `std::once_flag`, but callers
    # invoking it from two ad-hoc contexts should still observe a coherent
    # registry. Re-register on the second context and re-parse to prove it.
    result = _run_python("""
        import json

        from hc.mlir import ir, passmanager
        from hc.mlir.dialects import hc

        hc.register_passes()
        hc.register_passes()

        first = ir.Context()
        hc.register_dialects(first)
        passmanager.PassManager.parse(
            "builtin.module(convert-hc-front-to-hc)", context=first
        )

        second = ir.Context()
        hc.register_dialects(second)
        passmanager.PassManager.parse(
            "builtin.module(convert-hc-front-to-hc)", context=second
        )
        print(json.dumps({"ok": True}))
        """)

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
