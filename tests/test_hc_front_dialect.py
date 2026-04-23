# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

from build_tools.hc_native_tools import ensure_hc_native_tools_built
from build_tools.llvm_toolchain import ensure_llvm_toolchain

# Successful parse of real `hc_front.*` text is the stable contract that
# matters here.
_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)
_HC_OPT_TIMEOUT_SECONDS = 60.0

_HC_FRONT_MODULE_SOURCE = textwrap.dedent("""
    module {
      "hc_front.kernel"() <{name = "example"}> ({
        %target = "hc_front.target_name"() <{name = "lhs"}> : () -> !hc_front.value
        %value = "hc_front.constant"() <{value = 1 : i64}> : () -> !hc_front.value
        "hc_front.assign"(%target, %value)
          : (!hc_front.value, !hc_front.value) -> ()
        "hc_front.if"() ({
          %cond = "hc_front.name"() <{name = "pred"}> : () -> !hc_front.value
        }, {
          %lhs = "hc_front.name"() <{name = "lhs"}> : () -> !hc_front.value
          "hc_front.return"(%lhs) : (!hc_front.value) -> ()
        }, {
          %fallback = "hc_front.list"() : () -> !hc_front.value
        }) : () -> ()
        "hc_front.for"() ({
          %i = "hc_front.target_name"() <{name = "i"}> : () -> !hc_front.value
        }, {
          %rows = "hc_front.name"() <{name = "rows"}> : () -> !hc_front.value
        }, {
          %callee = "hc_front.name"() <{name = "consume"}> : () -> !hc_front.value
          %arg = "hc_front.name"() <{name = "i"}> : () -> !hc_front.value
          %call = "hc_front.call"(%callee, %arg)
            : (!hc_front.value, !hc_front.value) -> !hc_front.value
        }) : () -> ()
      }) : () -> ()
    }
    """)


@lru_cache(maxsize=1)
def _hc_opt_path() -> Path:
    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    return native_install_root / "bin" / "hc-opt"


def _run_hc_opt(
    args: list[str],
    *,
    input_text: str = "",
) -> subprocess.CompletedProcess[str]:
    command = [str(_hc_opt_path()), *args]
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=_HC_OPT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"hc-opt timed out after {_HC_OPT_TIMEOUT_SECONDS:.0f}s.\n"
            f"command: {command}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    if result.returncode == 0:
        return result
    raise AssertionError(
        "hc-opt failed.\n"
        f"command: {command}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_hc_opt_parses_and_prints_registered_hc_front_textual_ir() -> None:
    result = _run_hc_opt([], input_text=_HC_FRONT_MODULE_SOURCE)

    assert "hc_front.kernel" in result.stdout
    assert "hc_front.if" in result.stdout
    assert "hc_front.for" in result.stdout
    # The declarative assembly format infers `!hc_front.value` from the
    # parameterless, buildable `HCFront_ValueType`, so the pretty-printed
    # form never spells the type out — hence no string check for it here.
