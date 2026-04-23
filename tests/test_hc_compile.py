# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

import hc
from hc import Buffer, CompiledKernel, CurrentGroup, kernel
from hc._compile import _normalise_bindings
from hc.core import KernelMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
_COMPILE_SUBPROCESS_TIMEOUT_SECONDS = 60.0

_SKIP_NATIVE_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)


@lru_cache(maxsize=1)
def _native_env() -> dict[str, str]:
    # Import lazily: non-native tests should not pay for bootstrapping the
    # MLIR toolchain.
    from build_tools.hc_native_tools import (
        ensure_hc_native_tools_built,
        export_hc_native_environment,
    )
    from build_tools.llvm_toolchain import ensure_llvm_toolchain

    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    return export_hc_native_environment(native_install_root, os.environ.copy())


def _sym() -> Any:
    from hc import sym

    return sym


def test_compile_rejects_non_kernel() -> None:
    def not_a_kernel() -> None:
        return None

    with pytest.raises(TypeError, match="@kernel-decorated"):
        hc.compile(not_a_kernel)


def test_compile_rejects_unknown_literal_symbol() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    with pytest.raises(ValueError, match="not a declared literal symbol"):
        # str key path: symbol name "H" was never declared.
        _normalise_bindings({"H": 16}, foo.__hc_kernel__)


def test_compile_rejects_non_int_binding() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    # `True` is an int subclass but booleans are not intended shape values.
    with pytest.raises(TypeError, match="must bind to an int"):
        _normalise_bindings({sym.W: True}, foo.__hc_kernel__)

    with pytest.raises(TypeError, match="must bind to an int"):
        _normalise_bindings({sym.W: "16"}, foo.__hc_kernel__)


def test_compile_allows_missing_bindings() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # Partial specialization (the empty map here) must be valid; later
    # pipeline stages refine what is left — the doc calls this out.
    assert _normalise_bindings({}, metadata) == {}


def test_compile_allows_any_binding_when_no_literals_declared() -> None:
    metadata = KernelMetadata()

    assert _normalise_bindings({"wave_size": 32}, metadata) == {"wave_size": 32}


def test_compile_accepts_symbol_keys_and_string_keys_equivalently() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    by_symbol = _normalise_bindings({sym.W: 8}, metadata)
    by_string = _normalise_bindings({"W": 8}, metadata)
    assert by_symbol == by_string == {"W": 8}


def test_compiled_kernel_call_raises_until_pipeline_lands() -> None:
    def kfn() -> None:
        return None

    handle = CompiledKernel(
        kernel=kfn,
        bindings={"W": 16},
        front_ir=None,
        front_ir_text="",
    )

    with pytest.raises(NotImplementedError, match="frontend"):
        handle()


def test_compiled_kernel_repr_is_informative() -> None:
    def kfn() -> None:
        return None

    kfn.__name__ = "kfn"  # explicit for the assertion
    handle = CompiledKernel(
        kernel=kfn,
        bindings={"M": 1024, "N": 512},
        front_ir=None,
        front_ir_text="",
    )

    text = repr(handle)
    assert "CompiledKernel(kfn" in text
    assert "M=1024" in text
    assert "N=512" in text


@_SKIP_NATIVE_TESTS
def test_compile_returns_handle_with_front_ir_end_to_end(tmp_path: Path) -> None:
    # The real frontend needs both native MLIR bindings (hence the managed
    # env, same pattern as test_hc_front_python_bindings.py) and a kernel
    # function whose source file is on disk — `inspect.getsource` is used
    # to recover the text. `python -c '...'` scripts do not satisfy that.
    script = tmp_path / "smoke.py"
    script.write_text(textwrap.dedent("""
            import hc
            from hc import Buffer, CompiledKernel, CurrentGroup, kernel

            sym = hc.sym


            @kernel(work_shape=(sym.W,), literals={sym.W})
            def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                return None


            def main() -> None:
                handle = hc.compile(foo, {sym.W: 128})
                assert isinstance(handle, CompiledKernel)
                assert handle.bindings == {"W": 128}
                assert "hc_front.kernel" in handle.front_ir_text
                try:
                    handle()
                except NotImplementedError:
                    pass
                else:
                    raise AssertionError(
                        "handle() should raise NotImplementedError"
                    )
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=_native_env(),
        capture_output=True,
        text=True,
        timeout=_COMPILE_SUBPROCESS_TIMEOUT_SECONDS,
        check=True,
    )
    assert result.stdout.strip().endswith("OK"), result.stdout
