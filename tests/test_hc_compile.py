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

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.llvm_toolchain import ensure_llvm_toolchain
from hc import Buffer, CompiledKernel, CurrentGroup, compile, kernel
from hc._compile import _normalise_bindings, _symbol_name
from hc.core import KernelMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
_COMPILE_SUBPROCESS_TIMEOUT_SECONDS = 60.0

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)


@lru_cache(maxsize=1)
def _native_env() -> dict[str, str]:
    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    return export_hc_native_environment(native_install_root, os.environ.copy())


def _sym() -> Any:
    from hc import sym

    return sym


def _run_compile_smoke(script: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            env=_native_env(),
            capture_output=True,
            text=True,
            timeout=_COMPILE_SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "hc.compile smoke test timed out.\n"
            f"timeout: {_COMPILE_SUBPROCESS_TIMEOUT_SECONDS:.0f}s\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            "hc.compile smoke test failed.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


# --- public hc.compile entry point ----------------------------------------
# Validation runs before the frontend is imported, so these tests do not
# need the native toolchain.


def test_compile_rejects_non_kernel() -> None:
    def not_a_kernel() -> None:
        return None

    with pytest.raises(TypeError, match="@kernel-decorated"):
        compile(not_a_kernel)


def test_compile_rejects_non_mapping_symbols() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    # A list of pairs is a classic mistake; fail early with a message
    # that names the argument instead of a cryptic AttributeError later.
    with pytest.raises(TypeError, match="symbols must be a Mapping"):
        compile(foo, [("W", 16)])  # type: ignore[arg-type]


def test_compile_rejects_unknown_literal_symbol() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    with pytest.raises(ValueError, match="not a declared literal symbol"):
        compile(foo, {"H": 16})


def test_compile_rejects_non_int_binding() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    with pytest.raises(TypeError, match="must bind to an int"):
        compile(foo, {sym.W: "16"})


# --- _normalise_bindings ---------------------------------------------------


def test_normalise_bindings_rejects_unknown_literal() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    with pytest.raises(ValueError, match="not a declared literal symbol"):
        _normalise_bindings({"H": 16}, metadata)


def test_normalise_bindings_rejects_bool_and_str_values() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # `True` is an int subclass but booleans are not intended shape values.
    with pytest.raises(TypeError, match="must bind to an int"):
        _normalise_bindings({sym.W: True}, metadata)

    with pytest.raises(TypeError, match="must bind to an int"):
        _normalise_bindings({sym.W: "16"}, metadata)


def test_normalise_bindings_allows_empty_map() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # Partial specialization (empty here) must be valid; later pipeline
    # stages refine what is left — the doc calls this out.
    assert _normalise_bindings({}, metadata) == {}


def test_normalise_bindings_allows_any_key_when_no_literals_declared() -> None:
    metadata = KernelMetadata()

    # Deliberate: a kernel without a `literals=` whitelist lets any key
    # through. Doc calls this out; later stages will tighten it.
    assert _normalise_bindings({"wave_size": 32}, metadata) == {"wave_size": 32}


def test_normalise_bindings_symbol_and_string_keys_agree() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    by_symbol = _normalise_bindings({sym.W: 8}, metadata)
    by_string = _normalise_bindings({"W": 8}, metadata)
    assert by_symbol == by_string == {"W": 8}


def test_normalise_bindings_flags_conflicting_duplicate_keys() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # Same logical key via two forms pointing at different values is
    # ambiguous — fail loudly instead of last-write-wins.
    with pytest.raises(ValueError, match="bound twice"):
        _normalise_bindings({sym.W: 8, "W": 16}, metadata)


# --- _symbol_name name resolution ------------------------------------------


def test_symbol_name_accepts_string() -> None:
    assert _symbol_name("W") == "W"


def test_symbol_name_accepts_symbol_instance() -> None:
    sym = _sym()
    assert _symbol_name(sym.W) == "W"


def test_symbol_name_rejects_arbitrary_dot_name_objects() -> None:
    # A path-like object has a `.name` attribute but is not a Symbol;
    # rejecting it prevents bindings from silently using a surprising key.
    with pytest.raises(TypeError, match="cannot interpret"):
        _symbol_name(Path("/tmp/W"))


# --- CompiledKernel handle -------------------------------------------------


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


# --- end-to-end through the real frontend ----------------------------------


@_SKIP_HC_FRONT_DIALECT_TESTS
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

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_wmma_collects_deps_and_stamps_every_load(tmp_path: Path) -> None:
    # End-to-end assertion that what the resolver stamps flows through
    # ``hc.compile``: ``front_ir_symbols`` exposes the closed dep set and
    # every load-context ``hc_front.name`` carries a ``ref`` attribute.
    # The real WMMA example is the richest fixture we have for this check.
    script = tmp_path / "compile_wmma.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n" + textwrap.dedent("""
            import re

            import hc
            from examples.amdgpu_gfx11_wmma_matmul import tiled_gfx11_wmma_matmul


            def main() -> None:
                handle = hc.compile(tiled_gfx11_wmma_matmul)
                expected = {
                    "tiled_gfx11_wmma_matmul",
                    "init_wmma_acc",
                    "issue_wmma_tile",
                    "store_wmma_tile",
                    "load_wmma_a_fragment",
                    "load_wmma_b_fragment",
                    "wmma_gfx11",
                }
                assert set(handle.front_ir_symbols) == expected, (
                    handle.front_ir_symbols
                )
                assert handle.front_ir_symbols[0] == "tiled_gfx11_wmma_matmul"

                loads = re.findall(
                    r'(hc_front\\.name "[\\w_]+" \\{ctx = "load"[^\\n]*)\\n',
                    handle.front_ir_text,
                )
                assert loads, "expected at least one load-context name op"
                for line in loads:
                    assert "ref = {" in line, line
                print("OK")


            if __name__ == "__main__":
                main()
            """)
    )

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout
