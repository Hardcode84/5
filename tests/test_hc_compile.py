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
from hc._compile import normalise_bindings, symbol_name
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


# --- normalise_bindings ----------------------------------------------------


def test_normalise_bindings_rejects_unknown_literal() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    with pytest.raises(ValueError, match="not a declared literal symbol"):
        normalise_bindings({"H": 16}, metadata)


def test_normalise_bindings_rejects_bool_and_str_values() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # `True` is an int subclass but booleans are not intended shape values.
    with pytest.raises(TypeError, match="must bind to an int"):
        normalise_bindings({sym.W: True}, metadata)

    with pytest.raises(TypeError, match="must bind to an int"):
        normalise_bindings({sym.W: "16"}, metadata)


def test_normalise_bindings_allows_empty_map() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # Partial specialization (empty here) must be valid; later pipeline
    # stages refine what is left — the doc calls this out.
    assert normalise_bindings({}, metadata) == {}


def test_normalise_bindings_allows_any_key_when_no_literals_declared() -> None:
    metadata = KernelMetadata()

    # Deliberate: a kernel without a `literals=` whitelist lets any key
    # through. Doc calls this out; later stages will tighten it.
    assert normalise_bindings({"wave_size": 32}, metadata) == {"wave_size": 32}


def test_normalise_bindings_symbol_and_string_keys_agree() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    by_symbol = normalise_bindings({sym.W: 8}, metadata)
    by_string = normalise_bindings({"W": 8}, metadata)
    assert by_symbol == by_string == {"W": 8}


def test_normalise_bindings_flags_conflicting_duplicate_keys() -> None:
    sym = _sym()
    metadata = KernelMetadata(literals=frozenset({sym.W}))

    # Same logical key via two forms pointing at different values is
    # ambiguous — fail loudly instead of last-write-wins.
    with pytest.raises(ValueError, match="bound twice"):
        normalise_bindings({sym.W: 8, "W": 16}, metadata)


# --- symbol_name name resolution -------------------------------------------


def test_symbol_name_accepts_string() -> None:
    assert symbol_name("W") == "W"


def test_symbol_name_accepts_symbol_instance() -> None:
    sym = _sym()
    assert symbol_name(sym.W) == "W"


def test_symbol_name_rejects_arbitrary_dot_name_objects() -> None:
    # A path-like object has a `.name` attribute but is not a Symbol;
    # rejecting it prevents bindings from silently using a surprising key.
    with pytest.raises(TypeError, match="cannot interpret"):
        symbol_name(Path("/tmp/W"))


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
    #
    # The pipeline has to actually run for the `hc_ir` assertion below to
    # mean anything — a weaker version of this test that only poked at
    # `front_ir_text` would silently pass against a fully broken default
    # schedule.
    script = tmp_path / "smoke.py"
    script.write_text(textwrap.dedent("""
            from contextlib import suppress

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
                assert handle.hc_ir is not None, handle.pipeline_diagnostics
                assert handle.pipeline_diagnostics == ()
                with suppress(NotImplementedError):
                    handle()
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
def test_compile_runs_front_to_hc_pipeline_end_to_end(tmp_path: Path) -> None:
    # Happy path for the transform-schedule driver: compile a trivial
    # kernel and assert the `hc_ir_text` snapshot contains hc-dialect ops
    # (not hc_front.*), with no captured diagnostics. This is the gate we
    # care about for the CompiledKernel contract now that the pipeline
    # stage actually runs.
    script = tmp_path / "compile_pipeline.py"
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
                assert handle.hc_ir is not None, handle.pipeline_diagnostics
                assert handle.hc_ir_text is not None
                assert "hc.kernel" in handle.hc_ir_text, handle.hc_ir_text
                assert "hc_front." not in handle.hc_ir_text, handle.hc_ir_text
                # `front_ir_text` must remain the pre-pipeline snapshot
                # even after a successful run; the module clone in
                # hc._compile is the load-bearing mechanism.
                assert "hc_front.kernel" in handle.front_ir_text
                assert handle.pipeline_diagnostics == ()
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_default_compile_schedule_verifies_static_shapes(tmp_path: Path) -> None:
    script = tmp_path / "compile_static_shape_failure.py"
    script.write_text(textwrap.dedent("""
            import hc
            from hc import Buffer, CurrentGroup, kernel


            @kernel(work_shape=(4,), group_shape=(4,))
            def bad_shape(group: CurrentGroup, x: Buffer[4], n: int) -> None:
                tile = group.load(x, shape=(n,))
                _ = tile


            def main() -> None:
                handle = hc.compile(bad_shape)
                assert handle.hc_ir is None, handle.hc_ir_text
                assert handle.hc_ir_text is None
                assert handle.pipeline_diagnostics, (
                    "expected static shape verifier diagnostics"
                )
                joined = "\\n".join(handle.pipeline_diagnostics)
                assert "shape dimension #0" in joined, joined
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
@pytest.mark.parametrize(
    ("shape_source", "dtype_source", "diagnostic"),
    [
        ('"8"', "np.complex64", "unsupported dtype 'complex64'"),
        ('"M +"', "np.float32", "failed to parse hc.shape dim 'M +'"),
    ],
)
def test_compile_reports_invalid_intrinsic_type_contracts_from_python_metadata(
    tmp_path: Path,
    shape_source: str,
    dtype_source: str,
    diagnostic: str,
) -> None:
    script = tmp_path / "compile_bad_intrinsic_contract.py"
    script.write_text(textwrap.dedent(f"""
            import numpy as np

            import hc
            from hc import CurrentGroup, WorkItem, kernel, vector_type


            @kernel.intrinsic(
                scope=WorkItem,
                result_types=(vector_type(({shape_source},), {dtype_source}),),
            )
            def bad_contract_intrinsic():
                ...


            @kernel(work_shape=(1,), group_shape=(1,))
            def uses_bad_contract(group: CurrentGroup) -> None:
                _ = bad_contract_intrinsic()


            def main() -> None:
                handle = hc.compile(uses_bad_contract)
                assert handle.hc_ir is None, handle.hc_ir_text
                assert handle.hc_ir_text is None
                joined = "\\n".join(handle.pipeline_diagnostics)
                assert {diagnostic!r} in joined, joined
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_honors_inline_schedule_override(tmp_path: Path) -> None:
    # A custom schedule that only runs `-convert-hc-front-to-hc` — no
    # fold/inline — must still produce valid hc IR for a kernel without
    # inline helpers. Proves the override API threads inline text all the
    # way through the transform-dialect driver, not just the default.
    script = tmp_path / "compile_override.py"
    script.write_text(textwrap.dedent("""
            import hc
            from hc import Buffer, CompiledKernel, CurrentGroup, kernel

            sym = hc.sym

            SCHEDULE = \"\"\"
            module attributes {transform.with_named_sequence} {
              transform.named_sequence @__transform_main(%m: !transform.any_op) {
                %m1 = transform.apply_registered_pass "convert-hc-front-to-hc" to %m
                    : (!transform.any_op) -> !transform.any_op
                transform.yield
              }
            }
            \"\"\"


            @kernel(work_shape=(sym.W,), literals={sym.W})
            def bar(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                row = group.group_id[0]


            def main() -> None:
                handle = hc.compile(bar, {sym.W: 64}, schedule=SCHEDULE)
                assert handle.hc_ir is not None, handle.pipeline_diagnostics
                assert "hc.kernel" in handle.hc_ir_text
                # Skipping promote-names means hc.name_load/hc.assign
                # ops stay in the output — presence of `hc.name_load`
                # proves the override skipped that stage.
                assert "hc.name_load" in handle.hc_ir_text, handle.hc_ir_text
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_honors_path_schedule_override(tmp_path: Path) -> None:
    # The `str` and `None` paths through `schedule=` go through
    # `_schedule_file`'s tempfile branch; a real `Path` goes through
    # the resolve/exists/yield branch. Exercise that explicitly so the
    # path branch has an actual gate, including the resolve-to-absolute
    # behaviour (we pass a relative path via cwd and still expect a hit).
    script = tmp_path / "compile_path_override.py"
    schedule_file = tmp_path / "custom_schedule.mlir"
    schedule_file.write_text(textwrap.dedent("""
        module attributes {transform.with_named_sequence} {
          transform.named_sequence @__transform_main(%m: !transform.any_op) {
            %m1 = transform.apply_registered_pass "convert-hc-front-to-hc" to %m
                : (!transform.any_op) -> !transform.any_op
            transform.yield
          }
        }
        """))
    script.write_text(textwrap.dedent(f"""
            from pathlib import Path

            import hc
            from hc import Buffer, CompiledKernel, CurrentGroup, kernel

            sym = hc.sym


            @kernel(work_shape=(sym.W,), literals={{sym.W}})
            def bar(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                row = group.group_id[0]


            def main() -> None:
                schedule = Path({str(schedule_file)!r})
                handle = hc.compile(bar, {{sym.W: 64}}, schedule=schedule)
                assert handle.hc_ir is not None, handle.pipeline_diagnostics
                # Same schedule shape as the inline-override test —
                # name_load survives because promote-names is skipped.
                assert "hc.name_load" in handle.hc_ir_text, handle.hc_ir_text
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_rejects_missing_schedule_path(tmp_path: Path) -> None:
    # A Path that doesn't exist has to fail loudly with FileNotFoundError
    # rather than getting silently fed into the MLIR options parser and
    # producing a confusing diagnostic far from the source of the mistake.
    script = tmp_path / "compile_missing_path.py"
    missing = tmp_path / "not_a_real_schedule.mlir"
    script.write_text(textwrap.dedent(f"""
            from pathlib import Path

            import hc
            from hc import Buffer, CompiledKernel, CurrentGroup, kernel

            sym = hc.sym


            @kernel(work_shape=(sym.W,), literals={{sym.W}})
            def bar(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                row = group.group_id[0]


            def main() -> None:
                try:
                    hc.compile(bar, {{sym.W: 64}}, schedule=Path({str(missing)!r}))
                except FileNotFoundError as exc:
                    assert {str(missing)!r} in str(exc), str(exc)
                    print("OK")
                else:
                    raise AssertionError("expected FileNotFoundError")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_surfaces_pipeline_failure_non_fatal(tmp_path: Path) -> None:
    # Hand the driver an inline schedule that names a pass that doesn't
    # exist. We expect the handle to come back with hc_ir=None, the
    # front-IR snapshot preserved, and a non-empty diagnostics tuple —
    # not an exception. Callers want a value they can inspect.
    script = tmp_path / "compile_failure.py"
    script.write_text(textwrap.dedent("""
            import hc
            from hc import Buffer, CompiledKernel, CurrentGroup, kernel

            sym = hc.sym

            BAD_SCHEDULE = \"\"\"
            module attributes {transform.with_named_sequence} {
              transform.named_sequence @__transform_main(%m: !transform.any_op) {
                %m1 = transform.apply_registered_pass "this-pass-does-not-exist" to %m
                    : (!transform.any_op) -> !transform.any_op
                transform.yield
              }
            }
            \"\"\"


            @kernel(work_shape=(sym.W,), literals={sym.W})
            def baz(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                row = group.group_id[0]


            def main() -> None:
                handle = hc.compile(baz, {sym.W: 32}, schedule=BAD_SCHEDULE)
                assert handle.hc_ir is None, handle.hc_ir_text
                assert handle.hc_ir_text is None
                assert "hc_front.kernel" in handle.front_ir_text
                assert handle.pipeline_diagnostics, (
                    "expected at least one captured diagnostic"
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
    #
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
                assert handle.hc_ir is not None, handle.pipeline_diagnostics
                assert handle.hc_ir_text is not None
                assert (
                    '!hc.vector<f32, ["8"]>, !hc.idx<"$WI0">) -> '
                    '!hc.vector<f32, ["8"]>'
                ) in handle.hc_ir_text
                assert (
                    'to !hc.bare_vector<f32, ["8"]>, '
                    '!hc.bare_vector<!hc.pred, ["8"]>'
                    in handle.hc_ir_text
                    and ') -> !hc.bare_vector<f32, ["8"]>'
                    in handle.hc_ir_text
                    and ') -> !hc.bare_vector<!hc.pred, ["8"]>'
                    in handle.hc_ir_text
                    and 'to !hc.vector<f32, ["8"]>' in handle.hc_ir_text
                )
                assert re.search(
                    r'hc\\.call @store_wmma_tile\\([^\\n]+\\) : '
                    r'\\([^\\n]+!hc\\.bare_vector<f32, \\["8"\\]>, '
                    r'!hc\\.bare_vector<!hc\\.pred, \\["8"\\]>\\) -> \\(\\)',
                    handle.hc_ir_text,
                )
                assert re.search(
                    r'hc\\.store [^\\n]+, %[0-9]+, mask %[0-9]+ : '
                    r'\\([^\\n]+!hc\\.bare_vector<f32, \\["8"\\]>, '
                    r'!hc\\.bare_vector<!hc\\.pred, \\["8"\\]>\\) -> \\(\\)',
                    handle.hc_ir_text,
                )

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
