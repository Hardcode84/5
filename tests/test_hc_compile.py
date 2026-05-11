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


def test_compile_rejects_target_with_forbidden_chars() -> None:
    # The driver substitutes `target` into a literal MLIR string in the
    # default schedule; characters that would close the string early
    # ('"', '\\') or break the option parser ('\\n', '\\r') are rejected
    # up front so the user sees a clear error instead of a downstream
    # MLIR diagnostic.
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    with pytest.raises(ValueError, match="forbidden characters"):
        compile(foo, target='bad"target')


def test_compile_rejects_non_string_target() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    with pytest.raises(TypeError, match="target must be"):
        compile(foo, target=42)  # type: ignore[arg-type]


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


def test_compiled_kernel_call_raises_when_pipeline_did_not_run() -> None:
    # Handle with `hc_ir = None` means the lowering pipeline either
    # failed (diagnostics captured) or was never run (constructed
    # directly in tests). Either way the call surface is the same:
    # surface a `RuntimeError` instead of segfaulting on a missing
    # module.
    def kfn() -> None:
        return None

    handle = CompiledKernel(
        kernel=kfn,
        bindings={"W": 16},
        front_ir=None,
        front_ir_text="",
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
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
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_runs_front_to_hc_pipeline_end_to_end(tmp_path: Path) -> None:
    # Happy path for the transform-schedule driver: compile a trivial
    # kernel and assert the `hc_ir_text` snapshot has reached the
    # post-`gpu-to-llvm` stage where the host wrapper is `llvm.func`,
    # with no captured diagnostics. This trivial kernel never lays down
    # a `gpu.launch` (no real compute), so the schedule's gpu-outlining
    # path is a no-op and there's no `gpu.binary` to assert — the
    # surviving signal is "host-side IR is fully LLVM and the original
    # `hc_front.*` is gone".
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
                assert "llvm.func @foo" in handle.hc_ir_text, handle.hc_ir_text
                assert "hc.kernel" not in handle.hc_ir_text, handle.hc_ir_text
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


# End-to-end assertion script for the canonical WMMA pipeline. Pinned at
# module level (rather than dropped inline into the test) so lizard's
# function-length lint stays happy and the assertions read in one piece
# rather than wrapped in `textwrap.dedent` boilerplate.
_WMMA_COMPILE_SMOKE_SCRIPT = textwrap.dedent("""
    import re

    import hc
    from examples.amdgpu_gfx11_wmma_matmul import tiled_gfx11_wmma_matmul


    # `llvm.mlir.global ... constant @<name>("..."` carries the raw
    # bytes of the HSACO blob and the kernel name string. Either body
    # can contain literal "amdgpu", "gpu.module", etc. substrings (ELF
    # section names, escaped payload) that would false-positive the
    # negative scan below. Strip every quoted body on these lines so the
    # check sees only structural IR.
    _GLOBAL_RE = re.compile(
        r'(llvm\\.mlir\\.global[^(]*\\([^()"]*)"[^"]*"'
    )
    def _strip_global_string_body(line: str) -> str:
        return _GLOBAL_RE.sub(r'\\1"<stripped>"', line)


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
        assert set(handle.front_ir_symbols) == expected, handle.front_ir_symbols
        assert handle.front_ir_symbols[0] == "tiled_gfx11_wmma_matmul"
        assert handle.hc_ir is not None, handle.pipeline_diagnostics
        assert handle.hc_ir_text is not None
        # Default schedule lowers all the way to a self-contained LLVM
        # IR module: front-end → kernels-to-launch → intrinsic recipes
        # → kernel outlining → alloca-to-global + vector-transfer
        # reduction → rocdl attach → full ROCDL/LLVM lowering inside
        # `gpu.module` → host `gpu-to-llvm` → `hc-lower-gpu-to-binary`
        # (HSACO blob attached to a `gpu.binary`) →
        # `hc-lower-launch-func-to-runtime` (HSACO embedded as an LLVM
        # global, `gpu.launch_func` rewritten into `hc_rt_load_kernel`
        # + `hc_rt_launch_kernel` calls, source `gpu.binary` erased) →
        # symbol-dce. Nothing GPU-dialect-shaped survives in the post-
        # pipeline IR — only `gpu.container_module` (the module
        # attribute) is left, which the negative scan below excludes.
        # The HSACO blob ships in a `_data` global; its bytes can
        # carry literal "hc.", "amdgpu.", etc. substrings (ELF
        # metadata + escaped payload), so we strip every global string
        # body before the negative checks fire.
        ir_lines_no_blob = [
            _strip_global_string_body(line)
            for line in handle.hc_ir_text.splitlines()
        ]
        ir_no_blob = "\\n".join(ir_lines_no_blob)
        assert "!hc." not in ir_no_blob, ir_no_blob
        stray_hc_ops = [
            line for line in ir_lines_no_blob
            if " hc." in line or line.lstrip().startswith("hc.")
        ]
        assert not stray_hc_ops, stray_hc_ops
        assert "gpu.module" not in ir_no_blob, ir_no_blob
        assert "gpu.binary" not in ir_no_blob, ir_no_blob
        assert "gpu.launch" not in ir_no_blob, ir_no_blob
        assert "amdgpu." not in ir_no_blob, ir_no_blob
        assert "vector.transfer" not in ir_no_blob, ir_no_blob
        assert "unrealized_conversion_cast" not in ir_no_blob, ir_no_blob
        # Positive structural assertions: host wrapper landed as
        # `llvm.func` after `gpu-to-llvm`, takes one `PyObject *`
        # (lowered to `!llvm.ptr`) per kernel argument, calls the
        # `_mlir_ciface_hc_get_*` helpers (which `convert-func-to-llvm`
        # routes via the public-name wrappers `@hc_get_*`) to unpack
        # each tensor's data pointer (raw `!llvm.ptr` from
        # `hc_get_ptr`, addrspace-cast to `!llvm.ptr<1>` on the way to
        # the kernel) plus shape dims and strides, then dispatches via
        # the HIP shim (`hc_rt_load_kernel` + `hc_rt_launch_kernel`).
        # The HSACO blob and per-callsite handle/name globals live at
        # module scope.
        assert "module attributes {gpu.container_module}" in handle.hc_ir_text
        assert (
            "llvm.func @tiled_gfx11_wmma_matmul(%arg0: !llvm.ptr, "
            "%arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr)"
            in handle.hc_ir_text
        ), handle.hc_ir_text
        assert "@hc_get_ptr" in handle.hc_ir_text, handle.hc_ir_text
        assert (
            "@_mlir_ciface_hc_get_ptr" in handle.hc_ir_text
        ), handle.hc_ir_text
        assert "@hc_get_dim" in handle.hc_ir_text, handle.hc_ir_text
        assert (
            "@_mlir_ciface_hc_get_dim" in handle.hc_ir_text
        ), handle.hc_ir_text
        assert "@hc_get_stride" in handle.hc_ir_text, handle.hc_ir_text
        assert (
            "@_mlir_ciface_hc_get_stride" in handle.hc_ir_text
        ), handle.hc_ir_text
        assert "@hc_rt_load_kernel" in handle.hc_ir_text, handle.hc_ir_text
        assert "@hc_rt_launch_kernel" in handle.hc_ir_text, handle.hc_ir_text
        assert (
            "@tiled_gfx11_wmma_matmul_kernel_data" in handle.hc_ir_text
        ), handle.hc_ir_text
        assert (
            "@tiled_gfx11_wmma_matmul_kernel_handle" in handle.hc_ir_text
        ), handle.hc_ir_text
        # NUL-terminated kernel name string lands as a separate global.
        assert (
            '"tiled_gfx11_wmma_matmul_kernel\\\\00"' in handle.hc_ir_text
        ), handle.hc_ir_text

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


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_wmma_collects_deps_and_stamps_every_load(tmp_path: Path) -> None:
    # End-to-end assertion that what the resolver stamps flows through
    # ``hc.compile``: ``front_ir_symbols`` exposes the closed dep set and
    # every load-context ``hc_front.name`` carries a ``ref`` attribute.
    # The real WMMA example is the richest fixture we have for this check.
    script = tmp_path / "compile_wmma.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n"
        + _WMMA_COMPILE_SMOKE_SCRIPT
    )

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_target_selects_recipe(tmp_path: Path) -> None:
    # Three-way subprocess check on the `target=` plumbing:
    #   * `target=None` — recipe still fires (default empty target runs
    #     every named sequence) so the pipeline reaches a `gpu.binary`.
    #   * `target="amdgpu-gfx11"` — explicit match, same outcome plus
    #     the handle echoes the value back.
    #   * `target="amdgpu-gfx12"` — no recipe matches, so
    #     `hc-interpret-intrinsic-recipes` surfaces a hard "no
    #     intrinsic lowering recipe matched" diagnostic instead of
    #     silently passing the call through to whichever stage runs
    #     next.
    #
    # Pre-binary-emission this test pinned `amdgpu.wmma` as proof the
    # recipe fired. Now that the schedule lowers all the way through
    # to HSACO, `amdgpu.*` ops are gone — the structural signal of
    # success is `gpu.binary @<kernel>_kernel` with the rocdl target
    # attribute attached.
    script = tmp_path / "compile_target.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n" + textwrap.dedent("""
            import hc
            from examples.amdgpu_gfx11_wmma_matmul import tiled_gfx11_wmma_matmul


            def _check_compiled_to_binary(handle) -> None:
                # After `hc-lower-launch-func-to-runtime` the HSACO
                # ships in a `_data` global and the launch lands as a
                # `hc_rt_launch_kernel` call — the recipe-fired signal
                # is the per-callsite global + the runtime call.
                assert handle.hc_ir_text is not None, handle.pipeline_diagnostics
                assert (
                    "@tiled_gfx11_wmma_matmul_kernel_data"
                    in handle.hc_ir_text
                ), handle.hc_ir_text
                assert (
                    "@hc_rt_launch_kernel" in handle.hc_ir_text
                ), handle.hc_ir_text


            def main() -> None:
                default = hc.compile(tiled_gfx11_wmma_matmul)
                assert default.target is None, default.target
                _check_compiled_to_binary(default)

                matched = hc.compile(
                    tiled_gfx11_wmma_matmul, target="amdgpu-gfx11"
                )
                assert matched.target == "amdgpu-gfx11", matched.target
                _check_compiled_to_binary(matched)
                assert "target='amdgpu-gfx11'" in repr(matched), repr(matched)

                missed = hc.compile(
                    tiled_gfx11_wmma_matmul, target="amdgpu-gfx12"
                )
                assert missed.target == "amdgpu-gfx12", missed.target
                assert missed.hc_ir_text is None, missed.hc_ir_text
                joined = "\\n".join(missed.pipeline_diagnostics)
                assert "no intrinsic lowering recipe matched" in joined, joined
                assert "amdgpu-gfx12" in joined, joined
                print("OK")


            if __name__ == "__main__":
                main()
            """)
    )

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_invoke_dispatches_runtime_helpers(tmp_path: Path) -> None:
    # End-to-end: compile a no-op kernel that needs only the runtime
    # helpers (no `gpu.launch_func` survives the pipeline because the
    # body is empty), then invoke it through the JIT'd host wrapper. The
    # call must succeed without an actual GPU because the helpers are
    # all that runs.
    #
    # We deliberately use a kernel that bottoms out at the runtime
    # helpers — this exercises the whole invoke path (engine create,
    # shared-lib load, packed-args wrapper lookup, ctypes thunk) without
    # depending on libamdhip64.so being installed. WMMA-on-hardware
    # smoke is a separate task.
    script = tmp_path / "compile_invoke.py"
    script.write_text(textwrap.dedent("""
            import numpy as np

            import hc
            from hc import Buffer, CurrentGroup, kernel


            class _TensorView:
                # Minimal duck-type the runtime helpers require — we
                # depend on `data_ptr()` / `size(i)` / `stride(i)`
                # because those are the names BufferUtils.cpp looks up.
                def __init__(self, arr):
                    self._arr = arr

                def data_ptr(self):
                    return int(self._arr.ctypes.data)

                def size(self, dim):
                    return int(self._arr.shape[dim])

                def stride(self, dim):
                    return int(self._arr.strides[dim] // self._arr.itemsize)


            sym = hc.sym


            @kernel(work_shape=(sym.W,), literals={sym.W})
            def trivial(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                # Empty body — the lowered IR boils down to a host
                # wrapper that calls the runtime helpers and returns.
                return None


            def main() -> None:
                handle = hc.compile(trivial, {sym.W: 128})
                arr = np.zeros(128, dtype=np.float32)
                view = _TensorView(arr)

                handle(view)
                # Cache populated on the first call; second call must
                # reuse the same invoker (no fresh engine spin-up).
                cached = handle._invoker_cache.invoker
                assert cached is not None, "expected invoker cache populated"
                handle(view)
                assert handle._invoker_cache.invoker is cached

                # `stream=` is plumbed through to the host wrapper's
                # leading slot. This kernel has no gpu.launch_func so
                # the pointer never reaches the runtime — but the call
                # must accept it (None or 0) without complaint, which
                # is what tells us the ctypes thunk knows about the
                # extra leading slot.
                handle(view, stream=None)
                handle(view, stream=0)

                try:
                    handle(view, view)
                except TypeError as exc:
                    assert "takes 1 argument(s), got 2" in str(exc), str(exc)
                else:
                    raise AssertionError("expected TypeError on wrong arg count")

                try:
                    handle()
                except TypeError as exc:
                    assert "takes 1 argument(s), got 0" in str(exc), str(exc)
                else:
                    raise AssertionError("expected TypeError on wrong arg count")

                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


def test_compile_invoke_accepts_non_contiguous_tensor(tmp_path: Path) -> None:
    # The host wrapper pulls per-axis strides at runtime via
    # `_mlir_ciface_hc_get_stride` and feeds them into
    # `memref.reinterpret_cast`, so a numpy slice with stride > 1 must
    # flow through without complaint. The pre-stride lowering used
    # `memref.view` (which bakes identity strides) and would have
    # silently miscomputed against this input. The kernel body is empty
    # — we're not checking output values, just proving the strided
    # descriptor reaches the typed memref intact.
    script = tmp_path / "compile_invoke_strided.py"
    script.write_text(textwrap.dedent("""
            import numpy as np

            import hc
            from hc import Buffer, CurrentGroup, kernel


            class _TensorView:
                def __init__(self, arr):
                    self._arr = arr

                def data_ptr(self):
                    return int(self._arr.ctypes.data)

                def size(self, dim):
                    return int(self._arr.shape[dim])

                def stride(self, dim):
                    return int(self._arr.strides[dim] // self._arr.itemsize)


            sym = hc.sym


            @kernel(work_shape=(sym.W,), literals={sym.W})
            def trivial(group: CurrentGroup, x: Buffer[sym.W]) -> None:
                return None


            def main() -> None:
                handle = hc.compile(trivial, {sym.W: 128})
                strided_arr = np.zeros(256, dtype=np.float32)[::2]
                assert strided_arr.strides[0] // strided_arr.itemsize == 2
                handle(_TensorView(strided_arr))
                print("OK")


            if __name__ == "__main__":
                main()
            """))

    result = _run_compile_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout


def test_compile_invoke_raises_when_pipeline_failed() -> None:
    # `invoke` on a handle whose pipeline failed must surface the
    # captured diagnostics instead of segfaulting on a missing module.
    handle = CompiledKernel(
        kernel=lambda: None,
        bindings={},
        front_ir=None,
        front_ir_text="",
        hc_ir=None,
        hc_ir_text=None,
        pipeline_diagnostics=("simulated failure",),
    )
    with pytest.raises(RuntimeError, match="simulated failure"):
        handle.invoke()


def test_compile_invoke_rejects_kwargs() -> None:
    # Positional-only ABI for kernel arguments — the host wrapper has
    # no notion of named arguments beyond the dedicated `stream=` slot.
    # Reject other kwargs loudly instead of silently dropping them.
    handle = CompiledKernel(
        kernel=lambda: None,
        bindings={},
        front_ir=None,
        front_ir_text="",
        hc_ir=object(),
        hc_ir_text="",
    )
    with pytest.raises(TypeError, match="must be positional"):
        handle(x=1)
