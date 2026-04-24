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

import pytest

import hc.symbols as hs

REPO_ROOT = Path(__file__).resolve().parents[1]
_MISSING_IXSIMPL_ERROR_FRAGMENTS = (
    "Vendored ixsimpl backend is not built",
    "hc.symbols requires the ixsimpl submodule and a built vendored copy",
)
_SKIP_BUILD_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_IXSIMPL_BUILD_TESTS") == "1",
    reason="build-heavy ixsimpl integration tests disabled by env",
)


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


def _run_module(
    env: dict[str, str],
    module: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def _assert_missing_ixsimpl_error(stderr: str) -> None:
    assert any(fragment in stderr for fragment in _MISSING_IXSIMPL_ERROR_FRAGMENTS)


def test_import_hc_does_not_create_ixsimpl_vendor_dir(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    env = os.environ.copy()
    env["HC_IXSIMPL_VENDOR_DIR"] = str(vendor_dir)

    result = _run_python(
        env,
        """
        import json
        import os
        from pathlib import Path

        from hc import Buffer, sym

        print(
            json.dumps(
                {
                    "exists": Path(os.environ["HC_IXSIMPL_VENDOR_DIR"]).exists(),
                    "buffer": repr(Buffer[1]),
                    "sym": repr(sym),
                }
            )
        )
        """,
    )
    payload = json.loads(result.stdout)

    assert payload == {"exists": False, "buffer": "Buffer[1]", "sym": "sym"}


@_SKIP_BUILD_TESTS
def test_ixsimpl_toolchain_cli_bootstraps_vendor_dir(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    env = os.environ.copy()
    env["HC_IXSIMPL_VENDOR_DIR"] = str(vendor_dir)

    result = _run_module(env, "build_tools.ixsimpl_toolchain")

    assert Path(result.stdout.strip()) == vendor_dir
    assert (vendor_dir / ".hc-ixsimpl-stamp.json").exists()

    result = _run_python(
        env,
        """
        import json

        import hc.symbols as hs

        expr = str(hs.sym.W + 1)

        import ixsimpl

        print(json.dumps({"expr": expr, "module": ixsimpl.__file__}))
        """,
    )
    payload = json.loads(result.stdout)
    module_path = Path(payload["module"]).resolve()

    assert module_path.is_relative_to(vendor_dir.resolve())
    assert payload["expr"] == "1 + W"


def test_symbols_require_prebuilt_ixsimpl_vendor(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    env = os.environ.copy()
    env["HC_IXSIMPL_VENDOR_DIR"] = str(vendor_dir)

    result = _run_python(
        env,
        """
        import hc.symbols as hs

        hs.Context()
        """,
        check=False,
    )

    assert result.returncode != 0
    _assert_missing_ixsimpl_error(result.stderr)


def test_symbols_reject_invalid_ixsimpl_metadata(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    stamp_path = vendor_dir / ".hc-ixsimpl-stamp.json"
    env = os.environ.copy()
    env["HC_IXSIMPL_VENDOR_DIR"] = str(vendor_dir)

    vendor_dir.mkdir(parents=True)
    (vendor_dir / ".hc-ixsimpl-root").write_text("managed\n", encoding="utf-8")
    stamp_path.mkdir()

    result = _run_python(
        env,
        """
        import hc.symbols as hs

        hs.Context()
        """,
        check=False,
    )

    assert result.returncode != 0
    assert "Vendored ixsimpl backend metadata is invalid" in result.stderr


def test_default_namespace_caches_symbols() -> None:
    assert hs.sym.W is hs.sym.W


def test_context_namespace_caches_symbols() -> None:
    ctx = hs.Context()

    assert ctx.sym("W") is ctx.sym("W")
    assert ctx.sym("W") != hs.sym.W


def test_bindings_eval_shape_expression() -> None:
    expr = hs.ceil_div(hs.sym.W, 32) * hs.max_(hs.sym.H, 1)
    bindings = hs.Bindings()
    bindings.bind(hs.sym.W, 256, source="X.shape[0]")
    bindings.bind(hs.sym.H, 96, source="X.shape[1]")

    assert expr.eval(bindings.freeze()) == 768


def test_bindings_conflict_reports_sources() -> None:
    bindings = hs.Bindings()
    bindings.bind(hs.sym.W, 32, source="X.shape[0]")

    with pytest.raises(hs.SymbolConflictError, match=r"X.shape\[0\]"):
        bindings.bind(hs.sym.W, 64, source="Y.shape[0]")


def test_bindings_require_reports_mismatch() -> None:
    bindings = hs.Bindings()
    bindings.bind(hs.sym.W, 32, source="X.shape[0]")

    with pytest.raises(hs.SymbolConflictError, match="expected W == 64, got 32"):
        bindings.require(hs.sym.W, 64, source="launch")


def test_bindings_require_rejects_bare_bool() -> None:
    with pytest.raises(TypeError, match="bare bool"):
        hs.Bindings().require(True)


def test_check_uses_env_and_assumptions() -> None:
    pred = hs.sym.W > 0

    assert hs.check(pred, env={hs.sym.W: 3}) is hs.Truth.TRUE
    assert hs.check(pred, env={hs.sym.W: -1}) is hs.Truth.FALSE
    assert hs.check(pred, assumptions=[hs.sym.W >= 1]) is hs.Truth.TRUE


def test_ite_evaluates_branch_by_condition() -> None:
    expr = hs.ite(hs.sym.W > 0, hs.sym.W, -hs.sym.W)

    assert expr.eval({hs.sym.W: 5}) == 5
    assert expr.eval({hs.sym.W: -4}) == 4


def test_predicate_eval_rejects_unknown_truth_value() -> None:
    with pytest.raises(hs.NonConstantError, match="not constant"):
        (hs.sym.W > 0).eval({})


def test_explicit_eval_helpers_match_base_contracts() -> None:
    assert (hs.sym.W + 1).eval_int({hs.sym.W: 6}) == 7
    assert (hs.sym.W > 0).eval_bool({hs.sym.W: 6}) is True


def test_parse_free_symbols_and_subs_share_context() -> None:
    ctx = hs.Context()
    W = ctx.sym("W")
    expr = hs.parse("W + 2", ctx=ctx)

    assert expr.free_symbols() == frozenset({W})
    assert expr.subs({W: 7}).constant_value() == 9


def test_context_mismatch_raises() -> None:
    lhs = hs.Context().sym("W")
    rhs = hs.Context().sym("W")

    with pytest.raises(hs.ContextMismatchError):
        _ = lhs + rhs


def test_parse_reports_symbol_error() -> None:
    with pytest.raises(hs.SymbolError, match="parse error"):
        hs.parse("W +")


def test_eval_requires_all_symbol_bindings() -> None:
    expr = hs.sym.W + 1

    with pytest.raises(hs.UnboundSymbolError, match="W"):
        expr.eval({})


def test_expr_coercion_rejects_bool_operands() -> None:
    with pytest.raises(TypeError, match="boolean values"):
        _ = hs.sym.W + True


def test_predicate_apis_reject_expression_nodes() -> None:
    with pytest.raises(TypeError, match="expected predicate, got Symbol"):
        hs.check(hs.sym.W)


def test_rat_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        hs.rat(1, 0)


def test_preloaded_ixsimpl_from_unexpected_path_is_rejected(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HC_IXSIMPL_VENDOR_DIR"] = str(tmp_path / "vendor")

    result = _run_python(
        env,
        """
        import sys
        import types

        import hc.symbols as hs

        fake = types.ModuleType("ixsimpl")
        fake.__file__ = "/tmp/fake-ixsimpl.py"
        sys.modules["ixsimpl"] = fake

        hs.Context()
        """,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "pre-imported `ixsimpl` outside the managed vendor directory" in result.stderr
    )


def test_module_eq_ne_build_predicates() -> None:
    eq_pred = hs.eq(hs.sym.W, 4)
    ne_pred = hs.ne(hs.sym.W, 4)

    assert hs.check(eq_pred, env={hs.sym.W: 4}) is hs.Truth.TRUE
    assert hs.check(ne_pred, env={hs.sym.W: 4}) is hs.Truth.FALSE
