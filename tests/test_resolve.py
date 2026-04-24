# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.llvm_toolchain import ensure_llvm_toolchain
from examples.amdgpu_gfx11_wmma_matmul import tiled_gfx11_wmma_matmul
from hc._frontend import FrontendError
from hc._resolve import ResolvedFrontIR, resolve_front_ir

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)


@lru_cache(maxsize=1)
def _ensure_hc_front_bindings_available() -> None:
    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    os.environ.update(
        export_hc_native_environment(native_install_root, dict(os.environ))
    )


def _iter_ops(module: Any) -> Any:
    """Depth-first walk over every op in ``module``; includes toplevels."""
    stack = [(op, False) for op in module.body.operations]
    while stack:
        op, _ = stack.pop()
        yield op
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    stack.append((inner, False))


def _str_attr(attrs: Any, key: str) -> str | None:
    if key not in attrs:
        return None
    value = attrs[key]
    inner = getattr(value, "value", None)
    if isinstance(inner, str):
        return inner
    text = str(value)
    return text[1:-1] if text.startswith('"') and text.endswith('"') else text


def _ref_dict(op: Any) -> dict[str, str] | None:
    """Read the ``ref`` DictAttr off ``op`` into a plain dict-of-strings.

    Values are rendered via ``str`` so nested `ArrayAttr`s surface as their
    MLIR text form; identity is good enough for substring checks in tests
    and avoids hand-rolling a recursive decoder.
    """
    attrs = op.operation.attributes
    if "ref" not in attrs:
        return None
    out: dict[str, str] = {}
    for named in attrs["ref"]:
        inner = getattr(named.attr, "value", None)
        out[named.name] = inner if isinstance(inner, str) else str(named.attr)
    return out


def _name_refs(module: Any) -> dict[str, list[dict[str, str]]]:
    """{identifier: [ref-dict, ...]} for every load-context ``hc_front.name``."""
    out: dict[str, list[dict[str, str]]] = {}
    for op in _iter_ops(module):
        if op.operation.name != "hc_front.name":
            continue
        ref = _ref_dict(op)
        if ref is None:
            continue
        ident = _str_attr(op.operation.attributes, "name") or ""
        out.setdefault(ident, []).append(ref)
    return out


def _attr_refs(module: Any) -> dict[str, list[dict[str, str]]]:
    """{attr-method: [ref-dict, ...]} for every ``hc_front.attr`` with a ref."""
    out: dict[str, list[dict[str, str]]] = {}
    for op in _iter_ops(module):
        if op.operation.name != "hc_front.attr":
            continue
        ref = _ref_dict(op)
        if ref is None:
            continue
        method = _str_attr(op.operation.attributes, "name") or ""
        out.setdefault(method, []).append(ref)
    return out


def _ref_matches(ref: dict[str, str], needles: dict[str, str]) -> bool:
    return all(ref.get(key) == val for key, val in needles.items())


# --- input validation -------------------------------------------------------


def test_resolve_rejects_non_kernel() -> None:
    def not_a_kernel() -> None:
        return None

    with pytest.raises(TypeError, match="@kernel-decorated"):
        resolve_front_ir(not_a_kernel)


# --- end-to-end WMMA --------------------------------------------------------


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_wmma_collects_full_dep_set() -> None:
    _ensure_hc_front_bindings_available()

    resolved = resolve_front_ir(tiled_gfx11_wmma_matmul)

    assert isinstance(resolved, ResolvedFrontIR)
    # Order matters: the kernel emits first, then helpers/intrinsics and
    # undecorated inline helpers in BFS discovery order (globals +
    # closurevars). Pin both the set and the relative kernel-first
    # position so a regression that drops a dep is caught alongside one
    # that reshuffles the module. `inline_names` separately pins the
    # inline subset so the resolver stays honest about which top-levels
    # were marked `ref.kind = "inline"`.
    assert resolved.symbol_names[0] == "tiled_gfx11_wmma_matmul"
    assert set(resolved.decorated_symbol_names) == {
        "tiled_gfx11_wmma_matmul",
        "init_wmma_acc",
        "issue_wmma_tile",
        "store_wmma_tile",
        "load_wmma_a_fragment",
        "load_wmma_b_fragment",
        "wmma_gfx11",
    }
    assert resolved.inline_names == frozenset(
        {
            "_tile_origin",
            "_lane_a_row",
            "_lane_column",
            "_lane_output_rows",
            "_lane_output_row_step",
            "_lane_output_row_slice_args",
        }
    )


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_wmma_stamps_every_name_load_with_ref() -> None:
    _ensure_hc_front_bindings_available()

    resolved = resolve_front_ir(tiled_gfx11_wmma_matmul)

    # Every load-context ``hc_front.name`` must carry a ref. Stores
    # (``target_name``) are exempt.
    load_count = 0
    for op in _iter_ops(resolved.module):
        if op.operation.name != "hc_front.name":
            continue
        if _str_attr(op.operation.attributes, "ctx") != "load":
            continue
        load_count += 1
        assert _ref_dict(op) is not None, f"name op missing ref: {op}"
    assert load_count > 0, "expected at least one load-context name op"


@lru_cache(maxsize=1)
def _wmma_refs() -> (
    tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]
):
    """Cache the resolved WMMA refs — each parametrized case reuses one build."""
    _ensure_hc_front_bindings_available()
    resolved = resolve_front_ir(tiled_gfx11_wmma_matmul)
    return _name_refs(resolved.module), _attr_refs(resolved.module)


# One assertion per parametrize row: keeps lizard happy and makes failures
# point at exactly one ref kind instead of a 20-clause compound predicate.
@_SKIP_HC_FRONT_DIALECT_TESTS
@pytest.mark.parametrize(
    ("identifier", "needles"),
    [
        ("group", {"kind": "param"}),
        ("k0", {"kind": "iv"}),
        ("row0", {"kind": "local"}),
        ("init_wmma_acc", {"kind": "callee", "callee": "@init_wmma_acc"}),
        (
            "wmma_gfx11",
            {
                "kind": "intrinsic",
                "callee": "@wmma_gfx11",
                "effects": "pure",
                "const_kwargs": '["arch", "wave_size"]',
            },
        ),
        (
            "_tile_origin",
            {
                "kind": "inline",
                "qualified_name": "examples.amdgpu_gfx11_wmma_matmul._tile_origin",
            },
        ),
        ("range", {"kind": "builtin", "builtin": "range"}),
        ("np", {"kind": "module", "module": "numpy"}),
    ],
)
def test_resolve_wmma_name_ref_has_expected_kind(
    identifier: str, needles: dict[str, str]
) -> None:
    name_refs, _ = _wmma_refs()
    assert any(_ref_matches(ref, needles) for ref in name_refs[identifier])


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_wmma_constant_ref_is_stable() -> None:
    # ``WMMA_K`` is loaded at multiple sites; every occurrence must resolve
    # to the same single constant payload. A regression that emitted
    # per-site noise (location tokens, ids) would surface as >1 distinct
    # value here.
    name_refs, _ = _wmma_refs()
    distinct = {tuple(sorted(ref.items())) for ref in name_refs["WMMA_K"]}
    assert len(distinct) == 1, distinct
    const_ref = name_refs["WMMA_K"][0]
    assert const_ref["kind"] == "constant"
    assert const_ref["python_kind"] == "int"
    assert const_ref["value"] == "16"


@_SKIP_HC_FRONT_DIALECT_TESTS
@pytest.mark.parametrize(
    ("attr_name", "needles"),
    [
        ("load", {"kind": "dsl_method"}),
        ("group_id", {"kind": "dsl_method"}),
        ("float16", {"kind": "numpy_dtype_type", "dtype": "float16"}),
        # ``np.empty`` is a helper function, not a dtype — shouldn't mis-tag.
        ("empty", {"kind": "numpy_attr"}),
    ],
)
def test_resolve_wmma_attr_ref_has_expected_kind(
    attr_name: str, needles: dict[str, str]
) -> None:
    _, attr_refs = _wmma_refs()
    assert any(_ref_matches(ref, needles) for ref in attr_refs[attr_name])


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_wmma_symbols_get_symbol_ref() -> None:
    _ensure_hc_front_bindings_available()

    # Tiled WMMA doesn't load `M`/`N`/`K` directly inside its body (it reads
    # `a.shape[1]` instead), so build a minimal kernel that does reference a
    # `Symbol` in an expression context and check the payload shape.
    from hc import Buffer, kernel, sym
    from hc.symbols import ceil_div

    M = sym.M
    N = sym.N

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def refs_symbol(group, a: Buffer[M, N]) -> None:
        _ = (M, N)
        return None

    resolved = resolve_front_ir(refs_symbol)
    name_refs = _name_refs(resolved.module)

    (m_ref,) = name_refs["M"]
    assert m_ref["kind"] == "symbol"
    assert m_ref["name"] == "M"


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_recognizes_live_numpy_scalar_dtypes() -> None:
    # ``intp`` / ``uintp`` are platform-dependent aliases that a hardcoded
    # dtype list would miss. Check we delegate to the live numpy module so
    # every concrete scalar type surfaces as ``numpy_dtype_type`` without
    # per-platform edits to the resolver.
    _ensure_hc_front_bindings_available()

    from hc import Buffer, kernel, sym
    from hc.symbols import ceil_div

    M = sym.M

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def k(group, a: Buffer[M]) -> None:
        _ = np.intp
        _ = np.empty(1, dtype=np.intp)
        return None

    resolved = resolve_front_ir(k)
    attr_refs = _attr_refs(resolved.module)
    assert any(
        ref == {"kind": "numpy_dtype_type", "dtype": "intp"}
        for ref in attr_refs["intp"]
    ), attr_refs["intp"]


# --- diagnostics ------------------------------------------------------------


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_raises_on_unresolved_capture(tmp_path: Path) -> None:
    _ensure_hc_front_bindings_available()

    # Define the kernel in a real file so `inspect.getsourcelines` can see
    # it (identical constraint to test_hc_compile.py's end-to-end smoke).
    import importlib.util
    import sys

    script = tmp_path / "unresolved.py"
    script.write_text(
        "import hc\n"
        "from hc import Buffer, kernel, sym\n"
        "\n"
        "M = sym.M\n"
        "\n"
        "@kernel(work_shape=(M,), group_shape=(1,))\n"
        "def bad(group, a: Buffer[M]):\n"
        "    return mystery_undefined_name(a)\n"
    )
    spec = importlib.util.spec_from_file_location("unresolved_mod", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["unresolved_mod"] = module
    try:
        spec.loader.exec_module(module)
        with pytest.raises(FrontendError) as exc_info:
            resolve_front_ir(module.bad)
    finally:
        sys.modules.pop("unresolved_mod", None)

    msg = str(exc_info.value)
    assert "mystery_undefined_name" in msg
    assert "bad" in msg
    assert str(script) in msg


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_raises_on_unclassifiable_capture(tmp_path: Path) -> None:
    _ensure_hc_front_bindings_available()

    # Capture an unsupported Python value (a plain class instance) to exercise
    # the "unclassifiable capture" path. The error message should name the
    # offending identifier, its Python type, the kernel, and hint at what
    # captures *are* supported so the user can fix their code.
    import importlib.util
    import sys

    script = tmp_path / "unclassifiable.py"
    script.write_text(
        "import hc\n"
        "from hc import Buffer, kernel, sym\n"
        "\n"
        "M = sym.M\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    pass\n"
        "\n"
        "\n"
        "WIDGET = Widget()\n"
        "\n"
        "\n"
        "@kernel(work_shape=(M,), group_shape=(1,))\n"
        "def bad(group, a: Buffer[M]) -> None:\n"
        "    _ = WIDGET\n"
        "    return None\n"
    )
    spec = importlib.util.spec_from_file_location("unclassifiable_mod", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["unclassifiable_mod"] = module
    try:
        spec.loader.exec_module(module)
        with pytest.raises(FrontendError) as exc_info:
            resolve_front_ir(module.bad)
    finally:
        sys.modules.pop("unclassifiable_mod", None)

    msg = str(exc_info.value)
    assert "WIDGET" in msg
    assert "Widget" in msg
    assert "bad" in msg
    assert "@kernel.func" in msg or "hc.symbols" in msg
