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
from hc import Buffer, as_layout, kernel, sym
from hc._frontend import FrontendError
from hc._resolve import ResolvedFrontIR, resolve_front_ir
from hc.core import index_map
from hc.symbols import ceil_div

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


# Module-scope fixtures for layout-param / `as_layout` body-call tests.
# PEP 563 stringified annotations need the referenced names resolvable
# from function globals → kernel + captured `IndexMap` must be at
# module scope; function-local defs hide them from
# `inspect.get_annotations(eval_str=True)`. Decorator names match
# `kernel` literally in the AST, so unaliased imports above.
_FIXTURE_M = sym.M
_FIXTURE_N = sym.N
_FIXTURE_K = sym.K
_FIXTURE_LAYOUT = index_map(
    storage_size=lambda w, h: w * h,
    offset=lambda i, j, w, h: i * h + j,
)
# Non-injective: rank-3 logical shape, flat storage = one column.
# Offset drops two of three index syms → distinct logical indices
# share a slot. `LayoutAttr` requires
# `index_syms.size() == shape_syms.size()`; non-injectivity is in
# the offset formula, not in extra index syms.
_FIXTURE_NONINJECTIVE_LAYOUT = index_map(
    storage_size=lambda d0, d1, d2: d1,
    offset=lambda i, j, lane, d0, d1, d2: j,
)


@kernel(work_shape=(ceil_div(_FIXTURE_M, 16),), group_shape=(16,))
def _param_layout_kernel(
    group, a: Buffer[_FIXTURE_M, _FIXTURE_N, np.float32, _FIXTURE_LAYOUT]
) -> None:
    return


@kernel(work_shape=(ceil_div(_FIXTURE_M, 16),), group_shape=(16,))
def _as_layout_body_kernel(group, a: Buffer[_FIXTURE_M, _FIXTURE_N]) -> None:
    v = group.vzeros(shape=(16,))
    _ = as_layout(v, _FIXTURE_LAYOUT)
    return


_FIXTURE_LANE = sym.LANE


@kernel(work_shape=(ceil_div(_FIXTURE_M, 16),), group_shape=(16,))
def _noninjective_layout_vload_kernel(
    group,
    a: Buffer[
        _FIXTURE_M, _FIXTURE_K, _FIXTURE_LANE, np.float16, _FIXTURE_NONINJECTIVE_LAYOUT
    ],
    i: sym.idx,
    j: sym.idx,
    lane: sym.idx,
) -> None:
    _ = group.vload(a[i, j, lane], shape=(16,))
    return


def _iter_ops(module: Any) -> Any:
    """DFS over every op in `module`, toplevels included."""
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
    """Read `op`'s `ref` DictAttr into a dict-of-strings.

    Nested `ArrayAttr` surface as MLIR text — fine for substring checks,
    no recursive decoder needed.
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
    """{identifier: [ref-dict, ...]} for every load-context `hc_front.name`."""
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
    """{attr-method: [ref-dict, ...]} for every `hc_front.attr` with a ref."""
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
    # Order matters: kernel first, then helpers/intrinsics and inline
    # helpers in BFS discovery order (globals + closurevars). Pin the
    # set + kernel-first position so dropped deps and module reshuffles
    # both fail. `inline_names` separately pins which top-levels were
    # marked `ref.kind = "inline"`.
    assert resolved.symbol_names[0] == "tiled_gfx11_wmma_matmul"
    assert set(resolved.exported_symbol_names) == {
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
        }
    )


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_wmma_stamps_every_name_load_with_ref() -> None:
    _ensure_hc_front_bindings_available()

    resolved = resolve_front_ir(tiled_gfx11_wmma_matmul)

    # Every load-context `hc_front.name` carries a ref;
    # `target_name` (stores) exempt.
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
    """Cache resolved WMMA refs — parametrized cases share one build."""
    _ensure_hc_front_bindings_available()
    resolved = resolve_front_ir(tiled_gfx11_wmma_matmul)
    return _name_refs(resolved.module), _attr_refs(resolved.module)


# One assertion per parametrize row → lizard-happy, failures point at
# one ref kind instead of a compound predicate.
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
    # `WMMA_K` is loaded at multiple sites; every occurrence resolves
    # to the same constant payload. Per-site noise (locs, ids) would
    # show as >1 distinct value here.
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
        # `np.empty` is a helper, not a dtype — must not mis-tag.
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

    # Tiled WMMA reads `a.shape[1]`, never `M`/`N`/`K` directly. Use
    # a minimal kernel that references a `Symbol` in expression
    # context to check the payload shape.
    from hc import kernel, sym
    from hc.symbols import ceil_div

    M = sym.M
    N = sym.N

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def refs_symbol(group, a: Buffer[M, N]) -> None:
        _ = (M, N)
        return

    resolved = resolve_front_ir(refs_symbol)
    name_refs = _name_refs(resolved.module)

    (m_ref,) = name_refs["M"]
    assert m_ref["kind"] == "symbol"
    assert m_ref["name"] == "M"


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_recognizes_live_numpy_scalar_dtypes() -> None:
    # `intp`/`uintp` are platform-dependent aliases — a hardcoded
    # dtype list misses them. Resolver delegates to live numpy so
    # every scalar type surfaces as `numpy_dtype_type` without
    # per-platform edits.
    _ensure_hc_front_bindings_available()

    from hc import kernel, sym
    from hc.symbols import ceil_div

    M = sym.M

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def k(group, a: Buffer[M]) -> None:
        _ = np.intp
        _ = np.empty(1, dtype=np.intp)
        return

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

    # Kernel needs a real file so `inspect.getsourcelines` finds it.
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

    # Unsupported capture (plain class instance) → "unclassifiable
    # capture" diagnostic naming identifier, type, kernel, and the
    # supported-capture hint.
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


# --- layout descriptors -----------------------------------------------------


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_index_map_capture_serializes_layout_payload() -> None:
    """Module-level `IndexMap` capture → `kind = "layout"` ref payload
    built from typed MLIR attrs: `#hc.expr` for `storage_size`/`offset`
    and each `params` entry, `ArrayAttr` of `StringAttr` for
    `shape_syms`/`index_syms`. No textual round-trip across Python ->
    C++. Asserting on MLIR-text spellings catches ixsimpl
    normalization drift.
    """
    _ensure_hc_front_bindings_available()

    from hc import kernel, sym
    from hc.core import index_map
    from hc.symbols import ceil_div

    M = sym.M
    N = sym.N

    A_LAYOUT = index_map(
        params=lambda w, h: {"row_stride": h + 4},
        storage_size=lambda w, h, p: w * p["row_stride"],
        offset=lambda i, j, w, h, p: i * p["row_stride"] + j,
    )

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def uses_layout(group, a: Buffer[M, N]) -> None:
        _ = A_LAYOUT
        return

    resolved = resolve_front_ir(uses_layout)
    name_refs = _name_refs(resolved.module)

    (ref,) = name_refs["A_LAYOUT"]
    assert ref["kind"] == "layout"
    assert ref["shape_syms"] == '["w", "h"]'
    assert ref["index_syms"] == '["i", "j"]'
    assert ref["params"] == '{row_stride = #hc.expr<"4 + h">}'
    assert ref["storage_size"] == '#hc.expr<"row_stride*w">'
    assert ref["offset"] == '#hc.expr<"j + i*row_stride">'


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_index_map_without_params_emits_empty_table() -> None:
    """`IndexMap` without `params`: `storage_size`/`offset` see only
    shape/index syms, ref carries empty `params` dict. Default-strided
    layouts already do this C++-side; round-trip the same shape from
    Python.
    """
    _ensure_hc_front_bindings_available()

    from hc import kernel, sym
    from hc.core import index_map
    from hc.symbols import ceil_div

    M = sym.M
    N = sym.N

    DENSE = index_map(
        storage_size=lambda m, n: m * n,
        offset=lambda i, j, m, n: i * n + j,
    )

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def uses_dense(group, a: Buffer[M, N]) -> None:
        _ = DENSE
        return

    resolved = resolve_front_ir(uses_dense)
    name_refs = _name_refs(resolved.module)

    (ref,) = name_refs["DENSE"]
    assert ref["kind"] == "layout"
    assert ref["shape_syms"] == '["m", "n"]'
    assert ref["index_syms"] == '["i", "j"]'
    assert ref["params"] == "{}"
    assert ref["storage_size"] == '#hc.expr<"m*n">'
    assert ref["offset"] == '#hc.expr<"j + i*n">'


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_index_map_noninjective_layout_serializes() -> None:
    """Non-injective offset (multiple logical indices → one slot)
    serializes like any rank-balanced layout. `LayoutAttr` requires
    `index_syms.size() == shape_syms.size()`; offset/storage just
    drop some index axes — no selector-specific encoding.
    """
    _ensure_hc_front_bindings_available()

    from hc import kernel, sym
    from hc.core import index_map
    from hc.symbols import ceil_div

    M = sym.M
    K = sym.K
    LANE = sym.LANE

    # WMMA-A-fragment-ish: rank-3 logical (M, K, LANE), flat storage =
    # K column. Multiple (i, lane) share each slot → per-lane broadcast.
    A_FRAG = index_map(
        storage_size=lambda d0, d1, d2: d1,
        offset=lambda i, j, lane, d0, d1, d2: j,
    )

    @kernel(work_shape=(ceil_div(M, 16),), group_shape=(16,))
    def uses_a_frag(group, a: Buffer[M, K, LANE]) -> None:
        _ = A_FRAG
        return

    resolved = resolve_front_ir(uses_a_frag)
    name_refs = _name_refs(resolved.module)

    (ref,) = name_refs["A_FRAG"]
    assert ref["kind"] == "layout"
    assert ref["shape_syms"] == '["d0", "d1", "d2"]'
    assert ref["index_syms"] == '["i", "j", "lane"]'
    assert ref["params"] == "{}"
    assert ref["storage_size"] == '#hc.expr<"d1">'
    assert ref["offset"] == '#hc.expr<"j">'


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_resolve_as_layout_capture_classifies_as_layout_op() -> None:
    """`as_layout` is a DSL primitive, not an inlinable helper. Resolver
    emits `layout_op` ref so `hc-front-to-hc` recognizes the call site
    as `hc.as_layout`. Must NOT walk it as a BFS dep — that would
    re-parse its dispatcher body as kernel source.
    """
    _ensure_hc_front_bindings_available()

    from hc import as_layout, kernel, sym
    from hc.symbols import ceil_div

    M = sym.M
    N = sym.N

    @kernel(work_shape=(ceil_div(M, 4),), group_shape=(4,))
    def uses_as_layout(group, a: Buffer[M, N]) -> None:
        _ = as_layout
        return

    resolved = resolve_front_ir(uses_as_layout)
    name_refs = _name_refs(resolved.module)

    (ref,) = name_refs["as_layout"]
    assert ref == {"kind": "layout_op", "op": "as_layout"}
    assert "as_layout" not in resolved.inline_names


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_layout_kwarg_stamps_attr_on_producer() -> None:
    """End-to-end Python → hc_front → hc: kernel calling
    `group.vzeros(shape=..., layout=A_LAYOUT)` reaches
    `-convert-hc-front-to-hc` and surfaces as `hc.vzeros` with
    `layout = #hc.layout<...>` on the producer (no intermediate
    `hc.as_layout`). LIT pins the C++ consumer on hand-written
    hc_front IR; this test pins the Python-side boundary.
    """
    import subprocess

    from hc import kernel, sym
    from hc._native_paths import hc_opt_path
    from hc.core import index_map
    from hc.symbols import ceil_div

    _ensure_hc_front_bindings_available()

    M = sym.M
    N = sym.N

    A_LAYOUT = index_map(
        storage_size=lambda w, h: w * h,
        offset=lambda i, j, w, h: i * h + j,
    )

    @kernel(work_shape=(ceil_div(M, 16),), group_shape=(16,))
    def uses_layout_kwarg(group, a: Buffer[M, N]) -> None:
        _ = group.vzeros(shape=(16,), layout=A_LAYOUT)
        return

    resolved = resolve_front_ir(uses_layout_kwarg)
    front_text = str(resolved.module)

    result = subprocess.run(
        [str(hc_opt_path()), "--convert-hc-front-to-hc"],
        input=front_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
    ), f"hc-opt failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    hc_text = result.stdout
    assert "hc.vzeros" in hc_text, hc_text
    # Layout lands as `layout = #hc.layout<...>` on the producer; no
    # `hc.as_layout` for `layout=` kwargs.
    assert "hc.as_layout" not in hc_text, hc_text
    assert "layout = #hc.layout<" in hc_text, hc_text
    # ixsimpl reorders commuting factors (`h*w` vs `w*h`); assert
    # presence/shape, not verbatim spelling.
    assert 'index_syms = ["i", "j"]' in hc_text, hc_text
    assert 'shape_syms = ["w", "h"]' in hc_text, hc_text
    assert "storage_size = #hc.expr<" in hc_text, hc_text
    assert "offset = #hc.expr<" in hc_text, hc_text


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_noninjective_layout_lands_on_vload_with_full_bind() -> None:
    """Buffer with non-injective layout reaches `hc.vload` through
    standard subscript-then-vload. Slicing via `a[i, j, lane]` →
    `hc.vload` after `--convert-hc-front-to-hc` with one index
    operand per logical axis (full positional bind). Pins resolver +
    frontend lowering + layout serialization on the uniform
    `index_syms.size() == shape_syms.size()` invariant.
    """
    import subprocess

    from hc._native_paths import hc_opt_path

    _ensure_hc_front_bindings_available()

    resolved = resolve_front_ir(_noninjective_layout_vload_kernel)
    front_text = str(resolved.module)
    # Resolver: kernel-param layout carries all three logical axes;
    # body-call subscript stays positional.
    assert 'index_syms = ["i", "j", "lane"]' in front_text, front_text

    result = subprocess.run(
        [str(hc_opt_path()), "--convert-hc-front-to-hc"],
        input=front_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
    ), f"hc-opt failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    hc_text = result.stdout
    # BufferType's LayoutAttr is the captured non-injective layout
    # (no default-strided fallback).
    assert "$STRIDE_" not in hc_text, hc_text
    assert 'index_syms = ["i", "j", "lane"]' in hc_text, hc_text
    # vload wires three index operands — one per logical axis of the
    # rank-3 source.
    assert "hc.vload" in hc_text, hc_text
    vload_line = next(line for line in hc_text.splitlines() if "hc.vload" in line)
    operand_block = vload_line.split("[", 1)[1].split("]", 1)[0]
    assert operand_block.count(",") == 2, vload_line


def test_buffer_class_getitem_captures_trailing_index_map_as_layout() -> None:
    """`Buffer[..., IndexMap]` → `BufferSpec.layout = IndexMap`;
    `Buffer[..., dtype]` → `layout = None`. Position is detected by
    `isinstance(IndexMap)` — `[]` can't pass kwargs.
    """
    from hc import Buffer, sym
    from hc.core import BufferSpec, index_map

    L = index_map(
        storage_size=lambda w, h: w * h,
        offset=lambda i, j, w, h: i * h + j,
    )

    plain = Buffer[sym.M, sym.N, np.float32]
    assert isinstance(plain, BufferSpec)
    assert plain.layout is None

    with_layout = Buffer[sym.M, sym.N, np.float32, L]
    assert isinstance(with_layout, BufferSpec)
    assert with_layout.layout is L
    assert with_layout.dtype == "float32"
    assert tuple(str(d) for d in with_layout.dimensions) == ("M", "N")


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_buffer_layout_lands_on_kernel_parameter_dict() -> None:
    """Kernel param annotated `Buffer[..., IndexMap]` round-trips through
    `-convert-hc-front-to-hc`: captured layout becomes `BufferType`'s
    `LayoutAttr`, replacing the default `$STRIDE_<i>_<argname>`. Pins
    the resolver-side `layout` payload key set against the C++
    `layoutAttrFromRef` reader.
    """
    import subprocess

    from hc._native_paths import hc_opt_path

    _ensure_hc_front_bindings_available()

    resolved = resolve_front_ir(_param_layout_kernel)
    front_text = str(resolved.module)
    # Resolver: param dict carries the structured layout sub-dict
    # matching the body-level ref shape (kind + 5 keys).
    assert 'kind = "layout"' in front_text, front_text
    assert 'shape_syms = ["w", "h"]' in front_text, front_text
    assert 'index_syms = ["i", "j"]' in front_text, front_text

    result = subprocess.run(
        [str(hc_opt_path()), "--convert-hc-front-to-hc"],
        input=front_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
    ), f"hc-opt failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    hc_text = result.stdout
    # C++: `BufferType` `LayoutAttr` is the captured one, not the
    # default. Default-strided emits `$STRIDE_<i>_<argname>`
    # symbols; absence pins the override path.
    assert "$STRIDE_" not in hc_text, hc_text
    assert 'shape_syms = ["w", "h"]' in hc_text, hc_text
    assert 'index_syms = ["i", "j"]' in hc_text, hc_text


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_as_layout_body_call_emits_hc_as_layout() -> None:
    """Body-level `as_layout(value, A_LAYOUT)`: AST walker emits generic
    `hc_front.call`; resolver stamps callee `kind = "layout_op"` and
    descriptor `kind = "layout"`; `-convert-hc-front-to-hc` produces
    `hc.as_layout` carrying the structured layout attribute.
    """
    import subprocess

    from hc._native_paths import hc_opt_path

    _ensure_hc_front_bindings_available()

    resolved = resolve_front_ir(_as_layout_body_kernel)
    front_text = str(resolved.module)

    result = subprocess.run(
        [str(hc_opt_path()), "--convert-hc-front-to-hc"],
        input=front_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
    ), f"hc-opt failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    hc_text = result.stdout
    assert "hc.as_layout" in hc_text, hc_text
    assert 'shape_syms = ["w", "h"]' in hc_text, hc_text
    assert 'index_syms = ["i", "j"]' in hc_text, hc_text
    assert "storage_size = #hc.expr<" in hc_text, hc_text
    assert "offset = #hc.expr<" in hc_text, hc_text


def test_index_map_classifier_diagnoses_bad_signature() -> None:
    """Lambdas with varargs / kw-only / defaults reject with a located
    error. Both the simulator and the symbolic evaluator bind by
    positional slot; anything else makes slot semantics ambiguous.
    """
    from hc._resolve import _classify_index_map
    from hc.core import index_map

    L = index_map(
        storage_size=lambda *shape: shape[0],
        offset=lambda *args: args[0],
    )
    with pytest.raises(FrontendError) as exc_info:
        _classify_index_map("L", L)
    assert "L" in str(exc_info.value)
    assert "positional parameters only" in str(exc_info.value)


def test_index_map_classifier_diagnoses_mismatched_shape_names() -> None:
    """`offset`'s trailing shape slots must name-match
    `params`/`storage_size`. Typo would silently rebind the layout
    to the wrong shape sym; classifier rejects.
    """
    from hc._resolve import _classify_index_map
    from hc.core import index_map

    L = index_map(
        params=lambda w, h: {"s": w + h},
        storage_size=lambda w, h, p: w * p["s"],
        offset=lambda i, j, w, hh, p: i * p["s"] + j,
    )
    with pytest.raises(FrontendError) as exc_info:
        _classify_index_map("L", L)
    msg = str(exc_info.value)
    assert "shape parameters" in msg
    assert "disagree" in msg


def test_index_map_classifier_accepts_free_syms() -> None:
    """`free_syms` keeps names as free symbols in `offset`/`storage_size`
    so the lowering pipeline binds them from kernel scope (kernel-arg
    aux, ancestor block argument, launch geometry).
    """
    from hc._resolve import _index_map_ref
    from hc.core import index_map

    L = index_map(
        storage_size=lambda M, N: M * N,
        offset=lambda i, j, M, N, *, row0, col0: (row0 + i) * N + col0 + j,
        free_syms=("row0", "col0"),
    )
    ref = _index_map_ref(L)
    assert ref["kind"] == "layout"
    assert ref["shape_syms"] == ("M", "N")
    assert ref["index_syms"] == ("i", "j")
    # Free sym names appear as bare leaves in the offset expr. `ixsimpl`
    # rearranges terms; assert presence, not canonical form.
    offset_text = str(ref["offset"])
    assert "row0" in offset_text
    assert "col0" in offset_text
    # `storage_size` stays pure — no free syms.
    assert str(ref["storage_size"]) == "M*N"


def test_index_map_classifier_rejects_undeclared_kwonly() -> None:
    """Kw-only params on a layout lambda must be declared in `free_syms`
    — that's the only kw-only slot a layout claims. Undeclared name is
    almost always a typo for a shape sym or a missed `free_syms` entry.
    """
    from hc._resolve import _classify_index_map
    from hc.core import index_map

    L = index_map(
        storage_size=lambda M, N: M * N,
        offset=lambda i, j, M, N, *, row0: row0 + i * N + j,
        free_syms=(),
    )
    with pytest.raises(FrontendError) as exc_info:
        _classify_index_map("L", L)
    msg = str(exc_info.value)
    assert "row0" in msg
    assert "free_syms" in msg


def test_index_map_classifier_rejects_free_sym_collision() -> None:
    """`free_syms` shares the dialect-side `LayoutAttr` name pool with
    `shape_syms`/`index_syms`/`params`. Catch collisions Python-side
    so the diagnostic points at the classifier, not a generic MLIR
    dup-name error.
    """
    from hc._resolve import _classify_index_map
    from hc.core import index_map

    L = index_map(
        storage_size=lambda M, N: M * N,
        # `M` is a shape sym. `_layout_invoke`'s subset filter never
        # asks for it on the kw-only side, so the collision check has
        # to fire before lambda inspection.
        offset=lambda i, j, M, N: i * N + j,
        free_syms=("M",),
    )
    with pytest.raises(FrontendError) as exc_info:
        _classify_index_map("L", L)
    msg = str(exc_info.value)
    assert "free_syms" in msg
    assert "'M'" in msg
