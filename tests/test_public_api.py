# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from hc import (
    Buffer,
    CurrentGroup,
    Result,
    SubGroup,
    WorkGroup,
    WorkItem,
    as_layout,
    index_map,
    kernel,
    undef_type,
)


def _sym():
    from hc import sym

    return sym


def test_symbol_namespace_caches_by_name() -> None:
    sym = _sym()
    assert sym.W is sym.W
    assert str(sym.W) == "W"


def test_buffer_annotation_captures_dimensions() -> None:
    sym = _sym()
    spec = Buffer[sym.W, 3]
    assert spec.dimensions == (sym.W, 3)
    assert repr(spec) == "Buffer[W, 3]"


def test_kernel_decorator_stores_metadata() -> None:
    sym = _sym()

    @kernel(work_shape=(sym.W,), literals={sym.W})
    def foo(group: CurrentGroup, x: Buffer[sym.W]) -> None:
        return None

    metadata = foo.__hc_kernel__
    assert metadata.work_shape == (sym.W,)
    assert metadata.literals == frozenset({sym.W})


def test_helper_decorator_stores_scope() -> None:
    @kernel.func(scope=WorkGroup)
    def helper(x: int) -> int:
        return x

    assert helper.__hc_func__.scope == WorkGroup


def test_intrinsic_decorator_registers_hooks() -> None:
    @kernel.intrinsic(
        scope=SubGroup,
        effects="pure",
        const_attrs={"blocksz"},
        result_types=(undef_type(),),
    )
    def mfma(a, b, acc, *, blocksz):
        return acc

    @mfma.lower(target="amdgpu")
    def _lower(t, call):
        op = t.create(
            "test.mfma",
            operands=[call.operand("a"), call.operand(1), call.operand("acc")],
            result_types=[call.result_type(0)],
            attrs={"blocksz": call.attr("blocksz")},
        )
        return op.result(0)

    @mfma.verify
    def _verify(sig, target):
        return (sig, target)

    @mfma.infer
    def _infer(sig):
        return Result(type=sig)

    assert mfma.__hc_intrinsic__.scope == SubGroup
    assert mfma.__hc_intrinsic__.const_attrs == frozenset({"blocksz"})
    assert "amdgpu" in mfma.__hc_lowerings__
    recipe = mfma.__hc_lowerings__["amdgpu"]
    record = recipe.to_record()
    assert record["intrinsic"] == "mfma"
    assert record["target"] == "amdgpu"
    assert recipe.steps[0].op_name == "test.mfma"
    assert [value.name for value in recipe.steps[0].operands] == [
        "operand_a",
        "operand_b",
        "operand_acc",
    ]
    assert recipe.replacement[0].name == "created0_0"
    assert mfma.__hc_verify__ is _verify
    assert mfma.__hc_infer__ is _infer


def test_wmma_lowering_records_transform_recipe() -> None:
    from examples.amdgpu_gfx11_wmma_matmul import wmma_gfx11
    from hc._intrinsic_recipes import (
        RecipeCreateStep,
        RecipeRequireAttrStep,
        TypedIntAttr,
    )

    recipe = wmma_gfx11.__hc_lowerings__["amdgpu-gfx11"]
    assert recipe.intrinsic_name == "wmma_gfx11"
    require_steps = [s for s in recipe.steps if isinstance(s, RecipeRequireAttrStep)]
    create_steps = [s for s in recipe.steps if isinstance(s, RecipeCreateStep)]
    # The recipe asserts both `arch` and `wave_size` before letting the
    # rewrite touch the call. Match the literal types/values the frontend
    # emits at the call site (`arch` as a plain string, `wave_size` at i64
    # because the call-site attribute is `wave_size = 32 : i64`).
    require_by_name = {step.name: step for step in require_steps}
    assert set(require_by_name) == {"arch", "wave_size"}
    assert require_by_name["arch"].expected == "gfx11"
    wave_expected = require_by_name["wave_size"].expected
    assert isinstance(wave_expected, TypedIntAttr)
    assert wave_expected.width == 64
    assert wave_expected.value == 32

    create = create_steps[0]
    assert create.op_name == "amdgpu.wmma"
    assert [value.name for value in create.operands] == [
        "operand_a_frag",
        "operand_b_frag",
        "operand_acc_frag",
    ]
    # `amdgpu.wmma` only takes `m`/`n`/`k` (i32). `arch`/`wave_size` ride
    # on the call site for dispatch but never make it onto the created op.
    attrs = dict(create.attrs)
    assert set(attrs) == {"m", "n", "k"}
    for name, value in attrs.items():
        assert isinstance(value, TypedIntAttr), name
        assert value.width == 32, name
        assert value.value == 16, name
    assert recipe.replacement[0].name == "created0_0"
    text = recipe.to_mlir()
    assert 'transform.hc.create_op "amdgpu.wmma"' in text
    assert "transform.hc.require_intrinsic_attr" in text
    assert 'expected = "gfx11"' in text
    assert "expected = 32 : i64" in text
    assert "k = 16 : i32" in text
    assert "m = 16 : i32" in text
    assert "n = 16 : i32" in text


def test_index_map_records_callables() -> None:
    layout = index_map(
        params=lambda w, h: {"row_stride": h},
        storage_size=lambda w, h, p: w * p["row_stride"],
        offset=lambda i, j, w, h, p: i * p["row_stride"] + j,
    )
    assert layout.params(2, 4) == {"row_stride": 4}


def test_as_layout_delegates_to_value_method() -> None:
    class Dummy:
        def as_layout(self, layout):
            return layout

    layout = index_map(storage_size=lambda n: n, offset=lambda i, n: i)

    assert as_layout(Dummy(), layout) is layout


def test_current_group_reports_size() -> None:
    group = CurrentGroup(shape=(4, 8))
    assert group.size == 32


def test_region_decorators_return_original_function() -> None:
    group = CurrentGroup()

    @group.workitems
    def inner(wi: WorkItem) -> WorkItem:
        return wi

    assert inner is group.workitems(inner)
