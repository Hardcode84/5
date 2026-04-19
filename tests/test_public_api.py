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
    @kernel.intrinsic(scope=SubGroup, effects="pure", const_attrs={"blocksz"})
    def mfma(a, b, acc, *, blocksz):
        return acc

    @mfma.lower(target="amdgpu")
    def _lower(*args, **kwargs):
        return (args, kwargs)

    @mfma.verify
    def _verify(sig, target):
        return (sig, target)

    @mfma.infer
    def _infer(sig):
        return Result(type=sig)

    assert mfma.__hc_intrinsic__.scope == SubGroup
    assert mfma.__hc_intrinsic__.const_attrs == frozenset({"blocksz"})
    assert "amdgpu" in mfma.__hc_lowerings__
    assert mfma.__hc_verify__ is _verify
    assert mfma.__hc_infer__ is _infer


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
