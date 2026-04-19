# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import hc.simulator as sim
from hc import Buffer, kernel, sym

_EXPECTED_SUBGROUP_AND_WORKITEM_STATE = np.array(
    [
        [1, 3],
        [3, 5],
        [4, 6],
        [0, 2],
    ],
    dtype=np.int64,
)


@kernel(work_shape=(4, 2), group_shape=(4, 2), subgroup_size=2)
def _staged_subgroups_and_workitems_kernel(group, dst: Buffer[4, 2]) -> None:
    subgroup_tags = group.full(shape=group.shape, dtype=np.int64, fill_value=-1)
    scratch = group.zeros(shape=group.shape, dtype=np.int64)

    @group.subgroups
    def mark(sg) -> None:
        blocks_per_row = group.shape[0] // sg.size()
        block = sg.subgroup_id() % blocks_per_row
        row = sg.subgroup_id() // blocks_per_row
        start = block * sg.size()
        subgroup_tags[start : start + sg.size(), row] = sg.subgroup_id()

    mark()

    @group.workitems
    def inner(wi) -> None:
        lid0, lid1 = wi.local_id()
        gid0, gid1 = wi.global_id()
        scratch[lid0, lid1] = subgroup_tags[lid0, lid1] + gid0
        group.barrier()
        dst[gid0, gid1] = scratch[(lid0 + 1) % group.shape[0], lid1]

    inner()


def test_launch_runs_workgroup_pairwise_kernel() -> None:
    W1 = sym.W1
    W2 = sym.W2
    H = sym.H

    @kernel(work_shape=(W1, W2), group_shape=(4, 4))
    def pairwise(
        group,
        x1: Buffer[W1, H],
        x2: Buffer[W2, H],
        out: Buffer[W1, W2],
    ) -> None:
        gid = group.work_offset
        lhs = group.load(x1[gid[0] :], shape=(group.shape[0], x1.shape[1]))
        rhs = group.load(x2[gid[1] :], shape=(group.shape[1], x2.shape[1]))
        diff = ((lhs[:, None, :] - rhs[None, :, :]) ** 2).sum(axis=2)
        group.store(out[gid[0] :, gid[1] :], np.sqrt(diff))

    x1 = np.arange(15, dtype=np.float32).reshape(5, 3)
    x2 = np.arange(18, dtype=np.float32).reshape(6, 3) / 3.0
    out = np.zeros((5, 6), dtype=np.float32)

    sim.launch(pairwise, x1, x2, out)

    expected = np.sqrt(((x1[:, None, :] - x2[None, :, :]) ** 2).sum(axis=2))
    assert np.allclose(out, expected)


def test_zero_sized_work_shape_is_a_noop() -> None:
    touched: list[str] = []
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(4,))
    def noop(group, x: Buffer[W]) -> None:
        touched.append("ran")

    x = np.zeros((0,), dtype=np.float32)

    sim.launch(noop, x)

    assert touched == []


def test_launch_rejects_conflicting_symbol_bindings() -> None:
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(4,))
    def same_shape(group, x: Buffer[W], y: Buffer[W]) -> None:
        return None

    x = np.zeros((4,), dtype=np.float32)
    y = np.zeros((5,), dtype=np.float32)

    with pytest.raises(sim.LaunchError, match="conflicting binding"):
        sim.launch(same_shape, x, y)


def test_vector_load_and_store_preserve_active_elements_only() -> None:
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(4,))
    def add_vec(group, x: Buffer[W], y: Buffer[W]) -> None:
        gid = group.work_offset[0]
        vec = group.vload(x[gid:], shape=(4,))
        group.store(y[gid:], vec + 2)

    x = np.arange(6, dtype=np.float32)
    y = np.zeros((6,), dtype=np.float32)

    sim.launch(add_vec, x, y)

    assert np.array_equal(y, x + 2)


def test_poison_scalar_read_raises() -> None:
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(4,))
    def bad(group, x: Buffer[W]) -> None:
        tile = group.load(x, shape=(4,))
        _ = tile[3] + 1

    x = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(sim.PoisonError, match="inactive scalar"):
        sim.launch(bad, x)


def test_group_workitems_execute_in_deterministic_order() -> None:
    seen: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(2,))
    def visit(group, x: Buffer[W]) -> None:
        @group.workitems
        def inner(wi) -> None:
            seen.append((group.group_id, wi.local_id(), wi.global_id()))

        inner()

    x = np.arange(3, dtype=np.float32)

    sim.launch(visit, x)

    assert seen == [
        ((0,), (0,), (0,)),
        ((0,), (1,), (1,)),
        ((1,), (0,), (2,)),
        ((1,), (1,), (3,)),
    ]


def test_sim_tensor_mask_and_reductions_follow_mask_rules() -> None:
    tensor = sim.SimTensor(
        np.array([1.0, 99.0, 3.0], dtype=np.float32),
        np.array([True, False, True]),
    )

    assert tensor[1] is sim.poison
    assert tensor.mask[1] is False
    assert tensor.sum() == pytest.approx(4.0)
    assert tensor.with_inactive(value=-1.0)[1] == -1.0


def test_group_barrier_orders_workitem_tensor_accesses() -> None:
    out = np.zeros((4,), dtype=np.int64)

    @kernel(work_shape=(4,), group_shape=(4,))
    def phased(group, dst: Buffer[4]) -> None:
        scratch = group.zeros(shape=(4,), dtype=np.int64)

        @group.workitems
        def inner(wi) -> None:
            lid = wi.local_id()[0]
            gid = wi.global_id()[0]
            scratch[lid] = gid + 1
            group.barrier()
            dst[gid] = scratch.sum()

        inner()

    sim.launch(phased, out)

    assert np.array_equal(out, np.array([10, 10, 10, 10], dtype=np.int64))


def test_divergent_barrier_use_raises_simulator_error() -> None:
    @kernel(work_shape=(2,), group_shape=(2,))
    def bad(group, x: Buffer[2]) -> None:
        @group.workitems
        def inner(wi) -> None:
            if wi.local_id()[0] == 0:
                group.barrier()

        inner()

    x = np.zeros((2,), dtype=np.float32)

    with pytest.raises(sim.SimulatorError, match="divergent barrier"):
        sim.launch(bad, x)


def test_mask_view_is_read_only() -> None:
    tensor = sim.SimTensor(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([True, False]),
    )

    with pytest.raises(sim.SimulatorError, match="read-only"):
        tensor.mask[0] = False

    assert tensor.mask[0] is True
    assert tensor.mask[1] is False


def test_masked_load_respects_mask_value_and_mask_activity() -> None:
    out = np.zeros((4,), dtype=np.int64)

    @kernel(work_shape=(1,), group_shape=(1,))
    def masked(group, src: Buffer[3], dst: Buffer[4]) -> None:
        gate = group.load(src, shape=(4,)) > 15
        picked = group.load(src, mask=gate)
        group.store(dst, picked.with_inactive(value=-1))

    src = np.array([10, 20, 30], dtype=np.int64)

    sim.launch(masked, src, out)

    assert np.array_equal(out, np.array([-1, 20, 30, -1], dtype=np.int64))


def test_empty_tensor_reads_as_poison() -> None:
    out = np.zeros((1,), dtype=np.float32)

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group, dst: Buffer[1]) -> None:
        tile = group.empty(shape=(2,), dtype=np.float32)
        dst[0] = tile[0] + 1

    with pytest.raises(sim.PoisonError, match="inactive scalar"):
        sim.launch(bad, out)


def test_ufunc_out_updates_tensor_destination() -> None:
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(4,))
    def subtract(group, lhs: Buffer[W], rhs: Buffer[W], out: Buffer[W]) -> None:
        gid = group.work_offset[0]
        left = group.load(lhs[gid:], shape=(4,))
        right = group.load(rhs[gid:], shape=(4,))
        tmp = group.zeros(shape=(4,), dtype=lhs.dtype)
        np.subtract(left, right, out=tmp)
        group.store(out[gid:], tmp)

    lhs = np.arange(6, dtype=np.float32)
    rhs = np.arange(6, dtype=np.float32) / 2
    out = np.zeros((6,), dtype=np.float32)

    sim.launch(subtract, lhs, rhs, out)

    assert np.array_equal(out, lhs - rhs)


def test_launch_uses_local_string_annotations() -> None:
    W = sym.W
    touched: list[int] = []

    @kernel(work_shape=(W,), group_shape=(4,))
    def annotated(group, x: "Buffer[W]") -> None:
        touched.append(x.shape[0])

    x = np.zeros((4,), dtype=np.float32)

    sim.launch(annotated, x)

    assert touched == [4]


def test_simulator_target_chooses_default_group_shape_from_subgroup_size() -> None:
    shapes: list[tuple[int, ...]] = []

    @kernel(work_shape=(5,), subgroup_size=4)
    def capture(group, x: Buffer[5]) -> None:
        shapes.append(group.shape)

    x = np.zeros((5,), dtype=np.float32)

    sim.launch(capture, x)

    assert shapes == [(8,)]


def test_simulator_target_enforces_group_size_limit() -> None:
    @kernel(work_shape=(5,), subgroup_size=4)
    def capture(group, x: Buffer[5]) -> None:
        return None

    x = np.zeros((5,), dtype=np.float32)

    with pytest.raises(sim.LaunchError, match="group size exceeds"):
        sim.launch(capture, x, target=sim.SimulatorTarget(max_group_size=4))


def test_group_subgroups_execute_in_partition_order() -> None:
    seen: list[tuple[tuple[int, ...], int, int]] = []

    @kernel(work_shape=(4, 2), group_shape=(4, 2), subgroup_size=2)
    def visit(group, _x: Buffer[4, 2]) -> None:
        @group.subgroups
        def inner(sg) -> None:
            seen.append((group.group_id, sg.subgroup_id(), sg.size()))

        inner()

    src = np.zeros((4, 2), dtype=np.float32)

    sim.launch(visit, src)

    assert seen == [
        ((0, 0), 0, 2),
        ((0, 0), 1, 2),
        ((0, 0), 2, 2),
        ((0, 0), 3, 2),
    ]


def test_subgroups_and_workitems_share_tensor_state_across_regions() -> None:
    out = np.zeros((4, 2), dtype=np.int64)

    sim.launch(_staged_subgroups_and_workitems_kernel, out)

    assert np.array_equal(out, _EXPECTED_SUBGROUP_AND_WORKITEM_STATE)


def test_scalar_reduction_returns_scalar_result() -> None:
    tensor = sim.SimTensor(np.array(7.0, dtype=np.float32), np.array(True))
    inactive = sim.SimTensor(np.array(7.0, dtype=np.float32), np.array(False))

    assert tensor.sum() == pytest.approx(7.0)
    assert tensor.max() == pytest.approx(7.0)
    assert tensor.min() == pytest.approx(7.0)

    with pytest.raises(sim.PoisonError, match="inactive scalar"):
        _ = inactive.sum() + 1


def test_basic_indexing_subset_is_enforced() -> None:
    tensor = sim.SimTensor(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([True, True]),
    )

    with pytest.raises(sim.SimulatorError, match="basic indexing"):
        _ = tensor[[0, 1]]


def test_group_load_is_rejected_in_workitem_scope() -> None:
    @kernel(work_shape=(2,), group_shape=(2,))
    def bad(group, x: Buffer[2]) -> None:
        @group.workitems
        def inner(wi) -> None:
            _ = wi
            group.load(x, shape=(1,))

        inner()

    x = np.zeros((2,), dtype=np.float32)

    with pytest.raises(sim.ScopeError, match="group.load"):
        sim.launch(bad, x)


def test_group_vload_and_vector_store_work_in_workitem_scope() -> None:
    @kernel(work_shape=(4,), group_shape=(4,))
    def copy(group, src: Buffer[4], dst: Buffer[4]) -> None:
        @group.workitems
        def inner(wi) -> None:
            gid = wi.global_id()[0]
            vec = group.vload(src[gid:], shape=(1,))
            group.store(dst[gid:], vec + 2)

        inner()

    src = np.arange(4, dtype=np.float32)
    dst = np.zeros((4,), dtype=np.float32)

    sim.launch(copy, src, dst)

    assert np.array_equal(dst, src + 2)


def test_non_call_ufunc_methods_are_rejected() -> None:
    tensor = sim.SimTensor(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([True, True]),
    )

    with pytest.raises(sim.SimulatorError, match="ufunc method"):
        np.add.reduce(tensor)


def test_poison_blocks_additional_observation_channels() -> None:
    with pytest.raises(sim.PoisonError, match="inactive scalar"):
        hash(sim.poison)

    with pytest.raises(sim.PoisonError, match="inactive scalar"):
        format(sim.poison, "")
