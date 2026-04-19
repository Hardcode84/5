# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import hc.simulator as sim
from hc import Buffer, kernel, sym


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


def test_group_workitems_is_not_supported_in_milestone_zero() -> None:
    W = sym.W

    @kernel(work_shape=(W,), group_shape=(4,))
    def unsupported(group, x: Buffer[W]) -> None:
        @group.workitems
        def inner(wi):
            return wi

        inner()

    x = np.arange(4, dtype=np.float32)

    with pytest.raises(sim.ScopeError, match="@group.workitems"):
        sim.launch(unsupported, x)


def test_sim_tensor_mask_and_reductions_follow_mask_rules() -> None:
    tensor = sim.SimTensor(
        np.array([1.0, 99.0, 3.0], dtype=np.float32),
        np.array([True, False, True]),
    )

    assert tensor[1] is sim.poison
    assert tensor.mask[1] is False
    assert tensor.sum() == pytest.approx(4.0)
    assert tensor.with_inactive(value=-1.0)[1] == -1.0


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


def test_group_subgroups_is_not_supported_in_milestone_zero() -> None:
    @kernel(work_shape=(1,), group_shape=(1,))
    def unsupported(group, x: Buffer[1]) -> None:
        @group.subgroups
        def inner(sg):
            return sg

        inner()

    x = np.arange(1, dtype=np.float32)

    with pytest.raises(sim.ScopeError, match="@group.subgroups"):
        sim.launch(unsupported, x)


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
