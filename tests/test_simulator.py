# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading

import numpy as np
import pytest

import hc.core as hc_core
import hc.simulator as sim
from hc import Buffer, SubGroup, WorkGroup, WorkItem, as_layout, index_map, kernel, sym
from hc._sim_types import resolve_layout

_ROW_PADDED_LAYOUT = index_map(
    params=lambda w, h: {"row_stride": h + 1},
    storage_size=lambda w, h, p: w * p["row_stride"],
    offset=lambda i, j, w, h, p: i * p["row_stride"] + j,
)

_CONTIGUOUS_VECTOR_LAYOUT = index_map(
    storage_size=lambda n: n,
    offset=lambda i, n: i,
)

_REVERSED_VECTOR_LAYOUT = index_map(
    storage_size=lambda n: n,
    offset=lambda i, n: n - 1 - i,
)

_INVALID_VECTOR_LAYOUT = index_map(
    storage_size=lambda n: n,
    offset=lambda i, n: 0,
)

_EXPECTED_SUBGROUP_AND_WORKITEM_STATE = np.array(
    [
        [1, 3],
        [3, 5],
        [4, 6],
        [0, 2],
    ],
    dtype=np.int64,
)


def _wait_for_event(event: threading.Event, name: str) -> None:
    if not event.wait(timeout=2.0):
        raise RuntimeError(f"timed out waiting for {name}")


def _join_thread(thread: threading.Thread, name: str) -> None:
    thread.join(timeout=2.0)
    assert not thread.is_alive(), f"{name} did not finish"


def _launch_in_thread(fn, label: int, target, dst, errors: list[BaseException]) -> None:
    try:
        sim.launch(fn, label, dst, target=target)
    except BaseException as exc:  # pragma: no cover - surfaced in assertions below
        errors.append(exc)


def _make_thread_local_launch_case(verified):
    ready_a = threading.Event()
    ready_b = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()

    @kernel.intrinsic(scope=WorkGroup)
    def capture(x: int) -> int:
        return x

    @capture.verify
    def _verify(sig, target):
        verified.append((sig.arg(0), target))

    @kernel(work_shape=(1,), group_shape=(1,))
    def run(group, label: int, dst: Buffer[1]) -> None:
        _ = group
        if label == 1:
            ready_a.set()
            _wait_for_event(release_a, "resume thread A")
        else:
            ready_b.set()
            _wait_for_event(release_b, "resume thread B")
        dst[0] = capture(label)

    return run, ready_a, ready_b, release_a, release_b


def _run_thread_local_launch_pair(run, ready_a, ready_b, release_a, release_b):
    errors: list[BaseException] = []
    out_a = np.zeros((1,), dtype=np.int64)
    out_b = np.zeros((1,), dtype=np.int64)
    target_a = sim.SimulatorTarget(max_group_size=1)
    target_b = sim.SimulatorTarget(max_group_size=2)
    thread_a = threading.Thread(
        target=_launch_in_thread, args=(run, 1, target_a, out_a, errors)
    )
    thread_b = threading.Thread(
        target=_launch_in_thread, args=(run, 2, target_b, out_b, errors)
    )

    thread_a.start()
    _wait_for_event(ready_a, "thread A to enter the kernel")
    thread_b.start()
    _wait_for_event(ready_b, "thread B to enter the kernel")
    release_a.set()
    _join_thread(thread_a, "thread A")
    release_b.set()
    _join_thread(thread_b, "thread B")
    return target_a, target_b, out_a, out_b, errors


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


def test_group_load_accepts_layout_and_preserves_logical_contents() -> None:
    seen: list[tuple[int, int]] = []

    @kernel(work_shape=(1,), group_shape=(1,))
    def load_layout(group, src: Buffer[2, 3], dst: Buffer[2, 3]) -> None:
        tile = group.load(src, shape=(2, 3), layout=_ROW_PADDED_LAYOUT)
        seen.append((tile.layout.storage_size, tile.layout.params["row_stride"]))
        group.store(dst, tile + 1)

    src = np.arange(6, dtype=np.int64).reshape(2, 3)
    dst = np.zeros((2, 3), dtype=np.int64)

    sim.launch(load_layout, src, dst)

    assert np.array_equal(dst, src + 1)
    assert seen == [(8, 4)]


def test_group_zeros_accepts_layout_and_carries_metadata() -> None:
    seen: list[tuple[int, int]] = []

    @kernel(work_shape=(1,), group_shape=(1,))
    def alloc_layout(group, dst: Buffer[2, 3]) -> None:
        tile = group.zeros(shape=(2, 3), dtype=np.int64, layout=_ROW_PADDED_LAYOUT)
        seen.append((tile.layout.storage_size, tile.layout.params["row_stride"]))
        group.store(dst, tile + 1)

    dst = np.zeros((2, 3), dtype=np.int64)

    sim.launch(alloc_layout, dst)

    assert np.array_equal(dst, np.ones((2, 3), dtype=np.int64))
    assert seen == [(8, 4)]


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


def test_as_layout_preserves_logical_vector_contents() -> None:
    vec = sim.SimVector(
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([True, True, True, True]),
    )

    relaid = as_layout(vec, layout=_REVERSED_VECTOR_LAYOUT)

    assert relaid.layout.storage_size == 4
    assert relaid[0] == 1
    assert relaid[3] == 4


def test_laid_out_tensor_views_drop_layout_on_shape_change() -> None:
    layout = resolve_layout(_ROW_PADDED_LAYOUT, (2, 3))
    tensor = sim.SimTensor(
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.ones((2, 3), dtype=bool),
        layout=layout,
    )

    same_shape = tensor.reshape(2, 3)
    sliced = tensor[:, :2]
    reshaped = tensor.reshape(3, 2)
    transposed = tensor.transpose()
    flattened = tensor.vec(shape=(6,))

    assert same_shape.layout == layout
    assert sliced.layout is None
    assert reshaped.layout is None
    assert transposed.layout is None
    assert flattened.layout is None


def test_laid_out_vector_ops_preserve_compatible_layout() -> None:
    lhs = as_layout(
        sim.SimVector(
            np.array([1.0, 4.0, 9.0], dtype=np.float32),
            np.array([True, True, True]),
        ),
        layout=_REVERSED_VECTOR_LAYOUT,
    )
    rhs = as_layout(
        sim.SimVector(
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([True, True, True]),
        ),
        layout=_REVERSED_VECTOR_LAYOUT,
    )

    result = np.sqrt(lhs + rhs)

    assert result.layout == lhs.layout
    assert result.shape == lhs.shape
    assert result[0] == pytest.approx(np.sqrt(2.0))


def test_laid_out_vector_ops_reject_mismatched_layouts() -> None:
    lhs = as_layout(
        sim.SimVector(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([True, True, True]),
        ),
        layout=_CONTIGUOUS_VECTOR_LAYOUT,
    )
    rhs = as_layout(
        sim.SimVector(
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([True, True, True]),
        ),
        layout=_REVERSED_VECTOR_LAYOUT,
    )

    with pytest.raises(sim.SimulatorError, match="mismatched layouts"):
        _ = lhs + rhs


def test_invalid_layout_is_rejected() -> None:
    vec = sim.SimVector(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([True, True, True]),
    )

    with pytest.raises(sim.SimulatorError, match="injective"):
        as_layout(vec, layout=_INVALID_VECTOR_LAYOUT)


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


def test_masked_load_accepts_layout_and_preserves_metadata() -> None:
    seen: list[int] = []
    out = np.zeros((4,), dtype=np.int64)

    @kernel(work_shape=(1,), group_shape=(1,))
    def masked(group, src: Buffer[3], dst: Buffer[4]) -> None:
        gate = group.load(src, shape=(4,)) > 15
        picked = group.load(src, mask=gate, layout=_CONTIGUOUS_VECTOR_LAYOUT)
        seen.append(picked.layout.storage_size)
        group.store(dst, picked.with_inactive(value=-1))

    src = np.array([10, 20, 30], dtype=np.int64)

    sim.launch(masked, src, out)

    assert np.array_equal(out, np.array([-1, 20, 30, -1], dtype=np.int64))
    assert seen == [4]


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


def test_ufunc_out_updates_laid_out_tensor_destination() -> None:
    layout = resolve_layout(_CONTIGUOUS_VECTOR_LAYOUT, (3,))
    stale_layout = resolve_layout(_REVERSED_VECTOR_LAYOUT, (3,))
    lhs = sim.SimTensor(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([True, True, True]),
        layout=layout,
    )
    rhs = sim.SimTensor(
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
        np.array([True, True, True]),
        layout=layout,
    )
    out = sim.SimTensor(
        np.zeros((3,), dtype=np.float32),
        np.array([True, True, True]),
        layout=stale_layout,
    )

    np.add(lhs, rhs, out=(out,))

    assert out.layout == layout
    assert np.array_equal(out.with_inactive(value=0)._data, np.array([5.0, 7.0, 9.0]))


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


def test_group_workitems_collect_scalar_returns_into_vectors() -> None:
    out = np.zeros((4,), dtype=np.int64)

    @kernel(work_shape=(4,), group_shape=(4,))
    def collect(group, dst: Buffer[4]) -> None:
        @group.workitems
        def lanes(wi):
            return np.int64(wi.local_id()[0] + 1)

        group.store(dst, lanes())

    sim.launch(collect, out)

    assert np.array_equal(out, np.array([1, 2, 3, 4], dtype=np.int64))


def test_group_subgroups_collect_vector_returns_into_vectors() -> None:
    out = np.zeros((2, 2), dtype=np.int64)

    @kernel(work_shape=(4,), group_shape=(4,), subgroup_size=2)
    def collect(group, dst: Buffer[2, 2]) -> None:
        @group.subgroups
        def tiles(sg):
            return group.vfull(
                shape=(2,), dtype=np.int64, fill_value=sg.subgroup_id() + 1
            )

        group.store(dst, tiles())

    sim.launch(collect, out)

    assert np.array_equal(out, np.array([[1, 2], [1, 2]], dtype=np.int64))


def test_group_workitems_collect_tuple_returns_elementwise() -> None:
    acc_out = np.zeros((2, 4), dtype=np.int64)
    lane_out = np.zeros((4,), dtype=np.int32)

    @kernel(work_shape=(4,), group_shape=(4,))
    def collect(group, acc_dst: Buffer[2, 4], lane_dst: Buffer[4]) -> None:
        @group.workitems
        def init(wi):
            lane = wi.local_id()[0]
            return group.vfull(shape=(2,), dtype=np.int64, fill_value=lane), np.int32(
                lane
            )

        acc, lanes = init()
        group.store(acc_dst, acc)
        group.store(lane_dst, lanes)

    sim.launch(collect, acc_out, lane_out)

    expected_acc = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    expected_lanes = np.array([0, 1, 2, 3], dtype=np.int32)

    assert np.array_equal(acc_out, expected_acc)
    assert np.array_equal(lane_out, expected_lanes)


def test_collective_return_values_can_be_sliced_and_returned_again() -> None:
    out = np.zeros((2, 4), dtype=np.int64)

    @kernel(work_shape=(4,), group_shape=(4,))
    def staged(group, dst: Buffer[2, 4]) -> None:
        @group.workitems
        def init(wi):
            return group.vfull(shape=(2,), dtype=np.int64, fill_value=wi.local_id()[0])

        acc = init()

        @group.workitems
        def step(wi):
            lane = wi.local_id()[0]
            return acc[:, lane] + 1

        group.store(dst, step())

    sim.launch(staged, out)

    assert np.array_equal(out, np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.int64))


def test_collective_return_full_slice_preserves_suffix_metadata() -> None:
    seen: list[tuple[int, ...]] = []

    @kernel(work_shape=(4,), group_shape=(4,))
    def collect(group) -> None:
        @group.workitems
        def lanes(wi):
            return np.int64(wi.local_id()[0])

        seen.append(lanes()[:].collective_suffix)

    sim.launch(collect)

    assert seen == [(4,)]


def test_collective_return_reshape_preserves_trailing_suffix() -> None:
    seen: list[tuple[int, ...]] = []

    @kernel(work_shape=(4,), group_shape=(4,))
    def collect(group) -> None:
        @group.workitems
        def lanes(wi):
            return np.int64(wi.local_id()[0])

        seen.append(lanes().reshape(1, 4).collective_suffix)

    sim.launch(collect)

    assert seen == [(4,)]


def test_collective_return_mismatch_raises_simulator_error() -> None:
    @kernel(work_shape=(2,), group_shape=(2,))
    def bad(group) -> None:
        @group.workitems
        def inner(wi):
            if wi.local_id()[0] == 0:
                return np.int32(1)
            return group.vzeros(shape=(1,), dtype=np.int32)

        inner()

    with pytest.raises(
        sim.SimulatorError,
        match="return structure does not match: expected scalar, got vector",
    ):
        sim.launch(bad)


def test_collective_return_vectors_reject_implicit_broadcasting() -> None:
    out = np.zeros((4,), dtype=np.int64)

    @kernel(work_shape=(4,), group_shape=(4,))
    def bad(group, dst: Buffer[4]) -> None:
        @group.workitems
        def lanes(wi):
            return np.int64(wi.local_id()[0])

        lifted = lanes()
        base = group.vones(shape=(4,), dtype=np.int64)
        group.store(dst, lifted + base)

    with pytest.raises(sim.SimulatorError, match="matching collective suffixes"):
        sim.launch(bad, out)


def test_collective_return_nested_tuples_are_rejected() -> None:
    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        @group.workitems
        def inner(wi):
            lane = np.int32(wi.local_id()[0])
            return lane, (lane,)

        inner()

    with pytest.raises(
        sim.SimulatorError, match="collective tuple returns must be flat"
    ):
        sim.launch(bad)


def test_collective_return_tuple_elements_cannot_be_none() -> None:
    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        @group.workitems
        def inner(wi):
            return np.int32(wi.local_id()[0]), None

        inner()

    with pytest.raises(
        sim.SimulatorError,
        match="collective tuple returns may not contain None",
    ):
        sim.launch(bad)


def test_collective_return_tensors_are_rejected() -> None:
    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        scratch = group.zeros(shape=(1,), dtype=np.int32)

        @group.workitems
        def inner(wi):
            _ = wi
            return scratch

        inner()

    with pytest.raises(
        sim.SimulatorError,
        match="collective returns may only be scalars, vectors, or flat tuples",
    ):
        sim.launch(bad)


def test_collective_return_indexing_requires_trailing_suffix() -> None:
    @kernel(work_shape=(4,), group_shape=(4,))
    def bad(group) -> None:
        @group.workitems
        def init(wi):
            return group.vfull(shape=(2,), dtype=np.int64, fill_value=wi.local_id()[0])

        _ = init()[:, :, None]

    with pytest.raises(
        sim.SimulatorError,
        match="indexing must keep collective return dimensions as a trailing suffix",
    ):
        sim.launch(bad)


def test_collective_return_transpose_requires_trailing_suffix() -> None:
    @kernel(work_shape=(4,), group_shape=(4,))
    def bad(group) -> None:
        @group.workitems
        def init(wi):
            return group.vfull(shape=(2,), dtype=np.int64, fill_value=wi.local_id()[0])

        _ = init().transpose()

    with pytest.raises(
        sim.SimulatorError,
        match="transpose must keep collective return dimensions as a trailing suffix",
    ):
        sim.launch(bad)


def test_collective_return_reshape_requires_trailing_suffix() -> None:
    @kernel(work_shape=(4,), group_shape=(4,))
    def bad(group) -> None:
        @group.workitems
        def lanes(wi):
            return np.int64(wi.local_id()[0])

        _ = lanes().reshape(4, 1)

    with pytest.raises(
        sim.SimulatorError,
        match="reshape must keep collective return dimensions as a trailing suffix",
    ):
        sim.launch(bad)


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

    with pytest.raises(sim.ScopeError, match=r"group\.load"):
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


def test_helper_scope_is_rejected_outside_declared_scope() -> None:
    @kernel.func(scope=SubGroup)
    def subgroup_only(x: int) -> int:
        return x + 1

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        _ = group
        subgroup_only(1)

    with pytest.raises(sim.ScopeError, match=r"helper 'subgroup_only'.*SubGroup"):
        sim.launch(bad)


def test_helper_scope_allows_declared_subgroup_calls() -> None:
    seen: list[int] = []

    @kernel.func(scope=SubGroup)
    def subgroup_value(x: int) -> int:
        return x + 1

    @kernel(work_shape=(4,), group_shape=(4,), subgroup_size=2)
    def ok(group) -> None:
        @group.subgroups
        def inner(sg) -> None:
            seen.append(subgroup_value(sg.subgroup_id()))

        inner()

    sim.launch(ok)

    assert seen == [1, 2]


def test_intrinsic_scope_is_rejected_outside_declared_scope() -> None:
    @kernel.intrinsic(scope=WorkItem)
    def workitem_only(x: int) -> int:
        return x + 1

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        _ = group
        workitem_only(1)

    with pytest.raises(
        sim.ScopeError,
        match=r"intrinsic 'workitem_only'.*WorkItem",
    ):
        sim.launch(bad)


def test_bodyless_intrinsic_raises_when_reached() -> None:
    @kernel.intrinsic(scope=WorkGroup)
    def vendor_only(x: int) -> None:
        pass

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        _ = group
        vendor_only(1)

    with pytest.raises(sim.SimulatorError, match="no simulator fallback"):
        sim.launch(bad)


def test_intrinsic_verify_hook_receives_scope_and_arguments() -> None:
    verified: list[tuple[object, bool, int, sim.SimulatorTarget]] = []

    @kernel.intrinsic(scope=WorkItem, const_attrs={"delta"})
    def add_delta(vec, *, delta):
        return vec + delta

    @add_delta.verify
    def _verify(sig, target):
        verified.append((sig.scope, sig.arg(0).is_vector(), sig.kwarg("delta"), target))

    @kernel(work_shape=(1,), group_shape=(1,))
    def ok(group, src: Buffer[1], dst: Buffer[1]) -> None:
        @group.workitems
        def inner(wi) -> None:
            gid = wi.global_id()[0]
            vec = group.vload(src[gid:], shape=(1,))
            group.store(dst[gid:], add_delta(vec, delta=2))

        inner()

    src = np.array([3.0], dtype=np.float32)
    dst = np.zeros((1,), dtype=np.float32)
    target = sim.SimulatorTarget()

    sim.launch(ok, src, dst, target=target)

    assert np.array_equal(dst, np.array([5.0], dtype=np.float32))
    assert verified == [(WorkItem, True, 2, target)]


def test_intrinsic_verify_hook_can_reject_calls() -> None:
    @kernel.intrinsic(scope=WorkGroup, const_attrs={"blocksz"})
    def checked(x: int, *, blocksz: int) -> int:
        return x + blocksz

    @checked.verify
    def _verify(sig, target):
        _ = target
        assert sig.scope == WorkGroup
        if sig.kwarg("blocksz") != 4:
            raise ValueError("unsupported blocksz")

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        _ = group
        checked(1, blocksz=8)

    with pytest.raises(sim.SimulatorError, match="unsupported blocksz"):
        sim.launch(bad)


def test_intrinsic_verify_argument_capture_errors_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_signature = sim.inspect.signature

    @kernel.intrinsic(scope=WorkGroup)
    def checked(x: int) -> int:
        return x

    @checked.verify
    def _verify(sig, target):
        _ = (sig, target)

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        _ = group
        checked(1)

    def signature(fn):
        if fn is checked:
            raise ValueError("cannot inspect intrinsic")
        return original_signature(fn)

    monkeypatch.setattr(sim.inspect, "signature", signature)

    with pytest.raises(sim.SimulatorError, match="could not capture intrinsic call"):
        sim.launch(bad)


def test_bodyless_intrinsic_without_source_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_source(fn):
        _ = fn
        raise OSError("missing source")

    monkeypatch.setattr(hc_core.inspect, "getsource", missing_source)

    @kernel.intrinsic(scope=WorkGroup)
    def vendor_only(x: int) -> None:
        pass

    @kernel(work_shape=(1,), group_shape=(1,))
    def bad(group) -> None:
        _ = group
        vendor_only(1)

    with pytest.raises(sim.SimulatorError, match="no simulator fallback"):
        sim.launch(bad)


def test_nonempty_intrinsic_without_source_uses_bytecode_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_source(fn):
        _ = fn
        raise OSError("missing source")

    monkeypatch.setattr(hc_core.inspect, "getsource", missing_source)

    @kernel.intrinsic(scope=WorkGroup)
    def add_one(x: int) -> int:
        return x + 1

    @kernel(work_shape=(1,), group_shape=(1,))
    def ok(group, dst: Buffer[1]) -> None:
        _ = group
        dst[0] = add_one(1)

    out = np.zeros((1,), dtype=np.int64)

    sim.launch(ok, out)

    assert np.array_equal(out, np.array([2], dtype=np.int64))


def test_launch_execution_context_is_thread_local() -> None:
    verified: list[tuple[int, sim.SimulatorTarget]] = []
    run, ready_a, ready_b, release_a, release_b = _make_thread_local_launch_case(
        verified
    )
    target_a, target_b, out_a, out_b, errors = _run_thread_local_launch_pair(
        run, ready_a, ready_b, release_a, release_b
    )

    assert errors == []
    assert verified == [(1, target_a), (2, target_b)]
    assert np.array_equal(out_a, np.array([1], dtype=np.int64))
    assert np.array_equal(out_b, np.array([2], dtype=np.int64))


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
