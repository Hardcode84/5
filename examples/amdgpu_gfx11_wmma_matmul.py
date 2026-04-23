# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tiled AMDGPU matmul example built around a gfx11 WMMA intrinsic.

Run from the repository root with:

    python -m examples.amdgpu_gfx11_wmma_matmul

This version models the RDNA3/gfx11 `v_wmma_f32_16x16x16_f16` layout at
WorkItem scope. It also uses collective-return values to keep the WMMA
accumulator distributed across workitems at the WorkGroup-level K loop
boundary: `acc = init_wmma_acc(group)` before the loop,
`acc = issue_wmma_tile(...)` inside it, and `acc[:, lane]` when a lane consumes
its local fragment. For wave32, the calculator reports:

* A[i, k] lives in lanes `i` and `i + 16`, register `floor(k / 2)`.
* B[k, j] lives in lanes `j` and `j + 16`, register `floor(k / 2)`.
* D[i, j] lives in register `floor(i / 2)`, lane `16 * (i % 2) + j`.

That means each lane carries a duplicated A row fragment, a duplicated B column
fragment, and an 8-value accumulator fragment striped over either even or odd
output rows for one output column.
"""

from __future__ import annotations

import numpy as np

import hc.simulator as sim
from hc import Buffer, WorkGroup, WorkItem, kernel, sym
from hc.symbols import ceil_div

WAVE_LANES = 32
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
GFX_ARCH = "gfx11"
WMMA_ACC_ROW_STRIDE = WAVE_LANES // 16
WMMA_ACC_FRAGMENT = WMMA_M // WMMA_ACC_ROW_STRIDE
_WMMA_SIGNATURE_ARGS = (
    (1, "tensor", (WMMA_M, WMMA_K), np.float16, "A tile"),
    (2, "tensor", (WMMA_K, WMMA_N), np.float16, "B tile"),
    (3, "vector", (WMMA_K,), np.float16, "A lane fragment"),
    (4, "vector", (WMMA_K,), np.float16, "B lane fragment"),
    (5, "vector", (WMMA_ACC_FRAGMENT,), np.float32, "accumulator fragment"),
)
# The intrinsic returns one updated `(WMMA_ACC_FRAGMENT,)` accumulator fragment
# per workitem.

M = sym.M
N = sym.N
K = sym.K


def _lane_column(lane: int) -> int:
    return lane % WMMA_N


def _lane_a_row(lane: int) -> int:
    return lane % WMMA_M


def _lane_output_rows(lane: int, wave_size: int) -> tuple[int, ...]:
    start = lane // 16
    step = _lane_output_row_step(wave_size)
    return tuple(range(start, WMMA_M, step))


def _lane_output_row_step(wave_size: int) -> int:
    return wave_size // 16


def _lane_output_row_slice_args(lane: int, wave_size: int) -> tuple[int, int, int]:
    rows = _lane_output_rows(lane, wave_size)
    return rows[0], WMMA_M, _lane_output_row_step(wave_size)


def _tile_origin(tile_row: int, tile_col: int) -> tuple[int, int]:
    return tile_row * WMMA_M, tile_col * WMMA_N


def _require_signature_arg(sig, index, *, kind, shape, dtype, name):
    value = sig.arg(index)
    if kind == "tensor":
        is_expected = value.is_tensor()
    else:
        is_expected = value.is_vector()
    if not is_expected:
        raise ValueError(f"{name} must be a {kind}")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}")
    if value.type != np.dtype(dtype):
        raise ValueError(f"{name} dtype must be {np.dtype(dtype).name}")


def _require_wmma_context(sig) -> None:
    if sig.scope is not WorkItem:
        raise ValueError("wmma_gfx11 must execute in WorkItem scope")
    if sig.kwarg("arch") != GFX_ARCH:
        raise ValueError(f"wmma_gfx11 expects arch={GFX_ARCH!r}")
    if sig.kwarg("wave_size") != WAVE_LANES:
        raise ValueError(f"wmma_gfx11 expects wave_size={WAVE_LANES}")
    lane = sig.kwarg("lane")
    if not isinstance(lane, int) or not (0 <= lane < WAVE_LANES):
        raise ValueError(f"wmma_gfx11 lane must be in [0, {WAVE_LANES})")


def _require_wmma_operands(sig) -> None:
    for index, kind, shape, dtype, name in _WMMA_SIGNATURE_ARGS:
        _require_signature_arg(
            sig, index, kind=kind, shape=shape, dtype=dtype, name=name
        )


@kernel.intrinsic(
    scope=WorkItem,
    effects="pure",
    const_attrs={"wave_size", "arch"},
)
def wmma_gfx11(
    group,
    a_tile,
    b_tile,
    a_frag,
    b_frag,
    acc_frag,
    *,
    lane,
    wave_size,
    arch,
):
    """Simulator fallback for a gfx11 WMMA lane fragment.

    The simulator reconstructs each lane result from staged wave-distributed
    LDS tiles rather than from the explicit lane operands.
    """

    _ = (a_frag, b_frag)
    a_tile = a_tile.with_inactive(value=np.float16(0))
    b_tile = b_tile.with_inactive(value=np.float16(0))
    rows = _lane_output_rows(lane, wave_size)
    col = _lane_column(lane)
    values = np.empty((len(rows),), dtype=np.float32)
    for index in range(len(rows)):
        accum = np.float32(acc_frag[index])
        for k_idx in range(WMMA_K):
            accum += np.float32(a_tile[rows[index], k_idx]) * np.float32(
                b_tile[k_idx, col]
            )
        values[index] = accum
    return group.vload(values, shape=(len(rows),))


@wmma_gfx11.verify
def _verify_wmma(sig, target):
    _ = target
    _require_wmma_context(sig)
    _require_wmma_operands(sig)


@wmma_gfx11.lower(target="amdgpu-gfx11")
def _lower_wmma(
    ctx,
    group,
    a_tile,
    b_tile,
    a_frag,
    b_frag,
    acc_frag,
    *,
    lane,
    wave_size,
    arch,
):
    _ = (group, a_tile, b_tile, lane)
    op = ctx.builder.create(
        "amdgpu.wmma",
        results=[ctx.result_type(0)],
        operands=[a_frag, b_frag, acc_frag],
        attrs={
            "arch": arch,
            "wave_size": wave_size,
            "m": WMMA_M,
            "n": WMMA_N,
            "k": WMMA_K,
        },
        loc=ctx.loc,
    )
    return op.result(0)


@kernel.func(scope=WorkItem)
def load_wmma_a_fragment(wi, a_tile):
    return (
        a_tile[_lane_a_row(wi.local_id()[0]), :]
        .vec()
        .with_inactive(value=np.float16(0))
    )


@kernel.func(scope=WorkItem)
def load_wmma_b_fragment(wi, b_tile):
    return (
        b_tile[:, _lane_column(wi.local_id()[0])]
        .vec()
        .with_inactive(value=np.float16(0))
    )


@kernel.func(scope=WorkGroup)
def init_wmma_acc(group):
    @group.workitems
    def init(wi):
        _ = wi
        return group.vzeros(shape=(WMMA_ACC_FRAGMENT,), dtype=np.float32)

    return init()


@kernel.func(scope=WorkGroup)
def issue_wmma_tile(group, a_tile, b_tile, acc):
    @group.workitems
    def wave(wi):
        lane = wi.local_id()[0]
        a_frag = load_wmma_a_fragment(wi, a_tile)
        b_frag = load_wmma_b_fragment(wi, b_tile)
        # A 2D launch keeps a trailing singleton collective axis from
        # `group_shape=(WAVE_LANES, 1)`. Collapse it here so WMMA still sees the
        # per-lane `(WMMA_ACC_FRAGMENT,)` vector it expects.
        return wmma_gfx11(
            group,
            a_tile,
            b_tile,
            a_frag,
            b_frag,
            acc[:, lane, 0],
            lane=lane,
            wave_size=WAVE_LANES,
            arch=GFX_ARCH,
        )

    return wave()


@kernel.func(scope=WorkGroup)
def store_wmma_tile(group, c, row0, col0, acc):
    @group.workitems
    def wave(wi):
        lane = wi.local_id()[0]
        row_start, row_stop, row_step = _lane_output_row_slice_args(lane, WAVE_LANES)
        col = col0 + _lane_column(lane)
        # Keep the destination view slice-shaped so right-edge tiles degrade to
        # empty overlap instead of forming an out-of-bounds scalar column index.
        group.store(
            c[row0 + row_start : row0 + row_stop : row_step, col : col + 1],
            acc[:, lane, :],
        )

    wave()


@kernel(
    work_shape=(ceil_div(M, WMMA_M) * WAVE_LANES, ceil_div(N, WMMA_N)),
    group_shape=(WAVE_LANES, 1),
    subgroup_size=WAVE_LANES,
)
def tiled_gfx11_wmma_matmul(
    group,
    a: Buffer[M, K],
    b: Buffer[K, N],
    c: Buffer[M, N],
) -> None:
    row0, col0 = _tile_origin(group.group_id[0], group.group_id[1])
    acc = init_wmma_acc(group)

    for k0 in range(0, a.shape[1], WMMA_K):
        a_tile = group.load(
            a[row0 : row0 + WMMA_M, k0 : k0 + WMMA_K],
            shape=(WMMA_M, WMMA_K),
        )
        b_tile = group.load(
            b[k0 : k0 + WMMA_K, col0 : col0 + WMMA_N],
            shape=(WMMA_K, WMMA_N),
        )
        acc = issue_wmma_tile(group, a_tile, b_tile, acc)

    store_wmma_tile(group, c, row0, col0, acc)


def reference_blocked_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k0 in range(0, a.shape[1], WMMA_K):
        a_block = a[:, k0 : k0 + WMMA_K].astype(np.float32)
        b_block = b[k0 : k0 + WMMA_K, :].astype(np.float32)
        out += a_block @ b_block
    return out


def simulate_gfx11_wmma_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("A.shape[1] must match B.shape[0]")
    if a.dtype != np.float16 or b.dtype != np.float16:
        raise ValueError("this example expects float16 A and B operands")

    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    sim.launch(tiled_gfx11_wmma_matmul, a, b, c)
    return c


def make_demo_inputs(
    *,
    m: int = 32,
    n: int = 32,
    k: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float16)
    b = rng.uniform(-1.0, 1.0, size=(k, n)).astype(np.float16)
    return a, b


def main() -> None:
    a, b = make_demo_inputs()
    out = simulate_gfx11_wmma_matmul(a, b)
    reference = reference_blocked_matmul(a, b)
    np.testing.assert_allclose(out, reference, rtol=0.0, atol=2e-6)
    print("gfx11 WMMA tiled matmul example passed.")
    print(f"shape: A={a.shape}, B={b.shape}, C={out.shape}")
    max_diff = np.max(np.abs(out - reference))
    print(f"max abs diff vs blocked fallback reference: {max_diff}")


if __name__ == "__main__":
    main()
