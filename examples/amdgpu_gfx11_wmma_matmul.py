# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tiled AMDGPU matmul example built around a gfx11 WMMA intrinsic.

Run from the repository root with:

    python -m examples.amdgpu_gfx11_wmma_matmul

Pass ``--dump-front-ir`` to lower the kernel plus its transitive decorated
helpers through the Python frontend and print the resulting combined
``hc_front`` textual MLIR instead of running the simulator.

Pass ``--dump-hc-ir`` to run the current frontend-to-``hc`` pipeline and print
the resulting ``hc`` textual MLIR instead.

Pass ``--run-on-hw`` to compile the kernel for ``amdgpu-gfx11`` and dispatch
it through ``hc.compile().invoke()`` against ``torch.cuda`` tensors on a
physical gfx11 GPU, comparing the result to ``reference_blocked_matmul``.
Requires a working ROCm install with a gfx11 device visible to HIP and the
``torch`` package on the path.

Pass ``--bench`` to compile with ``bench=True``, dispatch through
``compiled.bench(...)`` against ``torch.cuda`` tensors, and print a small
stats table (median / mean / std + per-launch derivations). A correctness
smoke check runs first; on a mismatch we *warn but do not abort* — the
bench surface is for timing dispatch latency, not for verifying numerics,
and the launch/dispatch cost is still meaningful even when the kernel
itself is miscompiled (which is exactly when you want to triage the gap).
Requires the same ROCm + torch setup as ``--run-on-hw``.

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

The per-lane accumulator addressing is captured once as
``WAVE_ACC_FRAG_LAYOUT`` — an ``index_map`` whose offset
``(lane // 16 + fi * 2) * 16 + (lane % 16)`` maps each
``(lane, fragment_index)`` pair to a position inside the flat 16x16
output tile. ``init_wmma_acc`` and ``store_wmma_tile`` realise that
same formula as a strided two-dimensional ``c[row_slice, col_slice]``
view — the form the simulator handles via numpy strides and the gpu
path lowers through ``hc.generic``. Reading both sites off a single
declared layout removes the per-lane slice-arithmetic glue the
previous revision needed.

The direct layout-driven form
(``group.vload(c_tile, shape=(WAVE_LANES, WMMA_ACC_FRAGMENT),
layout=WAVE_ACC_FRAG_LAYOUT)`` plus a per-lane ``[lane, :]`` subscript)
trips a deeper substrate boundary than the join-on-layout work in this
revision: the layout's ``storage_size`` (256, the wave-wide backing)
diverges from the per-workitem dim product (8) when
``hc-flatten-with-layouts`` materialises the layout, while the bare
``wmma_gfx11`` intrinsic result flattens through the dim product —
the two sides of the K-tile loop's iter_init / iter_result pair end up
with incompatible flat extents. Reconciling that needs the intrinsic
recipe to plumb a matching layout (or the projection at the lane
boundary to strip the wave-wide backing); the strided-slice form here
stays as the right side of that substrate boundary in the meantime.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

import hc.simulator as sim
from hc import (
    Buffer,
    WorkGroup,
    WorkItem,
    idx_type,
    index_map,
    kernel,
    sym,
    tensor_type,
    undef_type,
    vector_type,
)
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


def _tile_origin(tile_row: int, tile_col: int) -> tuple[int, int]:
    return tile_row * WMMA_M, tile_col * WMMA_N


# Layout that captures the per-lane WMMA accumulator addressing
# inside a 16x16 C tile: lane `L` owns the elements at
# `(row = L // WMMA_N + fi * WMMA_ACC_ROW_STRIDE, col = L % WMMA_N)`
# for `fi in [0, WMMA_ACC_FRAGMENT)`. `storage_size = WMMA_M * WMMA_N`
# matches the flat 16x16 tile span; the rank-balanced `(lane, fi)`
# logical shape satisfies `LayoutAttr`'s
# `index_syms.size() == shape_syms.size()` contract.
#
# `init_wmma_acc` and `store_wmma_tile` realise the same arithmetic
# as a strided `c[row_slice, col_slice]` view. Single declared
# formula, two call sites that decode it by hand — see the module
# docstring for the substrate boundary the direct layout-driven form
# still trips.
WAVE_ACC_FRAG_LAYOUT = index_map(
    storage_size=lambda lc, fc: WMMA_M * WMMA_N,
    offset=lambda lane, fi, lc, fc: (lane // WMMA_N + fi * WMMA_ACC_ROW_STRIDE) * WMMA_N
    + (lane % WMMA_N),
)


def _require_signature_arg(sig, index, *, kind, shape, dtype, name):
    value = sig.arg(index)
    is_expected = value.is_tensor() if kind == "tensor" else value.is_vector()
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
    operand_types=(
        undef_type(),
        tensor_type((WMMA_M, WMMA_K), np.float16),
        tensor_type((WMMA_K, WMMA_N), np.float16),
        vector_type((WMMA_K,), np.float16),
        vector_type((WMMA_K,), np.float16),
        vector_type((WMMA_ACC_FRAGMENT,), np.float32),
        idx_type(),
    ),
    result_types=(vector_type((WMMA_ACC_FRAGMENT,), np.float32),),
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
    # `acc_frag.mask` carries the per-element output-validity bits stamped
    # in `init_wmma_acc` (true iff the corresponding `c[row, col]` is in
    # bounds) and forwarded through every iteration. Skip the multiply-add
    # for masked-off lanes so the fallback does not touch the poison
    # accumulator slots that bounds clipping leaves behind, and so the
    # simulator path mirrors the hardware lowering where the same mask gates
    # both the WMMA result and the eventual store.
    acc_mask = acc_frag.mask
    for index in range(len(rows)):
        accum = np.float32(0)
        if bool(acc_mask[index]):
            accum = np.float32(acc_frag[index])
            for k_idx in range(WMMA_K):
                accum += np.float32(a_tile[rows[index], k_idx]) * np.float32(
                    b_tile[k_idx, col]
                )
        values[index] = accum
    # Forward `acc_frag.mask` so the WMMA result keeps the input mask on its
    # output channel, matching what the hardware recipe does (see
    # `_lower_wmma`'s second return value).
    return group.vload(values, mask=acc_frag.mask)


@wmma_gfx11.verify
def _verify_wmma(sig, target):
    _ = target
    _require_wmma_context(sig)
    _require_wmma_operands(sig)


_FRAG_AB_TYPE = f"vector<{WMMA_K}xf16>"
_FRAG_ACC_TYPE = f"vector<{WMMA_ACC_FRAGMENT}xf32>"


@wmma_gfx11.lower(target="amdgpu-gfx11")
def _lower_wmma(t, call):
    # `amdgpu.wmma` only consumes the per-lane fragment vectors and the
    # accumulator; the staged tiles, lane index, group token, and the
    # `arch`/`wave_size` const_kwargs ride on the call site purely to
    # gate dispatch. We touch the unused handles to surface a recipe-time
    # error if the intrinsic signature ever drifts under us. The `.data`
    # / `.mask` suffixes mirror `hc-decompose-shaped-values`: by the time
    # `-hc-interpret-intrinsic-recipes` runs, every shaped operand has been
    # split into a data + mask pair and the call site exposes both.
    _ = (
        call.operand("group"),
        call.operand("a_tile.data"),
        call.operand("a_tile.mask"),
        call.operand("b_tile.data"),
        call.operand("b_tile.mask"),
        call.operand("lane"),
        call.attr("arch"),
        call.attr("wave_size"),
    )
    # Pre-rewrite assertions: target-dispatch already filters by the recipe's
    # `hc.target = "amdgpu-gfx11"`, but that says nothing about the call
    # site's *actual* `arch`/`wave_size`. A kernel whose `GFX_ARCH` drifted
    # to `"gfx12"` would otherwise smuggle a wrong-arch call into the
    # `amdgpu.wmma` rewrite and surface as a far-downstream verifier crash.
    # Match the `i64` width the frontend emits for `wave_size`; a width
    # mismatch counts as a value mismatch.
    t.require_attr(call, "arch", GFX_ARCH)
    t.require_attr(call, "wave_size", t.i64(WAVE_LANES))
    # `amdgpu.wmma` rejects HC bare types; bridge through
    # `unrealized_conversion_cast` at every operand and at the result. The
    # `expected_type=` shortcut on `call.operand` plants the operand-side
    # casts; `t.create(..., result_types=[upstream_str])` plus a closing
    # `t.cast` swap the result-side type back to the bare form the call
    # site exposes. `hc-lower-launch-body` already plants paired UCCs
    # around every `hc.call_intrinsic` boundary, so the recipe-inserted
    # casts pair with those existing ones and a post-rewrite
    # `--canonicalize` collapses the chains to identity — `amdgpu.wmma`
    # ends up sitting between plain upstream `vector<...>` values with
    # no leftover bridging machinery.
    op = t.create(
        "amdgpu.wmma",
        result_types=[_FRAG_ACC_TYPE],
        operands=[
            call.operand("a_frag.data", expected_type=_FRAG_AB_TYPE),
            call.operand("b_frag.data", expected_type=_FRAG_AB_TYPE),
            call.operand("acc_frag.data", expected_type=_FRAG_ACC_TYPE),
        ],
        # `amdgpu.wmma` declares `m`, `n`, `k` as `i32` attributes with
        # confined value sets; emit them at the right width so the upstream
        # verifier doesn't reject the freshly created op for "expected
        # 'i32' but got 'i64'".
        attrs={
            "m": t.i32(WMMA_M),
            "n": t.i32(WMMA_N),
            "k": t.i32(WMMA_K),
        },
    )
    # `amdgpu.wmma` returns a single fragment data vector. The call site
    # post-decomposition has two results — `acc.data` (the new accumulator)
    # and `acc.mask` (its validity bits). The mask channel is invariant
    # across the matmul step (the per-lane accumulator stays valid wherever
    # it was valid going in), so forward `acc_frag.mask` unchanged as the
    # second replacement.
    return (
        t.cast(op.result(0), to=call.result_type(0)),
        call.operand("acc_frag.mask"),
    )


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
def init_wmma_acc(group, c, row0, col0):
    # Stamp the per-element output-validity mask onto the accumulator
    # fragment at construction time. WMMA's recipe forwards the input
    # acc fragment's mask onto the result, so a bounds-aware mask stamped
    # here propagates through every loop iteration and the final
    # `group.store` keeps a real per-element `scf.if` guard instead of
    # canonicalize folding it to an unconditional store that walks past
    # `c[M, N]` for off-tile shapes.
    #
    # Why init and not `issue_wmma_tile`: putting the bounds-stamp inside
    # the body of the k-tile `for` loop trips an iterative-inference
    # fixed-point issue in `hc-infer-types`. Doing it once before the
    # loop sidesteps that issue while still pinning the mask into every
    # accumulator carried around the loop.
    @group.workitems
    def init(wi):
        lane = wi.local_id()[0]
        # Slice the per-lane strided fragment out of the 16x16 C tile.
        # The row and column expressions are the same arithmetic
        # `WAVE_ACC_FRAG_LAYOUT.offset` encodes — single source of truth
        # for the per-lane addressing, realised here as a strided
        # `hc.buffer_view`. `vload`'s clip-and-pad rule sets the mask
        # channel true exactly where the corresponding C element is
        # in-bounds, OOB positions zero out; `c` arrives zero-initialised
        # per the accumulator contract so the data channel is zero
        # everywhere and the loaded vector serves directly as the
        # running accumulator without poisoning the WMMA `a*b + acc`
        # math.
        bounds = group.vload(
            c[
                row0 + lane // WMMA_N : row0 + WMMA_M : WMMA_ACC_ROW_STRIDE,
                col0 + lane % WMMA_N : col0 + lane % WMMA_N + 1,
            ],
            shape=(WMMA_ACC_FRAGMENT, 1),
        )
        return bounds[:, 0]

    return init()


@kernel.func(scope=WorkGroup)
def issue_wmma_tile(group, a_tile, b_tile, acc):
    @group.workitems
    def wave(wi):
        lane = wi.local_id()[0]
        a_frag = load_wmma_a_fragment(wi, a_tile)
        b_frag = load_wmma_b_fragment(wi, b_tile)
        # `acc[:, lane, 0]` indexes out the trailing singleton collective axis
        # so the WMMA input matches its declared `(WMMA_ACC_FRAGMENT,)` vector
        # shape; the leading `:` keeps the fragment elements (and their
        # bounds-aware mask, see `init_wmma_acc`).
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
def store_wmma_tile(group, c, row0, col0, acc) -> None:
    @group.workitems
    def wave(wi):
        lane = wi.local_id()[0]
        # `WAVE_ACC_FRAG_LAYOUT.offset` applied in the inverse
        # direction: for each per-lane fragment index `fi`, the
        # element lives at `(row0 + lane // WMMA_N + fi *
        # WMMA_ACC_ROW_STRIDE, col0 + lane % WMMA_N)`. Expressed as a
        # strided `c[row_slice, col_slice]` view — the form the
        # simulator scatters via numpy strides and the gpu path lowers
        # through `hc.generic`. A symmetric layout-aware scatter on
        # `group.store` would let us shed the manual inverse; until
        # then the formula here must stay in lockstep with the
        # `WAVE_ACC_FRAG_LAYOUT` declaration above.
        group.store(
            c[
                row0 + lane // WMMA_N : row0 + WMMA_M : WMMA_ACC_ROW_STRIDE,
                col0 + lane % WMMA_N : col0 + lane % WMMA_N + 1,
            ],
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
    a: Buffer[M, K, np.float16],
    b: Buffer[K, N, np.float16],
    c: Buffer[M, N, np.float32],
) -> None:
    row0, col0 = _tile_origin(group.group_id[0], group.group_id[1])
    acc = init_wmma_acc(group, c, row0, col0)

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


def _require_torch_cuda(surface: str):
    """Import torch and confirm a HIP/ROCm device is visible.

    Both `--run-on-hw` and `--bench` need the same precondition;
    factoring the import + check keeps the error texts consistent
    (`surface=` names which flag the caller passed so the message is
    actionable instead of generic).
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"{surface} needs the `torch` package to allocate device "
            "buffers; `pip install torch` (with a ROCm-enabled wheel) and "
            "retry."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{surface} needs a HIP/ROCm device visible to torch.cuda; "
            "torch.cuda.is_available() returned False."
        )
    return torch


def run_on_hardware(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rtol: float = 0.0,
    atol: float = 2e-3,
) -> np.ndarray:
    """Compile for gfx11 and invoke through the bundled HIP shim.

    Mirrors `tests/test_examples.py::test_gfx11_wmma_example_invokes_on_real_hardware`,
    which is the executable spec for the same chain. We use `torch.cuda`
    tensors for the device buffers because `Tensor.data_ptr()` returns a
    HIP-allocated pointer that `_mlir_ciface_hc_get_ptr` hands straight
    to `gpu.launch_func` — the runtime helpers don't allocate or copy on
    their own. Raises a clear `RuntimeError` (not `ImportError`)
    when torch or torch.cuda is missing so callers see a single
    actionable message instead of a stack trace.
    """

    torch = _require_torch_cuda("--run-on-hw")

    import hc

    m, _ = a.shape
    _, n = b.shape
    a_dev = torch.from_numpy(a).cuda()
    b_dev = torch.from_numpy(b).cuda()
    c_dev = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    compiled = hc.compile(tiled_gfx11_wmma_matmul, target="amdgpu-gfx11")
    compiled.invoke(a_dev, b_dev, c_dev)

    out = c_dev.cpu().numpy()
    reference = reference_blocked_matmul(a, b)
    np.testing.assert_allclose(out, reference, rtol=rtol, atol=atol)
    return out


def bench_on_hardware(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_inner: int = 50,
    m_outer: int = 20,
    warmup: int = 3,
    rtol: float = 0.0,
    atol: float = 2e-3,
):
    """Compile with bench=True, smoke-check the output, then bench.

    Same device-allocation shape as `run_on_hardware`. We sample one
    invoke before timing and assert the output matches the numpy
    reference so the bench numbers below come from a known-correct
    kernel — timing a miscompile is worse than failing loudly, and the
    asserted diff floor is exactly the same `(atol, rtol)` pair the
    simulator and `run_on_hardware` paths use.
    """
    torch = _require_torch_cuda("--bench")

    import hc

    m, _ = a.shape
    _, n = b.shape
    a_dev = torch.from_numpy(a).cuda()
    b_dev = torch.from_numpy(b).cuda()
    c_dev = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    compiled = hc.compile(tiled_gfx11_wmma_matmul, target="amdgpu-gfx11", bench=True)
    # One untimed invoke to fill `c_dev`. The bench loop after this
    # overwrites it `m_outer*n_inner` times with the same value (inputs
    # are constant across samples), so the final readback reflects the
    # same kernel output as the smoke check below.
    compiled.invoke(a_dev, b_dev, c_dev)
    out = c_dev.cpu().numpy()
    reference = reference_blocked_matmul(a, b)
    np.testing.assert_allclose(out, reference, rtol=rtol, atol=atol)

    result = compiled.bench(
        (a_dev, b_dev, c_dev),
        n_inner=n_inner,
        m_outer=m_outer,
        warmup=warmup,
    )
    return result, out


def dump_front_ir() -> None:
    """Lower the kernel + its transitive deps to ``hc_front`` and print it.

    Uses the same resolver that ``hc.compile`` does so the dumped module
    matches what downstream lowering will see — one combined module with
    every `hc_front.name` load carrying a ``ref`` classification.

    Imports the resolver lazily so the simulator path stays independent of
    the managed ``hc_mlir`` native bindings.
    """

    from hc._resolve import resolve_front_ir
    from hc.mlir import ir

    with ir.Context() as context:
        resolved = resolve_front_ir(tiled_gfx11_wmma_matmul, context=context)
        print(str(resolved.module))


def dump_hc_ir() -> None:
    """Run the current ``hc_front`` -> ``hc`` pipeline and print its result."""

    from hc import compile as compile_kernel

    handle = compile_kernel(tiled_gfx11_wmma_matmul)
    if handle.hc_ir_text is None:
        print("hc_front -> hc pipeline failed.", file=sys.stderr)
        if handle.pipeline_diagnostics:
            for diagnostic in handle.pipeline_diagnostics:
                print(diagnostic, file=sys.stderr)
        else:
            print("No pipeline diagnostics were captured.", file=sys.stderr)
        raise SystemExit(1)
    for diagnostic in handle.pipeline_diagnostics:
        print(diagnostic, file=sys.stderr)
    print(handle.hc_ir_text)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gfx11 WMMA tiled matmul example.",
    )
    dump_group = parser.add_mutually_exclusive_group()
    dump_group.add_argument(
        "--dump-front-ir",
        action="store_true",
        help=(
            "lower the kernel + its transitive @kernel.func / @kernel.intrinsic "
            "helpers into one hc_front module, resolve every name reference, "
            "and print the MLIR to stdout instead of running the simulator"
        ),
    )
    dump_group.add_argument(
        "--dump-hc-ir",
        action="store_true",
        help=(
            "run the current hc_front -> hc pipeline over the WMMA kernel "
            "module and print the resulting hc MLIR to stdout instead of "
            "running the simulator"
        ),
    )
    dump_group.add_argument(
        "--run-on-hw",
        action="store_true",
        help=(
            "compile for amdgpu-gfx11 and dispatch through "
            "hc.compile().invoke() against torch.cuda tensors on a "
            "physical gfx11 GPU instead of running the simulator; "
            "needs torch + a HIP/ROCm device"
        ),
    )
    dump_group.add_argument(
        "--bench",
        action="store_true",
        help=(
            "compile for amdgpu-gfx11 with bench=True, smoke-check the "
            "output against the blocked numpy reference, then run "
            "compiled.bench(...) and print the stats table; needs torch + "
            "a HIP/ROCm device"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.dump_front_ir:
        dump_front_ir()
        return
    if args.dump_hc_ir:
        dump_hc_ir()
        return

    a, b = make_demo_inputs()
    if args.run_on_hw:
        out = run_on_hardware(a, b)
        reference = reference_blocked_matmul(a, b)
        max_diff = float(np.max(np.abs(out - reference)))
        print("gfx11 WMMA tiled matmul example passed on real hardware.")
        print(f"shape: A={a.shape}, B={b.shape}, C={out.shape}")
        # Round-off vs the blocked f32 reference; expected to be on the
        # order of f16 mantissa (~1e-3) for the demo's uniform [-1, 1]
        # inputs once we cross the f16 -> f32 accumulation boundary.
        print(f"max abs diff vs blocked fallback reference: {max_diff}")
        return
    if args.bench:
        result, out = bench_on_hardware(a, b)
        reference = reference_blocked_matmul(a, b)
        max_diff = float(np.max(np.abs(out - reference)))
        print(f"shape: A={a.shape}, B={b.shape}, C={out.shape}")
        print(f"max abs diff vs blocked fallback reference: {max_diff}")
        print(result.summary())
        return

    out = simulate_gfx11_wmma_matmul(a, b)
    reference = reference_blocked_matmul(a, b)
    np.testing.assert_allclose(out, reference, rtol=0.0, atol=2e-6)
    print("gfx11 WMMA tiled matmul example passed.")
    print(f"shape: A={a.shape}, B={b.shape}, C={out.shape}")
    max_diff = np.max(np.abs(out - reference))
    print(f"max abs diff vs blocked fallback reference: {max_diff}")


if __name__ == "__main__":
    main()
