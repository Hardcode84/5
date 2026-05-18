# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end coverage for Python bitwise `|` / `&` in kernels.

Boolean tiles are the natural surface -- shaped `hc.or` / `hc.and`
combine masks. Native compile (gated by absence of the signed-integer
fill bug) pins `hc_front.binop "BitOr"` -> `hc.or` -> the
`hc-decompose-shaped-values` data/mask split and the
`hc-elementwise-to-generic` body funnel. HIP invoke is gated by
`HC_RT_RUN_HIP_INVOKE_TEST=1`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import hc
import hc.simulator as sim
from hc import Buffer, kernel, sym

N = sym.N


# `N` literal so the partial-tile OOB predicate folds at compile time.
# Signless `i1` (Python `bool`) sidesteps the open signed-int constant
# fill bug; `|` is the elementwise mask combine surface.
@kernel(work_shape=(N,), group_shape=(8,), literals=[N])
def bitor_kernel(
    group,
    a: Buffer[N, np.bool_],
    b: Buffer[N, np.bool_],
    out: Buffer[N, np.bool_],
) -> None:
    gid = group.work_offset[0]
    g0 = group.shape[0]
    a_tile = group.load(a[gid:], shape=(g0,))
    b_tile = group.load(b[gid:], shape=(g0,))
    group.store(out[gid : gid + g0], a_tile | b_tile)


@kernel(work_shape=(N,), group_shape=(8,), literals=[N])
def bitand_kernel(
    group,
    a: Buffer[N, np.bool_],
    b: Buffer[N, np.bool_],
    out: Buffer[N, np.bool_],
) -> None:
    gid = group.work_offset[0]
    g0 = group.shape[0]
    a_tile = group.load(a[gid:], shape=(g0,))
    b_tile = group.load(b[gid:], shape=(g0,))
    group.store(out[gid : gid + g0], a_tile & b_tile)


def _make_inputs(n: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=(n,), dtype=np.bool_)
    b = rng.integers(0, 2, size=(n,), dtype=np.bool_)
    return a, b


_RUN_HIP_INVOKE_TESTS = pytest.mark.skipif(
    os.environ.get("HC_RT_RUN_HIP_INVOKE_TEST") != "1",
    reason="set HC_RT_RUN_HIP_INVOKE_TEST=1 on a HIP-visible host",
)


@pytest.mark.parametrize("n", [8, 16, 24])
def test_bitor_simulator_matches_numpy(n: int) -> None:
    a, b = _make_inputs(n, seed=11)
    out = np.zeros((n,), dtype=np.bool_)
    sim.launch(bitor_kernel, a, b, out)
    np.testing.assert_array_equal(out, a | b)


@pytest.mark.parametrize("n", [8, 16, 24])
def test_bitand_simulator_matches_numpy(n: int) -> None:
    a, b = _make_inputs(n, seed=13)
    out = np.zeros((n,), dtype=np.bool_)
    sim.launch(bitand_kernel, a, b, out)
    np.testing.assert_array_equal(out, a & b)


def test_bitor_compiles() -> None:
    a, _ = _make_inputs(16, seed=21)
    handle = hc.compile(bitor_kernel, symbols={N: a.shape[0]})
    assert (
        handle.hc_ir_text is not None
    ), "compile produced no hc_ir; pipeline diagnostics:\n  " + "\n  ".join(
        handle.pipeline_diagnostics
    )


@_RUN_HIP_INVOKE_TESTS
@pytest.mark.parametrize("n", [8, 16])
def test_bitor_invokes_on_real_hardware(n: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda unavailable")

    a, b = _make_inputs(n, seed=23)
    a_dev = torch.from_numpy(a).cuda()
    b_dev = torch.from_numpy(b).cuda()
    out_dev = torch.zeros(n, dtype=torch.bool, device="cuda")

    compiled = hc.compile(bitor_kernel, symbols={N: n}, target="amdgpu-gfx11")
    compiled.invoke(a_dev, b_dev, out_dev)

    np.testing.assert_array_equal(out_dev.cpu().numpy(), a | b)
