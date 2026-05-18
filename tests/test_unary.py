# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end coverage for Python unary operators in kernels.

Sim run pins the frontend resolver + `_MaskedValue.__neg__` path.
Native compile + HIP invoke (gated by `HC_RT_RUN_HIP_INVOKE_TEST=1`)
pins `hc_front.unaryop "USub"` -> `hc.neg` -> `hc-elementwise-to-generic`
-> AMDGPU binary.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import hc
import hc.simulator as sim
from hc import Buffer, kernel, sym

N = sym.N


# `N` literal so the partial-tile OOB predicate (`gid + i_0 < N`) folds
# at compile time. Runtime `N` would leave the predicate's free symbol
# unbound during apply lowering.
@kernel(work_shape=(N,), group_shape=(8,), literals=[N])
def negate_kernel(
    group,
    x: Buffer[N, np.float32],
    out: Buffer[N, np.float32],
) -> None:
    gid = group.work_offset[0]
    g0 = group.shape[0]
    tile = group.load(x[gid:], shape=(g0,))
    group.store(out[gid : gid + g0], -tile)


def _make_inputs(n: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)


_RUN_HIP_INVOKE_TESTS = pytest.mark.skipif(
    os.environ.get("HC_RT_RUN_HIP_INVOKE_TEST") != "1",
    reason="set HC_RT_RUN_HIP_INVOKE_TEST=1 on a HIP-visible host",
)


@pytest.mark.parametrize("n", [8, 16, 24])
def test_negate_simulator_matches_numpy(n: int) -> None:
    x = _make_inputs(n, seed=11)
    out = np.zeros((n,), dtype=np.float32)
    sim.launch(negate_kernel, x, out)
    np.testing.assert_array_equal(out, -x)


def test_negate_compiles() -> None:
    x = _make_inputs(16, seed=21)
    handle = hc.compile(negate_kernel, symbols={N: x.shape[0]})
    assert (
        handle.hc_ir_text is not None
    ), "compile produced no hc_ir; pipeline diagnostics:\n  " + "\n  ".join(
        handle.pipeline_diagnostics
    )


@_RUN_HIP_INVOKE_TESTS
@pytest.mark.parametrize("n", [8, 16])
def test_negate_invokes_on_real_hardware(n: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda unavailable")

    x = _make_inputs(n, seed=23)
    x_dev = torch.from_numpy(x).cuda()
    out_dev = torch.zeros(n, dtype=torch.float32, device="cuda")

    compiled = hc.compile(negate_kernel, symbols={N: n}, target="amdgpu-gfx11")
    compiled.invoke(x_dev, out_dev)

    np.testing.assert_allclose(out_dev.cpu().numpy(), -x, rtol=0.0, atol=0.0)
