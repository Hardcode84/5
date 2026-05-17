# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end coverage for Python `@` lowering to `hc.matmul`.

Sim run pins the frontend resolver + `_MaskedValue.__matmul__` path.
Native compile + HIP invoke (gated by `HC_RT_RUN_HIP_INVOKE_TEST=1`)
pins `hc_front.binop "MatMult"` -> `hc.matmul` -> generic-pipeline ->
AMDGPU binary.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import hc
import hc.simulator as sim
from hc import Buffer, kernel, sym

M = sym.M
N = sym.N
K = sym.K

_GROUP = (8, 8)


# `K` is a literal so the matmul reduction iter has a compile-time bound
# (LDS tile + generic-pipeline unrolling want concrete K). `M` / `N`
# stay symbolic and ride the work-shape.
@kernel(work_shape=(M, N), group_shape=_GROUP, literals=[K])
def matmul_at_kernel(
    group,
    a: Buffer[M, K, np.float32],
    b: Buffer[K, N, np.float32],
    c: Buffer[M, N, np.float32],
) -> None:
    gid = group.work_offset
    g0, g1 = group.shape
    a_tile = group.load(a[gid[0] : gid[0] + g0, :], shape=(g0, a.shape[1]))
    b_tile = group.load(b[:, gid[1] : gid[1] + g1], shape=(b.shape[0], g1))
    group.store(c[gid[0] : gid[0] + g0, gid[1] : gid[1] + g1], a_tile @ b_tile)


def _make_inputs(m: int, n: int, k: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(k, n)).astype(np.float32)
    return a, b


_RUN_HIP_INVOKE_TESTS = pytest.mark.skipif(
    os.environ.get("HC_RT_RUN_HIP_INVOKE_TEST") != "1",
    reason="set HC_RT_RUN_HIP_INVOKE_TEST=1 on a HIP-visible host",
)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(8, 8, 8), (16, 16, 8), (24, 16, 8)],
)
def test_matmul_at_simulator_matches_numpy(m: int, n: int, k: int) -> None:
    a, b = _make_inputs(m, n, k, seed=13)
    c = np.zeros((m, n), dtype=np.float32)

    sim.launch(matmul_at_kernel, a, b, c)

    np.testing.assert_allclose(c, a @ b, rtol=1e-5, atol=1e-5)


def test_matmul_at_compiles() -> None:
    a, _ = _make_inputs(16, 16, 8, seed=21)
    handle = hc.compile(matmul_at_kernel, symbols={K: a.shape[1]})
    assert (
        handle.hc_ir_text is not None
    ), "compile produced no hc_ir; pipeline diagnostics:\n  " + "\n  ".join(
        handle.pipeline_diagnostics
    )
    assert (
        "hc.matmul" not in handle.hc_ir_text
    ), "hc.matmul should be lowered away by the generic pipeline"


@_RUN_HIP_INVOKE_TESTS
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(8, 8, 8), (16, 16, 8)],
)
def test_matmul_at_invokes_on_real_hardware(m: int, n: int, k: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda unavailable")

    a, b = _make_inputs(m, n, k, seed=29)
    expected = a @ b

    a_dev = torch.from_numpy(a).cuda()
    b_dev = torch.from_numpy(b).cuda()
    c_dev = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    compiled = hc.compile(matmul_at_kernel, symbols={K: k}, target="amdgpu-gfx11")
    compiled.invoke(a_dev, b_dev, c_dev)

    np.testing.assert_allclose(c_dev.cpu().numpy(), expected, rtol=1e-4, atol=1e-4)
