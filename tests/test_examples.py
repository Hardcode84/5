# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import numpy as np
import pytest

from examples.amdgpu_gfx11_wmma_matmul import (
    dump_hc_ir,
    make_demo_inputs,
    reference_blocked_matmul,
    simulate_gfx11_wmma_matmul,
)

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 16, 16), (32, 32, 32), (17, 19, 18)],
)
def test_gfx11_wmma_example_matches_blocked_reference(m: int, n: int, k: int) -> None:
    a, b = make_demo_inputs(m=m, n=n, k=k, seed=7)

    out = simulate_gfx11_wmma_matmul(a, b)
    reference = reference_blocked_matmul(a, b)

    np.testing.assert_allclose(out, reference, rtol=0.0, atol=2e-6)


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_gfx11_wmma_example_dumps_current_pipeline_ir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dump_hc_ir()

    captured = capsys.readouterr()
    assert "func.func @tiled_gfx11_wmma_matmul" in captured.out
    # The default schedule outlines the launch into `gpu.launch_func`
    # (host) + `gpu.module @<kernel>_kernel` (device) and stamps the
    # module with the gfx1100 rocdl target. Pin both halves so the
    # dump test fails loudly if outlining/attach-target ever drops out.
    assert "gpu.launch_func" in captured.out
    assert "gpu.module @tiled_gfx11_wmma_matmul_kernel" in captured.out
    assert '#rocdl.target<chip = "gfx1100">' in captured.out
    assert "hc.kernel" not in captured.out
    assert "hc_front." not in captured.out
