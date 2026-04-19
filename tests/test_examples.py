# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from examples.amdgpu_gfx11_wmma_matmul import (
    make_demo_inputs,
    reference_blocked_matmul,
    simulate_gfx11_wmma_matmul,
)


@pytest.mark.parametrize(("m", "n", "k"), [(16, 16, 16), (32, 32, 32)])
def test_gfx11_wmma_example_matches_blocked_reference(m: int, n: int, k: int) -> None:
    a, b = make_demo_inputs(m=m, n=n, k=k, seed=7)

    out = simulate_gfx11_wmma_matmul(a, b)
    reference = reference_blocked_matmul(a, b)

    np.testing.assert_allclose(out, reference, rtol=0.0, atol=2e-6)
