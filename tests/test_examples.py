# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import numpy as np
import pytest

import hc.simulator as sim
from examples.amdgpu_gfx11_wmma_matmul import (
    dump_hc_ir,
    make_demo_inputs,
    reference_blocked_matmul,
    simulate_gfx11_wmma_matmul,
    tiled_gfx11_wmma_matmul,
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


def test_gfx11_wmma_example_does_not_write_past_c_extent() -> None:
    # Off-tile shape: M=24, N=24 are not multiples of WMMA_M=WMMA_N=16, so the
    # single right/bottom tile for each `(M, N)` covers a strict superset of
    # the live region. Pad `c` with sentinel values around the live `(M, N)`
    # extent and assert the kernel never touches the padding — this is the
    # "no OOB writes" property the bounds-aware accumulator mask is meant
    # to enforce.
    sentinel = np.float32(-1234.5)
    a, b = make_demo_inputs(m=24, n=24, k=32, seed=11)
    padded = np.full((40, 40), sentinel, dtype=np.float32)
    live = padded[:24, :24]
    live[...] = 0  # accumulator-output contract: caller zero-fills `c`.

    sim.launch(tiled_gfx11_wmma_matmul, a, b, live)

    reference = reference_blocked_matmul(a, b)
    np.testing.assert_allclose(live, reference, rtol=0.0, atol=2e-6)
    # Tail region untouched: any fragment element whose `(row, col)` lies
    # outside `c[:24, :24]` must be masked off by `init_wmma_acc` and thus
    # skipped by the per-element store guards.
    assert np.all(padded[24:, :] == sentinel)
    assert np.all(padded[:24, 24:] == sentinel)


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_gfx11_wmma_example_dumps_current_pipeline_ir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dump_hc_ir()

    captured = capsys.readouterr()
    # The default schedule lowers all the way through to a self-
    # contained LLVM module: `hc-lower-gpu-to-binary` produces a
    # `gpu.binary` HSACO blob, then `hc-lower-launch-func-to-runtime`
    # embeds that blob as an LLVM global and rewrites every
    # `gpu.launch_func` as a pair of `hc_rt_load_kernel` +
    # `hc_rt_launch_kernel` calls. Source `gpu.binary` is erased.
    # The host wrapper takes one `PyObject *` (lowered to `!llvm.ptr`)
    # per kernel argument and calls the `_mlir_ciface_hc_get_*`
    # helpers (declared with `llvm.emit_c_interface` so the wrapper
    # mangling lands on the symbols that libhc_rt_helpers.so exports)
    # to materialize each tensor's data pointer and shape dims before
    # dispatching. Pin each load-bearing milestone so the dump test
    # fails loudly if anything regresses.
    assert "module attributes {gpu.container_module}" in captured.out
    assert (
        "llvm.func @tiled_gfx11_wmma_matmul(%arg0: !llvm.ptr, "
        "%arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr)" in captured.out
    )
    assert "@hc_get_buffer" in captured.out
    assert "@_mlir_ciface_hc_get_buffer" in captured.out
    assert "@hc_get_dim" in captured.out
    assert "@_mlir_ciface_hc_get_dim" in captured.out
    assert "@hc_rt_load_kernel" in captured.out
    assert "@hc_rt_launch_kernel" in captured.out
    assert "@tiled_gfx11_wmma_matmul_kernel_data" in captured.out
    assert "@tiled_gfx11_wmma_matmul_kernel_handle" in captured.out
    assert "gpu.launch_func" not in captured.out
    assert "gpu.binary" not in captured.out
    assert "hc.kernel" not in captured.out
    assert "hc_front." not in captured.out
