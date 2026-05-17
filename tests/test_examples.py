# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import numpy as np
import pytest

import hc
import hc.simulator as sim
from examples.amdgpu_gfx11_wmma_matmul import (
    bench_on_hardware,
    dump_hc_ir,
    make_demo_inputs,
    reference_blocked_matmul,
    simulate_gfx11_wmma_matmul,
    tiled_gfx11_wmma_matmul,
)
from examples.pairwise_distance import (
    compile_pairwise_distance,
    reference_pairwise_distance,
    simulate_pairwise_distance,
)
from examples.pairwise_distance import (
    make_demo_inputs as make_pairwise_inputs,
)
from examples.pairwise_distance import (
    run_on_hardware as run_pairwise_on_hardware,
)

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)

# Real-hardware gate. Default-skipped — dlopens `libamdhip64.so` and
# dispatches on the GPU bound by `HIP_VISIBLE_DEVICES`.
_RUN_HIP_INVOKE_TESTS = pytest.mark.skipif(
    os.environ.get("HC_RT_RUN_HIP_INVOKE_TEST") != "1",
    reason="set HC_RT_RUN_HIP_INVOKE_TEST=1 on a host with a gfx11 GPU to run",
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
    # (24, 24, 32): M=N=24 not multiples of WMMA_M=WMMA_N=16, so the
    # last tile is a strict superset of the live region. Sentinel
    # padding outside `c[:24, :24]` pins the OOB-write property the
    # bounds-aware accumulator mask enforces.
    sentinel = np.float32(-1234.5)
    a, b = make_demo_inputs(m=24, n=24, k=32, seed=11)
    padded = np.full((40, 40), sentinel, dtype=np.float32)
    live = padded[:24, :24]
    live[...] = 0  # accumulator-output contract: caller zero-fills `c`.

    sim.launch(tiled_gfx11_wmma_matmul, a, b, live)

    reference = reference_blocked_matmul(a, b)
    np.testing.assert_allclose(live, reference, rtol=0.0, atol=2e-6)
    assert np.all(padded[24:, :] == sentinel)
    assert np.all(padded[:24, 24:] == sentinel)


@_RUN_HIP_INVOKE_TESTS
def test_gfx11_wmma_example_benches_on_real_hardware() -> None:
    """End-to-end acceptance for `CompiledKernel.bench(...)`.

    Same JIT + HIP path as invoke, through the bench wrapper.
    Verifies `BenchResult` shape and headline consistency; tighter
    bounds would flake against driver/queue jitter.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda unavailable")

    a, b = make_demo_inputs(m=32, n=32, k=32, seed=17)
    # Modest m_outer/n_inner keeps wall time bounded.
    # `bench_on_hardware` checks numerics before timing so reaching the
    # asserts below also covers numerics.
    result, _ = bench_on_hardware(a, b, n_inner=4, m_outer=5, warmup=1)
    assert result.kernel_name == "tiled_gfx11_wmma_matmul"
    assert result.m_outer == 5
    assert result.n_inner == 4
    assert result.samples_ns.shape == (5,)
    assert result.samples_ns.dtype == np.int64
    # Positive ns: zero/negative would mean the monotonic clock
    # regressed across at least one dispatch + sync.
    assert int(result.samples_ns.min()) > 0
    assert result.per_launch_median_ns == pytest.approx(
        result.median_ns / result.n_inner
    )
    summary = result.summary()
    assert "tiled_gfx11_wmma_matmul" in summary


@_RUN_HIP_INVOKE_TESTS
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 16, 16), (32, 32, 32)],
)
def test_gfx11_wmma_example_invokes_on_real_hardware(m: int, n: int, k: int) -> None:
    """End-to-end gfx11 WMMA dispatch through ORC LLJIT + HIP shim.

    `torch.cuda` backs device buffers: `Tensor.data_ptr()` returns the
    HIP-allocated pointer `_mlir_ciface_hc_get_ptr` hands to
    `gpu.launch_func`. Skip cleanly if torch isn't installed.
    """

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda unavailable")

    a, b = make_demo_inputs(m=m, n=n, k=k, seed=13)
    expected = reference_blocked_matmul(a, b)

    a_dev = torch.from_numpy(a).cuda()
    b_dev = torch.from_numpy(b).cuda()
    c_dev = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    compiled = hc.compile(tiled_gfx11_wmma_matmul, target="amdgpu-gfx11")
    compiled.invoke(a_dev, b_dev, c_dev)

    out = c_dev.cpu().numpy()
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=2e-3)


@pytest.mark.parametrize(
    ("w1", "w2", "h"),
    [(6, 5, 4), (8, 8, 3), (3, 7, 5)],
)
def test_pairwise_distance_simulator_matches_numpy(w1: int, w2: int, h: int) -> None:
    """Simulator pin for the langref WG-level pairwise Euclidean distance.

    Broadcasting axes here are corrected from the langref text (which
    writes the result transposed); per-element contract matches the
    workitem-level form in the same doc section.
    """

    x1, x2 = make_pairwise_inputs(w1=w1, w2=w2, h=h, seed=29)
    out = simulate_pairwise_distance(x1, x2)
    ref = reference_pairwise_distance(x1, x2)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("h", [3, 4, 8])
def test_pairwise_distance_native_compile(h: int) -> None:
    """Compile-only gate: hc pipeline with `H` bound via `symbols=`.

    Non-empty `hc_ir` proves every pass succeeded; diagnostics surface
    otherwise. Invoke path is covered by the matmul HIP gate.
    """

    x1, x2 = make_pairwise_inputs(w1=6, w2=5, h=h, seed=29)
    compiled = compile_pairwise_distance(x1, x2)
    assert (
        compiled.hc_ir_text is not None
    ), "compile produced no hc_ir; pipeline diagnostics:\n  " + "\n  ".join(
        compiled.pipeline_diagnostics
    )
    assert compiled.hc_ir_text


# Real-hardware gate for pairwise. Default-skipped.
#
# Shape coverage:
#   * Sub-tile (6, 5, h): single `group_shape=(8, 8)` workgroup with a
#     partial chunk — exercises the boundary-mask path.
#   * Tile-aligned (multiples of (8, 8)): full workgroups + cross-wave
#     LDS pre-fill against collective reductions.
#   * (16, 12, 8): partial-last-tile across multiple workgroups,
#     clipped in W2. Pins dst-bounds mask on store — without it stride
#     aliasing turns OOB column writes into next-row hits via
#     `hc.ptr_store_pred`.
@_RUN_HIP_INVOKE_TESTS
@pytest.mark.parametrize(
    ("w1", "w2", "h"),
    [
        (6, 5, 4),
        (8, 8, 4),
        (8, 8, 8),
        (16, 8, 4),
        (8, 16, 4),
        (16, 12, 8),
        (16, 16, 8),
        (32, 24, 16),
    ],
)
def test_pairwise_distance_invokes_on_real_hardware(w1: int, w2: int, h: int) -> None:
    x1, x2 = make_pairwise_inputs(w1=w1, w2=w2, h=h, seed=29)
    out = run_pairwise_on_hardware(x1, x2)
    ref = reference_pairwise_distance(x1, x2)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_gfx11_wmma_example_writes_dump_intermediates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`HC_DUMP_DIR` makes hc-lower-gpu-to-binary emit per-stage artifacts.

    Four files per `gpu.module`: pre-opt LLVM IR, post-opt LLVM IR,
    ISA, HSACO blob. Runs at compile time — no GPU needed.
    """

    monkeypatch.setenv("HC_DUMP_DIR", str(tmp_path))

    handle = hc.compile(tiled_gfx11_wmma_matmul, target="amdgpu-gfx11")
    assert handle.hc_ir_text is not None, handle.pipeline_diagnostics

    name = "tiled_gfx11_wmma_matmul_kernel"
    pre = tmp_path / f"{name}.0-pre-opt.ll"
    post = tmp_path / f"{name}.1-post-opt.ll"
    isa = tmp_path / f"{name}.2-isa.s"
    hsaco = tmp_path / f"{name}.3-binary.hsaco"
    for p in (pre, post, isa, hsaco):
        assert p.is_file(), f"missing {p.name}"
        assert p.stat().st_size > 0, f"empty {p.name}"

    # Shape check on LLVM module headers; IR body drifts with every
    # llvm bump.
    pre_text = pre.read_text(encoding="utf-8")
    post_text = post.read_text(encoding="utf-8")
    assert pre_text.startswith("; ModuleID")
    assert post_text.startswith("; ModuleID")
    assert 'target triple = "amdgcn-amd-amdhsa"' in pre_text
    assert 'target triple = "amdgcn-amd-amdhsa"' in post_text

    # ISA must name the kernel so we're not picking up stale state.
    isa_text = isa.read_text(encoding="utf-8")
    assert name in isa_text

    # ELF magic at offset 0.
    assert hsaco.read_bytes()[:4] == b"\x7fELF"


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_gfx11_wmma_example_dumps_current_pipeline_ir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dump_hc_ir()

    captured = capsys.readouterr()
    # Default schedule lowers to self-contained LLVM:
    # `hc-lower-gpu-to-binary` → HSACO `gpu.binary`,
    # `hc-lower-launch-func-to-runtime` embeds it as LLVM global +
    # rewrites `gpu.launch_func` → `hc_rt_load_kernel` +
    # `hc_rt_launch_kernel`. Host wrapper takes one `!llvm.ptr` per
    # kernel arg, calls `_mlir_ciface_hc_get_*` (via
    # `llvm.emit_c_interface` mangling onto `libhc_rt_helpers.so`)
    # before dispatch.
    assert "module attributes {gpu.container_module}" in captured.out
    assert (
        "llvm.func @tiled_gfx11_wmma_matmul(%arg0: !llvm.ptr, "
        "%arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr)" in captured.out
    )
    assert "@hc_get_ptr" in captured.out
    assert "@_mlir_ciface_hc_get_ptr" in captured.out
    assert "@hc_get_dim" in captured.out
    assert "@_mlir_ciface_hc_get_dim" in captured.out
    assert "@hc_get_stride" in captured.out
    assert "@_mlir_ciface_hc_get_stride" in captured.out
    assert "@hc_rt_load_kernel" in captured.out
    assert "@hc_rt_launch_kernel" in captured.out
    assert "@tiled_gfx11_wmma_matmul_kernel_data" in captured.out
    assert "@tiled_gfx11_wmma_matmul_kernel_handle" in captured.out
    assert "gpu.launch_func" not in captured.out
    assert "gpu.binary" not in captured.out
    assert "hc.kernel" not in captured.out
    assert "hc_front." not in captured.out
