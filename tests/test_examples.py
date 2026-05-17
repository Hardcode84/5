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

# Real-hardware end-to-end gate. Default-skipped — running it dlopens
# `libamdhip64.so` (via `hc_rt_init`) and dispatches a kernel onto the
# GPU bound by `HIP_VISIBLE_DEVICES`, so it only makes sense on a host
# with a working ROCm runtime and a gfx11-class device. Mirror the opt-in
# convention `tests/test_hip_runtime.py` already established for the
# init-path test.
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


@_RUN_HIP_INVOKE_TESTS
def test_gfx11_wmma_example_benches_on_real_hardware() -> None:
    """End-to-end acceptance for `CompiledKernel.bench(...)`.

    Goes through the same JIT + HIP path as the invoke test but via
    the bench wrapper: `hc_rt_launch_kernel_repeat` drives an inner
    loop of `n_inner` launches, `hipStreamSynchronize` closes the
    window, the runtime hands back monotonic ns. Verifies the shape
    of the returned `BenchResult` and that the headline numbers
    relate consistently — anything tighter (e.g. an absolute upper
    bound on per-launch latency) would be flaky against driver /
    queue jitter.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda unavailable")

    a, b = make_demo_inputs(m=32, n=32, k=32, seed=17)
    # Modest m_outer/n_inner so the test runs in a few hundred ms even
    # on a slow gfx11 host. `bench_on_hardware` asserts the kernel
    # output matches the numpy reference before timing, so reaching the
    # `BenchResult` assertions here is also implicit numerics coverage.
    result, _ = bench_on_hardware(a, b, n_inner=4, m_outer=5, warmup=1)
    assert result.kernel_name == "tiled_gfx11_wmma_matmul"
    assert result.m_outer == 5
    assert result.n_inner == 4
    assert result.samples_ns.shape == (5,)
    assert result.samples_ns.dtype == np.int64
    # Every sample must be positive — the C-side timer brackets at
    # least one host->device dispatch + one stream sync; a zero or
    # negative ns reading would mean the monotonic clock ran
    # backwards, which would be a bug worth catching here.
    assert int(result.samples_ns.min()) > 0
    # Per-launch median is the outer-sample median divided by n_inner;
    # mirror that contract from the C-side timing window down to the
    # Python aggregation.
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
    """End-to-end acceptance for the gfx11 WMMA execution epic.

    Compiles `tiled_gfx11_wmma_matmul` for `amdgpu-gfx11`, drives it
    through the full ORC LLJIT + HIP shim stack (`hc.compile().invoke()`
    -> JIT'd host wrapper -> `hc_rt_load_kernel` / `hc_rt_launch_kernel`
    -> `libamdhip64`), and checks the result matches the numpy reference
    to FP32 round-off. Requires GPU memory for the inputs/outputs;
    `torch.cuda` is the path of least resistance because its
    `Tensor.data_ptr()` returns a HIP-allocated device pointer that
    `_mlir_ciface_hc_get_ptr` hands straight to `gpu.launch_func` —
    the runtime helpers don't allocate or copy on their own. Skip
    cleanly if torch isn't installed so the gate stays usable
    on minimal Python envs.
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
    """End-to-end simulator pass for the langref WG-level pairwise distance.

    Pins the workgroup-level pairwise Euclidean distance kernel from
    `doc/langref.md` against a plain NumPy reference. Broadcasting axes
    are corrected from the langref text (which writes the result
    transposed); the per-element contract matches the workitem-level
    form in the same doc section.
    """

    x1, x2 = make_pairwise_inputs(w1=w1, w2=w2, h=h, seed=29)
    out = simulate_pairwise_distance(x1, x2)
    ref = reference_pairwise_distance(x1, x2)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("h", [3, 4, 8])
def test_pairwise_distance_native_compile(h: int) -> None:
    """Native compile pass for the langref WG-level pairwise distance.

    Drives `hc.compile` through the full hc pipeline with `H` bound via
    `symbols={H: h}`; asserts the handle exposes a non-empty hc_ir
    artifact (= every pass succeeded) and surfaces any pipeline
    diagnostics if it doesn't. This is the compile-only gate — the
    `invoke` path needs a HIP-visible build and is covered by the
    matmul example's `_RUN_HIP_INVOKE_TESTS` surface.
    """

    x1, x2 = make_pairwise_inputs(w1=6, w2=5, h=h, seed=29)
    compiled = compile_pairwise_distance(x1, x2)
    assert (
        compiled.hc_ir_text is not None
    ), "compile produced no hc_ir; pipeline diagnostics:\n  " + "\n  ".join(
        compiled.pipeline_diagnostics
    )
    assert compiled.hc_ir_text


# Real-hardware end-to-end gate for pairwise. Default-skipped — see the
# `_RUN_HIP_INVOKE_TESTS` comment block above. The matching shape is
# the smallest one whose work_shape stays strictly inside one
# `group_shape=(8, 8)` workgroup; larger shapes (those that hit the
# tile boundary at `W1==8` or `W2==8`) currently miscompare against
# the numpy reference — see the open bead on the workgroup-collective
# boundary race so the test surface tracks where the GPU output is
# actually trusted.
@_RUN_HIP_INVOKE_TESTS
def test_pairwise_distance_invokes_on_real_hardware() -> None:
    x1, x2 = make_pairwise_inputs(w1=6, w2=5, h=4, seed=29)
    out = run_pairwise_on_hardware(x1, x2)
    ref = reference_pairwise_distance(x1, x2)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_gfx11_wmma_example_writes_dump_intermediates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`HC_DUMP_DIR` makes hc-lower-gpu-to-binary emit per-stage artifacts.

    Compiling for `amdgpu-gfx11` runs the WMMA kernel through the full
    device chain, which includes `hc-lower-gpu-to-binary`. With
    `HC_DUMP_DIR` set Python-side, the pass should drop four files per
    `gpu.module`: pre-opt LLVM IR, post-opt LLVM IR, ISA assembly, and
    the linked HSACO blob. Doesn't need a real GPU — the LLD linker
    runs at compile time, not on a HIP device. Pin both that the
    files exist and that they look plausible (LLVM module headers,
    nonzero ELF blob) so a regression in the placeholder substitution
    or per-stage dump call site fails this test, not the harder-to-
    diagnose downstream "where did my disassembly go" stage.
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

    # LLVM modules carry a `; ModuleID = '...'` header on line 1 and a
    # `target triple = "amdgcn-amd-amdhsa"` line. Cheap shape check —
    # we don't pin the IR contents (those drift with every llvm bump),
    # just that the file is what we claimed it is.
    pre_text = pre.read_text(encoding="utf-8")
    post_text = post.read_text(encoding="utf-8")
    assert pre_text.startswith("; ModuleID")
    assert post_text.startswith("; ModuleID")
    assert 'target triple = "amdgcn-amd-amdhsa"' in pre_text
    assert 'target triple = "amdgcn-amd-amdhsa"' in post_text

    # ISA is plain text from the AMDGPU MC stack; it must reference
    # the kernel symbol so we know it's *this* kernel's assembly and
    # not stale state from a previous run that landed in the dir.
    isa_text = isa.read_text(encoding="utf-8")
    assert name in isa_text

    # HSACO is an ELF; magic bytes at offset 0 are 0x7f 'E' 'L' 'F'.
    # File-typing the blob in full is overkill here.
    assert hsaco.read_bytes()[:4] == b"\x7fELF"


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
    # to materialize each tensor's data pointer, shape dims, and
    # strides before dispatching. The pointer arrives via
    # `hc_get_ptr` (raw `!llvm.ptr`, addrspace-cast on the way to the
    # kernel). Pin each load-bearing milestone so the dump test fails
    # loudly if anything regresses.
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
