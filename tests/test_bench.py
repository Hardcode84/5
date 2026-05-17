# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit + end-to-end tests for `hc.compile(bench=True).bench(...)`."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.llvm_toolchain import ensure_llvm_toolchain
from hc import BenchResult, CompiledKernel

REPO_ROOT = Path(__file__).resolve().parents[1]
_COMPILE_SUBPROCESS_TIMEOUT_SECONDS = 60.0

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)


@lru_cache(maxsize=1)
def _native_env() -> dict[str, str]:
    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    return export_hc_native_environment(native_install_root, os.environ.copy())


def _run_bench_smoke(script: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            env=_native_env(),
            capture_output=True,
            text=True,
            timeout=_COMPILE_SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            "hc.compile(bench=True) smoke test failed.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


# --- BenchResult validation -------------------------------------------------


def test_bench_result_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError, match=r"must be an np\.ndarray"):
        BenchResult(samples_ns=[1, 2, 3], n_inner=10, kernel_name="k")  # type: ignore[arg-type]


def test_bench_result_rejects_wrong_dtype() -> None:
    with pytest.raises(TypeError, match="int64"):
        BenchResult(
            samples_ns=np.array([1, 2, 3], dtype=np.float64),
            n_inner=10,
            kernel_name="k",
        )


def test_bench_result_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        BenchResult(
            samples_ns=np.array([[1, 2], [3, 4]], dtype=np.int64),
            n_inner=10,
            kernel_name="k",
        )


def test_bench_result_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        BenchResult(
            samples_ns=np.array([], dtype=np.int64), n_inner=10, kernel_name="k"
        )


def test_bench_result_rejects_non_positive_n_inner() -> None:
    samples = np.array([100, 200, 300], dtype=np.int64)
    with pytest.raises(ValueError, match="n_inner must be positive"):
        BenchResult(samples_ns=samples, n_inner=0, kernel_name="k")
    with pytest.raises(ValueError, match="n_inner must be positive"):
        BenchResult(samples_ns=samples, n_inner=-1, kernel_name="k")


# --- BenchResult stat accessors --------------------------------------------


def _result_with_known_samples() -> BenchResult:
    # Eyeballable numpy stats: median=550, mean=550, min=200, max=900,
    # p25 in [300, 400], p75 in [700, 800].
    samples = np.array([200, 300, 400, 500, 600, 700, 800, 900], dtype=np.int64)
    return BenchResult(samples_ns=samples, n_inner=10, kernel_name="kfn")


def test_bench_result_median_mean_min_max() -> None:
    result = _result_with_known_samples()
    assert result.m_outer == 8
    assert result.n_inner == 10
    assert result.median_ns == pytest.approx(550.0)
    assert result.mean_ns == pytest.approx(550.0)
    assert result.min_ns == 200
    assert result.max_ns == 900


def test_bench_result_std_matches_numpy_population() -> None:
    result = _result_with_known_samples()
    # Contract is population std (ddof=0).
    expected = float(np.std(result.samples_ns))
    assert result.std_ns == pytest.approx(expected)


def test_bench_result_percentiles() -> None:
    result = _result_with_known_samples()
    # Contract is numpy linear-interp default — mirror, don't pin.
    expected_p25 = float(np.percentile(result.samples_ns, 25))
    expected_p75 = float(np.percentile(result.samples_ns, 75))
    assert result.p25_ns == pytest.approx(expected_p25)
    assert result.p75_ns == pytest.approx(expected_p75)


def test_bench_result_per_launch_is_sample_divided_by_n_inner() -> None:
    # per_launch = outer_sample / n_inner. Headline for sub-µs kernels.
    result = _result_with_known_samples()
    assert result.per_launch_median_ns == pytest.approx(
        result.median_ns / result.n_inner
    )
    assert result.per_launch_mean_ns == pytest.approx(result.mean_ns / result.n_inner)


def test_bench_result_single_sample_does_not_crash_std() -> None:
    # m_outer=1: ddof=0 std collapses to 0 — no divide-by-zero.
    result = BenchResult(
        samples_ns=np.array([1234], dtype=np.int64),
        n_inner=5,
        kernel_name="k",
    )
    assert result.std_ns == 0.0
    assert result.mean_ns == 1234.0
    assert result.median_ns == 1234.0


# --- summary() and __repr__ -------------------------------------------------


def test_bench_result_summary_contains_headline_fields() -> None:
    result = _result_with_known_samples()
    text = result.summary()
    # Pin load-bearing pieces only — table layout may evolve.
    assert "kfn" in text
    assert "m_outer=8" in text
    assert "n_inner=10" in text
    assert "per-launch (ns)" in text
    assert "outer-sample (ns)" in text
    for label in ("median", "mean", "std", "min", "max", "p25", "p75"):
        assert label in text, f"summary missing {label!r}: {text}"


def test_bench_result_summary_is_cached() -> None:
    result = _result_with_known_samples()
    first = result.summary()
    second = result.summary()
    assert first is second


def test_bench_result_repr_is_one_line_with_headline() -> None:
    result = _result_with_known_samples()
    text = repr(result)
    assert "\n" not in text
    assert "kernel='kfn'" in text
    assert "n_outer=8" in text
    assert "n_inner=10" in text
    # 550 / 10 = 55ns; allow tiny formatting drift.
    assert re.search(r"per_launch_median=55\.\d+ns", text), text


def test_bench_result_summary_handles_n_inner_one() -> None:
    # n_inner=1: per-launch column equals outer-sample column.
    samples = np.array([10, 20, 30], dtype=np.int64)
    result = BenchResult(samples_ns=samples, n_inner=1, kernel_name="k")
    text = result.summary()
    assert "n_inner=1" in text
    assert "median" in text


# --- CompiledKernel.bench rejection paths ----------------------------------


def _make_handle(*, hc_ir: Any, bench_wrapper_name: str | None) -> CompiledKernel:
    def kfn() -> None:
        return None

    kfn.__name__ = "kfn"
    return CompiledKernel(
        kernel=kfn,
        bindings={},
        front_ir=None,
        front_ir_text="",
        hc_ir=hc_ir,
        hc_ir_text=None,
        bench_wrapper_name=bench_wrapper_name,
    )


def test_compiled_kernel_bench_raises_when_pipeline_did_not_run() -> None:
    handle = _make_handle(hc_ir=None, bench_wrapper_name=None)
    with pytest.raises(RuntimeError, match="pipeline failed"):
        handle.bench((), n_inner=1, m_outer=1)


def test_compiled_kernel_bench_raises_when_compiled_without_bench_flag() -> None:
    # `hc_ir` set + `bench_wrapper_name=None` means no `bench=True` —
    # diagnostic must name it.
    handle = _make_handle(hc_ir=object(), bench_wrapper_name=None)
    with pytest.raises(RuntimeError, match=r"bench=True"):
        handle.bench((), n_inner=1, m_outer=1)


def test_compiled_kernel_bench_validates_loop_counts() -> None:
    handle = _make_handle(hc_ir=object(), bench_wrapper_name="kfn_bench")

    with pytest.raises(ValueError, match="n_inner"):
        handle.bench((), n_inner=0, m_outer=1)
    with pytest.raises(ValueError, match="m_outer"):
        handle.bench((), n_inner=1, m_outer=0)
    with pytest.raises(ValueError, match="warmup"):
        handle.bench((), n_inner=1, m_outer=1, warmup=-1)


# --- compile(bench=True) end-to-end pipeline -------------------------------


# End-to-end needs a kernel whose lowering emits a real
# `hc_rt_launch_kernel` — bench pass only mints a sibling when the
# host wrapper dispatches one. WMMA covers it.
_BENCH_SMOKE_SCRIPT = textwrap.dedent("""
    import hc
    from examples.amdgpu_gfx11_wmma_matmul import tiled_gfx11_wmma_matmul


    def main() -> None:
        handle_bench = hc.compile(tiled_gfx11_wmma_matmul, bench=True)
        text_bench = handle_bench.hc_ir_text
        assert handle_bench.hc_ir is not None, handle_bench.pipeline_diagnostics
        assert handle_bench.bench_wrapper_name == "tiled_gfx11_wmma_matmul_bench"
        assert "@tiled_gfx11_wmma_matmul(" in text_bench, text_bench
        assert "@tiled_gfx11_wmma_matmul_bench(" in text_bench, text_bench
        assert "hc_rt_launch_kernel_repeat" in text_bench, text_bench

        # Negative: no `bench=True` -> no bench sibling, no repeat symbol.
        handle = hc.compile(tiled_gfx11_wmma_matmul)
        text = handle.hc_ir_text
        assert handle.hc_ir is not None, handle.pipeline_diagnostics
        assert handle.bench_wrapper_name is None
        assert "@tiled_gfx11_wmma_matmul(" in text, text
        assert "@tiled_gfx11_wmma_matmul_bench" not in text, text
        assert "hc_rt_launch_kernel_repeat" not in text, text
        print("OK")


    if __name__ == "__main__":
        main()
""")


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_compile_bench_flag_round_trips_through_pipeline(
    tmp_path: Path,
) -> None:
    """`hc.compile(bench=True/False)` lands the right post-pipeline IR.

    `bench=True` -> sibling `<name>_bench`, `hc_rt_launch_kernel_repeat`,
    populated `bench_wrapper_name`. `bench=False` -> none of those.
    """
    script = tmp_path / "compile_bench.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n" + _BENCH_SMOKE_SCRIPT
    )

    result = _run_bench_smoke(script)
    assert result.stdout.strip().endswith("OK"), result.stdout
