# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for `bench=` placeholder plumbing in `hc._pipeline`.

Pins Python-side substitution: `bench=False` → byte-identical to
pre-bench shape (modulo the empty placeholder); `bench=True` splices
the pass into the right slot; placeholder appears exactly once in
`_GPU_LOWERING_PIPELINE`. End-to-end `-hc-emit-bench-wrapper` lives in
`test/HC/emit-bench-wrapper.mlir`.
"""

from __future__ import annotations

from hc._pipeline import (
    _BENCH_PASS_FRAGMENT,
    _BENCH_PLACEHOLDER,
    _GPU_LOWERING_PIPELINE,
    _substitute_bench,
)


def test_bench_placeholder_appears_once_in_gpu_lowering_pipeline() -> None:
    # String-replace substitution: a second placeholder would silently
    # double-splice the pass.
    assert _GPU_LOWERING_PIPELINE.count(_BENCH_PLACEHOLDER) == 1


def test_bench_placeholder_sits_between_runtime_lowering_and_symbol_dce() -> None:
    # Pass needs wrappers in post-runtime-lowering shape
    # (`hc_rt_launch_kernel` present) and the bench symbol survives
    # `symbol-dce`.
    runtime_idx = _GPU_LOWERING_PIPELINE.index("hc-lower-launch-func-to-runtime")
    placeholder_idx = _GPU_LOWERING_PIPELINE.index(_BENCH_PLACEHOLDER)
    dce_idx = _GPU_LOWERING_PIPELINE.index("symbol-dce")
    assert runtime_idx < placeholder_idx < dce_idx


def test_substitute_bench_false_drops_placeholder() -> None:
    # Default: byte-identical to pre-bench shape — no stray comma,
    # no empty pass slot, no leftover placeholder.
    out = _substitute_bench(_GPU_LOWERING_PIPELINE, bench=False)
    assert _BENCH_PLACEHOLDER not in out
    assert "hc-emit-bench-wrapper" not in out
    # Surrounding passes abut through a single comma.
    assert "hc-lower-launch-func-to-runtime,symbol-dce" in out


def test_substitute_bench_true_splices_pass_with_trailing_comma() -> None:
    out = _substitute_bench(_GPU_LOWERING_PIPELINE, bench=True)
    assert _BENCH_PLACEHOLDER not in out
    assert "hc-lower-launch-func-to-runtime,hc-emit-bench-wrapper,symbol-dce" in out


def test_substitute_bench_is_idempotent_on_text_without_placeholder() -> None:
    # User-supplied schedules without the placeholder round-trip
    # unchanged regardless of the bench flag.
    text = "canonicalize,cse"
    assert _substitute_bench(text, bench=False) == text
    assert _substitute_bench(text, bench=True) == text


def test_bench_pass_fragment_ends_with_comma() -> None:
    # Fragment carries its own trailing comma → placeholder lands in
    # `prev_pass,__HC_BENCH__next_pass` without a surrounding-comma
    # rewrite.
    assert _BENCH_PASS_FRAGMENT.endswith(",")
    assert _BENCH_PASS_FRAGMENT.strip(",") == "hc-emit-bench-wrapper"
