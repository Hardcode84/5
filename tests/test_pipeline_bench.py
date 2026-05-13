# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for the `bench=` placeholder plumbing in `hc._pipeline`.

The actual `-hc-emit-bench-wrapper` lowering is exercised end-to-end by
`test/HC/emit-bench-wrapper.mlir` against the C++ pass. These tests
pin the Python-side substitution: bench=False produces a pipeline
string byte-identical to the pre-bench shape (modulo the now-empty
placeholder), bench=True splices the pass into the right slot, and
the placeholder appears exactly once in `_GPU_LOWERING_PIPELINE` so a
future stray substitution can't silently double-fire.
"""

from __future__ import annotations

from hc._pipeline import (
    _BENCH_PASS_FRAGMENT,
    _BENCH_PLACEHOLDER,
    _GPU_LOWERING_PIPELINE,
    _substitute_bench,
)


def test_bench_placeholder_appears_once_in_gpu_lowering_pipeline() -> None:
    # The substitution is a string-replace; a second placeholder would
    # silently double-splice the pass. Pinning the count here keeps a
    # future drive-by edit from breaking the contract.
    assert _GPU_LOWERING_PIPELINE.count(_BENCH_PLACEHOLDER) == 1


def test_bench_placeholder_sits_between_runtime_lowering_and_symbol_dce() -> None:
    # The pass needs the wrappers to be in their post-runtime-lowering
    # shape (hc_rt_launch_kernel call present) and the new bench symbol
    # has to survive symbol-dce.
    runtime_idx = _GPU_LOWERING_PIPELINE.index("hc-lower-launch-func-to-runtime")
    placeholder_idx = _GPU_LOWERING_PIPELINE.index(_BENCH_PLACEHOLDER)
    dce_idx = _GPU_LOWERING_PIPELINE.index("symbol-dce")
    assert runtime_idx < placeholder_idx < dce_idx


def test_substitute_bench_false_drops_placeholder() -> None:
    # bench=False is the default; the post-substitution string must be
    # byte-identical to the pre-bench shape, i.e. no stray comma, no
    # empty pass slot, no leftover placeholder.
    out = _substitute_bench(_GPU_LOWERING_PIPELINE, bench=False)
    assert _BENCH_PLACEHOLDER not in out
    assert "hc-emit-bench-wrapper" not in out
    # Sanity: the surrounding passes still abut directly through a
    # single comma.
    assert "hc-lower-launch-func-to-runtime,symbol-dce" in out


def test_substitute_bench_true_splices_pass_with_trailing_comma() -> None:
    out = _substitute_bench(_GPU_LOWERING_PIPELINE, bench=True)
    assert _BENCH_PLACEHOLDER not in out
    assert "hc-lower-launch-func-to-runtime,hc-emit-bench-wrapper,symbol-dce" in out


def test_substitute_bench_is_idempotent_on_text_without_placeholder() -> None:
    # User-supplied schedules / pipelines that don't carry the
    # placeholder must round-trip unchanged regardless of the bench
    # flag; the substitution is a pure string-replace.
    text = "canonicalize,cse"
    assert _substitute_bench(text, bench=False) == text
    assert _substitute_bench(text, bench=True) == text


def test_bench_pass_fragment_ends_with_comma() -> None:
    # The fragment carries its own trailing comma so the placeholder
    # can land in `prev_pass,__HC_BENCH__next_pass` without a separate
    # surrounding-comma rewrite. Locking it here keeps the call site
    # convention explicit even though the rest of the test suite would
    # catch a regression structurally.
    assert _BENCH_PASS_FRAGMENT.endswith(",")
    assert _BENCH_PASS_FRAGMENT.strip(",") == "hc-emit-bench-wrapper"
