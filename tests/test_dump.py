# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the IR-dump knobs (`HC_DUMP_PASSES`, `HC_DUMP_DIR`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hc._dump import (
    DUMP_PASSES_ENV,
    dump_passes_enabled,
    splice_dump_passes,
)
from hc._pipeline import (
    _DUMP_DIR_ENV,
    _DUMP_DIR_PLACEHOLDER,
    _GPU_LOWERING_PIPELINE,
    _maybe_splice_dump_passes,
    _resolve_dump_dir,
    _resolve_schedule_text,
    _substitute_chip,
    _substitute_dump_dir,
    _substitute_features,
    _substitute_target,
    prepared_context,
)

# Minimal schedule covering every transform op shape the splicer
# touches. Small enough to assert exact post-splice structure without
# rotting when real schedule grows.
_MINI_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.apply_registered_pass "canonicalize" to %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %0 : !transform.any_op
    transform.apply_dce to %0 : !transform.any_op
    transform.yield
  }
}
"""


def _spliced(text: str) -> tuple[str, int]:
    from hc.mlir import ir

    with prepared_context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        n = splice_dump_passes(module)
        return str(module), n


def test_dump_passes_env_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DUMP_PASSES_ENV, raising=False)
    assert dump_passes_enabled() is False


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "on"])
def test_dump_passes_env_only_one_enables(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # MLIR-tooling convention: only "1" enables.
    monkeypatch.setenv(DUMP_PASSES_ENV, value)
    assert dump_passes_enabled() is False


def test_dump_passes_env_one_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DUMP_PASSES_ENV, "1")
    assert dump_passes_enabled() is True


def test_splicer_inserts_one_print_per_payload_mutator() -> None:
    out, n = _spliced(_MINI_SCHEDULE)
    # 4 payload mutators. Nested pattern descriptors and `transform.yield`
    # don't get probes.
    assert n == 4
    assert "transform.apply_patterns.canonicalization" in out
    yield_idx = out.index("transform.yield")
    last_print_idx = out.rfind("transform.print")
    assert last_print_idx < yield_idx


def test_splicer_uses_result_handle_for_apply_registered_pass() -> None:
    out, _ = _spliced(_MINI_SCHEDULE)
    # Probe latches onto result (%0), not input (%arg0) -- device-PM
    # "IR Dump Before" covers pre-pass.
    assert 'transform.print %0 {name = "after-canonicalize"}' in out


def test_splicer_uses_operand_handle_for_in_place_ops() -> None:
    out, _ = _spliced(_MINI_SCHEDULE)
    # In-place ops produce no new handle -- probe reuses %0.
    assert 'transform.print %0 {name = "after-apply_patterns"}' in out
    assert 'transform.print %0 {name = "after-apply_cse"}' in out
    assert 'transform.print %0 {name = "after-apply_dce"}' in out


def test_maybe_splice_no_op_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DUMP_PASSES_ENV, raising=False)
    # Fast path: no parse, text unchanged.
    out = _maybe_splice_dump_passes(_MINI_SCHEDULE, context=None)
    assert out == _MINI_SCHEDULE
    assert "transform.print" not in out


def test_maybe_splice_runs_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DUMP_PASSES_ENV, "1")
    out = _maybe_splice_dump_passes(_MINI_SCHEDULE, context=None)
    assert out.count("transform.print") == 4


# --- HC_DUMP_DIR (`hc-lower-gpu-to-binary --dump-intermediates=`) ---


def test_resolve_dump_dir_unset_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_DUMP_DIR_ENV, raising=False)
    assert _resolve_dump_dir() == ""


def test_resolve_dump_dir_returns_env_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_DUMP_DIR_ENV, str(tmp_path))
    assert _resolve_dump_dir() == str(tmp_path)


@pytest.mark.parametrize("ch", ['"', "\\", "\n", "\r"])
def test_resolve_dump_dir_rejects_forbidden_chars(
    monkeypatch: pytest.MonkeyPatch, ch: str
) -> None:
    # Reject chars that close the MLIR option string or split on
    # whitespace -- clean ValueError beats a downstream parse failure.
    monkeypatch.setenv(_DUMP_DIR_ENV, f"/tmp/with{ch}bad")
    with pytest.raises(ValueError, match=_DUMP_DIR_ENV):
        _resolve_dump_dir()


def test_substitute_dump_dir_replaces_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_DUMP_DIR_ENV, str(tmp_path))
    text = f"x={_DUMP_DIR_PLACEHOLDER},"
    assert _substitute_dump_dir(text) == f"x={tmp_path},"


def test_substitute_dump_dir_empty_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_DUMP_DIR_ENV, raising=False)
    text = f"x={_DUMP_DIR_PLACEHOLDER},"
    # Pass reads empty as "dump disabled" -- substitute to empty, not raw placeholder.
    assert _substitute_dump_dir(text) == "x=,"


def test_gpu_lowering_pipeline_carries_dump_dir_placeholder() -> None:
    # Catch placeholder loss before env var silently no-ops.
    assert _DUMP_DIR_PLACEHOLDER in _GPU_LOWERING_PIPELINE
    assert "dump-intermediates=" + _DUMP_DIR_PLACEHOLDER in _GPU_LOWERING_PIPELINE


def test_default_schedule_round_trips_through_splicer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Post-substitution module must verify against every shape the real
    # schedule emits.
    monkeypatch.setenv(DUMP_PASSES_ENV, "1")
    text = _resolve_schedule_text(None)
    text = _substitute_target(text, None)
    text = _substitute_chip(text, None)
    text = _substitute_features(text, None)
    out = _maybe_splice_dump_passes(text, context=None)
    # Schedule grows -- assert >=, don't pin exact count.
    n_passes = text.count("transform.apply_registered_pass")
    n_prints = out.count("transform.print")
    assert n_prints >= n_passes
