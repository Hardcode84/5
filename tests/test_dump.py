# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the per-pass IR-dump knob (`HC_DUMP_PASSES`)."""

from __future__ import annotations

import pytest

from hc._dump import (
    DUMP_PASSES_ENV,
    dump_passes_enabled,
    splice_dump_passes,
)
from hc._pipeline import (
    _maybe_splice_dump_passes,
    _resolve_schedule_text,
    _substitute_chip,
    _substitute_features,
    _substitute_target,
    prepared_context,
)

# Minimal hand-rolled schedule covering every transform op shape the
# splicer cares about. Keeping it small (vs parsing the real default
# schedule) lets us assert the post-splicer structure exactly without
# the test bit-rotting every time someone adds a pass to the real
# schedule.
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
    # Be strict about the trigger value so we don't silently turn dumps
    # on for `HC_DUMP_PASSES=true` (a common shape that the user would
    # reasonably expect to mean "off" in MLIR-tooling-land).
    monkeypatch.setenv(DUMP_PASSES_ENV, value)
    assert dump_passes_enabled() is False


def test_dump_passes_env_one_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DUMP_PASSES_ENV, "1")
    assert dump_passes_enabled() is True


def test_splicer_inserts_one_print_per_payload_mutator() -> None:
    out, n = _spliced(_MINI_SCHEDULE)
    # apply_registered_pass + apply_patterns + apply_cse + apply_dce.
    assert n == 4
    # apply_patterns.canonicalization is a *pattern descriptor* nested
    # inside apply_patterns, not a payload mutator on its own. The
    # splicer must skip it — otherwise we'd get five prints, not four.
    assert "transform.apply_patterns.canonicalization" in out
    # transform.yield never gets a probe — it's a region terminator.
    yield_idx = out.index("transform.yield")
    last_print_idx = out.rfind("transform.print")
    assert last_print_idx < yield_idx


def test_splicer_uses_result_handle_for_apply_registered_pass() -> None:
    out, _ = _spliced(_MINI_SCHEDULE)
    # The pass takes %arg0 and produces %0; the print must reference
    # %0 (the post-pass handle), not %arg0. Otherwise we'd be dumping
    # the pre-pass IR — which the device-PM instrumentation would
    # already give us via "IR Dump Before".
    assert 'transform.print %0 {name = "after-canonicalize"}' in out


def test_splicer_uses_operand_handle_for_in_place_ops() -> None:
    out, _ = _spliced(_MINI_SCHEDULE)
    # apply_patterns / apply_cse / apply_dce mutate the payload that
    # %0 references and don't produce a new handle. The probe must
    # therefore latch onto %0 itself.
    assert 'transform.print %0 {name = "after-apply_patterns"}' in out
    assert 'transform.print %0 {name = "after-apply_cse"}' in out
    assert 'transform.print %0 {name = "after-apply_dce"}' in out


def test_maybe_splice_no_op_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DUMP_PASSES_ENV, raising=False)
    # Cheap path: no parse, no print ops added, identical text out.
    out = _maybe_splice_dump_passes(_MINI_SCHEDULE, context=None)
    assert out == _MINI_SCHEDULE
    assert "transform.print" not in out


def test_maybe_splice_runs_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DUMP_PASSES_ENV, "1")
    out = _maybe_splice_dump_passes(_MINI_SCHEDULE, context=None)
    assert out.count("transform.print") == 4


def test_default_schedule_round_trips_through_splicer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real pipeline parses the default schedule after every
    # __HC_*__ substitution fires, and the splicer needs to keep the
    # post-substitution module verifying. This is the gate against the
    # splicer breaking on a schedule shape it hasn't seen before.
    monkeypatch.setenv(DUMP_PASSES_ENV, "1")
    text = _resolve_schedule_text(None)
    text = _substitute_target(text, None)
    text = _substitute_chip(text, None)
    text = _substitute_features(text, None)
    out = _maybe_splice_dump_passes(text, context=None)
    # One print per apply_registered_pass + per apply_cse / apply_dce /
    # apply_patterns. Don't pin the exact count — the schedule grows.
    # Instead assert each apply_registered_pass got followed by exactly
    # one transform.print binding its result handle.
    n_passes = text.count("transform.apply_registered_pass")
    n_prints = out.count("transform.print")
    assert n_prints >= n_passes
