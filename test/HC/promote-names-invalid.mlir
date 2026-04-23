// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative-path coverage for `-hc-promote-names`: every `hc.name_load`
// must have a reaching `hc.assign` in the same function body, or the
// pass fails with a diagnostic. The frontend is expected to emit a
// seed-assign for every bound name; an unresolved read here is always a
// bug upstream.
//
// RUN: hc-opt -hc-promote-names -split-input-file -verify-diagnostics %s

// A `hc.name_load` with no prior `hc.assign` is a "read before write":
// the frontend must have emitted a seed-assign (or the IR is malformed).
hc.func @read_before_write() -> !hc.undef {
  // expected-error @+1 {{read of name 'x' that has no reaching `hc.assign`}}
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Distinct names are independent: a write to "y" does not satisfy a
// read of "x".
hc.func @different_name(%a: !hc.undef) -> !hc.undef {
  hc.assign "y", %a : !hc.undef
  // expected-error @+1 {{read of name 'x' that has no reaching `hc.assign`}}
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// A body-read of a name that has no reaching outer `hc.assign` must
// fail with the reaching-def diagnostic. The error is reported on
// the transient outer-scope snap load the pass emits at the
// `hc.for_range`'s location — that load is what fails to resolve.
hc.func @for_body_read_before_write(%lo: !hc.undef, %hi: !hc.undef,
                                    %step: !hc.undef) -> !hc.undef {
  // expected-error @+1 {{read of name 'acc' that has no reaching `hc.assign`}}
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %cur = hc.name_load "acc" : !hc.undef
    hc.yield
  }
  %v = hc.name_load "acc" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Writes inside `hc.workitem_region` / `hc.subgroup_region` shadow and
// don't leak out: the enclosing scope's read can't see them. Reads
// capture, writes don't — so an outer `hc.name_load` without any
// outer-level `hc.assign` still fails, even though the region body
// assigns the same name.
hc.kernel @workitem_assign_does_not_leak(%a: !hc.undef) {
  hc.workitem_region {
    hc.assign "x", %a : !hc.undef
  }
  // expected-error @+1 {{read of name 'x' that has no reaching `hc.assign`}}
  %v = hc.name_load "x" : !hc.undef
  hc.return
}

// -----

// `hc.subgroup_region` mirrors `hc.workitem_region`: writes inside
// shadow and don't leak, so the outer read of `"x"` falls through to
// an unbound name even though the region body assigns it.
hc.kernel @subgroup_assign_does_not_leak(%a: !hc.undef) {
  hc.subgroup_region {
    hc.assign "x", %a : !hc.undef
  }
  // expected-error @+1 {{read of name 'x' that has no reaching `hc.assign`}}
  %v = hc.name_load "x" : !hc.undef
  hc.return
}

// -----

// Reads inside a nested scope capture via a lazy outer snapshot —
// but the snapshot itself still needs a reaching outer assign. An
// inner read of an entirely unbound name therefore still errors,
// reported on the pass-emitted outer snap load (which inherits the
// region op's source location).
hc.kernel @workitem_read_unbound_outer() {
  // expected-error @+1 {{read of name 'x' that has no reaching `hc.assign`}}
  hc.workitem_region {
    %v = hc.name_load "x" : !hc.undef
  }
  hc.return
}

// -----

// Atomic-scan regression coverage. The earlier `hc.assign "x"` +
// `hc.name_load "x"` pair would promote cleanly on its own; the
// later `hc.name_load "y"` has no reaching assign. Under a single-
// phase scan, the first pair would already be rewritten (RAUW'd
// and erased) by the time the second load returns `failure()`,
// leaving a torn body behind. The two-phase scan walks the block
// without mutation first, so the diagnostic fires on 'y' before
// anything is rewritten.
hc.func @fail_after_valid_prefix(%a: !hc.undef) -> !hc.undef {
  hc.assign "x", %a : !hc.undef
  %x = hc.name_load "x" : !hc.undef
  // expected-error @+1 {{read of name 'y' that has no reaching `hc.assign`}}
  %y = hc.name_load "y" : !hc.undef
  hc.return %y : !hc.undef
}
