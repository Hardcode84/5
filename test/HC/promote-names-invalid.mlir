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

// `hc.workitem_region` doesn't yet carry results; any `hc.assign` /
// `hc.name_load` inside it survives region promotion, and the flat
// sweep that follows must refuse to leave a half-lowered module.
hc.kernel @workitem_region_unsupported(%a: !hc.undef) {
  // expected-error @+1 {{region-carrying op still contains `hc.assign` / `hc.name_load`}}
  hc.workitem_region {
    hc.assign "x", %a : !hc.undef
  }
  hc.return
}

// -----

// Same defensive diagnostic for `hc.subgroup_region`.
hc.kernel @subgroup_region_unsupported(%a: !hc.undef) {
  // expected-error @+1 {{region-carrying op still contains `hc.assign` / `hc.name_load`}}
  hc.subgroup_region {
    hc.assign "x", %a : !hc.undef
  }
  hc.return
}

// -----

// A body-read of a name that has no reaching outer `hc.assign` must
// fail with the same reaching-def diagnostic. The error is reported on
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
