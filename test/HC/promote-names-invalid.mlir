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
