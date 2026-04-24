// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pipeline-handoff coverage. `-convert-hc-front-to-hc` emits
// `hc.assign` / `hc.name_load` placeholders unconditionally;
// unbound-name diagnostics now fire in `-hc-promote-names`. This file
// pins the *two-pass* path: the same driver bug that used to surface
// as "hc_front.name … not bound" must still reach the user via the
// composite pipeline, just with the new wording. That locks the
// handoff against a regression where conversion drops a parameter
// assign, mis-orders block args, or promote-names' diagnostic text
// drifts without updating the LIT.
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names --split-input-file -verify-diagnostics %s

// A `param`-classified name with no matching entry in `parameters`
// lowers to `hc.name_load "x"` (no diagnostic from the converter) and
// then fails the promotion pass's reaching-def check. The error points
// at the loaded-from-nowhere name_load, which inherits its source
// location from the original `hc_front.name`.
module {
  hc_front.kernel "unbound_param_via_pipeline" attributes {parameters = []} {
    // expected-error@+1 {{read of name 'x' that has no reaching `hc.assign`}}
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return
  }
}
