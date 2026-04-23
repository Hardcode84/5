// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative-path coverage: every top-level module below expects a
// specific diagnostic so the pass's guardrails stay wired. `--split-input-file`
// keeps each module independent; `-verify-diagnostics` matches the
// `expected-error` annotations against the actual diagnostics.
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file -verify-diagnostics %s

// Missing `parameters` attribute: the driver is required to stamp one,
// so a bare kernel op without it surfaces a hard error.
module {
  // expected-error@+1 {{missing `parameters` attribute}}
  hc_front.kernel "no_params" {
    hc_front.return
  }
}

// -----

// `param` name that never got bound — caller built a bad ref (or forgot
// to populate parameters). The diagnostic names the identifier so the
// classification can be debugged.
module {
  hc_front.kernel "unbound_param" attributes {parameters = []} {
    // expected-error@+1 {{hc_front.name 'x' (kind=param) not bound}}
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return
  }
}

// -----

// Unknown binop kind — `Pow` and friends are unsupported today.
module {
  hc_front.kernel "bad_binop" attributes {
    parameters = [{name = "a"}, {name = "b"}]
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    // expected-error@+1 {{unsupported hc_front.binop kind 'Pow'}}
    %c = hc_front.binop "Pow"(%a, %b)
    hc_front.return
  }
}

// -----

// Out-of-range launch-geometry axis.
module {
  hc_front.kernel "bad_axis" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %gid = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %ax = hc_front.constant<99 : i64>
    // expected-error@+1 {{axis index 99 out of range}}
    %out = hc_front.subscript %gid[%ax]
    %t = hc_front.target_name "t"
    hc_front.assign %t = %out
    hc_front.return
  }
}

// -----

// Unsupported DSL method — spelled correctly enough to reach dsl dispatch
// but not one we implement.
module {
  hc_front.kernel "bad_dsl" attributes {parameters = [{name = "x"}]} {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %m = hc_front.attr %x, "no_such_method" {ref = {kind = "dsl_method", method = "no_such_method"}}
    // expected-error@+1 {{unsupported dsl_method 'no_such_method'}}
    %v = hc_front.call %m()
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
    hc_front.return
  }
}
