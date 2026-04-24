// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative-path coverage for `-hc-front-inline`: every module below is
// expected to surface a specific diagnostic so the pass's guardrails
// stay wired. `--split-input-file` keeps each module independent;
// `-verify-diagnostics` matches the `expected-error` annotations
// against the actual diagnostics.
// RUN: hc-opt --hc-front-inline --split-input-file -verify-diagnostics %s

// Self-recursive inline helper. The inliner inlines each marker func's
// own body first (so nested inline calls expand inside the clone
// before it lands); a direct self-call trips the `chain` cycle guard
// at the call site inside the helper body.
module {
  hc_front.func "self_rec" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg.self_rec"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %rec = hc_front.name "self_rec" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.self_rec"}}
    // expected-error@+1 {{recursive inline through `self_rec'}}
    %v = hc_front.call %rec(%x)
    hc_front.return %v
  }
}

// -----

// Duplicate inline func names. The Python driver is expected to emit
// a unique top-level name per inline helper; a collision at setup
// time means the resolver has two paths claiming the same symbol.
module {
  hc_front.func "dup" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg1.dup"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %x
  }
  // expected-error@+1 {{duplicate inlinable func name `dup'; names must be unique within a module}}
  hc_front.func "dup" attributes {
    parameters = [{name = "y"}],
    ref = {kind = "inline", qualified_name = "pkg2.dup"}
  } {
    %y = hc_front.name "y" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %y
  }
}

// -----

// Inline helper with no `hc_front.return` at the top of its entry
// block. The pass requires exactly one top-level return so the
// region-result arity + mapped return values have a single source of
// truth. The diagnostic is stamped on the helper `hc_front.func`.
module {
  // expected-error@+1 {{inlinable func missing top-level `hc_front.return`}}
  hc_front.func "no_ret" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg.no_ret"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
  }
  hc_front.kernel "caller" attributes {parameters = [{name = "g"}]} {
    %one = hc_front.constant <1 : i64>
    %n = hc_front.name "no_ret" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.no_ret"}}
    %v = hc_front.call %n(%one)
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
  }
}

// -----

// Inline helper with two top-level returns: the `findSingleReturn`
// invariant forbids this — the arity story collapses if the pass has
// to pick one.
module {
  // expected-error@+1 {{inlinable func must have exactly one top-level `hc_front.return`}}
  hc_front.func "two_ret" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg.two_ret"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %x
    hc_front.return %x
  }
  hc_front.kernel "caller" attributes {parameters = [{name = "g"}]} {
    %one = hc_front.constant <1 : i64>
    %n = hc_front.name "two_ret" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.two_ret"}}
    %v = hc_front.call %n(%one)
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
  }
}

// -----

// Call-site arity that doesn't match the callee's `parameters`.
module {
  hc_front.func "takes_one" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg.takes_one"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %x
  }
  hc_front.kernel "caller" attributes {parameters = [{name = "g"}]} {
    %one = hc_front.constant <1 : i64>
    %two = hc_front.constant <2 : i64>
    %n = hc_front.name "takes_one" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.takes_one"}}
    // expected-error@+1 {{inline call arity 2 does not match callee `takes_one' parameter count 1}}
    %v = hc_front.call %n(%one, %two)
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
  }
}

// -----

// `ref.kind = "inline"` call with no matching `hc_front.func`
// definition in the module. The Python driver is expected to emit the
// marker func alongside every call it stamps; a missing callee is a
// driver bug, so the pass surfaces it at the call site. `_other`
// exists so the pass doesn't early-return on an empty marker-func
// set, which is how we reach the call-site check.
module {
  hc_front.func "_other" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg._other"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %x
  }
  hc_front.kernel "caller" attributes {parameters = [{name = "g"}]} {
    %one = hc_front.constant <1 : i64>
    %n = hc_front.name "missing" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.missing"}}
    // expected-error@+1 {{no inlinable `hc_front.func` named `missing'}}
    %v = hc_front.call %n(%one)
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
  }
}
