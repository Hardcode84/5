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
  hc_front.kernel "bad_no_params" {
    hc_front.return
  }
}

// -----

// `param` name that never got bound — caller built a bad ref (or forgot
// to populate parameters). The diagnostic names the identifier so the
// classification can be debugged.
module {
  hc_front.kernel "bad_unbound_param" attributes {parameters = []} {
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

// Out-of-range launch-geometry axis. `kMaxLaunchAxis` (32) is the cap;
// anything at or above is rejected before the unsigned cast.
module {
  hc_front.kernel "bad_axis" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %gid = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %ax = hc_front.constant<99 : i64>
    // expected-error@+1 {{launch-geo axis 99 out of range}}
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

// -----

// Tuple-unpack arity mismatch: RHS tuple has 2 elements, target has 3.
module {
  hc_front.kernel "bad_unpack_arity" attributes {
    parameters = [{name = "a"}, {name = "b"}]
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    %rhs = hc_front.tuple (%a, %b)
    %ta = hc_front.target_name "x"
    %tb = hc_front.target_name "y"
    %tc = hc_front.target_name "z"
    %tgt = hc_front.target_tuple (%ta, %tb, %tc)
    // expected-error@+1 {{tuple-unpack arity mismatch: rhs has 2, target has 3}}
    hc_front.assign %tgt = %rhs
    hc_front.return
  }
}

// -----

// Tuple-unpack against a non-tuple, single-result RHS with multi-target:
// exercises the "must be tuple literal, or a call producing one result per
// target" branch (a binop returns a single hc value — not an arity-2
// multi-result op, and not a tuple).
module {
  hc_front.kernel "bad_unpack_scalar_rhs" attributes {
    parameters = [{name = "a"}, {name = "b"}]
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    %scalar = hc_front.binop "Add"(%a, %b)
    %ta = hc_front.target_name "x"
    %tb = hc_front.target_name "y"
    %tgt = hc_front.target_tuple (%ta, %tb)
    // expected-error@+1 {{tuple-unpack rhs must be a tuple literal, or a call producing one result per target}}
    hc_front.assign %tgt = %scalar
    hc_front.return
  }
}

// -----

// Single-target tuple binding: `x = (a,)` with arity > 1 is rejected, not
// silently null-bound. Arity 2+ has no single-target meaning.
module {
  hc_front.kernel "bad_single_target_tuple" attributes {
    parameters = [{name = "a"}, {name = "b"}]
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    %rhs = hc_front.tuple (%a, %b)
    %t = hc_front.target_name "x"
    // expected-error@+1 {{cannot bind 'x' to a tuple rhs of arity 2}}
    hc_front.assign %t = %rhs
    hc_front.return
  }
}

// -----

// Slice operand count disagrees with has_* flags — hand-crafted inconsistency.
module {
  hc_front.kernel "bad_slice_arity" attributes {parameters = [{name = "a"}]} {
    %c0 = hc_front.constant<0 : i64>
    %c1 = hc_front.constant<1 : i64>
    %c2 = hc_front.constant<2 : i64>
    // expected-error@+1 {{slice operand count 3 does not match}}
    %s = hc_front.slice (%c0, %c1, %c2) {has_lower = true, has_upper = true, has_step = false}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.subscript %a[%s]
    %t = hc_front.target_name "t"
    hc_front.assign %t = %out
    hc_front.return
  }
}

// -----

// Unary DSL method with a base that didn't lower (the attr base is a
// callee-classified name, so `lowerValueOperand` returns null). `requireBase`
// must catch this and diagnose rather than build a null-operand hc op.
module {
  hc_front.kernel "bad_null_base_dsl" attributes {parameters = []} {
    // A name with `kind = "builtin"` lowers to null (consumed by parent).
    // Feeding it to `x.vec()` exercises the requireBase guard.
    %n = hc_front.name "some_builtin" {ctx = "load", ref = {kind = "builtin", builtin = "some_builtin"}}
    %m = hc_front.attr %n, "vec" {ref = {kind = "dsl_method", method = "vec"}}
    // expected-error@+1 {{vec: base did not lower}}
    %v = hc_front.call %m()
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
    hc_front.return
  }
}

// -----

// Malformed `ref` on a name: present dict but no (or non-string) `kind`.
// The diagnostic fingers the driver rather than silently returning a
// sentinel null.
module {
  hc_front.kernel "bad_ref_name" attributes {parameters = []} {
    // expected-error@+1 {{`ref` dict with missing or non-string `kind`}}
    %n = hc_front.name "x" {ctx = "load", ref = {notkind = "oops"}}
    hc_front.return
  }
}

// -----

// Same malformed-ref guardrail on an attr op: a dict without a string
// `kind` must blame the attr at source, not the downstream call or
// subscript trying to read `method`.
module {
  hc_front.kernel "bad_ref_attr" attributes {parameters = [{name = "x"}]} {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    // expected-error@+1 {{`ref` dict with missing or non-string `kind`}}
    %m = hc_front.attr %x, "foo" {ref = {method = "foo"}}
    hc_front.return
  }
}
