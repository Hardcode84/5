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

// Note: a "param" name that never gets bound is no longer a
// conversion-time error. This pass emits `hc.name_load "x"`
// unconditionally; a name with no reaching `hc.assign` surfaces in
// `-hc-promote-names`, which owns the diagnostic path. The exact
// wording is pinned in `test/HC/promote-names-invalid.mlir`
// (promote-names in isolation) and `test/HCFrontToHC/pipeline-invalid.mlir`
// (the end-to-end conversion + promote pipeline, i.e. the supported
// user-facing shape). Keeping the coverage split that way lets each
// pass assert its own contract.

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

// -----

// `ref.kind = "inline"` is the exclusive business of `-hc-front-inline`;
// if one survives to the conversion boundary the pass ordering is wrong
// and we want a located error rather than a silent placeholder.
module {
  hc_front.kernel "inline_not_inlined" attributes {parameters = [{name = "x"}]} {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %inl = hc_front.name "helper" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.helper"}}
    // expected-error@+1 {{`ref.kind = "inline"` call survived to conversion; run `-hc-front-inline` before `-convert-hc-front-to-hc`}}
    %v = hc_front.call %inl(%x)
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
    hc_front.return
  }
}

// -----

// A call whose callee is a `ref.kind = "local"` name is the ghost trail
// Python emits for `@group.workitems def inner(...); inner()`. The
// `-hc-front-fold-region-defs` pass erases it upstream; if one
// survives to the converter the pipeline ordering is wrong — the
// diagnostic points directly at the missing pass, parallel to the
// inline case above.
module {
  hc_front.kernel "local_callee_survives" attributes {parameters = []} {
    hc_front.workitem_region attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
    }
    %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
    // expected-error@+1 {{`ref.kind = "local"` call survived to conversion; run `-hc-front-fold-region-defs` before `-convert-hc-front-to-hc`}}
    %1 = hc_front.call %0()
    hc_front.return
  }
}

// -----

// `numpy_dtype_type` with a dtype name the converter doesn't know about
// (`float128` isn't in `resolveNumpyDtypeType` because it's platform-
// dependent and not part of HC's supported scalar set). The attr lowering
// rejects it here rather than later at the call site: the diagnostic
// should finger the attr op, which is where the offending string lives.
module {
  hc_front.kernel "bad_numpy_dtype" attributes {parameters = []} {
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    // expected-error@+1 {{unsupported numpy_dtype_type 'float128'}}
    %bad = hc_front.attr %np, "float128" {ref = {dtype = "float128", kind = "numpy_dtype_type"}}
    hc_front.return
  }
}

// -----

// `group.load(a[i][j], ...)` — chained Python subscripts lower to
// nested `hc.buffer_view`s whose index lists cannot be safely spliced
// (outer slice re-indexes the already-reduced view, not the original
// buffer). `peelBufferView` handles the single-level case; when it
// leaves a `buffer_view` behind, `lowerDslMethodCall` surfaces the
// rewrite suggestion before we emit wrong IR. Store shares the path;
// `vload` uses the same guard but covering one method variant is
// enough to pin the diagnostic wording.
module {
  hc_front.kernel "chained_subscript_load" attributes {
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M,K]", kind = "buffer", name = "a", shape = ["M", "K"]}
    ]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %load_attr = hc_front.attr %grp, "load" {ref = {kind = "dsl_method", method = "load"}}
    %zero = hc_front.constant<0 : i64>
    %m = hc_front.constant<16 : i64>
    %k = hc_front.constant<16 : i64>
    %row_sl = hc_front.slice (%zero, %m) {has_lower = true, has_upper = true, has_step = false}
    %col_sl = hc_front.slice (%zero, %k) {has_lower = true, has_upper = true, has_step = false}
    %sub1 = hc_front.subscript %a[%row_sl]
    %sub2 = hc_front.subscript %sub1[%col_sl]
    %m_dim = hc_front.constant<16 : i64>
    %k_dim = hc_front.constant<16 : i64>
    %shape_tuple = hc_front.tuple(%m_dim, %k_dim)
    %shape_kw = hc_front.keyword "shape" = %shape_tuple
    // expected-error@+1 {{chained subscript into `load` is not supported}}
    %tile = hc_front.call %load_attr(%sub2, %shape_kw)
    %t_tile = hc_front.target_name "a_tile"
    hc_front.assign %t_tile = %tile
    hc_front.return
  }
}
