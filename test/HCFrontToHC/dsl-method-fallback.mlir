// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// DSL-method dispatch reads `hc_front.attr`'s `$name` — the op-level
// spelling the frontend always stamps — rather than `ref.method`. The
// Python resolver only fills in `ref.method` when the attr's base was
// classifiable (param/iv/local); chained attrs on subscript or call
// results arrive with no `ref` dict. Companion coverage for
// `numpy_dtype_type` attrs used as both a call argument and a
// value-constructor callee.
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names %s | hc-opt | FileCheck %s

// `.vec()` on a subscript base and `.with_inactive(value=...)` on a
// `.vec()` result: both attrs lack a `ref` dict because their bases
// are unclassifiable at resolver time. Dispatch must fall back to the
// attr's `$name` to land on `hc.vec` / `hc.with_inactive`.
// CHECK-LABEL: hc.kernel @chained_dsl_methods
// CHECK-NOT: hc_front.
module {
  hc_front.kernel "chained_dsl_methods" attributes {
    parameters = [{annotation = "Buffer[M,K]", kind = "buffer", name = "a", shape = ["M", "K"]}]
  } {
    // CHECK: hc.buffer_view
    // CHECK: %[[V:.*]] = hc.vec %{{.*}} : !hc.undef -> !hc.undef
    // CHECK: hc.with_inactive %[[V]] {inactive = 0 : i64}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %zero = hc_front.constant<0 : i64>
    %one = hc_front.constant<1 : i64>
    %slice = hc_front.slice (%zero, %one) {has_lower = true, has_upper = true, has_step = false}
    %sub = hc_front.subscript %a[%slice]
    %vec_attr = hc_front.attr %sub, "vec"
    %vec = hc_front.call %vec_attr()
    %wi_attr = hc_front.attr %vec, "with_inactive"
    %fill = hc_front.constant<0 : i64>
    %kw_value = hc_front.keyword "value" = %fill
    %masked = hc_front.call %wi_attr(%kw_value)
    %t_masked = hc_front.target_name "m"
    hc_front.assign %t_masked = %masked
    hc_front.return
  }

  // `.astype(np.float32)` on a classified base: the method attr carries
  // `ref.method = "astype"` but the `numpy_dtype_type` arg still has to
  // materialize as an `hc.const` wrapping a `TypeAttr` for astype to
  // pick up. Covers the "numpy_dtype_type attr used as an argument"
  // shape — this used to fail at `collectCallArgs`' null-arg guard
  // because `lowerAttr` returned an empty value for every attr.
  // CHECK-LABEL: hc.kernel @astype_numpy
  hc_front.kernel "astype_numpy" attributes {
    parameters = [{name = "x"}]
  } {
    // CHECK: hc.const<f32> : !hc.undef
    // CHECK: hc.astype %{{.*}}, target = f32
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %astype_attr = hc_front.attr %x, "astype" {ref = {kind = "dsl_method", method = "astype"}}
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %f32 = hc_front.attr %np, "float32" {ref = {dtype = "float32", kind = "numpy_dtype_type"}}
    %cast = hc_front.call %astype_attr(%f32)
    %t_cast = hc_front.target_name "c"
    hc_front.assign %t_cast = %cast
    hc_front.return
  }

  // `np.<dtype>(lit)` as a value-constructor callee: the call folds
  // the `ref.dtype` from the attr with the integer/float payload of
  // the positional literal, emitting a typed `hc.const` — `f16` zero
  // here — so consumers that need a numeric scalar (like
  // `hc.with_inactive`'s `$inactive`) pick up a real payload instead
  // of a dtype handle. The dtype-only `hc.const<f16>` materialized by
  // `lowerAttr` stays in the IR (the `np.float16` attr access may
  // still be reused elsewhere); the call result is the typed literal.
  // CHECK-LABEL: hc.kernel @numpy_dtype_call
  hc_front.kernel "numpy_dtype_call" attributes {parameters = []} {
    // CHECK: hc.const<0.000000e+00 : f16> : !hc.undef
    // CHECK-NOT: hc_front.
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %f16 = hc_front.attr %np, "float16" {ref = {dtype = "float16", kind = "numpy_dtype_type"}}
    %placeholder = hc_front.constant<0 : i64>
    %val = hc_front.call %f16(%placeholder)
    %t_val = hc_front.target_name "v"
    hc_front.assign %t_val = %val
    hc_front.return
  }

  // Value-constructor with no positional arg falls back to the dtype
  // handle (`hc.const<TypeAttr>`) emitted by `lowerAttr`. Keeping the
  // degradation path tested means the branch doesn't silently start
  // diagnosing legitimate type-only users (`.astype` et al route
  // through the attribute position, but this guard is the cheap
  // insurance that the fallback itself works).
  // CHECK-LABEL: hc.kernel @numpy_dtype_call_bare
  hc_front.kernel "numpy_dtype_call_bare" attributes {parameters = []} {
    // CHECK: hc.const<f16> : !hc.undef
    // CHECK-NOT: hc_front.
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %f16 = hc_front.attr %np, "float16" {ref = {dtype = "float16", kind = "numpy_dtype_type"}}
    %val = hc_front.call %f16()
    %t_val = hc_front.target_name "v"
    hc_front.assign %t_val = %val
    hc_front.return
  }

  // `buf_view.shape[0]` — `shape` attr rooted in an unclassified
  // subscript base. The resolver leaves `ref.method` empty; the
  // subscript-fold path must still recognize `shape` from the attr's
  // `$name` and emit `hc.buffer_dim`.
  // CHECK-LABEL: hc.kernel @shape_on_chain
  hc_front.kernel "shape_on_chain" attributes {
    parameters = [{annotation = "Buffer[M,K]", kind = "buffer", name = "a", shape = ["M", "K"]}]
  } {
    // CHECK: %[[V:.*]] = hc.buffer_view
    // CHECK: hc.buffer_dim %[[V]], axis = 0 : !hc.undef
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %zero = hc_front.constant<0 : i64>
    %one = hc_front.constant<1 : i64>
    %slice = hc_front.slice (%zero, %one) {has_lower = true, has_upper = true, has_step = false}
    %view = hc_front.subscript %a[%slice]
    %shape_attr = hc_front.attr %view, "shape"
    %axis = hc_front.constant<0 : i64>
    %dim = hc_front.subscript %shape_attr[%axis]
    %t_dim = hc_front.target_name "d"
    hc_front.assign %t_dim = %dim
    hc_front.return
  }
}
