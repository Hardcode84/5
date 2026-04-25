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
    // CHECK: %[[FILL:.*]] = hc.const<0 : i64> : !hc.undef
    // CHECK: hc.with_inactive %[[V]], %[[FILL]] : (!hc.undef, !hc.undef) -> !hc.undef
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
  // here — so consumers that need a numeric scalar SSA value (like
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
  // handle (`hc.const<TypeAttr>`) emitted by `lowerAttr`. Pins the
  // degradation path so the coercion branch can't silently start
  // diagnosing legitimate type-only users (`.astype` et al route
  // through the attribute position).
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

  // `np.bool_(lit)` — NumPy's `bool_` is *truthiness* semantics, not
  // bit-pattern truncation. A naive `APInt(1, lit)` would fold the
  // low bit, so `np.bool_(2)` would store 0 and `np.bool_(0.5)` would
  // store 0 too. Coercion has to detect i1 targets and route through
  // `!= 0` instead. Covers both the integer-payload (`2 → 1`) and
  // float-payload (`0.5 → 1`, `0.0 → 0`) branches.
  // CHECK-LABEL: hc.kernel @numpy_bool_truthiness
  hc_front.kernel "numpy_bool_truthiness" attributes {parameters = []} {
    // CHECK: hc.const<true> : !hc.undef
    // CHECK: hc.const<true> : !hc.undef
    // CHECK: hc.const<false> : !hc.undef
    // CHECK-NOT: hc_front.
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %two_i = hc_front.constant<2 : i64>
    %b1 = hc_front.attr %np, "bool_" {ref = {dtype = "bool_", kind = "numpy_dtype_type"}}
    %v1 = hc_front.call %b1(%two_i)
    %t1 = hc_front.target_name "b1"
    hc_front.assign %t1 = %v1
    %half = hc_front.constant<5.000000e-01 : f64>
    %b2 = hc_front.attr %np, "bool_" {ref = {dtype = "bool_", kind = "numpy_dtype_type"}}
    %v2 = hc_front.call %b2(%half)
    %t2 = hc_front.target_name "b2"
    hc_front.assign %t2 = %v2
    %zero_f = hc_front.constant<0.000000e+00 : f64>
    %b3 = hc_front.attr %np, "bool_" {ref = {dtype = "bool_", kind = "numpy_dtype_type"}}
    %v3 = hc_front.call %b3(%zero_f)
    %t3 = hc_front.target_name "b3"
    hc_front.assign %t3 = %v3
    hc_front.return
  }

  // Float -> int coercion: NaN/Inf and values outside `[-2^63, 2^63)`
  // hit UB under a plain `static_cast<int64_t>(double)` (C++
  // [conv.fpint]). The coercion rejects those silently (returns null
  // attr) and falls back to the `hc.const<TypeAttr>` dtype handle, so
  // downstream users either accept the dtype or diagnose themselves
  // instead of reading a poisoned literal. Finite in-range literals
  // still produce a typed `IntegerAttr` as before.
  // CHECK-LABEL: hc.kernel @numpy_int_from_float
  hc_front.kernel "numpy_int_from_float" attributes {parameters = []} {
    // CHECK: hc.const<3 : si32> : !hc.undef
    // CHECK: hc.const<si32> : !hc.undef
    // CHECK-NOT: hc_front.
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %three_f = hc_front.constant<3.000000e+00 : f64>
    %i32a = hc_front.attr %np, "int32" {ref = {dtype = "int32", kind = "numpy_dtype_type"}}
    %va = hc_front.call %i32a(%three_f)
    %ta = hc_front.target_name "ia"
    hc_front.assign %ta = %va
    %inf = hc_front.constant<0x7FF0000000000000 : f64>
    %i32b = hc_front.attr %np, "int32" {ref = {dtype = "int32", kind = "numpy_dtype_type"}}
    %vb = hc_front.call %i32b(%inf)
    %tb = hc_front.target_name "ib"
    hc_front.assign %tb = %vb
    hc_front.return
  }

  // Unsigned dtype names lower to unsigned builtin integer types instead
  // of collapsing onto signless `iN`, and integer constructor literals wrap
  // to the target width.
  // CHECK-LABEL: hc.kernel @numpy_unsigned_dtype
  hc_front.kernel "numpy_unsigned_dtype" attributes {parameters = []} {
    // CHECK: hc.const<ui32> : !hc.undef
    // CHECK: hc.const<255 : ui8> : !hc.undef
    // CHECK: hc.const<1 : ui16> : !hc.undef
    // CHECK: hc.const<ui64> : !hc.undef
    // CHECK-NOT: hc_front.
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %u32 = hc_front.attr %np, "uint32" {ref = {dtype = "uint32", kind = "numpy_dtype_type"}}
    %t_u32 = hc_front.target_name "u32"
    hc_front.assign %t_u32 = %u32
    %u8 = hc_front.attr %np, "uint8" {ref = {dtype = "uint8", kind = "numpy_dtype_type"}}
    %neg_one = hc_front.constant<-1 : i64>
    %wrapped_neg = hc_front.call %u8(%neg_one)
    %t_neg = hc_front.target_name "neg"
    hc_front.assign %t_neg = %wrapped_neg
    %u16 = hc_front.attr %np, "uint16" {ref = {dtype = "uint16", kind = "numpy_dtype_type"}}
    %wide = hc_front.constant<65537 : i64>
    %wrapped_wide = hc_front.call %u16(%wide)
    %t_wide = hc_front.target_name "wide"
    hc_front.assign %t_wide = %wrapped_wide
    %u64 = hc_front.attr %np, "uint64" {ref = {dtype = "uint64", kind = "numpy_dtype_type"}}
    %t_u64 = hc_front.target_name "u64"
    hc_front.assign %t_u64 = %u64
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
