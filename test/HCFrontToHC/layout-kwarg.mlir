// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tensor-producer calls (`group.zeros / ones / full / empty / load /
// vload`, plus the `vec` method on a base value) accept an optional
// `layout=<captured-IndexMap>` keyword. The Python resolver classifies
// the kwarg's value as `ref.kind = "layout"` carrying typed `#hc.expr`
// pieces; this pass stamps the captured layout directly on the
// producer op via its `layout` attribute. Inference later bakes the
// attribute into the result type; `-hc-canonicalize-layouts` still
// collapses any explicit `hc.as_layout` users when the layout is the
// identity. No intermediate `hc.as_layout` is emitted for `layout=`
// kwargs — the old overlay path tripped the `hc.as_layout`
// storage_size verifier for non-injective layouts the moment the
// load's bare result type was pinned.
//
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file %s | hc-opt --split-input-file | FileCheck %s

// CHECK-LABEL: hc.kernel @zeros_with_layout
// CHECK: hc.zeros shape %{{.*}} {layout = #hc.layout<shape_syms = ["w", "h"], index_syms = ["i", "j"], params = {row_stride = #hc.expr<"4 + h">}, storage_size = #hc.expr<"row_stride*w">, offset = #hc.expr<"j + i*row_stride">>}
// CHECK-NOT: hc.as_layout
module {
  hc_front.kernel "zeros_with_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %zeros = hc_front.attr %grp, "zeros" {ref = {kind = "dsl_method", method = "zeros"}}
    %m = hc_front.constant<16 : i64>
    %n = hc_front.constant<16 : i64>
    %shape = hc_front.tuple(%m, %n)
    %shape_kw = hc_front.keyword "shape" = %shape
    %lay = hc_front.name "A_LAYOUT" {ctx = "load", ref = {
      kind = "layout", shape_syms = ["w", "h"], index_syms = ["i", "j"],
      params = {row_stride = #hc.expr<"4 + h">},
      storage_size = #hc.expr<"row_stride*w">,
      offset = #hc.expr<"j + i*row_stride">}}
    %lay_kw = hc_front.keyword "layout" = %lay
    %t = hc_front.call %zeros(%shape_kw, %lay_kw)
    %tn = hc_front.target_name "t"
    hc_front.assign %tn = %t
    hc_front.return
  }
}

// -----

// `vzeros`: vector allocator, same overlay shape — keeps the
// rank-1 case honest.

// CHECK-LABEL: hc.kernel @vzeros_with_layout
// CHECK: hc.vzeros shape %{{.*}} {layout = #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">>}
// CHECK-NOT: hc.as_layout
module {
  hc_front.kernel "vzeros_with_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %vz = hc_front.attr %grp, "vzeros" {ref = {kind = "dsl_method", method = "vzeros"}}
    %n = hc_front.constant<16 : i64>
    %shape = hc_front.tuple(%n)
    %shape_kw = hc_front.keyword "shape" = %shape
    %lay = hc_front.name "DENSE" {ctx = "load", ref = {
      kind = "layout", shape_syms = ["d0"], index_syms = ["i0"],
      params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">}}
    %lay_kw = hc_front.keyword "layout" = %lay
    %t = hc_front.call %vz(%shape_kw, %lay_kw)
    %tn = hc_front.target_name "t"
    hc_front.assign %tn = %t
    hc_front.return
  }
}

// -----

// `full` / `vfull`: fill_value via positional + layout= attribute both
// hold on the `hc.full` / `hc.vfull` op directly.

// CHECK-LABEL: hc.kernel @full_with_layout
// CHECK: hc.full %{{.*}}, shape %{{.*}} {layout = #hc.layout<shape_syms = ["d0"]
// CHECK-NOT: hc.as_layout
module {
  hc_front.kernel "full_with_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %full = hc_front.attr %grp, "full" {ref = {kind = "dsl_method", method = "full"}}
    %fill = hc_front.constant<3.14 : f32>
    %n = hc_front.constant<8 : i64>
    %shape = hc_front.tuple(%n)
    %shape_kw = hc_front.keyword "shape" = %shape
    %lay = hc_front.name "DENSE" {ctx = "load", ref = {
      kind = "layout", shape_syms = ["d0"], index_syms = ["i0"],
      params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">}}
    %lay_kw = hc_front.keyword "layout" = %lay
    %t = hc_front.call %full(%fill, %shape_kw, %lay_kw)
    %tn = hc_front.target_name "t"
    hc_front.assign %tn = %t
    hc_front.return
  }
}

// -----

// `load` / `vload`: the captured layout applies to the loaded tile,
// not the source buffer. Source-buffer layout still flows from the
// kernel parameter dict; the producer attribute only describes the
// result type's layout.

// CHECK-LABEL: hc.kernel @vload_with_layout
// CHECK: hc.vload %{{.*}}, shape %{{.*}} {layout = #hc.layout<shape_syms = ["d0"]
// CHECK-NOT: hc.as_layout
module {
  hc_front.kernel "vload_with_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M]", kind = "buffer", name = "a", shape = ["M"]}
    ],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %vload = hc_front.attr %grp, "vload" {ref = {kind = "dsl_method", method = "vload"}}
    %zero = hc_front.constant<0 : i64>
    %n = hc_front.constant<16 : i64>
    %shape = hc_front.tuple(%n)
    %shape_kw = hc_front.keyword "shape" = %shape
    %lay = hc_front.name "DENSE" {ctx = "load", ref = {
      kind = "layout", shape_syms = ["d0"], index_syms = ["i0"],
      params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">}}
    %lay_kw = hc_front.keyword "layout" = %lay
    %t = hc_front.call %vload(%a, %zero, %shape_kw, %lay_kw)
    %tn = hc_front.target_name "t"
    hc_front.assign %tn = %t
    hc_front.return
  }
}

// -----

// `x.vec()` with `layout=`: same attribute pattern on a unary-base
// DSL method.

// CHECK-LABEL: hc.kernel @vec_with_layout
// CHECK: hc.vec %{{.*}} {layout = #hc.layout<shape_syms = ["d0"]
// CHECK-NOT: hc.as_layout
module {
  hc_front.kernel "vec_with_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}, {name = "x"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %vec_attr = hc_front.attr %x, "vec" {ref = {kind = "dsl_method", method = "vec"}}
    %lay = hc_front.name "DENSE" {ctx = "load", ref = {
      kind = "layout", shape_syms = ["d0"], index_syms = ["i0"],
      params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">}}
    %lay_kw = hc_front.keyword "layout" = %lay
    %v = hc_front.call %vec_attr(%lay_kw)
    %tn = hc_front.target_name "v"
    hc_front.assign %tn = %v
    hc_front.return
  }
}
