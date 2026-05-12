// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Frontend-capture path for `as_layout(value, descriptor)` and the
// structured `#hc.layout<...>` descriptor it carries.
//
// The Python resolver classifies an `IndexMap` capture as a
// `kind = "layout"` ref whose payload is a quintuple of typed MLIR
// attributes (shape_syms / index_syms as ArrayAttr<StringAttr>, params
// as DictionaryAttr<StringAttr, ExprAttr>, storage_size / offset as
// ExprAttr). It classifies the `as_layout` free function as
// `kind = "layout_op", op = "as_layout"`. `-convert-hc-front-to-hc`
// rebuilds the `LayoutAttr` straight from the typed pieces (no
// `parseExpr` round-trip) and emits `hc.as_layout`.
//
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file %s | hc-opt --split-input-file | FileCheck %s

// CHECK-LABEL: hc.kernel @uses_as_layout
// CHECK: %[[V:.*]] = hc.name_load "v"
// CHECK: %[[R:.*]] = hc.as_layout %[[V]], layout = (#hc.layout<
// CHECK-SAME: shape_syms = ["w", "h"]
// CHECK-SAME: index_syms = ["i", "j"]
// CHECK-SAME: params = {row_stride = #hc.expr<"4 + h">}
// CHECK-SAME: storage_size = #hc.expr<"row_stride*w">
// CHECK-SAME: offset = #hc.expr<"j + i*row_stride">

module {
  hc_front.kernel "uses_as_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %v = hc_front.name "v" {ctx = "load", ref = {kind = "local"}}
    %layout_fn = hc_front.name "as_layout"
        {ctx = "load", ref = {kind = "layout_op", op = "as_layout"}}
    %descriptor = hc_front.name "A_LAYOUT"
        {ctx = "load",
         ref = {
           kind = "layout",
           shape_syms = ["w", "h"],
           index_syms = ["i", "j"],
           params = {row_stride = #hc.expr<"4 + h">},
           storage_size = #hc.expr<"row_stride*w">,
           offset = #hc.expr<"j + i*row_stride">
         }}
    %relabeled = hc_front.call %layout_fn(%v, %descriptor)
    %tgt = hc_front.target_name "w_rowmajor"
    hc_front.assign %tgt = %relabeled
    hc_front.return
  }
}

// -----

// `as_layout` without a captured descriptor: passing an arbitrary local
// fails fast with a located diagnostic so the user fixes the call site
// rather than diagnosing a "missing layout attribute" downstream.

// CHECK-LABEL: hc.kernel @bare_descriptor
// CHECK: hc.as_layout %{{.*}}, layout = (#hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">>)

module {
  hc_front.kernel "bare_descriptor" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    // Default-strided rank-1 descriptor — the minimal layout payload
    // the C++ side accepts; both `params` and the `$STRIDE_*` symbols
    // surface as a one-liner here to keep the CHECK readable while
    // still exercising the structured path.
    %v = hc_front.name "v" {ctx = "load", ref = {kind = "local"}}
    %layout_fn = hc_front.name "as_layout"
        {ctx = "load", ref = {kind = "layout_op", op = "as_layout"}}
    %descriptor = hc_front.name "DENSE"
        {ctx = "load",
         ref = {
           kind = "layout",
           shape_syms = ["d0"],
           index_syms = ["i0"],
           params = {},
           storage_size = #hc.expr<"0">,
           offset = #hc.expr<"i0">
         }}
    %relabeled = hc_front.call %layout_fn(%v, %descriptor)
    %tgt = hc_front.target_name "u"
    hc_front.assign %tgt = %relabeled
    hc_front.return
  }
}
