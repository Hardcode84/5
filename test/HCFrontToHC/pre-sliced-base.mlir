// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `group.load(a[row_sl, col_sl], shape=(M, K))` — the WMMA pattern — has
// to land as `hc.load %a[%row_sl, %col_sl] {shape = ...}`, not as a
// chained `hc.buffer_view` + zero-index `hc.load`. Same story for
// `vload` and `store`. Before the peel fix, the frontend subscript
// would lower into an `hc.buffer_view` whose SSA result then arrived
// at the load branch as a single positional with no trailing indices,
// and the shape-rank verifier on `hc.load` would reject the rank-0
// index list against a rank-2 `shape` attr.
// The second RUN parses the output back through `hc-opt` so the `hc`
// verifier sees the same IR FileCheck does; a rank mismatch surfaces
// as a parse-time failure there.
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names %s | hc-opt | FileCheck %s

// CHECK-LABEL: hc.kernel @load_presliced
// CHECK-NOT: hc_front.
module {
  hc_front.kernel "load_presliced" attributes {
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M,K]", kind = "buffer", name = "a", shape = ["M", "K"]}
    ]
  } {
    // CHECK: %[[ROW:.*]] = hc.slice_expr
    // CHECK: %[[COL:.*]] = hc.slice_expr
    // CHECK: hc.load %arg1[%[[ROW]], %[[COL]]] {shape = #hc.shape<["16", "16"]>}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %load_attr = hc_front.attr %grp, "load" {ref = {kind = "dsl_method", method = "load"}}
    %zero = hc_front.constant<0 : i64>
    %m = hc_front.constant<16 : i64>
    %k = hc_front.constant<16 : i64>
    %row_sl = hc_front.slice(%zero, %m) {has_lower = true, has_step = false, has_upper = true}
    %col_sl = hc_front.slice(%zero, %k) {has_lower = true, has_step = false, has_upper = true}
    %idx = hc_front.tuple(%row_sl, %col_sl)
    %sub = hc_front.subscript %a[%idx]
    %m_dim = hc_front.constant<16 : i64>
    %k_dim = hc_front.constant<16 : i64>
    %shape_tuple = hc_front.tuple(%m_dim, %k_dim)
    %shape_kw = hc_front.keyword "shape" = %shape_tuple
    %tile = hc_front.call %load_attr(%sub, %shape_kw)
    %t_tile = hc_front.target_name "a_tile"
    hc_front.assign %t_tile = %tile
    hc_front.return
  }

  // Same peel on `group.vload` — identical front-IR shape, different
  // terminal op. Covers the vload branch of the shared `peelBufferView`
  // helper in `lowerDslMethodCall`.
  // CHECK-LABEL: hc.kernel @vload_presliced
  // CHECK-NOT: hc_front.
  hc_front.kernel "vload_presliced" attributes {
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M,K]", kind = "buffer", name = "a", shape = ["M", "K"]}
    ]
  } {
    // CHECK: %[[ROW:.*]] = hc.slice_expr
    // CHECK: %[[COL:.*]] = hc.slice_expr
    // CHECK: hc.vload %arg1[%[[ROW]], %[[COL]]] {shape = #hc.shape<["16", "16"]>}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %vload_attr = hc_front.attr %grp, "vload" {ref = {kind = "dsl_method", method = "vload"}}
    %zero = hc_front.constant<0 : i64>
    %m = hc_front.constant<16 : i64>
    %k = hc_front.constant<16 : i64>
    %row_sl = hc_front.slice(%zero, %m) {has_lower = true, has_step = false, has_upper = true}
    %col_sl = hc_front.slice(%zero, %k) {has_lower = true, has_step = false, has_upper = true}
    %idx = hc_front.tuple(%row_sl, %col_sl)
    %sub = hc_front.subscript %a[%idx]
    %m_dim = hc_front.constant<16 : i64>
    %k_dim = hc_front.constant<16 : i64>
    %shape_tuple = hc_front.tuple(%m_dim, %k_dim)
    %shape_kw = hc_front.keyword "shape" = %shape_tuple
    %tile = hc_front.call %vload_attr(%sub, %shape_kw)
    %t_tile = hc_front.target_name "v"
    hc_front.assign %t_tile = %tile
    hc_front.return
  }

  // `group.store(c[row_sl, col_sl], tile)` — subscript dest, raw source.
  // `hc.store` has no shape attr and no rank verifier of its own, but
  // the target IR in doc/lowering.md still threads slices into the
  // op's index list rather than through a throwaway `hc.buffer_view`.
  // CHECK-LABEL: hc.kernel @store_presliced
  // CHECK-NOT: hc_front.
  hc_front.kernel "store_presliced" attributes {
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M,K]", kind = "buffer", name = "c", shape = ["M", "K"]},
      {name = "tile"}
    ]
  } {
    // CHECK: %[[ROW:.*]] = hc.slice_expr
    // CHECK: %[[COL:.*]] = hc.slice_expr
    // CHECK: hc.store %arg1[%[[ROW]], %[[COL]]], %{{.*}} : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> ()
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %c = hc_front.name "c" {ctx = "load", ref = {kind = "param"}}
    %tile = hc_front.name "tile" {ctx = "load", ref = {kind = "param"}}
    %store_attr = hc_front.attr %grp, "store" {ref = {kind = "dsl_method", method = "store"}}
    %zero = hc_front.constant<0 : i64>
    %m = hc_front.constant<16 : i64>
    %k = hc_front.constant<16 : i64>
    %row_sl = hc_front.slice(%zero, %m) {has_lower = true, has_step = false, has_upper = true}
    %col_sl = hc_front.slice(%zero, %k) {has_lower = true, has_step = false, has_upper = true}
    %idx = hc_front.tuple(%row_sl, %col_sl)
    %sub = hc_front.subscript %c[%idx]
    %call = hc_front.call %store_attr(%sub, %tile)
    hc_front.return
  }
}
