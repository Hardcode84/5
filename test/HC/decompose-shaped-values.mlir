// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-decompose-shaped-values -split-input-file %s | hc-opt | FileCheck %s

// CHECK-LABEL: func.func @straight_line
// CHECK: %[[LOAD:.*]] = hc.load {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[LOAD_MASK:.*]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[VIEW:.*]] = hc.buffer_view %[[LOAD]]
// CHECK-SAME: -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[VIEW_MASK:.*]] = hc.buffer_view %[[LOAD_MASK]]
// CHECK-SAME: -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[VEC:.*]] = hc.vec %[[VIEW]]
// CHECK-SAME: -> !hc.bare_vector<f32, ["4"]>
// CHECK: %[[VEC_MASK:.*]] = hc.vec %[[VIEW_MASK]]
// CHECK-SAME: -> !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: hc.select %[[VEC_MASK]], %[[VEC]], %{{.*}} : (!hc.bare_vector<!hc.pred, ["4"]>, !hc.bare_vector<f32, ["4"]>, f32) -> !hc.bare_vector<f32, ["4"]>
// CHECK: hc.full_mask : !hc.bare_vector<!hc.pred, ["4"]>
// CHECK-NOT: !hc.tensor
// CHECK-NOT: !hc.vector
func.func @straight_line(%buf: !hc.buffer<f32, ["M"]>,
                         %i: !hc.idx<"0">,
                         %fill: f32) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %full = hc.slice_expr() : () -> !hc.slice
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  %view = hc.buffer_view %tile[%full]
      : (!hc.tensor<f32, ["4"]>, !hc.slice) -> !hc.tensor<f32, ["4"]>
  %vec = hc.vec %view : !hc.tensor<f32, ["4"]> -> !hc.vector<f32, ["4"]>
  %filled = hc.with_inactive %vec, %fill
      : (!hc.vector<f32, ["4"]>, f32) -> !hc.vector<f32, ["4"]>
  return
}

// -----

// CHECK-LABEL: func.func @vload_from_tensor
// CHECK: %[[TILE:.*]] = hc.load {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[TILE_MASK:.*]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[VEC:.*]] = hc.vload %[[TILE]]
// CHECK-SAME: -> !hc.bare_vector<f32, ["4"]>
// CHECK: hc.vload %[[TILE_MASK]]
// CHECK-SAME: -> !hc.bare_vector<!hc.pred, ["4"]>
// CHECK-NOT: !hc.tensor
// CHECK-NOT: !hc.vector
func.func @vload_from_tensor(%buf: !hc.buffer<f32, ["M"]>,
                             %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %full = hc.slice_expr() : () -> !hc.slice
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  %vec = hc.vload %tile[%full], shape %shape
      : (!hc.tensor<f32, ["4"]>, !hc.slice, tuple<!hc.idx<"4">>)
        -> !hc.vector<f32, ["4"]>
  return
}

// -----

// CHECK-LABEL: func.func @buffer_view_from_buffer
// CHECK: %[[VIEW:.*]] = hc.buffer_view {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: hc.full_mask : !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK-NOT: !hc.tensor
// CHECK-NOT: !hc.vector
func.func @buffer_view_from_buffer(%buf: !hc.buffer<f32, ["M"]>) {
  %full = hc.slice_expr() : () -> !hc.slice
  %view = hc.buffer_view %buf[%full]
      : (!hc.buffer<f32, ["M"]>, !hc.slice) -> !hc.tensor<f32, ["4"]>
  return
}

// -----

// CHECK-LABEL: func.func @getitem_vector
// CHECK: %[[DATA:.*]] = hc.vec {{.*}} -> !hc.bare_vector<f32, ["4"]>
// CHECK: %[[MASK:.*]] = hc.vec {{.*}} -> !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: %[[ITEM:.*]] = hc.getitem %[[DATA]]
// CHECK-SAME: -> !hc.bare_vector<f32, ["4"]>
// CHECK: hc.getitem %[[MASK]]
// CHECK-SAME: -> !hc.bare_vector<!hc.pred, ["4"]>
// CHECK-NOT: !hc.tensor
// CHECK-NOT: !hc.vector
func.func @getitem_vector(%buf: !hc.buffer<f32, ["M"]>,
                          %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %full = hc.slice_expr() : () -> !hc.slice
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  %vec = hc.vec %tile : !hc.tensor<f32, ["4"]> -> !hc.vector<f32, ["4"]>
  %item = hc.getitem %vec[%full]
      : (!hc.vector<f32, ["4"]>, !hc.slice) -> !hc.vector<f32, ["4"]>
  return
}

// -----

// CHECK-LABEL: func.func @store_to_buffer
// CHECK: %[[DATA:.*]] = hc.load {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[MASK:.*]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: hc.store {{.*}}, %[[DATA]], mask %[[MASK]]
// CHECK-SAME: !hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @store_to_buffer(%buf: !hc.buffer<f32, ["M"]>,
                           %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  hc.store %buf[%i], %tile
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.tensor<f32, ["4"]>) -> ()
  return
}

// -----

// CHECK-LABEL: hc.func @store_to_tensor
// CHECK-SAME: %[[DST_DATA:[^:]+]]: !hc.bare_tensor<f32, ["4"]>
// CHECK-SAME: %[[DST_MASK:[^:]+]]: !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK-SAME: %[[SRC_DATA:[^:]+]]: !hc.bare_tensor<f32, ["4"]>
// CHECK-SAME: %[[SRC_MASK:[^:]+]]: !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: hc.store %[[DST_DATA]][], %[[SRC_DATA]]
// CHECK-SAME: (!hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<f32, ["4"]>) -> ()
// CHECK: hc.store %[[DST_MASK]][], %[[SRC_MASK]]
// CHECK-SAME: (!hc.bare_tensor<!hc.pred, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>) -> ()
// CHECK-NOT: builtin.unrealized_conversion_cast
hc.func @store_to_tensor(%dst: !hc.tensor<f32, ["4"]>,
                         %src: !hc.tensor<f32, ["4"]>) {
  hc.store %dst[], %src
      : (!hc.tensor<f32, ["4"]>, !hc.tensor<f32, ["4"]>) -> ()
  hc.return
}

// -----

// CHECK-LABEL: hc.func @passthrough
// CHECK-SAME: %[[ARG_DATA:.*]]: !hc.bare_tensor<f32, ["4"]>
// CHECK-SAME: %[[ARG_MASK:.*]]: !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>)
// CHECK: hc.return %[[ARG_DATA]], %[[ARG_MASK]]
// CHECK-SAME: !hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>
hc.func @passthrough(%tile: !hc.tensor<f32, ["4"]>) -> !hc.tensor<f32, ["4"]> {
  hc.return %tile : !hc.tensor<f32, ["4"]>
}

// CHECK-LABEL: func.func @call_boundary
// CHECK: %[[LOAD:.*]] = hc.load {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[MASK:.*]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[CALL:.*]]:2 = hc.call @passthrough(%[[LOAD]], %[[MASK]])
// CHECK-SAME: (!hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>) -> (!hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>)
// CHECK: hc.vec %[[CALL]]#0
// CHECK-SAME: -> !hc.bare_vector<f32, ["4"]>
// CHECK: hc.vec %[[CALL]]#1
// CHECK-SAME: -> !hc.bare_vector<!hc.pred, ["4"]>
func.func @call_boundary(%buf: !hc.buffer<f32, ["M"]>,
                         %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  %out = hc.call @passthrough(%tile)
      : (!hc.tensor<f32, ["4"]>) -> !hc.tensor<f32, ["4"]>
  %vec = hc.vec %out : !hc.tensor<f32, ["4"]> -> !hc.vector<f32, ["4"]>
  return
}

// -----

// CHECK-LABEL: hc.intrinsic @intrinsic_passthrough
// CHECK-SAME: %{{[^:]+}}: !hc.bare_vector<f32, ["4"]>
// CHECK-SAME: %{{[^:]+}}: !hc.bare_vector<!hc.pred, ["4"]>
// CHECK-SAME: -> (!hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>)
hc.intrinsic @intrinsic_passthrough(%vec: !hc.vector<f32, ["4"]>)
    -> !hc.vector<f32, ["4"]>
    scope = #hc.scope<"WorkItem"> parameters = ["vec"] {}

// CHECK-LABEL: func.func @intrinsic_boundary
// CHECK: %[[LOAD:.*]] = hc.load {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[LOAD_MASK:.*]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[VEC:.*]] = hc.vec %[[LOAD]]
// CHECK-SAME: -> !hc.bare_vector<f32, ["4"]>
// CHECK: %[[VEC_MASK:.*]] = hc.vec %[[LOAD_MASK]]
// CHECK-SAME: -> !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: %[[CALL:.*]]:2 = hc.call_intrinsic @intrinsic_passthrough(%[[VEC]], %[[VEC_MASK]])
// CHECK-SAME: (!hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>) -> (!hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>)
// CHECK: hc.store {{.*}}, %[[CALL]]#0, mask %[[CALL]]#1
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @intrinsic_boundary(%buf: !hc.buffer<f32, ["M"]>,
                              %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  %vec = hc.vec %tile : !hc.tensor<f32, ["4"]> -> !hc.vector<f32, ["4"]>
  %out = hc.call_intrinsic @intrinsic_passthrough(%vec)
      : (!hc.vector<f32, ["4"]>) -> !hc.vector<f32, ["4"]>
  hc.store %buf[%i], %out
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.vector<f32, ["4"]>) -> ()
  return
}

// -----

// CHECK-LABEL: hc.func @for_range_iter_arg
// CHECK-SAME: %[[INIT_DATA:.*]]: !hc.bare_vector<f32, ["4"]>
// CHECK-SAME: %[[INIT_MASK:.*]]: !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: %[[LOOP:.*]]:2 = hc.for_range {{.*}} iter_args(%[[INIT_DATA]], %[[INIT_MASK]])
// CHECK-SAME: -> (!hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>)
// CHECK: ^bb0(%{{.*}}: !hc.idx<"0">, %[[ARG_DATA:.*]]: !hc.bare_vector<f32, ["4"]>, %[[ARG_MASK:.*]]: !hc.bare_vector<!hc.pred, ["4"]>):
// CHECK: hc.yield %[[ARG_DATA]], %[[ARG_MASK]]
// CHECK-SAME: !hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: hc.return %[[LOOP]]#0, %[[LOOP]]#1
hc.func @for_range_iter_arg(%init: !hc.vector<f32, ["4"]>)
    -> !hc.vector<f32, ["4"]> {
  %lo = hc.const<0 : i64> : !hc.idx<"0">
  %hi = hc.const<4 : i64> : !hc.idx<"4">
  %step = hc.const<1 : i64> : !hc.idx<"1">
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.idx<"0">, !hc.idx<"4">, !hc.idx<"1">)
        -> (!hc.vector<f32, ["4"]>) {
  ^bb0(%iv: !hc.idx<"0">, %carried: !hc.vector<f32, ["4"]>):
    hc.yield %carried : !hc.vector<f32, ["4"]>
  }
  hc.return %loop : !hc.vector<f32, ["4"]>
}

// -----

// CHECK-LABEL: hc.func @if_result
// CHECK-SAME: %[[LHS_DATA:[^:]+]]: !hc.bare_vector<f32, ["4"]>
// CHECK-SAME: %[[LHS_MASK:[^:]+]]: !hc.bare_vector<!hc.pred, ["4"]>
// CHECK-SAME: %[[RHS_DATA:[^:]+]]: !hc.bare_vector<f32, ["4"]>
// CHECK-SAME: %[[RHS_MASK:[^:]+]]: !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: %[[IF:.*]]:2 = hc.if {{.*}} -> (!hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>)
// CHECK: hc.yield %[[LHS_DATA]], %[[LHS_MASK]]
// CHECK-SAME: !hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: hc.yield %[[RHS_DATA]], %[[RHS_MASK]]
// CHECK-SAME: !hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: hc.return %[[IF]]#0, %[[IF]]#1
hc.func @if_result(%cond: i1,
                   %lhs: !hc.vector<f32, ["4"]>,
                   %rhs: !hc.vector<f32, ["4"]>)
    -> !hc.vector<f32, ["4"]> {
  %choice = hc.if %cond -> (!hc.vector<f32, ["4"]>) : i1 {
    hc.yield %lhs : !hc.vector<f32, ["4"]>
  } else {
    hc.yield %rhs : !hc.vector<f32, ["4"]>
  }
  hc.return %choice : !hc.vector<f32, ["4"]>
}

// -----

// CHECK-LABEL: hc.func @workitem_region_result
// CHECK-SAME: %[[LOCAL_DATA:[^:]+]]: !hc.bare_vector<f32, ["8"]>
// CHECK-SAME: %[[LOCAL_MASK:[^:]+]]: !hc.bare_vector<!hc.pred, ["8"]>
// CHECK-SAME: -> (!hc.bare_vector<f32, ["8", "32", "1"]>, !hc.bare_vector<!hc.pred, ["8", "32", "1"]>)
// CHECK: %[[REGION:.*]]:2 = hc.workitem_region -> (!hc.bare_vector<f32, ["8", "32", "1"]>, !hc.bare_vector<!hc.pred, ["8", "32", "1"]>)
// CHECK: ^bb0(%{{.*}}: !hc.workitem<group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>):
// CHECK: hc.yield %[[LOCAL_DATA]], %[[LOCAL_MASK]]
// CHECK-SAME: !hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>
// CHECK: hc.return %[[REGION]]#0, %[[REGION]]#1
hc.func @workitem_region_result(%local: !hc.vector<f32, ["8"]>)
    -> !hc.vector<f32, ["8", "32", "1"]> {
  %region = hc.workitem_region -> (!hc.vector<f32, ["8", "32", "1"]>) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.vector<f32, ["8"]>
  }
  hc.return %region : !hc.vector<f32, ["8", "32", "1"]>
}

// -----

// CHECK-LABEL: hc.func @subgroup_region_result
// CHECK-SAME: %[[LOCAL_DATA:[^:]+]]: !hc.bare_vector<f32, ["4"]>
// CHECK-SAME: %[[LOCAL_MASK:[^:]+]]: !hc.bare_vector<!hc.pred, ["4"]>
// CHECK-SAME: -> (!hc.bare_vector<f32, ["4", "2"]>, !hc.bare_vector<!hc.pred, ["4", "2"]>)
// CHECK: %[[REGION:.*]]:2 = hc.subgroup_region -> (!hc.bare_vector<f32, ["4", "2"]>, !hc.bare_vector<!hc.pred, ["4", "2"]>)
// CHECK: ^bb0(%{{.*}}: !hc.subgroup<group_shape = #hc.shape<["64"]>, subgroup_size = #hc.expr<"32">>):
// CHECK: hc.yield %[[LOCAL_DATA]], %[[LOCAL_MASK]]
// CHECK-SAME: !hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: hc.return %[[REGION]]#0, %[[REGION]]#1
hc.func @subgroup_region_result(%local: !hc.vector<f32, ["4"]>)
    -> !hc.vector<f32, ["4", "2"]> {
  %region = hc.subgroup_region -> (!hc.vector<f32, ["4", "2"]>) {
  ^bb0(%sg: !hc.subgroup<group_shape = #hc.shape<["64"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.vector<f32, ["4"]>
  }
  hc.return %region : !hc.vector<f32, ["4", "2"]>
}

// -----

// Non-injective layout on the vload result. The layout encodes a broadcast
// (`storage_size < product(shape)`); decompose must preserve it on both the
// bare data and the bare mask carriers so `hc-flatten-with-layouts` can later
// compose the per-iter offset against the same broadcast — dropping the
// layout here would silently materialise the broadcast into a full identity
// tile before flatten ever sees it. Empty indices route the mask through
// `hc.full_mask` instead of `hc.load_mask`: with no per-axis slice carriers
// there is nothing to plant a predicate from, and the layout is the user's
// contract for legal addressing into the flat source.
//
// CHECK-LABEL: func.func @noninjective_layout_preserved
// CHECK: %[[VLOAD:.*]] = hc.vload
// CHECK-SAME: -> !hc.bare_vector<si64, ["M", "K", "LANE"],
// CHECK-SAME: storage_size = #hc.expr<"K">
// CHECK-SAME: offset = #hc.expr<"j">
// CHECK: hc.full_mask
// CHECK-SAME: : !hc.bare_vector<!hc.pred, ["M", "K", "LANE"],
// CHECK-SAME: storage_size = #hc.expr<"K">
// CHECK-SAME: offset = #hc.expr<"j">
// CHECK-NOT: hc.load_mask
// CHECK-NOT: !hc.vector
func.func @noninjective_layout_preserved(%buf: !hc.buffer<si64, ["K"]>) {
  %m = hc.const<2 : i64> : !hc.idx<"M">
  %k = hc.const<4 : i64> : !hc.idx<"K">
  %lane = hc.const<3 : i64> : !hc.idx<"LANE">
  %shape = hc.tuple(%m, %k, %lane)
      : (!hc.idx<"M">, !hc.idx<"K">, !hc.idx<"LANE">)
        -> tuple<!hc.idx<"M">, !hc.idx<"K">, !hc.idx<"LANE">>
  %vec = hc.vload %buf[], shape %shape
      : (!hc.buffer<si64, ["K"]>,
         tuple<!hc.idx<"M">, !hc.idx<"K">, !hc.idx<"LANE">>)
        -> !hc.vector<si64, ["M", "K", "LANE"],
                      #hc.layout<shape_syms = ["M", "K", "L"],
                                 index_syms = ["i", "j", "lane"],
                                 params = {},
                                 storage_size = #hc.expr<"K">,
                                 offset = #hc.expr<"j">>>
  return
}

// -----

// `hc.load` mirrors the vload broadcast rule: an empty index list has no
// per-axis slice to plant a predicate from, so the mask becomes
// `hc.full_mask` (whole-shape-valid by the layout contract) instead of
// `hc.load_mask` (which would later trip `rewriteLoadMask` on the rank
// mismatch between zero indices and the rank-1 source).
//
// CHECK-LABEL: func.func @noninjective_load_empty_indices
// CHECK: hc.load
// CHECK-SAME: -> !hc.bare_tensor<si64, ["M", "K"]
// CHECK: hc.full_mask
// CHECK-SAME: : !hc.bare_tensor<!hc.pred, ["M", "K"]
// CHECK-NOT: hc.load_mask
func.func @noninjective_load_empty_indices(%buf: !hc.buffer<si64, ["K"]>) {
  %m = hc.const<2 : i64> : !hc.idx<"M">
  %k = hc.const<4 : i64> : !hc.idx<"K">
  %shape = hc.tuple(%m, %k)
      : (!hc.idx<"M">, !hc.idx<"K">)
        -> tuple<!hc.idx<"M">, !hc.idx<"K">>
  %tile = hc.load %buf[], shape %shape
      : (!hc.buffer<si64, ["K"]>, tuple<!hc.idx<"M">, !hc.idx<"K">>)
        -> !hc.tensor<si64, ["M", "K"],
                      #hc.layout<shape_syms = ["M", "K"],
                                 index_syms = ["i", "j"],
                                 params = {},
                                 storage_size = #hc.expr<"K">,
                                 offset = #hc.expr<"j">>>
  return
}

// -----

// `hc.astype` on a semantic tensor splits into a bare data astype +
// mask pass-through. Element type on the data half tracks `target`;
// the mask half rides unchanged because element-cast doesn't gate
// lanes. The `target` attribute threads through the pattern; the
// generic unary template can't reuse-build the op because the
// constructor signature carries the type attribute.
// CHECK-LABEL: func.func @astype_decomposes
// CHECK: %[[LOAD:.+]] = hc.load {{.*}} -> !hc.bare_tensor<f16, ["4"]>
// CHECK: %[[MASK:.+]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[CAST:.+]] = hc.astype %[[LOAD]], target = f32 : !hc.bare_tensor<f16, ["4"]> -> !hc.bare_tensor<f32, ["4"]>
// CHECK-NOT: !hc.tensor<
// CHECK-NOT: !hc.vector<
func.func @astype_decomposes(%buf: !hc.buffer<f16, ["M"]>,
                              %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f16, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f16, ["4"]>
  %cast = hc.astype %tile, target = f32
      : !hc.tensor<f16, ["4"]> -> !hc.tensor<f32, ["4"]>
  return
}
