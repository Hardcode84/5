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
