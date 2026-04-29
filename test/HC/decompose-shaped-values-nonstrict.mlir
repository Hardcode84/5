// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-decompose-shaped-values=strict=false %s | FileCheck %s

// The default frontend schedule uses non-strict decomposition while helper-call,
// store, and region-boundary consumers are still being taught to preserve masks.
// Dialect conversion should bridge converted and unconverted boundaries with
// unrealized casts instead of failing during `hc.compile`.

// CHECK-LABEL: func.func @source_materialization
// CHECK: %[[DATA:.*]] = hc.load {{.*}} -> !hc.bare_tensor<f32, ["4"]>
// CHECK: %[[MASK:.*]] = hc.load_mask {{.*}} -> !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: %[[TILE:.*]] = builtin.unrealized_conversion_cast %[[DATA]], %[[MASK]] : !hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]> to !hc.tensor<f32, ["4"]>
// CHECK: hc.store {{.*}}, %[[TILE]]
func.func @source_materialization(%buf: !hc.buffer<f32, ["M"]>,
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

// CHECK-LABEL: func.func @target_materialization
// CHECK-SAME: %[[ARG:.*]]: !hc.tensor<f32, ["4"]>
// CHECK: %[[SPLIT:.*]]:2 = builtin.unrealized_conversion_cast %[[ARG]] : !hc.tensor<f32, ["4"]> to !hc.bare_tensor<f32, ["4"]>, !hc.bare_tensor<!hc.pred, ["4"]>
// CHECK: hc.vec %[[SPLIT]]#0
// CHECK: hc.vec %[[SPLIT]]#1
// CHECK: return
func.func @target_materialization(%tile: !hc.tensor<f32, ["4"]>) {
  %vec = hc.vec %tile : !hc.tensor<f32, ["4"]> -> !hc.vector<f32, ["4"]>
  return
}
