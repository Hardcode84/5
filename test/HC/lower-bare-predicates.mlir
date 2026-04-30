// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-lower-bare-predicates -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @signature
// CHECK-SAME: %{{.*}}: i1
// CHECK-SAME: %{{.*}}: vector<4xi1>
// CHECK-SAME: %{{.*}}: vector<4xf32>
// CHECK-SAME: -> i1
func.func @signature(%pred: !hc.pred,
                     %mask: !hc.bare_vector<!hc.pred, ["4"]>,
                     %data: !hc.bare_vector<f32, ["4"]>) -> !hc.pred {
  return %pred : !hc.pred
}

// -----

// CHECK-LABEL: func.func @full_vector_mask
// CHECK: arith.constant dense<true> : vector<4xi1>
// CHECK-NOT: hc.full_mask
func.func @full_vector_mask() {
  %mask = hc.full_mask : !hc.bare_vector<!hc.pred, ["4"]>
  return
}

// -----

// CHECK-LABEL: func.func @full_tensor_mask_stays_hc
// CHECK: hc.full_mask : !hc.bare_tensor<!hc.pred, ["2", "4"]>
// CHECK-NOT: vector<2x4xi1>
func.func @full_tensor_mask_stays_hc() {
  %mask = hc.full_mask : !hc.bare_tensor<!hc.pred, ["2", "4"]>
  return
}

// -----

// CHECK-LABEL: func.func @store_boundary
// CHECK-SAME: %{{.*}}: !hc.buffer<f32, []>, %[[DATA:[^:]+]]: vector<4xf32>
// CHECK: %[[DATA_CAST:.*]] = builtin.unrealized_conversion_cast %[[DATA]] : vector<4xf32> to !hc.bare_vector<f32, ["4"]>
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<4xi1>
// CHECK: %[[MASK_CAST:.*]] = builtin.unrealized_conversion_cast %[[MASK]] : vector<4xi1> to !hc.bare_vector<!hc.pred, ["4"]>
// CHECK: hc.store {{.*}}, %[[DATA_CAST]], mask %[[MASK_CAST]]
func.func @store_boundary(%buf: !hc.buffer<f32, []>,
                          %data: !hc.bare_vector<f32, ["4"]>) {
  %mask = hc.full_mask : !hc.bare_vector<!hc.pred, ["4"]>
  hc.store %buf[], %data, mask %mask
      : (!hc.buffer<f32, []>, !hc.bare_vector<f32, ["4"]>,
         !hc.bare_vector<!hc.pred, ["4"]>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @select_vector
// CHECK-SAME: %[[MASK:.*]]: vector<4xi1>
// CHECK-SAME: %[[DATA:.*]]: vector<4xf32>
// CHECK: %[[FILL:.*]] = vector.broadcast %{{.*}} : f32 to vector<4xf32>
// CHECK: arith.select %[[MASK]], %[[DATA]], %[[FILL]] : vector<4xi1>, vector<4xf32>
// CHECK-NOT: hc.select
func.func @select_vector(%mask: !hc.bare_vector<!hc.pred, ["4"]>,
                         %data: !hc.bare_vector<f32, ["4"]>,
                         %fill: f32) {
  %selected = hc.select %mask, %data, %fill
      : (!hc.bare_vector<!hc.pred, ["4"]>, !hc.bare_vector<f32, ["4"]>, f32)
        -> !hc.bare_vector<f32, ["4"]>
  return
}

// -----

// CHECK-LABEL: func.func @select_tensor_stays_hc
// CHECK-SAME: %[[MASK:.*]]: !hc.bare_tensor<!hc.pred, ["2", "4"]>
// CHECK-SAME: %[[DATA:.*]]: !hc.bare_tensor<f32, ["2", "4"]>
// CHECK: hc.select %[[MASK]], %[[DATA]], %{{.*}} : {{.*}} -> !hc.bare_tensor<f32, ["2", "4"]>
// CHECK-NOT: vector<2x4xi1>
func.func @select_tensor_stays_hc(%mask: !hc.bare_tensor<!hc.pred, ["2", "4"]>,
                                  %data: !hc.bare_tensor<f32, ["2", "4"]>,
                                  %fill: f32) {
  %selected = hc.select %mask, %data, %fill
      : (!hc.bare_tensor<!hc.pred, ["2", "4"]>,
         !hc.bare_tensor<f32, ["2", "4"]>, f32)
        -> !hc.bare_tensor<f32, ["2", "4"]>
  return
}
