// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-shaped-constants`: materialise shaped constants
// through the launch-body type converter ahead of the main lowering.
// Vector flavours -> `arith.constant` splat. LDS flavours -> `hc.alloc
// workgroup` + `hc.generic` per-element fill.
//
// RUN: hc-opt %s --hc-lower-launch-shaped-constants | FileCheck %s

module {
  // `hc.vzeros` on a `!hc.bare_vector` lowers to an arith vector splat
  // via the converter's bare_vector -> vector<NxT> rule.
  // CHECK-LABEL: func.func @vzeros_vector
  // CHECK: %{{.*}} = arith.constant dense<0> : vector<8xi32>
  // CHECK-NOT: hc.vzeros
  func.func @vzeros_vector() -> !hc.bare_vector<i32, ["8"]> {
    %eight = hc.idx_apply () : () -> !hc.idx<"8">
    %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
    %v = hc.vzeros shape %shape
        : (tuple<!hc.idx<"8">>) -> !hc.bare_vector<i32, ["8"]>
    return %v : !hc.bare_vector<i32, ["8"]>
  }

  // `hc.full_mask` on a bare_vector lowers to a `dense<true>` splat.
  // CHECK-LABEL: func.func @full_mask_vector
  // CHECK: %{{.*}} = arith.constant dense<true> : vector<4xi1>
  // CHECK-NOT: hc.full_mask
  func.func @full_mask_vector() -> !hc.bare_vector<!hc.pred, ["4"]> {
    %m = hc.full_mask : !hc.bare_vector<!hc.pred, ["4"]>
    return %m : !hc.bare_vector<!hc.pred, ["4"]>
  }

  // `hc.zeros` on a `!hc.bare_tensor` lowers to LDS: `hc.alloc workgroup`
  // plus an `hc.generic` per-element fill of the splat scalar. Downstream
  // `hc-lower-generic` consumes the generic; barriers run between them.
  // CHECK-LABEL: func.func @zeros_lds
  // CHECK: %[[LDS:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
  // CHECK: hc.generic
  // CHECK-SAME: ins ()
  // CHECK-SAME: outs (%[[LDS]] at [{{.*}}] : !hc.ptr<workgroup, f32>)
  // CHECK:   hc.yield %{{.+}} : f32
  // CHECK-NOT: hc.zeros
  func.func @zeros_lds() {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
      %eight = hc.idx_apply () : () -> !hc.idx<"8">
      %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
      %tile = hc.zeros shape %shape
          : (tuple<!hc.idx<"8">>) -> !hc.bare_tensor<f32, ["8"]>
      gpu.terminator
    }
    return
  }
}
