// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-apply`: same `ExprLowerer` plumbing as
// `hc-lower-launch-body`, just for the apply ops. Runs after
// `hc-lower-generic` plants fresh applies on the per-lane offsets.
// The launch-body test covers the full lowering surface; this file
// pins the apply-only path so the scheduling cut stays intact.
//
// RUN: hc-opt %s --hc-lower-apply | FileCheck %s

module {
  // `hc.idx_apply` with one operand binds `K`; ambient `$WG0` binds
  // through the launch's block ids. Empty `hc.pred_apply` resolves
  // entirely via the launch context.
  // CHECK-LABEL: func.func @apply_offset(
  // CHECK-SAME: %[[K:[^:)]+]]: index
  // CHECK: gpu.launch blocks(%[[BX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
  // CHECK: arith.addi %{{.*}}, %[[K]] : index
  // CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : index
  // CHECK-NOT: hc.idx_apply
  // CHECK-NOT: hc.pred_apply
  func.func @apply_offset(%ext_k: index) {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %k = builtin.unrealized_conversion_cast %ext_k
          : index to !hc.idx<"K">
      %off = hc.idx_apply (%k as "K")
           : (!hc.idx<"K">) -> !hc.idx<"K + $WG0">
      %p = hc.pred_apply (%k as "K")
         : (!hc.idx<"K">) -> !hc.pred<"K < $WG0">
      gpu.terminator
    }
    return
  }

  // Apply inside an `hc.generic` body is dyn-legal (lowered when
  // `hc-lower-generic` clones the body out). The pass leaves it.
  // CHECK-LABEL: func.func @apply_inside_generic_survives(
  // CHECK: hc.generic
  // CHECK: hc.idx_apply
  // CHECK: hc.pred_apply
  func.func @apply_inside_generic_survives(%ptr: !hc.ptr<global, f32>,
                                            %out: !hc.ptr<workgroup, f32>) {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %bound = hc.idx_apply () : () -> !hc.idx<"8">
      hc.generic
          iter (parallel i_0 = %bound : !hc.idx<"8">)
          ins (%ptr at [#hc.expr<"i_0">] : !hc.ptr<global, f32>)
          outs (%out at [#hc.expr<"i_0">] : !hc.ptr<workgroup, f32>)
          -> () {
        ^bb0(%v: f32, %_unused: f32):
          %inside_idx = hc.idx_apply () : () -> !hc.idx<"i_0">
          %inside_pred = hc.pred_apply () : () -> !hc.pred<"i_0 < 8">
          hc.yield %v : f32
      }
      gpu.terminator
    }
    return
  }
}
