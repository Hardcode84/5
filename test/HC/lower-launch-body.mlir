// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s --hc-lower-launch-body | FileCheck %s

module {
  // CHECK-LABEL: func.func @scalar_and_loop(
  // CHECK-SAME: %[[A:.*]]: memref<?x?xf32>)
  func.func @scalar_and_loop(%a: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c8, %gy = %c2, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
      // CHECK: gpu.launch blocks(%[[BX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
      // CHECK-SAME: threads(%[[TX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
      %buffer = builtin.unrealized_conversion_cast %a
          : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
      // CHECK: %[[M:.*]] = memref.dim %[[A]], %{{.*}} : memref<?x?xf32>
      // CHECK: %[[N:.*]] = memref.dim %[[A]], %{{.*}} : memref<?x?xf32>
      // CHECK: %[[ROW:.*]] = arith.muli %{{.*}}, %[[BX]] : index
      // CHECK: %[[LANE_TILE:.*]] = arith.divui %[[TX]], %{{.*}} : index
      // CHECK: %[[COL:.*]] = arith.addi %{{.*}}, %[[LANE_TILE]] : index
      %row = hc.materialize_bound_expr : !hc.idx<"16*$WG0">
      %col = hc.materialize_bound_expr : !hc.idx<"16*$WG1 + 1/16*$WI0">
      %m = hc.materialize_bound_expr : !hc.idx<"M">
      %n = hc.buffer_dim %buffer, axis = 1
          : !hc.buffer<f32, ["M", "N"]> -> !hc.idx<"N">
      %one = hc.const<1 : i64> : !hc.idx<"1">
      %row_next = hc.add %row, %one
          : (!hc.idx<"16*$WG0">, !hc.idx<"1">) -> !hc.idx<"1 + 16*$WG0">
      %shape = hc.tuple(%m, %n)
          : (!hc.idx<"M">, !hc.idx<"N">) -> tuple<!hc.idx<"M">, !hc.idx<"N">>
      // CHECK: hc.tuple(%{{.*}}, %{{.*}}) : (index, index) -> tuple<index, index>
      %slice = hc.slice_expr(lower = %row upper = %row_next step = %one)
          : (!hc.idx<"16*$WG0">, !hc.idx<"1 + 16*$WG0">, !hc.idx<"1">)
            -> !hc.slice<lower = !hc.idx<"16*$WG0">, upper = !hc.idx<"1 + 16*$WG0">, step = !hc.idx<"1">>
      // CHECK: hc.slice_expr(lower = %[[ROW]] upper = %{{[^ ]+}} step = %{{[^)]+}}) : (index, index, index)
      // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      hc.for_range %one to %n step %one
          : (!hc.idx<"1">, !hc.idx<"N">, !hc.idx<"1">) {
      ^bb0(%i: !hc.idx<"$join0">):
        %i_next = hc.add %i, %one
            : (!hc.idx<"$join0">, !hc.idx<"1">) -> !hc.idx<"1 + $join0">
        hc.yield
      }
      // CHECK-NOT: hc.materialize_bound_expr
      // CHECK-NOT: hc.for_range
      gpu.terminator
    }
    return
  }
}
