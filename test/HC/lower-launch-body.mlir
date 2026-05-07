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

  // CHECK-LABEL: func.func @tile_memory(
  // CHECK: vector.transfer_read
  // CHECK-SAME: vector<4x4xf32>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_write
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: vector.create_mask
  // CHECK-SAME: vector<4x4xi1>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_write
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: memref.subview
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>> to memref<4xf32,
  // CHECK: memref.subview
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>> to memref<4xi1,
  // CHECK: vector.transfer_read
  // CHECK-SAME: vector<4xf32>
  // CHECK: vector.transfer_read
  // CHECK-SAME: vector<4xi1>
  // CHECK: vector.broadcast
  // CHECK-SAME: f32 to vector<4xf32>
  // CHECK: arith.select
  // CHECK-SAME: vector<4xi1>, vector<4xf32>
  // CHECK: arith.constant dense<true> : vector<4xi1>
  // CHECK-NOT: hc.load
  // CHECK-NOT: hc.load_mask
  // CHECK-NOT: hc.buffer_view
  // CHECK-NOT: hc.vec
  // CHECK-NOT: hc.select
  // CHECK-NOT: hc.full_mask
  func.func @tile_memory(%a: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buffer = builtin.unrealized_conversion_cast %a
        : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c4, %sy = %c1, %sz = %c1) {
      %zero = hc.const<0 : i64> : !hc.idx<"0">
      %four = hc.const<4 : i64> : !hc.idx<"4">
      %row_slice = hc.slice_expr(lower = %zero upper = %four)
          : (!hc.idx<"0">, !hc.idx<"4">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>
      %col_slice = hc.slice_expr(lower = %zero upper = %four)
          : (!hc.idx<"0">, !hc.idx<"4">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>
      %shape = hc.tuple(%four, %four)
          : (!hc.idx<"4">, !hc.idx<"4">) -> tuple<!hc.idx<"4">, !hc.idx<"4">>
      %tile = hc.load %buffer[%row_slice, %col_slice], shape %shape
          : (!hc.buffer<f32, ["M", "N"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             tuple<!hc.idx<"4">, !hc.idx<"4">>)
            -> !hc.bare_tensor<f32, ["4", "4"]>
      %mask = hc.load_mask %buffer[%row_slice, %col_slice], shape %shape
          : (!hc.buffer<f32, ["M", "N"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             tuple<!hc.idx<"4">, !hc.idx<"4">>)
            -> !hc.bare_tensor<!hc.pred, ["4", "4"]>
      %lane = hc.materialize_bound_expr : !hc.idx<"$WI0">
      %full = hc.slice_expr() : () -> !hc.slice
      %frag = hc.buffer_view %tile[%lane, %full]
          : (!hc.bare_tensor<f32, ["4", "4"]>, !hc.idx<"$WI0">, !hc.slice)
            -> !hc.bare_tensor<f32, ["4"]>
      %frag_mask = hc.buffer_view %mask[%lane, %full]
          : (!hc.bare_tensor<!hc.pred, ["4", "4"]>, !hc.idx<"$WI0">, !hc.slice)
            -> !hc.bare_tensor<!hc.pred, ["4"]>
      %vec = hc.vec %frag
          : !hc.bare_tensor<f32, ["4"]> -> !hc.bare_vector<f32, ["4"]>
      %vec_mask = hc.vec %frag_mask
          : !hc.bare_tensor<!hc.pred, ["4"]> -> !hc.bare_vector<!hc.pred, ["4"]>
      %zero_f = arith.constant 0.0 : f32
      %selected = hc.select %vec_mask, %vec, %zero_f
          : (!hc.bare_vector<!hc.pred, ["4"]>, !hc.bare_vector<f32, ["4"]>, f32)
            -> !hc.bare_vector<f32, ["4"]>
      %full_mask = hc.full_mask : !hc.bare_vector<!hc.pred, ["4"]>
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @tensor_mask_and_select(
  // CHECK: vector.transfer_read
  // CHECK-SAME: vector<4x4xf32>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_write
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: vector.create_mask
  // CHECK-SAME: vector<4x4xi1>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_write
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_read
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_read
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: arith.select
  // CHECK-SAME: vector<4x4xi1>, vector<4x4xf32>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_write
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK-NOT: hc.load_mask
  // CHECK-NOT: hc.select
  func.func @tensor_mask_and_select(%a: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buffer = builtin.unrealized_conversion_cast %a
        : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %zero = hc.const<0 : i64> : !hc.idx<"0">
      %four = hc.const<4 : i64> : !hc.idx<"4">
      %row_slice = hc.slice_expr(lower = %zero upper = %four)
          : (!hc.idx<"0">, !hc.idx<"4">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>
      %col_slice = hc.slice_expr(lower = %zero upper = %four)
          : (!hc.idx<"0">, !hc.idx<"4">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>
      %shape = hc.tuple(%four, %four)
          : (!hc.idx<"4">, !hc.idx<"4">) -> tuple<!hc.idx<"4">, !hc.idx<"4">>
      %tile = hc.load %buffer[%row_slice, %col_slice], shape %shape
          : (!hc.buffer<f32, ["M", "N"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             tuple<!hc.idx<"4">, !hc.idx<"4">>)
            -> !hc.bare_tensor<f32, ["4", "4"]>
      %mask = hc.load_mask %tile[%row_slice, %col_slice], shape %shape
          : (!hc.bare_tensor<f32, ["4", "4"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"4">>,
             tuple<!hc.idx<"4">, !hc.idx<"4">>)
            -> !hc.bare_tensor<!hc.pred, ["4", "4"]>
      %zero_f = arith.constant 0.0 : f32
      %selected = hc.select %mask, %tile, %zero_f
          : (!hc.bare_tensor<!hc.pred, ["4", "4"]>, !hc.bare_tensor<f32, ["4", "4"]>, f32)
            -> !hc.bare_tensor<f32, ["4", "4"]>
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @masked_vector_store(
  // CHECK: vector.extract
  // CHECK-SAME: f32 from vector<4xf32>
  // CHECK: vector.extract
  // CHECK-SAME: i1 from vector<4xi1>
  // CHECK: scf.if
  // CHECK: memref.store
  // CHECK-SAME: memref<?xf32>
  // CHECK-NOT: hc.store
  func.func @masked_vector_store(%a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %a
        : memref<?xf32> to !hc.buffer<f32, ["M"]>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %data_vector = arith.constant dense<1.000000e+00> : vector<4xf32>
      %mask_vector = arith.constant dense<true> : vector<4xi1>
      %data = builtin.unrealized_conversion_cast %data_vector
          : vector<4xf32> to !hc.bare_vector<f32, ["4"]>
      %mask = builtin.unrealized_conversion_cast %mask_vector
          : vector<4xi1> to !hc.bare_vector<!hc.pred, ["4"]>
      %zero = hc.const<0 : i64> : !hc.idx<"0">
      %eight = hc.const<8 : i64> : !hc.idx<"8">
      %two = hc.const<2 : i64> : !hc.idx<"2">
      %rows = hc.slice_expr(lower = %zero upper = %eight step = %two)
          : (!hc.idx<"0">, !hc.idx<"8">, !hc.idx<"2">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"2">>
      hc.store %buffer[%rows], %data, mask %mask
          : (!hc.buffer<f32, ["M"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"2">>,
             !hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>) -> ()
      gpu.terminator
    }
    return
  }
}
