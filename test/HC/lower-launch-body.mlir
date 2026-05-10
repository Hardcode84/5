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
      %row = hc.idx_apply () {symbols = []} : () -> !hc.idx<"16*$WG0">
      %col = hc.idx_apply () {symbols = []} : () -> !hc.idx<"16*$WG1 + 1/16*$WI0">
      %m = hc.idx_apply () {symbols = []} : () -> !hc.idx<"M">
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
      // CHECK-NOT: hc.idx_apply
      // CHECK-NOT: hc.pred_apply
      // CHECK-NOT: hc.for_range
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @tile_memory(
  // The cooperative load lowers `hc.load` into a per-lane `scf.for` chunk
  // loop that copies the source slice into LDS one element at a time, with
  // a closing `gpu.barrier` to publish the writes to the rest of the wave.
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: arith.ceildivui
  // CHECK: scf.for
  // CHECK: arith.cmpi ult
  // CHECK: scf.if
  // CHECK: arith.remui
  // CHECK: arith.divui
  // CHECK: arith.cmpi ult
  // CHECK: arith.andi
  // CHECK: scf.if {{.*}} -> (f32)
  // CHECK: memref.load
  // CHECK-SAME: memref<?x?xf32>
  // CHECK: memref.store
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: gpu.barrier
  // CHECK: vector.create_mask
  // CHECK-SAME: vector<4x4xi1>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // i1 stores go through per-element vector.extract + memref.store rather than
  // a packed vector.transfer_write — see the writeVectorToMemRef i1 case.
  // CHECK: vector.extract
  // CHECK: memref.store
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: memref.subview
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>> to memref<4xf32,
  // CHECK: memref.subview
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>> to memref<4xi1,
  // CHECK: vector.transfer_read
  // CHECK-SAME: vector<4xf32>
  // The per-lane i1 read mirrors the write: scalar memref.load + vector.insert
  // instead of vector.transfer_read of vector<4xi1>.
  // CHECK: memref.load
  // CHECK-SAME: memref<4xi1
  // CHECK: vector.insert
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
      %lane = hc.idx_apply () {symbols = []} : () -> !hc.idx<"$WI0">
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
  // Cooperative load of the source slice into LDS, gated by a barrier.
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: scf.for
  // CHECK: memref.load
  // CHECK-SAME: memref<?x?xf32>
  // CHECK: memref.store
  // CHECK-SAME: memref<4x4xf32, #gpu.address_space<workgroup>>
  // CHECK: gpu.barrier
  // CHECK: vector.create_mask
  // CHECK-SAME: vector<4x4xi1>
  // CHECK: memref.alloca
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // i1 mask materializes via per-element vector.extract + memref.store; the
  // matching read uses memref.load + vector.insert. Both ends agree on the
  // byte-per-element layout LLVM emits for memref<NxI1>, sidestepping the
  // bit-packed `vector<NxI1>` encoding the upstream lowering would otherwise
  // pick for the contiguous transfer_write/read pair.
  // CHECK: vector.extract
  // CHECK: memref.store
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: memref.load
  // CHECK-SAME: memref<4x4xi1, #gpu.address_space<workgroup>>
  // CHECK: vector.insert
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

  // CHECK-LABEL: func.func @strided_vload(
  // CHECK-SAME: %[[A:.*]]: memref<?x?xf32>)
  // The `step = !hc.idx<"2">` slice on the row axis must lower via a
  // `memref.subview` that absorbs the stride into the result memref's
  // affine layout; the subsequent `transfer_read` then reads from the
  // subview at logical zero offsets.
  // CHECK: %[[SUB:.*]] = memref.subview %[[A]][0, 0] [8, 1] [2, 1]
  // CHECK-SAME: memref<?x?xf32> to memref<8x1xf32, strided<[?, 1]>>
  // CHECK: vector.transfer_read %[[SUB]]
  // CHECK-SAME: vector<8x1xf32>
  // CHECK-NOT: hc.vload
  func.func @strided_vload(%a: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %a
        : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %zero = hc.const<0 : i64> : !hc.idx<"0">
      %one = hc.const<1 : i64> : !hc.idx<"1">
      %two = hc.const<2 : i64> : !hc.idx<"2">
      %sixteen = hc.const<16 : i64> : !hc.idx<"16">
      %eight = hc.const<8 : i64> : !hc.idx<"8">
      %rows = hc.slice_expr(lower = %zero upper = %sixteen step = %two)
          : (!hc.idx<"0">, !hc.idx<"16">, !hc.idx<"2">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">, step = !hc.idx<"2">>
      %col = hc.slice_expr(lower = %zero upper = %one)
          : (!hc.idx<"0">, !hc.idx<"1">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>
      %shape = hc.tuple(%eight, %one)
          : (!hc.idx<"8">, !hc.idx<"1">) -> tuple<!hc.idx<"8">, !hc.idx<"1">>
      %vec = hc.vload %buffer[%rows, %col], shape %shape
          : (!hc.buffer<f32, ["M", "N"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">, step = !hc.idx<"2">>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>,
             tuple<!hc.idx<"8">, !hc.idx<"1">>)
            -> !hc.bare_vector<f32, ["8", "1"]>
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @strided_load_mask(
  // CHECK-SAME: %[[A:.*]]: memref<?x?xf32>)
  // For a stride-2 slice into row axis of extent `M`, the in-bounds count
  // is `ceildiv(M - offset, 2)` rather than `M - offset`. Unit-stride
  // axes still emit the simpler `extent - offset` form (no extra ops).
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[M:.*]] = memref.dim %[[A]], %[[C0]]
  // CHECK: %[[REM_M:.*]] = arith.subi %[[M]], %{{.*}} : index
  // CHECK: %[[STEPM1:.*]] = arith.subi %{{.*}}, %{{.*}} : index
  // CHECK: %[[ADJ:.*]] = arith.addi %[[REM_M]], %[[STEPM1]] : index
  // CHECK: %[[ROWSZ:.*]] = arith.divsi %[[ADJ]], %{{.*}} : index
  // CHECK: %[[N:.*]] = memref.dim %[[A]], %[[C1]]
  // CHECK: %[[COLSZ:.*]] = arith.subi %[[N]], %{{.*}} : index
  // CHECK: vector.create_mask %[[ROWSZ]], %[[COLSZ]] : vector<8x1xi1>
  // CHECK-NOT: hc.load_mask
  func.func @strided_load_mask(%a: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %a
        : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %zero = hc.const<0 : i64> : !hc.idx<"0">
      %one = hc.const<1 : i64> : !hc.idx<"1">
      %two = hc.const<2 : i64> : !hc.idx<"2">
      %sixteen = hc.const<16 : i64> : !hc.idx<"16">
      %eight = hc.const<8 : i64> : !hc.idx<"8">
      %rows = hc.slice_expr(lower = %zero upper = %sixteen step = %two)
          : (!hc.idx<"0">, !hc.idx<"16">, !hc.idx<"2">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">, step = !hc.idx<"2">>
      %col = hc.slice_expr(lower = %zero upper = %one)
          : (!hc.idx<"0">, !hc.idx<"1">)
            -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>
      %shape = hc.tuple(%eight, %one)
          : (!hc.idx<"8">, !hc.idx<"1">) -> tuple<!hc.idx<"8">, !hc.idx<"1">>
      %mask = hc.load_mask %buffer[%rows, %col], shape %shape
          : (!hc.buffer<f32, ["M", "N"]>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">, step = !hc.idx<"2">>,
             !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>,
             tuple<!hc.idx<"8">, !hc.idx<"1">>)
            -> !hc.bare_vector<!hc.pred, ["8", "1"]>
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

  // `hc.idx_apply` substitutes its named operands into the carried
  // expression while leaving unlisted free symbols (e.g. `$WG0` here)
  // to the launch-context binding. The lowering composes the same
  // arith ops an empty-binding `hc.idx_apply` would for the same
  // expression text, but the per-symbol SSA edges come straight from
  // the op rather than via an ambient `unrealized_conversion_cast`
  // walk over the launch.
  // CHECK-LABEL: func.func @apply_offset(
  // CHECK-SAME: %[[A:[^:]+]]: memref<?xf32>
  // CHECK-SAME: %[[K:[^:)]+]]: index
  // CHECK: gpu.launch blocks(%[[BX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
  // CHECK: %[[OFF:.*]] = arith.addi %{{.*}}, %[[K]] : index
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
  // CHECK-NOT: hc.idx_apply
  // CHECK-NOT: hc.pred_apply
  func.func @apply_offset(%a: memref<?xf32>, %ext_k: index) {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %buffer = builtin.unrealized_conversion_cast %a
          : memref<?xf32> to !hc.buffer<f32, ["M"]>
      %k = builtin.unrealized_conversion_cast %ext_k
          : index to !hc.idx<"K">
      %off = hc.idx_apply (%k) {symbols = ["K"]}
           : (!hc.idx<"K">) -> !hc.idx<"K + $WG0">
      %p = hc.pred_apply (%k) {symbols = ["K"]}
         : (!hc.idx<"K">) -> !hc.pred<"K < $WG0">
      gpu.terminator
    }
    return
  }
}
