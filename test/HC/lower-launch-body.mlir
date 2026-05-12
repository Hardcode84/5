// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Workgroup-AS storage (`!hc.bare_tensor`) lowers to flat
// `!hc.ptr<workgroup, T>` instead of `memref<..., workgroup>`. Kernel-arg
// buffers come in as `(ptr, dim, stride)` bundles UCC'd to
// `!hc.buffer<...>`; the per-axis dim values feed `hc.buffer_dim` /
// `hc.load_mask` extents and the stride values feed offset linearization
// for `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`.
// See `doc/layouts.md` "hc.ptr and memory ops".
//
// RUN: hc-opt %s --hc-lower-launch-body | FileCheck %s

module {
  // CHECK-LABEL: func.func @scalar_and_loop(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[M:[^:]+]]: index,
  // CHECK-SAME: %[[N:[^:]+]]: index,
  // CHECK-SAME: %[[SM:[^:]+]]: index,
  // CHECK-SAME: %[[SN:[^:]+]]: index)
  func.func @scalar_and_loop(%ptr: !hc.ptr<global, f32>,
                             %m: index, %n: index,
                             %sm: index, %sn: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c8, %gy = %c2, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
      // CHECK: gpu.launch blocks(%[[BX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
      // CHECK-SAME: threads(%[[TX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
      %buffer = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
          : !hc.ptr<global, f32>, index, index, index, index
          to !hc.buffer<f32, ["M", "N"]>
      // The UCC dissolves: per-axis dim/stride values flow straight into
      // the rest of the body without any `memref.dim` chain.
      // CHECK: %[[ROW:.*]] = arith.muli %{{.*}}, %[[BX]] : index
      // CHECK: %[[LANE_TILE:.*]] = arith.divui %[[TX]], %{{.*}} : index
      // CHECK: %[[COL:.*]] = arith.addi %{{.*}}, %[[LANE_TILE]] : index
      %row = hc.idx_apply () : () -> !hc.idx<"16*$WG0">
      %col = hc.idx_apply () : () -> !hc.idx<"16*$WG1 + 1/16*$WI0">
      %ms = hc.idx_apply () : () -> !hc.idx<"M">
      %ns = hc.buffer_dim %buffer, axis = 1
          : !hc.buffer<f32, ["M", "N"]> -> !hc.idx<"N">
      %one = hc.const<1 : i64> : !hc.idx<"1">
      %row_next = hc.add %row, %one
          : (!hc.idx<"16*$WG0">, !hc.idx<"1">) -> !hc.idx<"1 + 16*$WG0">
      %shape = hc.tuple(%ms, %ns)
          : (!hc.idx<"M">, !hc.idx<"N">) -> tuple<!hc.idx<"M">, !hc.idx<"N">>
      // CHECK: hc.tuple(%[[M]], %[[N]]) : (index, index) -> tuple<index, index>
      %slice = hc.slice_expr(lower = %row upper = %row_next step = %one)
          : (!hc.idx<"16*$WG0">, !hc.idx<"1 + 16*$WG0">, !hc.idx<"1">)
            -> !hc.slice<lower = !hc.idx<"16*$WG0">, upper = !hc.idx<"1 + 16*$WG0">, step = !hc.idx<"1">>
      // CHECK: hc.slice_expr(lower = %[[ROW]] upper = %{{[^ ]+}} step = %{{[^)]+}}) : (index, index, index)
      // CHECK: scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
      hc.for_range %one to %ns step %one
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
  // The cooperative load is unchanged in shape but its LDS write target is
  // now a flat `!hc.ptr<workgroup, f32>` — `hc.alloc` mints the storage,
  // and the per-thread element loop lands a `hc.ptr_offset` + `hc.ptr_store`
  // at the same linear position the chunk loop already computes. The
  // closing `gpu.barrier` is the same publish-to-the-wave fence as before.
  // CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup, f32>
  // CHECK: arith.ceildivui
  // CHECK: scf.for
  // CHECK: arith.cmpi ult
  // CHECK: scf.if
  // CHECK: arith.remui
  // CHECK: arith.divui
  // CHECK: arith.cmpi ult
  // CHECK: arith.andi
  // The kernel-arg load now goes through `hc.ptr_offset` + `hc.ptr_load`
  // against the global pointer — the source-side offset is the per-axis
  // `index * stride` sum, not a `memref.load` index list. Pointer math
  // is hoisted above the in-bounds `scf.if` (it's a no-trap operation
  // on OOB indices); only the actual load and the OOB-pad fallback live
  // inside the if's branches.
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<global, f32>
  // CHECK: scf.if {{.*}} -> (f32)
  // CHECK: hc.ptr_load
  // CHECK-SAME: !hc.ptr<global, f32> -> f32
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<workgroup, f32>
  // CHECK: hc.ptr_store
  // CHECK-SAME: f32, !hc.ptr<workgroup, f32>
  // CHECK: gpu.barrier
  // CHECK: vector.create_mask
  // CHECK-SAME: vector<4x4xi1>
  // The bare-tensor mask materializes as `!hc.ptr<workgroup, i1>` plus a
  // `vector.extract` + `hc.ptr_offset` + `hc.ptr_store` per lane.
  // CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup, i1>
  // CHECK: vector.extract
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<workgroup, i1>
  // CHECK: hc.ptr_store
  // CHECK-SAME: i1, !hc.ptr<workgroup, i1>
  // CHECK: arith.constant dense<0.000000e+00> : vector<4xf32>
  // CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.ptr<workgroup, f32>, index) -> !hc.ptr<workgroup, f32>
  // CHECK: hc.ptr_load %{{.*}} : !hc.ptr<workgroup, f32> -> f32
  // CHECK: vector.insert {{.*}} : f32 into vector<4xf32>
  // CHECK: arith.constant dense<false> : vector<4xi1>
  // CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.ptr<workgroup, i1>, index) -> !hc.ptr<workgroup, i1>
  // CHECK: hc.ptr_load %{{.*}} : !hc.ptr<workgroup, i1> -> i1
  // CHECK: vector.insert {{.*}} : i1 into vector<4xi1>
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
  // CHECK-NOT: memref{{.*}}workgroup
  func.func @tile_memory(%ptr: !hc.ptr<global, f32>,
                         %m: index, %n: index,
                         %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"]>
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
      %lane = hc.idx_apply () : () -> !hc.idx<"$WI0">
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
  // CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup, f32>
  // CHECK: scf.for
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<global, f32>
  // CHECK: scf.if {{.*}} -> (f32)
  // CHECK: hc.ptr_load
  // CHECK-SAME: !hc.ptr<global, f32> -> f32
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<workgroup, f32>
  // CHECK: hc.ptr_store
  // CHECK-SAME: f32, !hc.ptr<workgroup, f32>
  // CHECK: gpu.barrier
  // CHECK: vector.create_mask {{.*}} : vector<4x4xi1>
  // CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup, i1>
  // CHECK: vector.extract
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<workgroup, i1>
  // CHECK: hc.ptr_store
  // CHECK-SAME: i1, !hc.ptr<workgroup, i1>
  // CHECK: arith.constant dense<false> : vector<4x4xi1>
  // CHECK: hc.ptr_load %{{.*}} : !hc.ptr<workgroup, i1> -> i1
  // CHECK: arith.constant dense<0.000000e+00> : vector<4x4xf32>
  // CHECK: hc.ptr_load %{{.*}} : !hc.ptr<workgroup, f32> -> f32
  // CHECK: vector.broadcast
  // CHECK: arith.select
  // CHECK-SAME: vector<4x4xi1>, vector<4x4xf32>
  // CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup, f32>
  // CHECK: hc.ptr_store
  // CHECK-SAME: f32, !hc.ptr<workgroup, f32>
  // CHECK-NOT: hc.load_mask
  // CHECK-NOT: hc.select
  // CHECK-NOT: memref{{.*}}workgroup
  func.func @tensor_mask_and_select(%ptr: !hc.ptr<global, f32>,
                                    %m: index, %n: index,
                                    %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"]>
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
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>
  // Per-lane vector loads from a kernel-arg ptr now use per-element
  // `hc.ptr_offset` + `hc.ptr_load`s. LLVM's SLP recombines the unit-stride
  // lanes; the explicit per-element form keeps unit and non-unit stride
  // paths uniform and sidesteps the bit/byte i1 mismatch
  // `vector.transfer_read` of `vector<Nxi1>` triggered.
  // CHECK: hc.ptr_offset %[[PTR]], %{{.*}} : (!hc.ptr<global, f32>, index) -> !hc.ptr<global, f32>
  // CHECK: %[[V0:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<global, f32> -> f32
  // CHECK: vector.insert %[[V0]], %{{.*}} [0, 0]
  // CHECK: hc.ptr_offset %[[PTR]], %{{.*}} : (!hc.ptr<global, f32>, index) -> !hc.ptr<global, f32>
  // CHECK: hc.ptr_load %{{.*}} : !hc.ptr<global, f32> -> f32
  // CHECK: vector.insert {{.*}} [1, 0]
  // CHECK-NOT: memref.subview
  // CHECK-NOT: vector.transfer_read
  // CHECK-NOT: hc.vload
  func.func @strided_vload(%ptr: !hc.ptr<global, f32>,
                           %m: index, %n: index,
                           %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"]>
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
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[M:[^:]+]]: index,
  // CHECK-SAME: %[[N:[^:]+]]: index,
  // For a stride-2 slice into row axis of extent `M`, the in-bounds count
  // is `ceildiv(M - offset, 2)` rather than `M - offset`. Unit-stride
  // axes still emit the simpler `extent - offset` form. The dim values
  // come straight from the kernel-arg UCC (no `memref.dim` chain).
  // CHECK: %[[REM_M:.*]] = arith.subi %[[M]], %{{.*}} : index
  // CHECK: %[[STEPM1:.*]] = arith.subi %{{.*}}, %{{.*}} : index
  // CHECK: %[[ADJ:.*]] = arith.addi %[[REM_M]], %[[STEPM1]] : index
  // CHECK: %[[ROWSZ:.*]] = arith.divsi %[[ADJ]], %{{.*}} : index
  // CHECK: %[[COLSZ:.*]] = arith.subi %[[N]], %{{.*}} : index
  // CHECK: vector.create_mask %[[ROWSZ]], %[[COLSZ]] : vector<8x1xi1>
  // CHECK-NOT: hc.load_mask
  func.func @strided_load_mask(%ptr: !hc.ptr<global, f32>,
                               %m: index, %n: index,
                               %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"]>
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
  // Kernel-argument ptr destination — masked store path emits per-element
  // `hc.ptr_store_pred` against the global pointer with the per-lane
  // mask carried as a first-class operand.
  // CHECK: vector.extract
  // CHECK-SAME: f32 from vector<4xf32>
  // CHECK: hc.ptr_offset
  // CHECK-SAME: !hc.ptr<global, f32>
  // CHECK: vector.extract
  // CHECK-SAME: i1 from vector<4xi1>
  // CHECK: hc.ptr_store_pred
  // CHECK-SAME: f32, !hc.ptr<global, f32>, i1
  // CHECK-NOT: hc.store
  func.func @masked_vector_store(%ptr: !hc.ptr<global, f32>,
                                 %m: index, %sm: index) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %m, %sm
        : !hc.ptr<global, f32>, index, index
        to !hc.buffer<f32, ["M"]>
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
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[M:[^:]+]]: index,
  // CHECK-SAME: %[[SM:[^:]+]]: index,
  // CHECK-SAME: %[[K:[^:)]+]]: index
  // CHECK: gpu.launch blocks(%[[BX:[^,]+]], %{{[^,]+}}, %{{[^)]+}})
  // CHECK: %[[OFF:.*]] = arith.addi %{{.*}}, %[[K]] : index
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
  // CHECK-NOT: hc.idx_apply
  // CHECK-NOT: hc.pred_apply
  func.func @apply_offset(%ptr: !hc.ptr<global, f32>,
                          %m: index, %sm: index,
                          %ext_k: index) {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %buffer = builtin.unrealized_conversion_cast %ptr, %m, %sm
          : !hc.ptr<global, f32>, index, index
          to !hc.buffer<f32, ["M"]>
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

  // Post-flatten kernel-arg ABI: `hc-flatten-with-layouts` collapses the
  // per-axis dim/stride bundle down to a `(ptr, total, 1)` UCC and folds
  // the access op's multi-index list into a single composed `!hc.idx<expr>`
  // base offset. The lowering's synthesized "lane stride = 1" axis is
  // what fires here instead of `collectAxes`: per-lane addresses come out
  // as `composed + lane * 1`, the kernel-arg `* stride_0 = * 1` rides
  // through `linearizeKernelArgOffset` unchanged, and the result vector
  // is the flat tile materialized one element at a time. The composed
  // offset has to ride a vanilla `index` operand (the `!hc.idx<...>`
  // input the converter produced from the `hc.idx_apply`) — that's how
  // `synthesizePostFlattenAxes` discriminates the post-flatten form from
  // a pre-flatten scalar subscript on a rank-1 buffer. The lane-0
  // `arith.addi (composed + 0)` and `arith.muli (_ * 1)` survive in the
  // unfolded IR because `hc-lower-launch-body` doesn't run canonicalize
  // on its own output; LLVM later folds them.
  // CHECK-LABEL: func.func @post_flatten_vload(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[TOTAL:[^:]+]]: index,
  // CHECK-SAME: %[[OFF:[^:)]+]]: index
  // CHECK: gpu.launch
  // CHECK: %[[AT0:.*]] = arith.addi %[[OFF]], %{{.*}} : index
  // CHECK: %[[SC0:.*]] = arith.muli %[[AT0]], %{{.*}} : index
  // CHECK: hc.ptr_offset %[[PTR]], %[[SC0]] : (!hc.ptr<global, f32>, index) -> !hc.ptr<global, f32>
  // CHECK: %[[E0:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<global, f32> -> f32
  // CHECK: vector.insert %[[E0]], %{{.*}} [0] : f32 into vector<8xf32>
  // CHECK: %[[AT1:.*]] = arith.addi %[[OFF]], %{{.*}} : index
  // CHECK: %[[SC1:.*]] = arith.muli %[[AT1]], %{{.*}} : index
  // CHECK: hc.ptr_offset %[[PTR]], %[[SC1]] : (!hc.ptr<global, f32>, index) -> !hc.ptr<global, f32>
  // CHECK: hc.ptr_load %{{.*}} : !hc.ptr<global, f32> -> f32
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<8xf32>
  // CHECK-NOT: hc.vload
  func.func @post_flatten_vload(%ptr: !hc.ptr<global, f32>,
                                %total: index, %composed: index) {
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %total, %one
        : !hc.ptr<global, f32>, index, index
        to !hc.buffer<f32, ["?"]>
    %off = builtin.unrealized_conversion_cast %composed
        : index to !hc.idx<"composed">
    %eight = hc.const<8 : i64> : !hc.idx<"8">
    %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %vec = hc.vload %buffer[%off], shape %shape
          : (!hc.buffer<f32, ["?"]>, !hc.idx<"composed">, tuple<!hc.idx<"8">>)
            -> !hc.bare_vector<f32, ["8"]>
      gpu.terminator
    }
    return
  }

  // `hc.load_mask` on the same post-flatten form: the composed offset
  // anchors a flat lane walk, the kernel-arg's only dim is the post-
  // flatten `total` element count, and the mask size collapses to
  // `total - composed` (no per-axis ceildiv because the synthesized lane
  // stride is the constant 1 that `maskAxisIsUnitStride` accepts).
  // CHECK-LABEL: func.func @post_flatten_load_mask(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[TOTAL:[^:]+]]: index,
  // CHECK-SAME: %[[OFF:[^:)]+]]: index
  // CHECK: %[[REM:.*]] = arith.subi %[[TOTAL]], %[[OFF]] : index
  // CHECK: vector.create_mask %[[REM]] : vector<8xi1>
  // CHECK-NOT: arith.divsi
  // CHECK-NOT: hc.load_mask
  func.func @post_flatten_load_mask(%ptr: !hc.ptr<global, f32>,
                                    %total: index, %composed: index) {
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %total, %one
        : !hc.ptr<global, f32>, index, index
        to !hc.buffer<f32, ["?"]>
    %off = builtin.unrealized_conversion_cast %composed
        : index to !hc.idx<"composed">
    %eight = hc.const<8 : i64> : !hc.idx<"8">
    %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %mask = hc.load_mask %buffer[%off], shape %shape
          : (!hc.buffer<f32, ["?"]>, !hc.idx<"composed">, tuple<!hc.idx<"8">>)
            -> !hc.bare_vector<!hc.pred, ["8"]>
      gpu.terminator
    }
    return
  }

  // Post-flatten masked store: same synthesized "lane stride = 1" axis
  // applies. Each lane's `hc.ptr_store_pred` takes the per-lane mask bit
  // and the per-lane data element side-by-side; no rank-N coordinate
  // bookkeeping survives flatten.
  // CHECK-LABEL: func.func @post_flatten_masked_store(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[TOTAL:[^:]+]]: index,
  // CHECK-SAME: %[[OFF:[^:)]+]]: index
  // CHECK: vector.extract
  // CHECK-SAME: f32 from vector<4xf32>
  // CHECK: arith.addi %[[OFF]], %{{.*}} : index
  // CHECK: hc.ptr_offset %[[PTR]]
  // CHECK-SAME: !hc.ptr<global, f32>
  // CHECK: vector.extract
  // CHECK-SAME: i1 from vector<4xi1>
  // CHECK: hc.ptr_store_pred
  // CHECK-SAME: f32, !hc.ptr<global, f32>, i1
  // CHECK-NOT: hc.store
  func.func @post_flatten_masked_store(%ptr: !hc.ptr<global, f32>,
                                       %total: index, %composed: index) {
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %total, %one
        : !hc.ptr<global, f32>, index, index
        to !hc.buffer<f32, ["?"]>
    %off = builtin.unrealized_conversion_cast %composed
        : index to !hc.idx<"composed">
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %data_vector = arith.constant dense<1.000000e+00> : vector<4xf32>
      %mask_vector = arith.constant dense<true> : vector<4xi1>
      %data = builtin.unrealized_conversion_cast %data_vector
          : vector<4xf32> to !hc.bare_vector<f32, ["4"]>
      %mask = builtin.unrealized_conversion_cast %mask_vector
          : vector<4xi1> to !hc.bare_vector<!hc.pred, ["4"]>
      hc.store %buffer[%off], %data, mask %mask
          : (!hc.buffer<f32, ["?"]>, !hc.idx<"composed">,
             !hc.bare_vector<f32, ["4"]>, !hc.bare_vector<!hc.pred, ["4"]>) -> ()
      gpu.terminator
    }
    return
  }

  // Post-flatten `hc.generic` with a kernel-arg buffer ins: the outer
  // UCC carries `(ptr, dim_M, dim_N, stride_M, stride_N) ->
  // !hc.buffer<f32, ["M","N"], <layout>>` and a chained
  // multi-output UCC retypes that rank-2 buffer to a rank-1 storage
  // carrier plus idx-typed aux ($STRIDE_*, M, N). `hc-lower-launch-
  // body` resolves the post-flatten 1D buffer back to its underlying
  // global ptr through the chain; the offset attribute, the body, and
  // the bare_vector outs are preserved verbatim for `hc-lower-generic`
  // to consume next.
  // CHECK-LABEL: func.func @post_flatten_generic_buffer_ins(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[M:[^:]+]]: index,
  // CHECK-SAME: %[[N:[^:]+]]: index,
  // CHECK-SAME: %[[SM:[^:]+]]: index,
  // CHECK-SAME: %[[SN:[^:]+]]: index
  // CHECK: %[[OUT:.*]] = hc.generic
  // CHECK-SAME: ins (%[[PTR]] at [{{.*}}] : !hc.ptr<global, f32>)
  // CHECK-SAME: outs (%{{.*}} at [{{.*}}] : !hc.bare_vector<f32, ["8"]>)
  // CHECK-NOT: !hc.buffer<f32, ["?"]>
  func.func @post_flatten_generic_buffer_ins(%ptr: !hc.ptr<global, f32>,
                                             %m: index, %n: index,
                                             %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %buf2d = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"]>
    %sym_sm = hc.const<0 : i64> : !hc.idx<"$STRIDE_0_x">
    %sym_sn = hc.const<0 : i64> : !hc.idx<"$STRIDE_1_x">
    %sym_m = hc.const<0 : i64> : !hc.idx<"M">
    %sym_n = hc.const<0 : i64> : !hc.idx<"N">
    %buf1d, %as_sm, %as_sn, %as_m, %as_n
        = builtin.unrealized_conversion_cast %buf2d
        : !hc.buffer<f32, ["M", "N"]>
        to !hc.buffer<f32, ["?"]>, !hc.idx<"$STRIDE_0_x">,
           !hc.idx<"$STRIDE_1_x">, !hc.idx<"M">, !hc.idx<"N">
    %eight = hc.const<8 : i64> : !hc.idx<"8">
    %one_idx = hc.const<1 : i64> : !hc.idx<"1">
    %vshape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %zeros = hc.vzeros shape %vshape
          : (tuple<!hc.idx<"8">>) -> !hc.bare_vector<f32, ["8"]>
      %res = hc.generic iter (parallel i_0 = %eight : !hc.idx<"8">,
                              parallel i_1 = %one_idx : !hc.idx<"1">)
          ins (%buf1d at [#hc.expr<"$STRIDE_0_x*i_0 + $STRIDE_1_x*i_1">]
                : !hc.buffer<f32, ["?"]>)
          outs (%zeros at [#hc.expr<"i_0 + i_1">]
                : !hc.bare_vector<f32, ["8"]>)
          -> (!hc.bare_vector<f32, ["8"]>) {
        ^bb0(%arg0: f32, %arg1: f32):
          hc.yield %arg0 : f32
      }
      gpu.terminator
    }
    return
  }
}
