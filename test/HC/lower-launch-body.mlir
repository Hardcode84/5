// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Workgroup-AS storage (`!hc.bare_tensor`) lowers to flat
// `!hc.ptr<workgroup, T>` instead of `memref<..., workgroup>`. Kernel-arg
// buffers come in as `(ptr, dim, stride)` bundles UCC'd to
// `!hc.buffer<...>`; the per-axis dim values feed `hc.buffer_dim` and
// the stride values feed offset linearization for `hc.ptr_offset` +
// `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`.
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

  // `idx_apply` and other apply-driven offset lowering paths route
  // `!hc.idx<sym>` references through the post-flatten retype UCC.
  // The launch-body pass binds each implicit sym to the kernel-arg
  // bundle's original `index`-typed input (shape dim or stride per the
  // frontend's default strided layout convention), so the resulting
  // arith chain reads straight off the bundle inputs instead of
  // bouncing through an `idx<sym> -> index` UCC. That UCC chain has
  // an HC-typed intermediate the `gpu-to-rocdl` block-arg conversion
  // can't see across, so any survivor would block LLVM translation
  // downstream — pin the short-circuit so a regression that
  // re-materializes the cast fails here, not at the GPU surface.
  // CHECK-LABEL: func.func @bundle_sym_short_circuits_idx_to_index(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>,
  // CHECK-SAME: %[[M:[^:]+]]: index,
  // CHECK-SAME: %[[N:[^:]+]]: index,
  // CHECK-SAME: %[[SM:[^:]+]]: index,
  // CHECK-SAME: %[[SN:[^:]+]]: index
  // The apply chain composes `$STRIDE_0_x*$WI0 + $STRIDE_1_x*1`. With
  // the bundle short-circuit the `$STRIDE_*_x` operands of the
  // resulting arith chain are the bundle inputs themselves (`%[[SM]]`
  // / `%[[SN]]`) rather than UCCs re-materialized from the retype
  // output.
  // CHECK: gpu.launch
  // CHECK-DAG: arith.muli %{{.*}}, %[[SM]] : index
  // CHECK-DAG: arith.addi %{{.*}}, %[[SN]] : index
  // No `idx<...> to index` cast — bundle short-circuit handles it.
  // CHECK-NOT: builtin.unrealized_conversion_cast {{.*}} : !hc.idx<{{.*}}> to index
  func.func @bundle_sym_short_circuits_idx_to_index(
      %ptr: !hc.ptr<global, f32>,
      %m: index, %n: index, %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %buf2d = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_x*i0 + $STRIDE_1_x*i1">>>
    %buf1d, %as_sm, %as_sn, %as_m, %as_n
        = builtin.unrealized_conversion_cast %buf2d
        : !hc.buffer<f32, ["M", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_x*i0 + $STRIDE_1_x*i1">>>
        to !hc.buffer<f32, ["?"]>, !hc.idx<"$STRIDE_0_x">,
           !hc.idx<"$STRIDE_1_x">, !hc.idx<"M">, !hc.idx<"N">
    %eight = hc.const<8 : i64> : !hc.idx<"8">
    %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      // `$STRIDE_*_x` symbol references inside the offset expression
      // must resolve to the bundle's `%sm` / `%sn` index inputs
      // through the apply lowering; the retype UCC's idx-typed
      // outputs would be the long way around.
      %composed = hc.idx_apply (%as_sm as "$STRIDE_0_x", %as_sn as "$STRIDE_1_x")
          : (!hc.idx<"$STRIDE_0_x">, !hc.idx<"$STRIDE_1_x">) -> !hc.idx<"$STRIDE_0_x*$WI0 + $STRIDE_1_x*1">
      %vec = hc.vload %buf1d[%composed], shape %shape
          : (!hc.buffer<f32, ["?"]>,
             !hc.idx<"$STRIDE_0_x*$WI0 + $STRIDE_1_x*1">,
             tuple<!hc.idx<"8">>)
            -> !hc.bare_vector<f32, ["8"]>
      gpu.terminator
    }
    return
  }

  // `group.work_offset[k]` lowers to a `hc.idx_apply` whose expression
  // names `$WO[k]` — the upper-left corner of the workgroup's tile in
  // the work grid. Launch-body materialises the binding as
  // `arith.muli %block_id_k, %block_size_k`, mirroring what every
  // other launch-geometry sym does (`$WG` / `$WI` / `$WGS` bind off
  // the gpu.launch operands directly; `$WO` is the only one that
  // needs an actual arithmetic op). Axes 0 and 1 in the same kernel
  // pin that the binding is per-axis, not just axis 0.
  // CHECK-LABEL: func.func @work_offset_binding
  // CHECK-SAME: %[[GX:[^:]+]]: index,
  // CHECK-SAME: %[[GY:[^:]+]]: index,
  // CHECK-SAME: %[[SX:[^:]+]]: index,
  // CHECK-SAME: %[[SY:[^:]+]]: index
  // CHECK: gpu.launch blocks(%[[BX:[^,]+]], %[[BY:[^,]+]], %{{[^)]+}})
  // CHECK-DAG: %[[WO0:.+]] = arith.muli %[[BX]], %[[SX]] : index
  // CHECK-DAG: %[[WO1:.+]] = arith.muli %[[BY]], %[[SY]] : index
  // CHECK-NOT: hc.idx_apply
  func.func @work_offset_binding(%gx: index, %gy: index,
                                 %sx: index, %sy: index) {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gxr = %gx, %gyr = %gy, %gzr = %c1)
               threads(%tx, %ty, %tz) in (%sxr = %sx, %syr = %sy, %szr = %c1) {
      %off0 = hc.idx_apply () : () -> !hc.idx<"$WO0">
      %off1 = hc.idx_apply () : () -> !hc.idx<"$WO1">
      gpu.terminator
    }
    return
  }

  // `hc.zeros` of a `!hc.bare_tensor` lowers to a workgroup LDS
  // allocation; the resulting `ptr<workgroup, T> -> bare_tensor` UCC
  // would otherwise leak past launch-body as the ins of a downstream
  // `hc.generic`. `AdaptGenericOp` resolves the bare_tensor ins to
  // the underlying ptr at the boundary so `hc-lower-generic` reads
  // the LDS through the same ptr-typed access path the kernel-arg
  // ptr / explicit `hc.alloc` ins already use — no UCC walk-back
  // needed at consume time.
  // CHECK-LABEL: func.func @workgroup_bare_tensor_ins_swap
  // CHECK: gpu.launch
  // CHECK: %[[LDS:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
  // CHECK: hc.generic
  // CHECK-SAME: ins (%[[LDS]] at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
  // CHECK-NOT: !hc.bare_tensor<f32, ["8"]>
  func.func @workgroup_bare_tensor_ins_swap(%dst: !hc.ptr<global, f32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gxr = %c1, %gyr = %c1, %gzr = %c1)
               threads(%tx, %ty, %tz) in (%sxr = %c8, %syr = %c1, %szr = %c1) {
      %eight = hc.idx_apply () : () -> !hc.idx<"8">
      %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
      %tile = hc.zeros shape %shape
          : (tuple<!hc.idx<"8">>) -> !hc.bare_tensor<f32, ["8"]>
      hc.generic
          iter (parallel i = %eight : !hc.idx<"8">)
          ins (%tile at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["8"]>)
          outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
          -> () {
      ^bb0(%v: f32, %init: f32):
        hc.yield %v : f32
      }
      gpu.terminator
    }
    return
  }
}
