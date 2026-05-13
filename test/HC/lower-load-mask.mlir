// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-load-mask`. The pass runs after
// `hc-lower-kernels-to-gpu-launch` (which plants the kernel-arg
// `(ptr, dim, stride)` UCC bundle) and before `hc-flatten-with-layouts`
// (which composes per-axis offsets into a single 1D `#hc.expr` and
// erases the per-axis bounds). For each `hc.load_mask` against the
// kernel-arg bundle we synthesise per-axis in-bounds counts off the
// multi-dim source dims, then plant `hc.mask_from_sizes` carrying those
// Index-typed sizes; the downstream launch-body lowering turns it into
// `vector.create_mask` + a bare-vector / -tensor materialisation. Without
// this pre-flatten snapshot the post-flatten 1D kernel-arg view's
// placeholder dim feeds a `0 - composed_offset` extent into the mask
// size, clamping every bound to all-false and silently masking off the
// kernel's stores.
//
// RUN: hc-opt --hc-lower-load-mask %s --split-input-file | FileCheck %s

// Strided row axis (step=2) plus a unit-stride column axis: the row
// in-bounds count emits `ceildiv(M - lower, 2)` (`arith.subi` /
// `arith.addi` / `arith.divsi`); the column emits the simpler
// `N - lower`. The original `hc.load_mask` goes away — the planted
// `hc.mask_from_sizes` carries the two sizes plus the multi-dim shape
// attribute for the downstream lowering to use as the `vector.create_mask`
// shape.
// CHECK-LABEL: func.func @strided_load_mask_to_mask_from_sizes(
// CHECK-SAME: %[[M:[^:]+]]: index,
// CHECK-SAME: %[[N:[^:]+]]: index,
// CHECK-DAG: %[[ROW_REM:.*]] = arith.subi %[[M]], %{{.*}} : index
// CHECK-DAG: %[[STEPM1:.*]] = arith.subi %{{.*}}, %{{.*}} : index
// CHECK-DAG: %[[ROW_ADJ:.*]] = arith.addi %[[ROW_REM]], %[[STEPM1]] : index
// CHECK-DAG: %[[ROW_SZ:.*]] = arith.divsi %[[ROW_ADJ]], %{{.*}} : index
// CHECK-DAG: %[[COL_SZ:.*]] = arith.subi %[[N]], %{{.*}} : index
// CHECK: hc.mask_from_sizes(%[[ROW_SZ]], %[[COL_SZ]]) shape [8, 1]
// CHECK-SAME: : !hc.bare_vector<!hc.pred, ["8", "1"]>
// CHECK-NOT: hc.load_mask
func.func @strided_load_mask_to_mask_from_sizes(%ptr: !hc.ptr<global, f32>,
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

// -----

// Bare-tensor source (workgroup-staged tile): the per-axis extents are
// the source bare tensor's static shape, not kernel-arg dims. The pass
// still applies — we use `arith.constant`s for the extents and emit the
// same `hc.mask_from_sizes` carrier so the launch-body LDS spill path
// can pick the op up later.
// CHECK-LABEL: func.func @bare_tensor_load_mask_to_mask_from_sizes(
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[ROW_LOWER_IDX:.*]] = builtin.unrealized_conversion_cast {{.*}} : !hc.idx<"4"> to index
// CHECK-DAG: %[[ROW_REM:.*]] = arith.subi %[[C16]], %[[ROW_LOWER_IDX]] : index
// CHECK-DAG: %[[COL_LOWER_IDX:.*]] = builtin.unrealized_conversion_cast {{.*}} : !hc.idx<"2"> to index
// CHECK-DAG: %[[COL_REM:.*]] = arith.subi %{{.*}}, %[[COL_LOWER_IDX]] : index
// CHECK: hc.mask_from_sizes
// CHECK-SAME: shape [4, 4]
// CHECK-SAME: : !hc.bare_tensor<!hc.pred, ["4", "4"]>
// CHECK-NOT: hc.load_mask
func.func @bare_tensor_load_mask_to_mask_from_sizes(
    %tile: !hc.bare_tensor<f16, ["16", "16"]>) {
  %two = hc.const<2 : i64> : !hc.idx<"2">
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %six = hc.const<6 : i64> : !hc.idx<"6">
  %eight = hc.const<8 : i64> : !hc.idx<"8">
  %rows = hc.slice_expr(lower = %four upper = %eight)
      : (!hc.idx<"4">, !hc.idx<"8">)
        -> !hc.slice<lower = !hc.idx<"4">, upper = !hc.idx<"8">>
  %cols = hc.slice_expr(lower = %two upper = %six)
      : (!hc.idx<"2">, !hc.idx<"6">)
        -> !hc.slice<lower = !hc.idx<"2">, upper = !hc.idx<"6">>
  %shape = hc.tuple(%four, %four)
      : (!hc.idx<"4">, !hc.idx<"4">) -> tuple<!hc.idx<"4">, !hc.idx<"4">>
  %mask = hc.load_mask %tile[%rows, %cols], shape %shape
      : (!hc.bare_tensor<f16, ["16", "16"]>,
         !hc.slice<lower = !hc.idx<"4">, upper = !hc.idx<"8">>,
         !hc.slice<lower = !hc.idx<"2">, upper = !hc.idx<"6">>,
         tuple<!hc.idx<"4">, !hc.idx<"4">>)
        -> !hc.bare_tensor<!hc.pred, ["4", "4"]>
  return
}

// -----

// Source isn't a kernel-arg UCC bundle or a static bare tensor —
// just a bare `!hc.buffer<>` argument with a symbolic shape. The pass
// leaves the op alone for the legacy launch-body handler to pick up
// (and for `hc-flatten-with-layouts` to compose its offsets first).
// CHECK-LABEL: func.func @non_resolvable_source_stays
// CHECK: hc.load_mask
// CHECK-NOT: hc.mask_from_sizes
func.func @non_resolvable_source_stays(%buffer: !hc.buffer<f32, ["M", "N"]>) {
  %zero = hc.const<0 : i64> : !hc.idx<"0">
  %one = hc.const<1 : i64> : !hc.idx<"1">
  %sixteen = hc.const<16 : i64> : !hc.idx<"16">
  %eight = hc.const<8 : i64> : !hc.idx<"8">
  %rows = hc.slice_expr(lower = %zero upper = %sixteen)
      : (!hc.idx<"0">, !hc.idx<"16">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">>
  %col = hc.slice_expr(lower = %zero upper = %one)
      : (!hc.idx<"0">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>
  %shape = hc.tuple(%eight, %one)
      : (!hc.idx<"8">, !hc.idx<"1">) -> tuple<!hc.idx<"8">, !hc.idx<"1">>
  %mask = hc.load_mask %buffer[%rows, %col], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">>,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>,
         tuple<!hc.idx<"8">, !hc.idx<"1">>)
        -> !hc.bare_vector<!hc.pred, ["8", "1"]>
  return
}
