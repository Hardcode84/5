// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-memory-access`: per-tile memory ops route through
// the kernel-arg UCC bundle and emit `hc.ptr_offset` +
// `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`. Standalone surface
// for the post-scalar / post-shaped-constants slice.
//
// RUN: hc-opt %s --hc-lower-launch-memory-access | FileCheck %s

module {
  // Slice indices flow through the type converter's index cast before
  // they feed the per-lane offset arithmetic. Regression for the
  // `arith.addi(!hc.idx, index)` bug where `collectSliceAxis` returned
  // raw `!hc.idx`-typed slice fields. CHECK pins the arith ops at the
  // index level.
  // CHECK-LABEL: func.func @vload_slice_kernel_arg(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>
  // CHECK-NOT: arith.addi %{{[^,]*}}, %{{.*}} : !hc.idx
  // CHECK: %[[OFF:.+]] = hc.ptr_offset %[[PTR]],
  // CHECK: hc.ptr_load %[[OFF]]
  // CHECK-NOT: hc.vload
  func.func @vload_slice_kernel_arg(%ptr: !hc.ptr<global, f32>,
                                    %m: index, %n: index,
                                    %sm: index, %sn: index) {
    %c1 = arith.constant 1 : index
    %buffer = builtin.unrealized_conversion_cast %ptr, %m, %n, %sm, %sn
        : !hc.ptr<global, f32>, index, index, index, index
        to !hc.buffer<f32, ["M", "N"]>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %zero = builtin.unrealized_conversion_cast %c1 : index to !hc.idx<"0">
      %one  = builtin.unrealized_conversion_cast %c1 : index to !hc.idx<"1">
      %two  = builtin.unrealized_conversion_cast %c1 : index to !hc.idx<"2">
      %eight = builtin.unrealized_conversion_cast %c1 : index to !hc.idx<"8">
      %sixteen = builtin.unrealized_conversion_cast %c1 : index to !hc.idx<"16">
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
}
