// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-verify-static-shapes -split-input-file %s | FileCheck %s

// CHECK-LABEL: hc.func @valid_loads_and_allocators
// CHECK: hc.load
// CHECK: hc.vload
// CHECK: hc.zeros
// CHECK: hc.vfull
hc.func @valid_loads_and_allocators(
    %buf: !hc.buffer<f32, ["M", "N"]>,
    %tensor: !hc.tensor<f16, ["16", "16"]>,
    %i: !hc.idx<"0">,
    %sl: !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">>,
    %fill: f32) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %eight = hc.const<8 : i64> : !hc.idx<"8">
  %shape2 = hc.tuple(%four, %eight)
      : (!hc.idx<"4">, !hc.idx<"8">)
        -> tuple<!hc.idx<"4">, !hc.idx<"8">>
  %shape1 = hc.tuple(%eight)
      : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
  %loaded = hc.load %buf[%i, %i], shape %shape2
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"0">, !hc.idx<"0">,
         tuple<!hc.idx<"4">, !hc.idx<"8">>)
        -> !hc.tensor<f32, ["4", "8"]>
  %vloaded = hc.vload %tensor[%sl, %i], shape %shape1
      : (!hc.tensor<f16, ["16", "16"]>,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">>,
         !hc.idx<"0">, tuple<!hc.idx<"8">>)
        -> !hc.vector<f16, ["8"]>
  %zeros = hc.zeros shape %shape2
      : (tuple<!hc.idx<"4">, !hc.idx<"8">>)
        -> !hc.tensor<f32, ["4", "8"]>
  %vfull = hc.vfull %fill, shape %shape1
      : (f32, tuple<!hc.idx<"8">>) -> !hc.vector<f32, ["8"]>
  hc.return
}
