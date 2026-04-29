// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-decompose-shaped-values -split-input-file -verify-diagnostics %s

// -----

// expected-error @+1 {{failed to legalize operation 'func.func'}}
func.func @semantic_block_arg(%arg0: !hc.tensor<f32, ["4"]>) {
  return
}

// -----

func.func @unsupported_shaped_result(%buf: !hc.buffer<f32, ["M"]>,
                                     %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  %tile = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  // expected-error @+1 {{failed to legalize operation 'hc.store'}}
  hc.store %buf[%i], %tile
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.tensor<f32, ["4"]>) -> ()
  return
}
