// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-verify-static-shapes -split-input-file -verify-diagnostics %s

// -----

hc.func @shape_not_resolved(%buf: !hc.buffer<f32, ["M"]>, %shape: !hc.undef,
                            %i: !hc.idx<"0">) {
  // expected-error @+1 {{shape operand is still !hc.undef; expected a concrete tuple of static !hc.idx dimensions}}
  %x = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.undef)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

hc.func @dynamic_shape_dim(%buf: !hc.buffer<f32, ["M"]>, %i: !hc.idx<"0">,
                           %dynamic: !hc.idx) {
  %shape = hc.tuple(%dynamic) : (!hc.idx) -> tuple<!hc.idx>
  // expected-error @+1 {{shape dimension #0 is dynamic; expected pinned !hc.idx expression}}
  %x = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

hc.func @non_idx_shape_dim(%buf: !hc.buffer<f32, ["M"]>, %i: !hc.idx<"0">,
                           %dim: f32) {
  %shape = hc.tuple(%dim) : (f32) -> tuple<f32>
  // expected-error @+1 {{shape dimension #0 must be !hc.idx with a static expression, got 'f32'}}
  %x = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<f32>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

hc.func @shape_result_mismatch(%buf: !hc.buffer<f32, ["M"]>,
                               %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %eight = hc.const<8 : i64> : !hc.idx<"8">
  %shape = hc.tuple(%four, %eight)
      : (!hc.idx<"4">, !hc.idx<"8">)
        -> tuple<!hc.idx<"4">, !hc.idx<"8">>
  // expected-error @+1 {{shape operand #hc.shape<["4", "8"]> does not match result type shape #hc.shape<["4"]>}}
  %x = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">,
         tuple<!hc.idx<"4">, !hc.idx<"8">>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

hc.func @too_many_indices(%buf: !hc.buffer<f32, ["M"]>,
                          %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  // expected-error @+1 {{has 2 index operand(s) for rank-1 source}}
  %x = hc.load %buf[%i, %i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.idx<"0">,
         tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

hc.func @bad_index_type(%buf: !hc.buffer<f32, ["M"]>, %i: f32) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  // expected-error @+1 {{index #0 must be !hc.idx, !hc.slice, or builtin integer/index, got 'f32'}}
  %x = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, f32, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

hc.func @allocator_shape_mismatch {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %eight = hc.const<8 : i64> : !hc.idx<"8">
  %shape = hc.tuple(%four, %eight)
      : (!hc.idx<"4">, !hc.idx<"8">)
        -> tuple<!hc.idx<"4">, !hc.idx<"8">>
  // expected-error @+1 {{shape operand #hc.shape<["4", "8"]> does not match result type shape #hc.shape<["4"]>}}
  %x = hc.vzeros shape %shape
      : (tuple<!hc.idx<"4">, !hc.idx<"8">>) -> !hc.vector<f32, ["4"]>
  hc.return
}

// -----

hc.func @wrong_result_shell(%buf: !hc.buffer<f32, ["M"]>,
                            %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  // expected-error @+1 {{expected result type !hc.vector, got '!hc.tensor<f32, ["4"]>'}}
  %x = hc.vload %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}
