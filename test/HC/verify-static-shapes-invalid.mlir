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

// Arity upper bound: number of indices must not exceed the source's
// bind count — that's the layout's `index_syms` size if a layout is
// attached, else the type rank (implicit identity layout).
hc.func @too_many_indices(%buf: !hc.buffer<f32, ["M"]>,
                          %i: !hc.idx<"0">) {
  %four = hc.const<4 : i64> : !hc.idx<"4">
  %shape = hc.tuple(%four) : (!hc.idx<"4">) -> tuple<!hc.idx<"4">>
  // expected-error @+1 {{has 2 index operand(s) for source bind count 1}}
  %x = hc.load %buf[%i, %i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.idx<"0">,
         tuple<!hc.idx<"4">>)
        -> !hc.tensor<f32, ["4"]>
  hc.return
}

// -----

// Selector-bearing layout: `index_syms = ["i0", "i1", "lane"]` admits
// up to rank + 1 = 3 indices. Passing a 4th overshoots; the attached
// note spells out the selector tail so the error report is self
// contained.
hc.func @selector_layout_too_many_indices(
    %t: !hc.tensor<f16, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1", "lane"], params = {}, storage_size = #hc.expr<"d1">, offset = #hc.expr<"i1">>>,
    %i: !hc.idx<"i">, %j: !hc.idx<"j">, %lane: !hc.idx<"lane">,
    %extra: !hc.idx<"extra">,
    %m: !hc.idx<"M">, %k: !hc.idx<"K">) {
  %shape = hc.tuple(%m, %k)
      : (!hc.idx<"M">, !hc.idx<"K">) -> tuple<!hc.idx<"M">, !hc.idx<"K">>
  // expected-error @+2 {{has 4 index operand(s) for source bind count 3}}
  // expected-note @+1 {{source layout's index_syms is 3 (rank 2 + 1 selector(s))}}
  %v = hc.vload %t[%i, %j, %lane, %extra], shape %shape
      : (!hc.tensor<f16, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1", "lane"], params = {}, storage_size = #hc.expr<"d1">, offset = #hc.expr<"i1">>>,
         !hc.idx<"i">, !hc.idx<"j">, !hc.idx<"lane">, !hc.idx<"extra">,
         tuple<!hc.idx<"M">, !hc.idx<"K">>)
        -> !hc.bare_vector<f16, ["K"]>
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
