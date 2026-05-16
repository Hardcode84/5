// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Verifier diagnostics for `hc.buffer_view`'s `unit_axes` attribute.
// Positions are output-rank coordinates; the op rejects negative,
// duplicate, and out-of-range entries before downstream passes ever
// see the malformed view. Output rank = (residual indices) +
// (unit_axes entries).
//
// RUN: hc-opt %s -verify-diagnostics -split-input-file

hc.func @unit_axes_negative(%t: !hc.tensor<f32, ["16", "16"]>) -> !hc.undef {
  %full = hc.slice_expr() : () -> !hc.undef
  // expected-error@+1 {{unit_axes entries must be non-negative, got -1}}
  %v = hc.buffer_view %t[%full, %full] {unit_axes = array<i64: -1>}
      : (!hc.tensor<f32, ["16", "16"]>, !hc.undef, !hc.undef) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

hc.func @unit_axes_out_of_range(%t: !hc.tensor<f32, ["16", "16"]>) -> !hc.undef {
  %full = hc.slice_expr() : () -> !hc.undef
  // expected-error@+1 {{unit_axes entry 3 is out of range for output rank 3 (= 2 residual indices + 1 unit axes)}}
  %v = hc.buffer_view %t[%full, %full] {unit_axes = array<i64: 3>}
      : (!hc.tensor<f32, ["16", "16"]>, !hc.undef, !hc.undef) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

hc.func @unit_axes_duplicate(%t: !hc.tensor<f32, ["16", "16"]>) -> !hc.undef {
  %full = hc.slice_expr() : () -> !hc.undef
  // expected-error@+1 {{unit_axes entries must be unique; 1 repeats}}
  %v = hc.buffer_view %t[%full, %full] {unit_axes = array<i64: 1, 1>}
      : (!hc.tensor<f32, ["16", "16"]>, !hc.undef, !hc.undef) -> !hc.undef
  hc.return %v : !hc.undef
}
