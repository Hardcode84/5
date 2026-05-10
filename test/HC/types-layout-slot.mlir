// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// All five shaped types accept an optional trailing `#hc.layout<...>`.
// Absent layout is the v0 identity contract; explicit layouts round-trip
// through canonical ixsimpl form.
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: @absent_legacy_form
// CHECK-SAME: !hc.buffer<f32, ["M", "N"]>
// CHECK-SAME: !hc.tensor<f16, ["M", "N"]>
// CHECK-SAME: !hc.vector<f32, ["8"]>
// CHECK-SAME: !hc.bare_tensor<f16, ["M", "N"]>
// CHECK-SAME: !hc.bare_vector<f16, ["16"]>
func.func @absent_legacy_form(%a: !hc.buffer<f32, ["M", "N"]>,
                              %b: !hc.tensor<f16, ["M", "N"]>,
                              %c: !hc.vector<f32, ["8"]>,
                              %d: !hc.bare_tensor<f16, ["M", "N"]>,
                              %e: !hc.bare_vector<f16, ["16"]>) {
  return
}

// CHECK-LABEL: @explicit_layout_buffer
// CHECK: !hc.buffer<f16, ["M", "K"], <
// CHECK-SAME: shape_syms = ["M", "K"]
// CHECK-SAME: index_syms = ["i", "j"]
// CHECK-SAME: storage_size = #hc.expr<"K*M">
// CHECK-SAME: offset = #hc.expr<"j + K*i">
func.func @explicit_layout_buffer(
    %a: !hc.buffer<f16, ["M", "K"],
                   #hc.layout<shape_syms = ["M", "K"],
                              index_syms = ["i", "j"],
                              params = {},
                              storage_size = #hc.expr<"M * K">,
                              offset = #hc.expr<"i * K + j">>>) {
  return
}

// CHECK-LABEL: @explicit_layout_tensor
// CHECK: !hc.tensor<f16, ["M", "K"], <
// CHECK-SAME: storage_size = #hc.expr<"K*M">
func.func @explicit_layout_tensor(
    %a: !hc.tensor<f16, ["M", "K"],
                   #hc.layout<shape_syms = ["M", "K"],
                              index_syms = ["i", "j"],
                              params = {},
                              storage_size = #hc.expr<"M * K">,
                              offset = #hc.expr<"i * K + j">>>) {
  return
}

// CHECK-LABEL: @explicit_layout_vector
// CHECK: !hc.vector<f32, ["8"], <
// CHECK-SAME: storage_size = #hc.expr<"8">
func.func @explicit_layout_vector(
    %a: !hc.vector<f32, ["8"],
                   #hc.layout<shape_syms = ["8"],
                              index_syms = ["i"],
                              params = {},
                              storage_size = #hc.expr<"8">,
                              offset = #hc.expr<"i">>>) {
  return
}

// CHECK-LABEL: @explicit_layout_bare_tensor
// CHECK: !hc.bare_tensor<f16, ["M", "K"], <
// CHECK-SAME: storage_size = #hc.expr<"K*M">
func.func @explicit_layout_bare_tensor(
    %a: !hc.bare_tensor<f16, ["M", "K"],
                        #hc.layout<shape_syms = ["M", "K"],
                                   index_syms = ["i", "j"],
                                   params = {},
                                   storage_size = #hc.expr<"M * K">,
                                   offset = #hc.expr<"i * K + j">>>) {
  return
}

// CHECK-LABEL: @explicit_layout_bare_vector
// CHECK: !hc.bare_vector<f16, ["16"], <
// CHECK-SAME: storage_size = #hc.expr<"16">
func.func @explicit_layout_bare_vector(
    %a: !hc.bare_vector<f16, ["16"],
                        #hc.layout<shape_syms = ["16"],
                                   index_syms = ["i"],
                                   params = {},
                                   storage_size = #hc.expr<"16">,
                                   offset = #hc.expr<"i">>>) {
  return
}

// Padded layout with a bound parameter survives the round-trip; ixsimpl
// canonicalizes the offset/size payloads.
// CHECK-LABEL: @padded_layout
// CHECK: !hc.buffer<f16, ["M", "K"], <
// CHECK-SAME: params = {row_stride = #hc.expr<"4 + K">}
// CHECK-SAME: storage_size = #hc.expr<"M*row_stride">
// CHECK-SAME: offset = #hc.expr<"j + i*row_stride">
func.func @padded_layout(
    %a: !hc.buffer<f16, ["M", "K"],
                   #hc.layout<shape_syms = ["M", "K"],
                              index_syms = ["i", "j"],
                              params = {row_stride = #hc.expr<"K + 4">},
                              storage_size = #hc.expr<"M * row_stride">,
                              offset = #hc.expr<"i * row_stride + j">>>) {
  return
}

// `?` is the surface spelling for `#hc.dyn`, the sentinel a shape entry
// uses when its size isn't derivable from the in-IR symbol set
// (canonical case: a buffer arg whose backing storage extent is owned
// by the host descriptor). Round-trips on every shaped shell that
// carries a `ShapeAttr`.
// CHECK-LABEL: @dyn_size_round_trip
// CHECK-SAME: !hc.buffer<f32, ["?"]>
// CHECK-SAME: !hc.tensor<f16, ["?", "M"]>
// CHECK-SAME: !hc.bare_tensor<f16, ["M", "?"]>
func.func @dyn_size_round_trip(%a: !hc.buffer<f32, ["?"]>,
                               %b: !hc.tensor<f16, ["?", "M"]>,
                               %c: !hc.bare_tensor<f16, ["M", "?"]>) {
  return
}

// Two equivalent spellings of the same layout (W vs 0+W, i vs i+0) hash
// to the same LayoutAttr in the type uniquer, so the two operands
// declared with each spelling print as the same type. This is the
// equality property the canonicalize pass leans on.
// CHECK-LABEL: @canonical_equality
// CHECK-SAME: %arg0: !hc.tensor<f32, ["W"], <shape_syms = ["W"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"W">, offset = #hc.expr<"i">>>
// CHECK-SAME: %arg1: !hc.tensor<f32, ["W"], <shape_syms = ["W"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"W">, offset = #hc.expr<"i">>>
func.func @canonical_equality(
    %a: !hc.tensor<f32, ["W"],
                   #hc.layout<shape_syms = ["W"], index_syms = ["i"],
                              params = {}, storage_size = #hc.expr<"W">,
                              offset = #hc.expr<"i">>>,
    %b: !hc.tensor<f32, ["W"],
                   #hc.layout<shape_syms = ["W"], index_syms = ["i"],
                              params = {}, storage_size = #hc.expr<"0 + W">,
                              offset = #hc.expr<"i + 0">>>) {
  return
}
