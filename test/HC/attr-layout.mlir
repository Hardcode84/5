// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Round-trip plus ixsimpl-canonical-equality smoke for #hc.layout<...>.
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: @padded_2d
// CHECK-SAME: test.layout = #hc.layout<
// CHECK-SAME: shape_syms = ["W", "H"]
// CHECK-SAME: index_syms = ["i", "j"]
// CHECK-SAME: params = {row_stride = #hc.expr<"4 + H">}
// CHECK-SAME: storage_size = #hc.expr<"W*row_stride">
// CHECK-SAME: offset = #hc.expr<"j + i*row_stride">
func.func @padded_2d() attributes {
  test.layout = #hc.layout<
    shape_syms = ["W", "H"],
    index_syms = ["i", "j"],
    params = {row_stride = #hc.expr<"H + 4">},
    storage_size = #hc.expr<"W * row_stride">,
    offset = #hc.expr<"i * row_stride + j">
  >
} {
  return
}

// CHECK-LABEL: @dense_2d
// CHECK-SAME: test.layout = #hc.layout<
// CHECK-SAME: shape_syms = ["M", "N"]
// CHECK-SAME: index_syms = ["i", "j"]
// CHECK-SAME: params = {}
// CHECK-SAME: storage_size = #hc.expr<"M*N">
// CHECK-SAME: offset = #hc.expr<"j + N*i">
func.func @dense_2d() attributes {
  test.layout = #hc.layout<
    shape_syms = ["M", "N"],
    index_syms = ["i", "j"],
    params = {},
    storage_size = #hc.expr<"M * N">,
    offset = #hc.expr<"i * N + j">
  >
} {
  return
}

// Rank-zero layouts (scalars) are admitted: empty name lists, constant
// storage_size and offset.
// CHECK-LABEL: @rank_zero
// CHECK-SAME: shape_syms = []
// CHECK-SAME: index_syms = []
// CHECK-SAME: params = {}
// CHECK-SAME: storage_size = #hc.expr<"1">
// CHECK-SAME: offset = #hc.expr<"0">
func.func @rank_zero() attributes {
  test.layout = #hc.layout<shape_syms = [], index_syms = [], params = {},
                           storage_size = #hc.expr<"1">,
                           offset = #hc.expr<"0">>
} {
  return
}

// Field order is normalized at print time: `params` ahead of `storage_size`
// regardless of input order.
// CHECK-LABEL: @reordered_input
// CHECK-SAME: test.layout = #hc.layout<shape_syms = ["W"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"W">, offset = #hc.expr<"i">>
func.func @reordered_input() attributes {
  test.layout = #hc.layout<offset = #hc.expr<"i">,
                           storage_size = #hc.expr<"W">,
                           params = {},
                           index_syms = ["i"],
                           shape_syms = ["W"]>
} {
  return
}

// Two source spellings whose `#hc.expr` payloads are ixsimpl-equal hash to
// the same LayoutAttr in the attribute uniquer; both keys therefore print
// identically. This is the type-uniquing-as-equality property that lets the
// canonicalization pass match layouts structurally.
// CHECK-LABEL: @canonical_equality
// CHECK-SAME: test.a = #hc.layout<shape_syms = ["W"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"W">, offset = #hc.expr<"i">>
// CHECK-SAME: test.b = #hc.layout<shape_syms = ["W"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"W">, offset = #hc.expr<"i">>
func.func @canonical_equality() attributes {
  test.a = #hc.layout<shape_syms = ["W"], index_syms = ["i"], params = {},
                      storage_size = #hc.expr<"W">,
                      offset = #hc.expr<"i">>,
  test.b = #hc.layout<shape_syms = ["W"], index_syms = ["i"], params = {},
                      storage_size = #hc.expr<"0 + W">,
                      offset = #hc.expr<"i + 0">>
} {
  return
}

// Layout grammar is intentionally unrestricted in v1: any expression
// ixsimpl can canonicalize is admitted. Non-affine and product-of-symbols
// payloads are accepted at parse time; downstream consumers (canonicalize,
// flatten) get to decide what they can prove about the offset.
// CHECK-LABEL: @nonlinear_payload
// CHECK-SAME: storage_size = #hc.expr<"H*W">
// CHECK-SAME: offset = #hc.expr<"i*j">
func.func @nonlinear_payload() attributes {
  test.layout = #hc.layout<shape_syms = ["W", "H"], index_syms = ["i", "j"],
                           params = {},
                           storage_size = #hc.expr<"W * H">,
                           offset = #hc.expr<"i * j">>
} {
  return
}
