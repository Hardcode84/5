// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// The handle-backed symbolic attrs print using the canonical ixsimpl form, not
// necessarily the exact source spelling from the input below.
// CHECK: func.func @use_types(
// CHECK: !hc.buffer<f32, ["M", "N"]>
// CHECK: !hc.tensor<f16, ["1 + WG0", "1 + WG1"]>
// CHECK: func.func @legacy_shape_syntax(%arg0: !hc.buffer<f32, ["TileM", "TileN"]>)
// CHECK: hc.kernel "wmma_matmul"
// CHECK: requirements = <[#hc.pred<"-1 + M >= 0">, #hc.pred<"Mod(N, 32) == 0">]> {
// CHECK: hc.subgroup_region captures = ["lhs", "rhs"] {
// CHECK: hc.workitem_region {
// CHECK: hc.return
// CHECK: hc.func "tile_helper" {

module {
  func.func @use_types(
      %arg0: !hc.buffer<f32, ["M", "N"]>,
      %arg1: !hc.tensor<f16, ["WG0 + 1", "WG1 + 1"]>) {
    return
  }

  func.func @legacy_shape_syntax(%arg0: !hc.buffer<f32, #hc.shape<["TileM", "TileN"]>>) {
    return
  }

  hc.kernel "wmma_matmul" requirements = #hc.constraints<[#hc.pred<"M >= 1">, #hc.pred<"Mod(N, 32) == 0">]> {
    hc.subgroup_region captures = ["lhs", "rhs"] {
      hc.workitem_region {
        hc.return
      }
    }
  }

  hc.func "tile_helper" {
    hc.return
  }
}
