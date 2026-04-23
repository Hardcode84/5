// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK: func.func @use_types(
// CHECK: !hc.buffer<f32, #hc.shape<[#hc.expr<"M">, #hc.expr<"N">]>>
// CHECK: !hc.tensor<f16, #hc.shape<[#hc.expr<"WG0 + 1">, #hc.expr<"WG1 + 1">]>>
// CHECK: hc.kernel "wmma_matmul"
// CHECK: requirements = <[#hc.pred<"M >= 1">, #hc.pred<"Mod(N, 32) == 0">]> {
// CHECK: hc.subgroup_region captures = ["lhs", "rhs"] {
// CHECK: hc.workitem_region {
// CHECK: hc.return
// CHECK: hc.func "tile_helper" {

module {
  func.func @use_types(
      %arg0: !hc.buffer<f32, #hc.shape<[#hc.expr<"M">, #hc.expr<"N">]>>,
      %arg1: !hc.tensor<f16, #hc.shape<[#hc.expr<"WG0 + 1">, #hc.expr<"WG1 + 1">]>>) {
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
