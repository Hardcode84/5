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
// CHECK: func.func @use_progressive_types
// CHECK-SAME: !hc.undef
// CHECK-SAME: !hc.idx
// CHECK-SAME: !hc.idx<"1 + M">
// CHECK-SAME: !hc.pred
// CHECK-SAME: !hc.pred<"M - N < 0">
// CHECK-SAME: !hc.slice
// CHECK-SAME: !hc.vector<f32, ["TileM", "TileN"]>
// CHECK-SAME: !hc.group<work_shape = #hc.shape<["M"]>, subgroup_size = 64 : i32>
// CHECK: func.func @use_scope_and_effects
// CHECK-SAME: effects = #hc<effects pure>
// CHECK-SAME: scope = #hc.scope<"WorkItem">
// CHECK: hc.kernel @wmma_matmul
// CHECK: requirements = <[#hc.pred<"-1 + M >= 0">, #hc.pred<"Mod(N, 32) == 0">]> {
// CHECK: hc.subgroup_region captures = ["lhs", "rhs"] {
// CHECK: hc.workitem_region {
// CHECK: hc.return
// CHECK: hc.kernel @full_kernel(%arg0: !hc.buffer<f32, ["M", "N"]>, %arg1: !hc.buffer<f32, ["M", "N"]>)
// CHECK-SAME: attributes {
// CHECK-SAME: group_shape = #hc.shape<["32", "1"]>
// CHECK-SAME: literals = ["WMMA_M", "WMMA_K"]
// CHECK-SAME: subgroup_size = 32 : i32
// CHECK-SAME: work_shape = #hc.shape<["M", "N"]>
// Launch-geometry-only kernel (no signature): confirms the two are orthogonal.
// CHECK: hc.kernel @geometry_only attributes {
// CHECK-SAME: subgroup_size = 64 : i32
// CHECK-SAME: work_shape = #hc.shape<["M"]>
// CHECK: hc.func @tile_helper {
// CHECK: hc.func @typed_with_return(%arg0: i32) -> i32 {
// CHECK-NEXT: hc.return %arg0 : i32

module {
  func.func @use_types(
      %arg0: !hc.buffer<f32, ["M", "N"]>,
      %arg1: !hc.tensor<f16, ["WG0 + 1", "WG1 + 1"]>) {
    return
  }

  func.func @legacy_shape_syntax(%arg0: !hc.buffer<f32, #hc.shape<["TileM", "TileN"]>>) {
    return
  }

  // Round-trips every new progressive-typing surface type and an example of
  // the bare/pinned spellings for `!hc.idx` and `!hc.pred`.
  func.func @use_progressive_types(
      %u: !hc.undef,
      %i: !hc.idx,
      %iexpr: !hc.idx<#hc.expr<"M + 1">>,
      %p: !hc.pred,
      %ppred: !hc.pred<#hc.pred<"M < N">>,
      %s: !hc.slice,
      %v: !hc.vector<f32, ["TileM", "TileN"]>,
      %g: !hc.group<work_shape = #hc.shape<["M"]>, subgroup_size = 64 : i32>) {
    return
  }

  func.func @use_scope_and_effects()
      attributes {
        scope = #hc.scope<"WorkItem">,
        effects = #hc<effects pure>
      } {
    return
  }

  hc.kernel @wmma_matmul requirements = #hc.constraints<[#hc.pred<"M >= 1">, #hc.pred<"Mod(N, 32) == 0">]> {
    hc.subgroup_region captures = ["lhs", "rhs"] {
      hc.workitem_region {
        hc.return
      }
    }
  }

  // Launch-geometry attrs + explicit signature ride together on the same
  // op; the attrs travel in the attr-dict so adding more of them later does
  // not change the keyword syntax.
  hc.kernel @full_kernel(%a: !hc.buffer<f32, ["M", "N"]>,
                         %b: !hc.buffer<f32, ["M", "N"]>)
      attributes {
        work_shape = #hc.shape<["M", "N"]>,
        group_shape = #hc.shape<["32", "1"]>,
        subgroup_size = 32 : i32,
        literals = ["WMMA_M", "WMMA_K"]
      } {
    hc.return
  }

  // Launch geometry without a signature: the two are independent attr groups,
  // so a signature-less kernel can still carry `work_shape`/`subgroup_size`.
  hc.kernel @geometry_only attributes {
    work_shape = #hc.shape<["M"]>,
    subgroup_size = 64 : i32
  } {
    hc.return
  }

  hc.func @tile_helper {
    hc.return
  }

  // Signatured func with a matching `hc.return`: the legal round-trip for
  // the return-parity verifier.
  hc.func @typed_with_return(%a: i32) -> i32 {
    hc.return %a : i32
  }
}
