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
// CHECK-SAME: !hc.slice<lower = !hc.idx<"M">, step = !hc.idx>
// CHECK-SAME: !hc.vector<f32, ["TileM", "TileN"]>
// CHECK-SAME: !hc.bare_tensor<!hc.pred, ["TileM", "TileN"]>
// CHECK-SAME: !hc.bare_vector<f32, ["TileM"]>
// CHECK-SAME: !hc.buffer<f32, []>
// CHECK-SAME: !hc.group<work_shape = #hc.shape<["M"]>, subgroup_size = #hc.expr<"64">>
// CHECK-SAME: tuple<!hc.idx<"M">, f32>
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
// CHECK-SAME: bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "M", "N"]
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
// CHECK: hc.func @apply_bound {
// CHECK: hc.idx_apply () : () -> !hc.idx<"$WI0">
// CHECK: hc.pred_apply () : () -> !hc.pred<"-32 + $WI0 < 0">
// CHECK: hc.func @apply_ops(%[[I:[^:]+]]: index, %[[K:[^:]+]]: !hc.idx<"K">, %[[J:[^:]+]]: index) {
// CHECK: hc.idx_apply (%[[I]] as "i", %[[K]] as "K", %[[J]] as "j")
// CHECK-SAME: : (index, !hc.idx<"K">, index) -> !hc.idx<"K + i*j">
// CHECK: hc.idx_apply () : () -> !hc.idx<"$WG0">
// CHECK: hc.pred_apply (%[[I]] as "i", %[[K]] as "K")
// CHECK-SAME: : (index, !hc.idx<"K">) -> !hc.pred<"-K + i < 0">

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
      %typed_slice: !hc.slice<lower = !hc.idx<"M">, step = !hc.idx>,
      %v: !hc.vector<f32, ["TileM", "TileN"]>,
      %bare_mask: !hc.bare_tensor<!hc.pred, ["TileM", "TileN"]>,
      %bare_vec: !hc.bare_vector<f32, ["TileM"]>,
      %scalar_buffer: !hc.buffer<f32, []>,
      %g: !hc.group<work_shape = #hc.shape<["M"]>, subgroup_size = #hc.expr<"64">>,
      %t: tuple<!hc.idx<"M">, f32>) {
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
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "M", "N"],
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

  hc.func @apply_bound {
    %idx = hc.idx_apply () : () -> !hc.idx<"$WI0">
    %pred = hc.pred_apply () : () -> !hc.pred<"$WI0 < 32">
    hc.return
  }

  // `idx_apply` / `pred_apply` carry an explicit symbol-to-operand
  // binding alongside the symbolic expr / pred. Listed names get
  // resolved from the matching operand; unlisted names (e.g. the
  // launch-geometry `$WG0` below) stay ambient and are bound by the
  // launch-body lowering. Operands are either `index` or
  // `!hc.idx<...>` per `HC_SymBindingValueType`.
  hc.func @apply_ops(%i: index, %k: !hc.idx<"K">, %j: index) {
    %off = hc.idx_apply (%i as "i", %k as "K", %j as "j")
         : (index, !hc.idx<"K">, index) -> !hc.idx<"i*j + K">
    %wg = hc.idx_apply () : () -> !hc.idx<"$WG0">
    %p = hc.pred_apply (%i as "i", %k as "K")
       : (index, !hc.idx<"K">) -> !hc.pred<"i < K">
    hc.return
  }

  // `hc.workitem_region` with results: the verifier accepts the
  // collective-lift relationship between the yield and the region
  // result. Both the pre-flatten "append suffix dims" form and the
  // post-flatten "storage equals yield_storage * product(suffix)"
  // form must round-trip cleanly so the verifier survives both
  // sides of `hc-flatten-with-layouts`.

  // CHECK: hc.func @collective_lift_preflatten_vector
  // CHECK: hc.workitem_region
  // CHECK-SAME: -> (!hc.bare_vector<f32, ["8", "1", "32", "1"]>)
  // CHECK: hc.yield {{.*}} : !hc.bare_vector<f32, ["8", "1"]>
  hc.func @collective_lift_preflatten_vector(
      %lane: !hc.bare_vector<f32, ["8", "1"]>) {
    %r = hc.workitem_region
        -> (!hc.bare_vector<f32, ["8", "1", "32", "1"]>) {
    ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                           subgroup_size = #hc.expr<"32">>):
      hc.yield %lane : !hc.bare_vector<f32, ["8", "1"]>
    }
    hc.return
  }

  // CHECK: hc.func @collective_lift_postflatten_vector
  // CHECK: hc.workitem_region
  // CHECK-SAME: -> (!hc.bare_vector<f32, ["256"]>)
  // CHECK: hc.yield {{.*}} : !hc.bare_vector<f32, ["8"]>
  hc.func @collective_lift_postflatten_vector(
      %lane: !hc.bare_vector<f32, ["8"]>) {
    %r = hc.workitem_region -> (!hc.bare_vector<f32, ["256"]>) {
    ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                           subgroup_size = #hc.expr<"32">>):
      hc.yield %lane : !hc.bare_vector<f32, ["8"]>
    }
    hc.return
  }

  // Symbolic post-flatten storage: `S * 32 * 1` canonicalizes to `32*S`
  // through the ixsimpl store, and the verifier compares the lift's
  // hash-consed handle against the result's storage handle.
  // CHECK: hc.func @collective_lift_postflatten_symbolic
  // CHECK: hc.workitem_region
  // CHECK-SAME: -> (!hc.bare_vector<f32, ["32*S"]>)
  // CHECK: hc.yield {{.*}} : !hc.bare_vector<f32, ["S"]>
  hc.func @collective_lift_postflatten_symbolic(
      %lane: !hc.bare_vector<f32, ["S"]>) {
    %r = hc.workitem_region -> (!hc.bare_vector<f32, ["32*S"]>) {
    ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                           subgroup_size = #hc.expr<"32">>):
      hc.yield %lane : !hc.bare_vector<f32, ["S"]>
    }
    hc.return
  }

  // Tuple yields lift element-wise both pre- and post-flatten; the
  // verifier walks the tuple recursively in both regimes.
  // CHECK: hc.func @collective_lift_postflatten_tuple
  // CHECK: hc.workitem_region
  // CHECK-SAME: -> (tuple<!hc.bare_vector<f32, ["256"]>, !hc.bare_vector<!hc.pred, ["256"]>>)
  hc.func @collective_lift_postflatten_tuple(
      %data: !hc.bare_vector<f32, ["8"]>,
      %mask: !hc.bare_vector<!hc.pred, ["8"]>) {
    %tup = hc.tuple(%data, %mask)
        : (!hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>)
        -> tuple<!hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>>
    %r = hc.workitem_region
        -> (tuple<!hc.bare_vector<f32, ["256"]>,
                  !hc.bare_vector<!hc.pred, ["256"]>>) {
    ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                           subgroup_size = #hc.expr<"32">>):
      hc.yield %tup
          : tuple<!hc.bare_vector<f32, ["8"]>,
                  !hc.bare_vector<!hc.pred, ["8"]>>
    }
    hc.return
  }

  // `hc.subgroup_region` runs the same lift logic; the suffix is the
  // workgroup-tile-size divided by the subgroup_size, so for a
  // group_shape of [32, 1] with subgroup_size 32 the suffix is `[1]`
  // and post-flatten storage equals the yield storage.
  // CHECK: hc.func @collective_lift_subgroup_postflatten
  // CHECK: hc.subgroup_region
  // CHECK-SAME: -> (!hc.bare_vector<f32, ["8"]>)
  hc.func @collective_lift_subgroup_postflatten(
      %lane: !hc.bare_vector<f32, ["8"]>) {
    %r = hc.subgroup_region -> (!hc.bare_vector<f32, ["8"]>) {
    ^bb0(%sg: !hc.subgroup<group_shape = #hc.shape<["32", "1"]>,
                           subgroup_size = #hc.expr<"32">>):
      hc.yield %lane : !hc.bare_vector<f32, ["8"]>
    }
    hc.return
  }
}
