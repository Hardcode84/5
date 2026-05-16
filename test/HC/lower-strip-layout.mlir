// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-strip-layout`. The pass rewrites the
// user-marked layout-drop boundary `hc.strip_layout` into the
// body-driven `hc.generic` surface so post-flatten codegen sees a
// uniform op family. See the design in `doc/layouts.md` and the
// op's documentation on `include/hc/IR/HCOps.td`.
//
// RUN: hc-opt --hc-lower-strip-layout %s --split-input-file | FileCheck %s

// Non-injective source on a bare vector — the canonical per-lane
// WMMA-fragment shape: layout's `storage_size > product(shape)`.
// Rewrite emits an `hc.vzeros` sized to the layout-less result
// + an `hc.generic` whose iter set spans the result shape, ins/outs
// carry identity per-axis offsets, body forwards the source element.
// Flatten's `composeAccessOffsetExpr` later substitutes the iter syms
// into the source layout's `offset` at `index_syms[k]`, producing the
// gather offset post-1D.
// CHECK-LABEL: func.func @strip_non_injective
// CHECK-DAG: %[[K:.+]] = hc.idx_apply () : () -> !hc.idx<"K">
// CHECK: %[[SH:.+]] = hc.tuple(%[[K]])
// CHECK: %[[FILL:.+]] = hc.vzeros shape %[[SH]]
// CHECK-SAME: -> !hc.bare_vector<f32, ["K"]>
// CHECK: %[[OUT:.+]] = hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[K]] : !hc.idx<"K">)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i_0">]
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">]
// CHECK: ^bb0(%[[BV:.+]]: f32, %{{.+}}: f32):
// CHECK:   hc.yield %[[BV]] : f32
// CHECK-NOT: hc.strip_layout
func.func @strip_non_injective(
    %src: !hc.bare_vector<f32, ["K"],
      #hc.layout<shape_syms = ["d0"], index_syms = ["i0"],
                 params = {}, storage_size = #hc.expr<"256">,
                 offset = #hc.expr<"i0 * 16">>>)
    -> !hc.bare_vector<f32, ["K"]> {
  %out = hc.strip_layout %src
      : !hc.bare_vector<f32, ["K"],
          #hc.layout<shape_syms = ["d0"], index_syms = ["i0"],
                     params = {}, storage_size = #hc.expr<"256">,
                     offset = #hc.expr<"i0 * 16">>>
      -> !hc.bare_vector<f32, ["K"]>
  return %out : !hc.bare_vector<f32, ["K"]>
}

// -----

// Layout-less / identical source and result: defensive no-op the user
// may emit against producers that may or may not carry a layout.
// Rewrite collapses to a direct forward; no allocator, no generic.
// CHECK-LABEL: func.func @strip_noop
// CHECK-NOT: hc.vzeros
// CHECK-NOT: hc.generic
// CHECK-NOT: hc.strip_layout
// CHECK: return %arg0 : !hc.bare_vector<f32, ["K"]>
func.func @strip_noop(%src: !hc.bare_vector<f32, ["K"]>)
    -> !hc.bare_vector<f32, ["K"]> {
  %out = hc.strip_layout %src
      : !hc.bare_vector<f32, ["K"]> -> !hc.bare_vector<f32, ["K"]>
  return %out : !hc.bare_vector<f32, ["K"]>
}

// -----

// Rank-2 source with a non-injective layout — same shape, two iter
// syms. Confirms the per-axis offset emission stays rank-correct so
// flatten's offset compose sees `[i_0, i_1]` to substitute into the
// layout's index syms.
// CHECK-LABEL: func.func @strip_rank2
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply () : () -> !hc.idx<"B">
// CHECK: %[[FILL:.+]] = hc.vzeros shape
// CHECK-SAME: -> !hc.bare_vector<f32, ["A", "B"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[A]] : !hc.idx<"A">, parallel i_1 = %[[B]] : !hc.idx<"B">)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">]
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">, #hc.expr<"i_1">]
// CHECK-NOT: hc.strip_layout
func.func @strip_rank2(
    %src: !hc.bare_vector<f32, ["A", "B"],
      #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"],
                 params = {}, storage_size = #hc.expr<"d0 * d1 + 64">,
                 offset = #hc.expr<"i0 * 8 + i1">>>)
    -> !hc.bare_vector<f32, ["A", "B"]> {
  %out = hc.strip_layout %src
      : !hc.bare_vector<f32, ["A", "B"],
          #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"],
                     params = {}, storage_size = #hc.expr<"d0 * d1 + 64">,
                     offset = #hc.expr<"i0 * 8 + i1">>>
      -> !hc.bare_vector<f32, ["A", "B"]>
  return %out : !hc.bare_vector<f32, ["A", "B"]>
}
