// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-load-store-to-generic`. The pass rewrites the
// per-tile memory ops `hc.load`, `hc.vload`, and `hc.store` into the
// body-driven `hc.generic` surface so post-flatten codegen has a
// single op family to lower. See the design in `doc/layouts.md`.
//
// RUN: hc-opt --hc-load-store-to-generic %s --split-input-file | FileCheck %s

// Plain rank-2 buffer load: iter syms `i_0`, `i_1`; the buffer
// operand carries `[i + i_0, j + i_1]`; the synthesised init has
// identity offsets. Body forwards the loaded element through
// `hc.yield`.
// CHECK-LABEL: func.func @load_basic
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply () : () -> !hc.idx<"B">
// CHECK: %[[SH:.+]] = hc.tuple(%[[A]], %[[B]])
// CHECK: %[[FILL:.+]] = hc.zeros shape %[[SH]] {{.*}} -> !hc.tensor<f32, ["A", "B"]>
// CHECK: %[[OUT:.+]] = hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[A]] : !hc.idx<"A">, parallel i_1 = %[[B]] : !hc.idx<"B">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i + i_0">, #hc.expr<"i_1 + j">] : !hc.buffer<f32, ["M", "N"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.tensor<f32, ["A", "B"]>)
// CHECK: ^bb0(%[[BV:.+]]: f32, %{{.+}}: f32):
// CHECK:   hc.yield %[[BV]] : f32
// CHECK-NOT: hc.load
func.func @load_basic(%buf: !hc.buffer<f32, ["M", "N"]>,
                      %i: !hc.idx<"i">, %j: !hc.idx<"j">) -> !hc.tensor<f32, ["A", "B"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %b = hc.const<1 : i64> : !hc.idx<"B">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"A">, !hc.idx<"B">) -> tuple<!hc.idx<"A">, !hc.idx<"B">>
  %t = hc.load %buf[%i, %j], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"i">, !hc.idx<"j">,
         tuple<!hc.idx<"A">, !hc.idx<"B">>) -> !hc.tensor<f32, ["A", "B"]>
  return %t : !hc.tensor<f32, ["A", "B"]>
}

// -----

// `hc.vload` on a tensor source: same shape, but the synthesised
// init is `hc.vzeros` because the result is a vector. Confirms the
// init-op picker keys off the result type, not the source.
// CHECK-LABEL: func.func @vload_basic
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply () : () -> !hc.idx<"B">
// CHECK: %[[FILL:.+]] = hc.vzeros shape %{{[^ ]+}} {{.*}} -> !hc.vector<f32, ["A", "B"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[A]] : !hc.idx<"A">, parallel i_1 = %[[B]] : !hc.idx<"B">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i + i_0">, #hc.expr<"i_1 + j">] : !hc.tensor<f32, ["M", "N"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.vector<f32, ["A", "B"]>)
// CHECK-NOT: hc.vload
func.func @vload_basic(%src: !hc.tensor<f32, ["M", "N"]>,
                       %i: !hc.idx<"i">, %j: !hc.idx<"j">) -> !hc.vector<f32, ["A", "B"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %b = hc.const<1 : i64> : !hc.idx<"B">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"A">, !hc.idx<"B">) -> tuple<!hc.idx<"A">, !hc.idx<"B">>
  %v = hc.vload %src[%i, %j], shape %shape
      : (!hc.tensor<f32, ["M", "N"]>, !hc.idx<"i">, !hc.idx<"j">,
         tuple<!hc.idx<"A">, !hc.idx<"B">>) -> !hc.vector<f32, ["A", "B"]>
  return %v : !hc.vector<f32, ["A", "B"]>
}

// -----

// Uniform layout (`shape_syms.size() == index_syms.size()`): access
// provides one operand per index_sym; each contributes `base + step *
// iter_k` to the per-axis offset, and flatten composes them against
// the layout's offset formula downstream. This is the only access
// shape the rewriter understands — no selector-bind sugar anymore.
// CHECK-LABEL: func.func @vload_uniform_layout
// CHECK: %[[FILL:[^ ]+]] = hc.vzeros shape %{{[^ ]+}} {{.*}} -> !hc.bare_vector<f16, ["A", "B"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"A">, parallel i_1 = %{{[^ ]+}} : !hc.idx<"B">)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i + i_0">, #hc.expr<"i_1 + j">]
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.bare_vector<f16, ["A", "B"]>)
// CHECK-NOT: hc.vload
func.func @vload_uniform_layout(
    %t: !hc.tensor<f16, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0*d1">, offset = #hc.expr<"i0*d1 + i1">>>,
    %i: !hc.idx<"i">, %j: !hc.idx<"j">) -> !hc.bare_vector<f16, ["A", "B"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %b = hc.const<1 : i64> : !hc.idx<"B">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"A">, !hc.idx<"B">) -> tuple<!hc.idx<"A">, !hc.idx<"B">>
  %v = hc.vload %t[%i, %j], shape %shape
      : (!hc.tensor<f16, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0*d1">, offset = #hc.expr<"i0*d1 + i1">>>,
         !hc.idx<"i">, !hc.idx<"j">, tuple<!hc.idx<"A">, !hc.idx<"B">>)
        -> !hc.bare_vector<f16, ["A", "B"]>
  return %v : !hc.bare_vector<f16, ["A", "B"]>
}

// -----

// Store into a buffer: ptr-out `hc.generic` with no SSA result.
// `%src` has identity offsets; `%dst` carries the multi-index
// addressing. Body still forwards the source element so the
// downstream `hc.ptr_store` materialisation has the value to write.
// CHECK-LABEL: func.func @store_buffer
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply () : () -> !hc.idx<"B">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[A]] : !hc.idx<"A">, parallel i_1 = %[[B]] : !hc.idx<"B">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.tensor<f32, ["A", "B"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i + i_0">, #hc.expr<"i_1 + j">] : !hc.buffer<f32, ["M", "N"]>)
// CHECK-NOT: hc.yield {{.*}}, {{.*}} :
// CHECK: ^bb0(%[[SV:.+]]: f32, %{{.+}}: f32):
// CHECK:   hc.yield %[[SV]] : f32
// CHECK-NOT: hc.store
func.func @store_buffer(%dst: !hc.buffer<f32, ["M", "N"]>,
                        %src: !hc.tensor<f32, ["A", "B"]>,
                        %i: !hc.idx<"i">, %j: !hc.idx<"j">) {
  hc.store %dst[%i, %j], %src
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"i">, !hc.idx<"j">,
         !hc.tensor<f32, ["A", "B"]>) -> ()
  return
}

// -----

// Bare-typed result (post-decompose): `hc.zeros` happily produces a
// `bare_tensor` because its result type comes from the create call,
// not the inferer. Confirms the rewrite stays consistent across
// pre- and post-decompose IR.
// CHECK-LABEL: func.func @load_bare
// CHECK: %[[FILL:.+]] = hc.zeros shape %{{[^ ]+}} {{.*}} -> !hc.bare_tensor<f32, ["A"]>
// CHECK: hc.generic
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">] : !hc.bare_tensor<f32, ["A"]>)
func.func @load_bare(%buf: !hc.buffer<f32, ["M"]>, %i: !hc.idx<"i">)
    -> !hc.bare_tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %r = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"i">, tuple<!hc.idx<"A">>)
        -> !hc.bare_tensor<f32, ["A"]>
  return %r : !hc.bare_tensor<f32, ["A"]>
}

// -----

// Empty index list: whole-tensor / whole-buffer access. The
// memory-side offset collapses to identity over the iter syms (no
// addressing addend). Body and structure stay the same.
// CHECK-LABEL: func.func @load_empty_indices
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">] : !hc.buffer<f32, ["A"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["A"]>)
func.func @load_empty_indices(%buf: !hc.buffer<f32, ["A"]>) -> !hc.tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %r = hc.load %buf[], shape %shape
      : (!hc.buffer<f32, ["A"]>, tuple<!hc.idx<"A">>) -> !hc.tensor<f32, ["A"]>
  return %r : !hc.tensor<f32, ["A"]>
}

// -----

// Untyped index operand (raw `index`): the rewriter has no
// symbolic name to bind, so it bails and the op survives for the
// downstream lowering or the type-only flatten retype.
// CHECK-LABEL: func.func @load_untyped_index_falls_through
// CHECK: hc.load
// CHECK-NOT: hc.generic
func.func @load_untyped_index_falls_through(%buf: !hc.buffer<f32, ["M"]>, %i: index)
    -> !hc.tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %r = hc.load %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, index, tuple<!hc.idx<"A">>) -> !hc.tensor<f32, ["A"]>
  return %r : !hc.tensor<f32, ["A"]>
}

// -----

// Partial indices: `X[gid[0]:]` against a rank-2 `X` lowers to a
// single slice for the leading axis with the trailing axis
// implicit-full. The funnel pads the missing axis with a default
// `base=0, step=1` slot so the per-axis offset list rank-matches the
// tile and the leading axis carries the slice's `i + i_0` offset.
// CHECK-LABEL: func.func @load_partial_indices
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply () : () -> !hc.idx<"B">
// CHECK: hc.generic iter (parallel i_0 = %[[A]] : !hc.idx<"A">, parallel i_1 = %[[B]] : !hc.idx<"B">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i + i_0">, #hc.expr<"i_1">] : !hc.buffer<f32, ["M", "N"]>)
// CHECK-NOT: hc.load
func.func @load_partial_indices(%buf: !hc.buffer<f32, ["M", "N"]>,
                                  %i: !hc.idx<"i">)
    -> !hc.tensor<f32, ["A", "B"]> {
  %slice = hc.slice_expr(lower = %i)
      : (!hc.idx<"i">) -> !hc.slice<lower = !hc.idx<"i">>
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %b = hc.const<1 : i64> : !hc.idx<"B">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"A">, !hc.idx<"B">) -> tuple<!hc.idx<"A">, !hc.idx<"B">>
  %r = hc.load %buf[%slice], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>, !hc.slice<lower = !hc.idx<"i">>,
         tuple<!hc.idx<"A">, !hc.idx<"B">>) -> !hc.tensor<f32, ["A", "B"]>
  return %r : !hc.tensor<f32, ["A", "B"]>
}

// -----

// Masked store: the mask rides as a second `ins` slot with identity
// offsets (same shape as `%src`, both tile-local). The body now has
// three block args (`src`, `mask`, `dst`) and terminates with
// `hc.yield_predicated`, which `hc-lower-generic` routes to
// `hc.ptr_store_pred` at the dst's ins-slot offset — masked-out
// lanes leave the existing dst contents in place.
// CHECK-LABEL: func.func @store_masked
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK: hc.generic iter (parallel i_0 = %[[A]] : !hc.idx<"A">{{[^)]*}})
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">] : !hc.bare_tensor<f32, ["A"]>, %{{.+}} at [#hc.expr<"i_0">] : !hc.bare_tensor<!hc.pred, ["A"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i + i_0">] : !hc.buffer<f32, ["M"]>)
// CHECK: ^bb0(%[[SV:.+]]: f32, %[[MV:.+]]: !hc.pred, %{{.+}}: f32):
// CHECK:   hc.yield_predicated %[[SV]] mask %[[MV]] : (f32), (!hc.pred)
// CHECK-NOT: hc.store
func.func @store_masked(%dst: !hc.buffer<f32, ["M"]>,
                        %src: !hc.bare_tensor<f32, ["A"]>,
                        %mask: !hc.bare_tensor<!hc.pred, ["A"]>,
                        %i: !hc.idx<"i">) {
  hc.store %dst[%i], %src, mask %mask
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"i">, !hc.bare_tensor<f32, ["A"]>,
         !hc.bare_tensor<!hc.pred, ["A"]>) -> ()
  return
}

// -----

// Rank-2 masked store: the mask carries the same shape as the source
// tile and rides as a second ins slot with identity offsets. The dst
// keeps the per-axis composed `base + i_k` offsets, same as the
// unmasked rank-2 store.
// CHECK-LABEL: func.func @store_masked_rank2
// CHECK: hc.generic iter (parallel i_0 = {{[^,]+}}, parallel i_1 = {{[^)]+}})
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.bare_tensor<f32, ["A", "B"]>, %{{.+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.bare_tensor<!hc.pred, ["A", "B"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i + i_0">, #hc.expr<"i_1 + j">] : !hc.buffer<f32, ["M", "N"]>)
// CHECK: ^bb0(%[[SV:.+]]: f32, %[[MV:.+]]: !hc.pred, %{{.+}}: f32):
// CHECK:   hc.yield_predicated %[[SV]] mask %[[MV]] : (f32), (!hc.pred)
func.func @store_masked_rank2(%dst: !hc.buffer<f32, ["M", "N"]>,
                              %src: !hc.bare_tensor<f32, ["A", "B"]>,
                              %mask: !hc.bare_tensor<!hc.pred, ["A", "B"]>,
                              %i: !hc.idx<"i">, %j: !hc.idx<"j">) {
  hc.store %dst[%i, %j], %src, mask %mask
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"i">, !hc.idx<"j">,
         !hc.bare_tensor<f32, ["A", "B"]>,
         !hc.bare_tensor<!hc.pred, ["A", "B"]>) -> ()
  return
}

// -----

// Bare-vector source + bare-vector mask: same rewrite shape — the
// mask rides as a second ins slot with identity offsets. Confirms
// the rewrite isn't pinned to bare_tensor on the src/mask pair.
// CHECK-LABEL: func.func @store_masked_bare_vector
// CHECK: hc.generic iter (parallel i_0 = {{[^)]+}})
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">] : !hc.bare_vector<f32, ["A"]>, %{{.+}} at [#hc.expr<"i_0">] : !hc.bare_vector<!hc.pred, ["A"]>)
// CHECK: ^bb0(%[[SV:.+]]: f32, %[[MV:.+]]: !hc.pred, %{{.+}}: f32):
// CHECK:   hc.yield_predicated %[[SV]] mask %[[MV]] : (f32), (!hc.pred)
func.func @store_masked_bare_vector(%dst: !hc.buffer<f32, ["M"]>,
                                    %src: !hc.bare_vector<f32, ["A"]>,
                                    %mask: !hc.bare_vector<!hc.pred, ["A"]>,
                                    %i: !hc.idx<"i">) {
  hc.store %dst[%i], %src, mask %mask
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"i">,
         !hc.bare_vector<f32, ["A"]>,
         !hc.bare_vector<!hc.pred, ["A"]>) -> ()
  return
}

// -----

// Tensor / bare_tensor dst is workgroup-shared LDS storage with
// in-place mutation semantics, but the IR types it value-typed.
// Producing an SSA result + propagating it through every subsequent
// use of `%dst` is non-local; the v0 rewrite bails and a separate
// slice handles tensor-dst stores.
// CHECK-LABEL: func.func @store_tensor_falls_through
// CHECK: hc.store
// CHECK-NOT: hc.generic
func.func @store_tensor_falls_through(%dst: !hc.tensor<f32, ["A"]>,
                                      %src: !hc.tensor<f32, ["A"]>) {
  hc.store %dst[], %src
      : (!hc.tensor<f32, ["A"]>, !hc.tensor<f32, ["A"]>) -> ()
  return
}

// -----

// Pre-inference IR with `!hc.undef` operands: the rewriter has no
// shape to source bounds from and leaves the op alone.
// CHECK-LABEL: func.func @load_undef_falls_through
// CHECK: hc.load
// CHECK-NOT: hc.generic
func.func @load_undef_falls_through(%buf: !hc.undef, %i: !hc.undef, %sh: !hc.undef)
    -> !hc.undef {
  %r = hc.load %buf[%i], shape %sh
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// `hc.load_mask` is the bare-predicate companion to `hc.load`; it
// stays a primitive in this slice — only data movement gets
// rewritten here.
// CHECK-LABEL: func.func @load_mask_untouched
// CHECK: hc.load_mask
// CHECK-NOT: hc.generic
func.func @load_mask_untouched(%buf: !hc.buffer<f32, ["M"]>, %i: !hc.idx<"i">)
    -> !hc.bare_tensor<!hc.pred, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %m = hc.load_mask %buf[%i], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"i">, tuple<!hc.idx<"A">>)
        -> !hc.bare_tensor<!hc.pred, ["A"]>
  return %m : !hc.bare_tensor<!hc.pred, ["A"]>
}

// -----

// Slice-indexed load (unit step): the slice carries pinned
// `!hc.idx<"lo">` lower and `!hc.idx<"1">` step. Per-axis offset is
// `lo + i_0` — step = const 1 folds out, matching the scalar-idx
// printed form. This is the case the WMMA cooperative-load path emits
// when both fragment dims are walked as `range(0, k)`-style slices.
// CHECK-LABEL: func.func @load_slice_unit_step
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{.+}} : !hc.idx<"A">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0 + lo">] : !hc.buffer<f32, ["M"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["A"]>)
// CHECK-NOT: hc.load
func.func @load_slice_unit_step(%buf: !hc.buffer<f32, ["M"]>,
                                %lo: !hc.idx<"lo">, %hi: !hc.idx<"hi">,
                                %step: !hc.idx<"1">)
    -> !hc.tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %s = hc.slice_expr(lower = %lo upper = %hi step = %step)
      : (!hc.idx<"lo">, !hc.idx<"hi">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"1">>
  %r = hc.load %buf[%s], shape %shape
      : (!hc.buffer<f32, ["M"]>,
         !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"1">>,
         tuple<!hc.idx<"A">>) -> !hc.tensor<f32, ["A"]>
  return %r : !hc.tensor<f32, ["A"]>
}

// -----

// Non-unit step slice: per-axis offset composes to
// `lo + i_0 * step`. The Mul rides on the iter sym because step is a
// generic `!hc.idx<"step">` — the unit-step fold only triggers on
// integer literal 1. Canonical ixsimpl ordering puts the bare term
// (`lo`) first and the product (`i_0*step`) second.
// CHECK-LABEL: func.func @load_slice_non_unit_step
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"lo + i_0*step">] : !hc.buffer<f32, ["M"]>)
// CHECK-NOT: hc.load
func.func @load_slice_non_unit_step(%buf: !hc.buffer<f32, ["M"]>,
                                    %lo: !hc.idx<"lo">, %hi: !hc.idx<"hi">,
                                    %step: !hc.idx<"step">)
    -> !hc.tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %s = hc.slice_expr(lower = %lo upper = %hi step = %step)
      : (!hc.idx<"lo">, !hc.idx<"hi">, !hc.idx<"step">)
        -> !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"step">>
  %r = hc.load %buf[%s], shape %shape
      : (!hc.buffer<f32, ["M"]>,
         !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"step">>,
         tuple<!hc.idx<"A">>) -> !hc.tensor<f32, ["A"]>
  return %r : !hc.tensor<f32, ["A"]>
}

// -----

// Slice with no lower / no step (`x[:]`-shaped): per-axis offset
// reduces to identity over the iter sym. Confirms the default-step /
// default-lower path picks the `0 + iter_sym` form, which after the
// ixsimpl additive-identity fold prints as just `iter_sym`.
// CHECK-LABEL: func.func @load_slice_default_parts
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">] : !hc.buffer<f32, ["M"]>)
// CHECK-NOT: hc.load
func.func @load_slice_default_parts(%buf: !hc.buffer<f32, ["M"]>)
    -> !hc.tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %s = hc.slice_expr() : () -> !hc.slice
  %r = hc.load %buf[%s], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.slice, tuple<!hc.idx<"A">>)
        -> !hc.tensor<f32, ["A"]>
  return %r : !hc.tensor<f32, ["A"]>
}

// -----

// Slice mixed with scalar idx: rank-2 access where axis 0 is a slice
// and axis 1 is a pinned scalar. The mixed case is what the WMMA
// fragment loads emit (block tile origin pinned, lane walk striped
// across the other axis).
// CHECK-LABEL: func.func @load_mixed_slice_and_scalar
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0 + lo">, #hc.expr<"i_1 + j">] : !hc.buffer<f32, ["M", "N"]>)
// CHECK-NOT: hc.load
func.func @load_mixed_slice_and_scalar(%buf: !hc.buffer<f32, ["M", "N"]>,
                                       %lo: !hc.idx<"lo">,
                                       %hi: !hc.idx<"hi">,
                                       %step: !hc.idx<"1">,
                                       %j: !hc.idx<"j">)
    -> !hc.tensor<f32, ["A", "B"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %b = hc.const<1 : i64> : !hc.idx<"B">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"A">, !hc.idx<"B">) -> tuple<!hc.idx<"A">, !hc.idx<"B">>
  %s = hc.slice_expr(lower = %lo upper = %hi step = %step)
      : (!hc.idx<"lo">, !hc.idx<"hi">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"1">>
  %r = hc.load %buf[%s, %j], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>,
         !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"1">>,
         !hc.idx<"j">,
         tuple<!hc.idx<"A">, !hc.idx<"B">>) -> !hc.tensor<f32, ["A", "B"]>
  return %r : !hc.tensor<f32, ["A", "B"]>
}

// -----

// Slice with a non-pinned step (raw `index`, not `!hc.idx<expr>`):
// the rewrite needs a symbolic name to compose `step * iter` and
// bails. The op survives for a later slice-pinning pass / downstream
// lowering.
// CHECK-LABEL: func.func @load_slice_unpinned_step_falls_through
// CHECK: hc.load
// CHECK-NOT: hc.generic
func.func @load_slice_unpinned_step_falls_through(%buf: !hc.buffer<f32, ["M"]>,
                                                  %lo: !hc.idx<"lo">,
                                                  %hi: !hc.idx<"hi">,
                                                  %step: index)
    -> !hc.tensor<f32, ["A"]> {
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %shape = hc.tuple(%a) : (!hc.idx<"A">) -> tuple<!hc.idx<"A">>
  %s = hc.slice_expr(lower = %lo upper = %hi step = %step)
      : (!hc.idx<"lo">, !hc.idx<"hi">, index)
        -> !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = index>
  %r = hc.load %buf[%s], shape %shape
      : (!hc.buffer<f32, ["M"]>,
         !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = index>,
         tuple<!hc.idx<"A">>) -> !hc.tensor<f32, ["A"]>
  return %r : !hc.tensor<f32, ["A"]>
}

// -----

// Slice-indexed store: same `lo + step*iter` composition fires on
// the outs side of `hc.store`. Confirms the helper is shared between
// load and store paths.
// CHECK-LABEL: func.func @store_slice_buffer
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["A"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i_0 + lo">] : !hc.buffer<f32, ["M"]>)
// CHECK-NOT: hc.store
func.func @store_slice_buffer(%dst: !hc.buffer<f32, ["M"]>,
                              %src: !hc.tensor<f32, ["A"]>,
                              %lo: !hc.idx<"lo">, %hi: !hc.idx<"hi">,
                              %step: !hc.idx<"1">) {
  %s = hc.slice_expr(lower = %lo upper = %hi step = %step)
      : (!hc.idx<"lo">, !hc.idx<"hi">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"1">>
  hc.store %dst[%s], %src
      : (!hc.buffer<f32, ["M"]>,
         !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">, step = !hc.idx<"1">>,
         !hc.tensor<f32, ["A"]>) -> ()
  return
}

// -----

// `hc.load_mask` rewrites to an `hc.generic` whose body computes
// the per-axis in-bounds predicate from the slice's `lo` + `step *
// iter_k` against the source dim. Iter sym `i_k` matches the result
// tile axis order. The body emits a single `hc.pred_apply` carrying
// the conjuncted predicate (one comparison per slice axis, joined by
// `&`) and feeds its result through an `UnrealizedConversionCast`
// from the pinned form `!hc.pred<expr>` to the unpinned `!hc.pred`
// the bare-tile carrier element type wants — the per-thread / LDS
// materialisation downstream sees a single op that pulls every
// runtime-dependent dim from the kernel-arg bundle through the
// usual ambient-sym route.
// CHECK-LABEL: func.func @load_mask_basic
// CHECK: %[[OUT:.+]] = hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"8">, parallel i_1 = %{{[^ ]+}} : !hc.idx<"1">)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">]
// CHECK-SAME: -> (!hc.bare_vector<!hc.pred, ["8", "1"]>)
// CHECK: ^bb0(%{{[^:]+}}: !hc.pred):
// CHECK:   %[[P:.+]] = hc.pred_apply () : () -> !hc.pred<"-M + 2*i_0 < 0 & -N + i_1 < 0">
// CHECK:   %[[U:.+]] = builtin.unrealized_conversion_cast %[[P]] : !hc.pred<"-M + 2*i_0 < 0 & -N + i_1 < 0"> to !hc.pred
// CHECK:   hc.yield %[[U]] : !hc.pred
// CHECK-NOT: hc.load_mask
func.func @load_mask_basic(%buf: !hc.buffer<f32, ["M", "N"]>) -> !hc.bare_vector<!hc.pred, ["8", "1"]> {
  %zero = hc.const<0 : i64> : !hc.idx<"0">
  %one = hc.const<1 : i64> : !hc.idx<"1">
  %two = hc.const<2 : i64> : !hc.idx<"2">
  %sixteen = hc.const<16 : i64> : !hc.idx<"16">
  %eight = hc.const<8 : i64> : !hc.idx<"8">
  %rows = hc.slice_expr(lower = %zero upper = %sixteen step = %two)
      : (!hc.idx<"0">, !hc.idx<"16">, !hc.idx<"2">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">, step = !hc.idx<"2">>
  %col = hc.slice_expr(lower = %zero upper = %one)
      : (!hc.idx<"0">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>
  %shape = hc.tuple(%eight, %one)
      : (!hc.idx<"8">, !hc.idx<"1">) -> tuple<!hc.idx<"8">, !hc.idx<"1">>
  %m = hc.load_mask %buf[%rows, %col], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"16">, step = !hc.idx<"2">>,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"1">>,
         tuple<!hc.idx<"8">, !hc.idx<"1">>) -> !hc.bare_vector<!hc.pred, ["8", "1"]>
  return %m : !hc.bare_vector<!hc.pred, ["8", "1"]>
}

// -----

// `hc.load_mask` with a partial index list (the `X[gid[0]:]` form
// against a rank-2 source) gets the same trailing-full-slice
// padding the data-side `hc.load` rewriter uses: the trailing axis
// contributes a structurally trivial bound (`0 + 1*iter < N`),
// completing the rank-matched slice-axis form `composeMaskConjunction`
// needs. The conjunction carries the user-pinned `i + i_0 < M`
// alongside the synthesized `i_1 < N`. The decomposed-load broadcast
// shape `(A,B)` flattened against the rank-1 mask carrier post-flatten
// rides on the same pre-flatten rewrite: by the time launch-body
// looks, the op is already an `hc.generic` and the rank-1-vs-tuple
// mismatch never happens.
// CHECK-LABEL: func.func @load_mask_partial_indices
// CHECK-DAG: %[[A:.+]] = hc.idx_apply () : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply () : () -> !hc.idx<"B">
// CHECK: hc.generic iter (parallel i_0 = %[[A]] : !hc.idx<"A">, parallel i_1 = %[[B]] : !hc.idx<"B">)
// CHECK: ^bb0(%{{[^:]+}}: !hc.pred):
// CHECK:   %[[P:.+]] = hc.pred_apply () : () -> !hc.pred<"-N + i_1 < 0 & -M + i + i_0 < 0">
// CHECK:   %[[U:.+]] = builtin.unrealized_conversion_cast %[[P]] : !hc.pred<"-N + i_1 < 0 & -M + i + i_0 < 0"> to !hc.pred
// CHECK:   hc.yield %[[U]] : !hc.pred
// CHECK-NOT: hc.load_mask
func.func @load_mask_partial_indices(%buf: !hc.buffer<f32, ["M", "N"]>,
                                       %i: !hc.idx<"i">)
    -> !hc.bare_tensor<!hc.pred, ["A", "B"]> {
  %slice = hc.slice_expr(lower = %i)
      : (!hc.idx<"i">) -> !hc.slice<lower = !hc.idx<"i">>
  %a = hc.const<1 : i64> : !hc.idx<"A">
  %b = hc.const<1 : i64> : !hc.idx<"B">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"A">, !hc.idx<"B">) -> tuple<!hc.idx<"A">, !hc.idx<"B">>
  %m = hc.load_mask %buf[%slice], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>, !hc.slice<lower = !hc.idx<"i">>,
         tuple<!hc.idx<"A">, !hc.idx<"B">>) -> !hc.bare_tensor<!hc.pred, ["A", "B"]>
  return %m : !hc.bare_tensor<!hc.pred, ["A", "B"]>
}

// -----

// Layout-driven gather load. The result type carries a layout whose
// `storage_size` matches the rank-2 slice's flat capacity (`16 * 16
// = 256`), so the rewriter inverts the layout's `offset` to
// per-axis source coordinates instead of the trivial `lo + iter`
// identity. The decomposition is `base_k + Mod(floor(flat /
// inner_prod_k), extent_k)`, with the last axis skipping the floor
// div. This is the path the AMD gfx11 WMMA accumulator init takes
// — the C-side tile is laid out per-lane, per-fragment via the
// `WAVE_ACC_FRAG_LAYOUT` (lane = i_0, fi = i_1; `32*i_1 + i_0`
// folds onto a flat 256-cell tile span), and the layout-driven
// decomposition recovers the `(lane // 16 + fi * 2, lane % 16)`
// per-element coords that the matching `store_wmma_tile` writes
// back through a strided buffer-view. Without this path the load
// would carry identity offsets and misaddress every lane > 0.
// CHECK-LABEL: func.func @vload_layout_gather
// CHECK: %[[FILL:.+]] = hc.vzeros shape %{{[^ ]+}} {{.*}} -> !hc.bare_vector<f32, ["16", "16"]
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"16">, parallel i_1 = %{{[^ ]+}} : !hc.idx<"16">)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"row0 + Mod(2*i_1 + floor(1/16*i_0), 16)">, #hc.expr<"col0 + Mod(i_0, 16)">] : !hc.buffer<f32, ["M", "N"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">, #hc.expr<"i_1">]
// CHECK-NOT: hc.vload
func.func @vload_layout_gather(%buf: !hc.buffer<f32, ["M", "N"]>,
                               %lo0: !hc.idx<"row0">,
                               %hi0: !hc.idx<"row0 + 16">,
                               %lo1: !hc.idx<"col0">,
                               %hi1: !hc.idx<"col0 + 16">,
                               %step: !hc.idx<"1">)
    -> !hc.bare_vector<f32, ["16", "16"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>> {
  %a = hc.const<16 : i64> : !hc.idx<"16">
  %b = hc.const<16 : i64> : !hc.idx<"16">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"16">, !hc.idx<"16">) -> tuple<!hc.idx<"16">, !hc.idx<"16">>
  %s0 = hc.slice_expr(lower = %lo0 upper = %hi0 step = %step)
      : (!hc.idx<"row0">, !hc.idx<"row0 + 16">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"row0">, upper = !hc.idx<"row0 + 16">, step = !hc.idx<"1">>
  %s1 = hc.slice_expr(lower = %lo1 upper = %hi1 step = %step)
      : (!hc.idx<"col0">, !hc.idx<"col0 + 16">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"col0">, upper = !hc.idx<"col0 + 16">, step = !hc.idx<"1">>
  %v = hc.vload %buf[%s0, %s1], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>,
         !hc.slice<lower = !hc.idx<"row0">, upper = !hc.idx<"row0 + 16">, step = !hc.idx<"1">>,
         !hc.slice<lower = !hc.idx<"col0">, upper = !hc.idx<"col0 + 16">, step = !hc.idx<"1">>,
         tuple<!hc.idx<"16">, !hc.idx<"16">>)
        -> !hc.bare_vector<f32, ["16", "16"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>
  return %v : !hc.bare_vector<f32, ["16", "16"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>
}

// -----

// Layout-driven gather mask: mirror of `@vload_layout_gather` for
// `hc.load_mask`. The per-axis bound check becomes
// `decomposed_offset_k < srcDim_k` instead of the per-iter
// `lo + step*iter < srcDim`, so the predicate's "in-bounds" view
// matches the data path's layout-driven addressing. The WMMA
// accumulator init relies on this: without the layout-aware mask,
// lanes whose decomposed `(row, col)` runs off the partial-tile
// `c[M, N]` would still mask True and the final store would scatter
// past `c`'s end on M/N not multiples of `WMMA_M`/`WMMA_N`.
// CHECK-LABEL: func.func @load_mask_layout_gather
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"16">, parallel i_1 = %{{[^ ]+}} : !hc.idx<"16">)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">]
// CHECK: ^bb0(%{{[^:]+}}: !hc.pred):
// CHECK:   %[[P:.+]] = hc.pred_apply () : () -> !hc.pred<"-M + row0 + Mod(2*i_1 + floor(1/16*i_0), 16) < 0 & -N + col0 + Mod(i_0, 16) < 0">
// CHECK:   %[[U:.+]] = builtin.unrealized_conversion_cast %[[P]]
// CHECK:   hc.yield %[[U]] : !hc.pred
// CHECK-NOT: hc.load_mask
func.func @load_mask_layout_gather(%buf: !hc.buffer<f32, ["M", "N"]>,
                                   %lo0: !hc.idx<"row0">,
                                   %hi0: !hc.idx<"row0 + 16">,
                                   %lo1: !hc.idx<"col0">,
                                   %hi1: !hc.idx<"col0 + 16">,
                                   %step: !hc.idx<"1">)
    -> !hc.bare_vector<!hc.pred, ["16", "16"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>> {
  %a = hc.const<16 : i64> : !hc.idx<"16">
  %b = hc.const<16 : i64> : !hc.idx<"16">
  %shape = hc.tuple(%a, %b)
      : (!hc.idx<"16">, !hc.idx<"16">) -> tuple<!hc.idx<"16">, !hc.idx<"16">>
  %s0 = hc.slice_expr(lower = %lo0 upper = %hi0 step = %step)
      : (!hc.idx<"row0">, !hc.idx<"row0 + 16">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"row0">, upper = !hc.idx<"row0 + 16">, step = !hc.idx<"1">>
  %s1 = hc.slice_expr(lower = %lo1 upper = %hi1 step = %step)
      : (!hc.idx<"col0">, !hc.idx<"col0 + 16">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"col0">, upper = !hc.idx<"col0 + 16">, step = !hc.idx<"1">>
  %m = hc.load_mask %buf[%s0, %s1], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>,
         !hc.slice<lower = !hc.idx<"row0">, upper = !hc.idx<"row0 + 16">, step = !hc.idx<"1">>,
         !hc.slice<lower = !hc.idx<"col0">, upper = !hc.idx<"col0 + 16">, step = !hc.idx<"1">>,
         tuple<!hc.idx<"16">, !hc.idx<"16">>)
        -> !hc.bare_vector<!hc.pred, ["16", "16"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>
  return %m : !hc.bare_vector<!hc.pred, ["16", "16"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>
}

// -----

// Layout-bearing buffer source. `hc.as_layout` with a `shape=`
// operand declares the buffer carries a `(32, 8)` view through
// `WAVE_ACC_FRAG_LAYOUT`, and the access subscripts that view with a
// pinned lane + slice. The rewriter peels the as_layout, binds
// `index_syms[0] = lane` and `index_syms[1] = i_0`, composes
// `LAY.offset(lane, i_0)` against the declared `(32, 8)` shape, and
// decomposes the flat offset against the underlying tile's `(16,
// 16)` shape (row-major). The generic's source operand becomes the
// peeled buffer; the per-axis offsets re-route the layout-driven
// gather through the underlying `c_tile` storage. No `% extent`
// guard on the leading axis because the verifier already pins
// `storage_size(LAY) == product(underlying.shape)` at the as_layout
// boundary.
// CHECK-LABEL: func.func @vload_layout_bearing_buffer_source
// CHECK: %[[FILL:.+]] = hc.vzeros shape %{{[^ ]+}} {{.*}} -> !hc.bare_vector<f32, ["8"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"8">)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"Mod(2*i_0 + floor(1/16*lane), 16)">, #hc.expr<"Mod(lane, 16)">] : !hc.buffer<f32, ["16", "16"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">] : !hc.bare_vector<f32, ["8"]>)
// CHECK-NOT: hc.vload
func.func @vload_layout_bearing_buffer_source(
    %c_tile: !hc.buffer<f32, ["16", "16"]>,
    %lane: !hc.idx<"lane">,
    %hi: !hc.idx<"8">,
    %zero_i: !hc.idx<"0">,
    %one_i: !hc.idx<"1">)
    -> !hc.bare_vector<f32, ["8"]> {
  %d0 = hc.const<32 : i64> : !hc.idx<"32">
  %d1 = hc.const<8 : i64> : !hc.idx<"8">
  %lay_shape = hc.tuple(%d0, %d1)
      : (!hc.idx<"32">, !hc.idx<"8">) -> tuple<!hc.idx<"32">, !hc.idx<"8">>
  %c_lane = hc.as_layout %c_tile,
      layout = (#hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>),
      shape = %lay_shape : tuple<!hc.idx<"32">, !hc.idx<"8">>
      : !hc.buffer<f32, ["16", "16"]>
        -> !hc.buffer<f32, ["32", "8"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>
  %tile_shape_a = hc.const<1 : i64> : !hc.idx<"8">
  %tile_shape = hc.tuple(%tile_shape_a)
      : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
  %s = hc.slice_expr(lower = %zero_i upper = %hi step = %one_i)
      : (!hc.idx<"0">, !hc.idx<"8">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"1">>
  %v = hc.vload %c_lane[%lane, %s], shape %tile_shape
      : (!hc.buffer<f32, ["32", "8"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>,
         !hc.idx<"lane">,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"1">>,
         tuple<!hc.idx<"8">>)
        -> !hc.bare_vector<f32, ["8"]>
  return %v : !hc.bare_vector<f32, ["8"]>
}

// -----

// Symmetric scatter through a layout-bearing buffer dest. Mirrors
// `@vload_layout_bearing_buffer_source` on the store side: the dest
// is `as_layout` with `shape=` declared `(32, 8)`, the access has a
// pinned lane + slice, and the rewriter peels the as_layout so the
// `outs` operand of the planted generic targets the underlying
// `(16, 16)` tile. Same decomposition as the load side — the layout
// composes `index_syms` to a flat offset and the row-major split
// recovers `(row, col)`. This is the bd-npj3 path; the matching
// init form in `@vload_layout_bearing_buffer_source` exercises the
// load side.
// CHECK-LABEL: func.func @store_layout_bearing_buffer_dest
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"8">)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i_0">] : !hc.bare_vector<f32, ["8"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"Mod(2*i_0 + floor(1/16*lane), 16)">, #hc.expr<"Mod(lane, 16)">] : !hc.buffer<f32, ["16", "16"]>)
// CHECK-NOT: hc.store
func.func @store_layout_bearing_buffer_dest(
    %c_tile: !hc.buffer<f32, ["16", "16"]>,
    %frag: !hc.bare_vector<f32, ["8"]>,
    %lane: !hc.idx<"lane">,
    %hi: !hc.idx<"8">,
    %zero_i: !hc.idx<"0">,
    %one_i: !hc.idx<"1">) {
  %d0 = hc.const<32 : i64> : !hc.idx<"32">
  %d1 = hc.const<8 : i64> : !hc.idx<"8">
  %lay_shape = hc.tuple(%d0, %d1)
      : (!hc.idx<"32">, !hc.idx<"8">) -> tuple<!hc.idx<"32">, !hc.idx<"8">>
  %c_lane = hc.as_layout %c_tile,
      layout = (#hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>),
      shape = %lay_shape : tuple<!hc.idx<"32">, !hc.idx<"8">>
      : !hc.buffer<f32, ["16", "16"]>
        -> !hc.buffer<f32, ["32", "8"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>
  %s = hc.slice_expr(lower = %zero_i upper = %hi step = %one_i)
      : (!hc.idx<"0">, !hc.idx<"8">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"1">>
  hc.store %c_lane[%lane, %s], %frag
      : (!hc.buffer<f32, ["32", "8"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"256">, offset = #hc.expr<"32*i1 + i0">>>,
         !hc.idx<"lane">,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"1">>,
         !hc.bare_vector<f32, ["8"]>) -> ()
  return
}
