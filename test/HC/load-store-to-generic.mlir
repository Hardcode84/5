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
// CHECK-DAG: %[[A:.+]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"B">
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
// CHECK-DAG: %[[A:.+]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"B">
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

// Store into a buffer: ptr-out `hc.generic` with no SSA result.
// `%src` has identity offsets; `%dst` carries the multi-index
// addressing. Body still forwards the source element so the
// downstream `hc.ptr_store` materialisation has the value to write.
// CHECK-LABEL: func.func @store_buffer
// CHECK-DAG: %[[A:.+]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"A">
// CHECK-DAG: %[[B:.+]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"B">
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

// Masked store: scf.if-shaped body the v0 rewrite doesn't emit; the
// op stays for the masked-store follow-up.
// CHECK-LABEL: func.func @store_masked_falls_through
// CHECK: hc.store {{.*}} mask
// CHECK-NOT: hc.generic
func.func @store_masked_falls_through(%dst: !hc.buffer<f32, ["M"]>,
                                      %src: !hc.bare_tensor<f32, ["A"]>,
                                      %mask: !hc.bare_tensor<!hc.pred, ["A"]>,
                                      %i: !hc.idx<"i">) {
  hc.store %dst[%i], %src, mask %mask
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"i">, !hc.bare_tensor<f32, ["A"]>,
         !hc.bare_tensor<!hc.pred, ["A"]>) -> ()
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
