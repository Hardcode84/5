// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-generic` at the v0 scalar baseline (`U = 1`).
// The pass lowers `hc.generic` over typed-`!hc.ptr` operands to
// `scf.parallel` (parallel iters) + `scf.for` (reduction iters) with
// `iter_args` carrying the per-output accumulators, plus
// `hc.ptr_offset` + `hc.ptr_load` / `hc.ptr_store` at the body
// boundary.
//
// RUN: hc-opt --hc-lower-generic %s --split-input-file | FileCheck %s

// Pure-parallel 1D elementwise. One parallel iter, one ptr in, one
// ptr out, identity offsets. The body adds the loaded init to the
// loaded input. Lowers to a single `scf.parallel` with the body
// emitting load(init out) + load(in) + body + store(out) per
// iteration.
// CHECK-LABEL: func.func @elementwise_add_1d
// CHECK: scf.parallel (%[[I:[^)]+]])
// CHECK:   %[[PCI:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   %[[CV:[^ ]+]] = hc.ptr_load %[[PCI]]
// CHECK:   %[[PAI:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   %[[AV:[^ ]+]] = hc.ptr_load %[[PAI]]
// CHECK:   %[[S:[^ ]+]] = hc.add %[[CV]], %[[AV]]
// CHECK:   %[[PCS:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   hc.ptr_store %[[S]], %[[PCS]]
// CHECK:   scf.reduce
// CHECK-NOT: hc.generic
func.func @elementwise_add_1d(%n: index,
                              %a: !hc.ptr<global, f32>,
                              %c: !hc.ptr<global, f32>) {
  hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%c at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// Pure-reduction sum into a scalar slot. One reduction iter; the
// input rides the iter sym, the output sits at constant offset 0.
// Lowers with no outer parallel — the load of the initial
// accumulator happens once at the function level, then `scf.for`
// threads the running value through `iter_args`, then a final store
// at the same constant offset.
// CHECK-LABEL: func.func @reduce_sum_1d
// CHECK: %[[INIT:[^ ]+]] = hc.ptr_load %{{[^ ]+}}
// CHECK: %[[FINAL:[^ ]+]] = scf.for %[[K:[^ ]+]] =
// CHECK-SAME: iter_args(%[[ACC:[^ ]+]] = %[[INIT]])
// CHECK:   %[[PA:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[K]]
// CHECK:   %[[AV:[^ ]+]] = hc.ptr_load %[[PA]]
// CHECK:   %[[NEXT:[^ ]+]] = hc.add %[[ACC]], %[[AV]]
// CHECK:   scf.yield %[[NEXT]] : f32
// CHECK: hc.ptr_store %[[FINAL]]
// CHECK-NOT: hc.generic
func.func @reduce_sum_1d(%n: index,
                         %src: !hc.ptr<global, f32>,
                         %dst: !hc.ptr<global, f32>) {
  hc.generic
      iter (reduction k = %n : index)
      ins (%src at [#hc.expr<"k">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"0">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %dv: f32):
    %s = hc.add %dv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// Per-row reduction (parallel + reduction) using an outer-scope
// shape-symbol binding. The iter bound for `j` is typed
// `!hc.idx<"N">`, which the lowering picks up as the runtime value
// of the free symbol "N" when evaluating the input's `i*N + j`
// linear offset. Lowers to an outer `scf.parallel(i)` with an inner
// `scf.for(j)` reducing into the per-row accumulator.
// CHECK-LABEL: func.func @reduce_sum_2d_per_row
// CHECK: %[[N:[^ ]+]] = builtin.unrealized_conversion_cast %{{[^ ]+}} : !hc.idx<"N"> to index
// CHECK: scf.parallel (%[[I:[^)]+]])
// CHECK:   %[[PD0:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   %[[INIT:[^ ]+]] = hc.ptr_load %[[PD0]]
// CHECK:   %[[FINAL:[^ ]+]] = scf.for %[[J:[^ ]+]] =
// CHECK-SAME: iter_args(%[[ACC:[^ ]+]] = %[[INIT]])
// CHECK:     arith.muli {{.*}}%[[N]]
// CHECK:     %[[PA:[^ ]+]] = hc.ptr_offset
// CHECK:     %[[AV:[^ ]+]] = hc.ptr_load %[[PA]]
// CHECK:     %[[NEXT:[^ ]+]] = hc.add %[[ACC]], %[[AV]]
// CHECK:     scf.yield %[[NEXT]]
// CHECK:   hc.ptr_store %[[FINAL]]
// CHECK:   scf.reduce
// CHECK-NOT: hc.generic
func.func @reduce_sum_2d_per_row(%m: index, %nsym: !hc.idx<"N">,
                                 %src: !hc.ptr<global, f32>,
                                 %dst: !hc.ptr<global, f32>) {
  hc.generic
      iter (parallel i = %m : index, reduction j = %nsym : !hc.idx<"N">)
      ins (%src at [#hc.expr<"i*N + j">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %dv: f32):
    %s = hc.add %dv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// Multiple ptr outs in a single op. Each out gets its own initial
// load, its own accumulator slot in the (here trivial, no
// reduction) body, and its own final store. Confirms the
// declaration-order block-arg layout (ins first, then outs in
// order). Body does sum into one out and product into the other —
// both consume the same per-iteration in.
// CHECK-LABEL: func.func @elementwise_two_outs
// CHECK: scf.parallel (%[[I:[^)]+]])
// CHECK:   %[[PB:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   %[[BV:[^ ]+]] = hc.ptr_load %[[PB]]
// CHECK:   %[[PC:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   %[[CV:[^ ]+]] = hc.ptr_load %[[PC]]
// CHECK:   %[[PA:[^ ]+]] = hc.ptr_offset %{{[^,]+}}, %[[I]]
// CHECK:   %[[AV:[^ ]+]] = hc.ptr_load %[[PA]]
// CHECK:   %[[Y0:[^ ]+]] = hc.add %[[BV]], %[[AV]]
// CHECK:   %[[Y1:[^ ]+]] = hc.mul %[[CV]], %[[AV]]
// CHECK:   %[[PB2:[^ ]+]] = hc.ptr_offset
// CHECK:   hc.ptr_store %[[Y0]], %[[PB2]]
// CHECK:   %[[PC2:[^ ]+]] = hc.ptr_offset
// CHECK:   hc.ptr_store %[[Y1]], %[[PC2]]
// CHECK-NOT: hc.generic
func.func @elementwise_two_outs(%n: index,
                                %a: !hc.ptr<global, f32>,
                                %b: !hc.ptr<global, f32>,
                                %c: !hc.ptr<global, f32>) {
  hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%b at [#hc.expr<"i">] : !hc.ptr<global, f32>,
            %c at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %bv: f32, %cv: f32):
    %s = hc.add %bv, %av : (f32, f32) -> f32
    %p = hc.mul %cv, %av : (f32, f32) -> f32
    hc.yield %s, %p : f32, f32
  }
  return
}

// -----

// Out-of-v0-scope ops survive untouched. v0 only handles all-ptr
// operands; a bare-tensor on either side keeps the op around for a
// later slice (bufferization → all-ptr, then this pass picks it
// up). LIT pins the bail behaviour so a future scope expansion is
// an explicit, reviewable change.
// CHECK-LABEL: func.func @bail_bare_tensor_out
// CHECK: hc.generic
// CHECK-NOT: scf.parallel
// CHECK-NOT: scf.for
func.func @bail_bare_tensor_out(%n: index,
                                %src: !hc.ptr<global, f32>,
                                %dst: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield %sv : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}
