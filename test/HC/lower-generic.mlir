// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-generic`: the unroll-and-merge codegen
// frame for `hc.generic`. Tests below exercise both the scalar
// fallback (when ixsimpl can't prove divisibility, partition pins to
// `(1, ..., 1)`) and the vec path (per-axis unroll factors picked by
// the partition search, vector-typed `hc.ptr_load` / `hc.ptr_store`
// at merged contig groups).
//
// RUN: hc-opt --hc-lower-generic %s --split-input-file | FileCheck %s

// Pure-parallel 1D elementwise. One parallel iter, one ptr in, one
// ptr out, identity offsets. The body adds the loaded init to the
// loaded input. Lowers to a single `scf.parallel` with the body
// emitting load(init out) + load(in) + body + store(out) per
// iteration.
// CHECK-LABEL: func.func @elementwise_add_1d
// CHECK: scf.parallel (%[[I:[^)]+]])
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   %[[PCI:[^ ]+]] = hc.ptr_offset
// CHECK:   %[[CV:[^ ]+]] = hc.ptr_load %[[PCI]]
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   %[[PAI:[^ ]+]] = hc.ptr_offset
// CHECK:   %[[AV:[^ ]+]] = hc.ptr_load %[[PAI]]
// CHECK:   %[[S:[^ ]+]] = hc.add %[[CV]], %[[AV]]
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   %[[PCS:[^ ]+]] = hc.ptr_offset
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
// CHECK: hc.idx_apply ()
// CHECK-SAME: -> !hc.idx<"0">
// CHECK: %[[INIT:[^ ]+]] = hc.ptr_load %{{[^ ]+}}
// CHECK: %[[FINAL:[^ ]+]] = scf.for %[[K:[^ ]+]] =
// CHECK-SAME: iter_args(%[[ACC:[^ ]+]] = %[[INIT]])
// CHECK:   hc.idx_apply (%[[K]] as "k")
// CHECK:   %[[PA:[^ ]+]] = hc.ptr_offset
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

// Per-row reduction (parallel + reduction). Iter syms `i` and `j`
// surface as explicit operand bindings in the per-row offset's
// `hc.idx_apply`; the shape sym `N` stays unlisted and gets
// resolved ambiently by the launch-body lowering downstream. The
// out offset only references the parallel iter, so its apply lists
// `["i"]` alone.
// CHECK-LABEL: func.func @reduce_sum_2d_per_row
// CHECK: scf.parallel (%[[I:[^)]+]])
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK-SAME: -> !hc.idx<"i">
// CHECK:   %[[INIT:[^ ]+]] = hc.ptr_load
// CHECK:   %[[FINAL:[^ ]+]] = scf.for %[[J:[^ ]+]] =
// CHECK-SAME: iter_args(%[[ACC:[^ ]+]] = %[[INIT]])
// CHECK:     hc.idx_apply (%{{[^ ]+}} as "i", %{{[^ ]+}} as "j")
// CHECK-SAME: -> !hc.idx<"j + N*i">
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
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   %[[PB:[^ ]+]] = hc.ptr_offset
// CHECK:   %[[BV:[^ ]+]] = hc.ptr_load %[[PB]]
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   %[[PC:[^ ]+]] = hc.ptr_offset
// CHECK:   %[[CV:[^ ]+]] = hc.ptr_load %[[PC]]
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   %[[PA:[^ ]+]] = hc.ptr_offset
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

// -----

// Pure-parallel 1D with a divisible bound (`!hc.idx<"32">`). The
// partition search proves `32 % 32 == 0`, picks `(32,)`, and merges
// every lane into one vector group of width 32. The outer
// `scf.parallel` steps by 32 (one iteration over the bound), the
// boundary loads/stores are `vector<32xf32>`, and `vector.extract` /
// `vector.from_elements` plumb the per-lane scalars into the body.
// CHECK-LABEL: func.func @vec_pure_par_unroll_1d
// CHECK: %[[STEP:[^ ]+]] = arith.constant 32 : index
// CHECK: scf.parallel (%[[I:[^)]+]]) = {{.*}} step (%[[STEP]])
// CHECK:   hc.idx_apply (%[[I]] as "i")
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   vector.extract
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   hc.add
// CHECK:   vector.from_elements {{.*}} : vector<32xf32>
// CHECK:   hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : vector<32xf32>, !hc.ptr<global, f32>
func.func @vec_pure_par_unroll_1d(%a: !hc.ptr<global, f32>,
                                  %c: !hc.ptr<global, f32>) {
  %n = hc.idx_apply () : () -> !hc.idx<"32">
  hc.generic
      iter (parallel i = %n : !hc.idx<"32">)
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

// 2D row-major (offset `i*32 + j`, j contig). The merge analyzer
// finds a 32-wide contig run on the j-axis and picks partition
// `(1, 32)` — i steps by 1 in the parallel loop, j steps by 32, and
// every lane folds into one vector load / store of width 32.
// CHECK-LABEL: func.func @vec_rowmaj_2d
// CHECK: %[[STEPI:[^ ]+]] = arith.constant 1 : index
// CHECK: %[[STEPJ:[^ ]+]] = arith.constant 32 : index
// CHECK: scf.parallel (%[[I:[^,]+]], %[[J:[^)]+]]) = {{.*}} step (%[[STEPI]], %[[STEPJ]])
// CHECK:   hc.idx_apply (%[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: !hc.idx<"32*i + j">
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : vector<32xf32>, !hc.ptr<global, f32>
func.func @vec_rowmaj_2d(%a: !hc.ptr<global, f32>, %c: !hc.ptr<global, f32>) {
  %m = hc.idx_apply () : () -> !hc.idx<"8">
  %n = hc.idx_apply () : () -> !hc.idx<"32">
  hc.generic
      iter (parallel i = %m : !hc.idx<"8">,
            parallel j = %n : !hc.idx<"32">)
      ins (%a at [#hc.expr<"i*32 + j">] : !hc.ptr<global, f32>)
      outs (%c at [#hc.expr<"i*32 + j">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// 2D column-major (offset `i + j*8`, i contig). With axis-order
// search the pass finds the order in which i is the rightmost-fastest
// axis, picks partition `(8, 1)` — i fully covered as a contig group
// of 8, j stepped scalar. The vector width is 8 (i's bound), not 32,
// because i's bound caps the per-axis factor.
// CHECK-LABEL: func.func @vec_colmaj_2d
// CHECK: %[[STEPI:[^ ]+]] = arith.constant 8 : index
// CHECK: %[[STEPJ:[^ ]+]] = arith.constant 1 : index
// CHECK: scf.parallel (%[[I:[^,]+]], %[[J:[^)]+]]) = {{.*}} step (%[[STEPI]], %[[STEPJ]])
// CHECK:   hc.idx_apply (%[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: !hc.idx<"i + 8*j">
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<8xf32>
// CHECK:   hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : vector<8xf32>, !hc.ptr<global, f32>
func.func @vec_colmaj_2d(%a: !hc.ptr<global, f32>, %c: !hc.ptr<global, f32>) {
  %m = hc.idx_apply () : () -> !hc.idx<"8">
  %n = hc.idx_apply () : () -> !hc.idx<"32">
  hc.generic
      iter (parallel i = %m : !hc.idx<"8">,
            parallel j = %n : !hc.idx<"32">)
      ins (%a at [#hc.expr<"i + j*8">] : !hc.ptr<global, f32>)
      outs (%c at [#hc.expr<"i + j*8">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// Reduction with a divisible bound. Partition `(32,)` on the
// reduction axis collapses the inner loop to one iteration; the body
// runs 32 sequential `hc.add` clones threading a single accumulator.
// Outs sit at constant offset 0 with a single scalar load/store.
// CHECK-LABEL: func.func @vec_reduce_unroll_1d
// CHECK: hc.idx_apply () : () -> !hc.idx<"0">
// CHECK: %[[INIT:[^ ]+]] = hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> f32
// CHECK: %[[STEP:[^ ]+]] = arith.constant 32 : index
// CHECK: %[[FINAL:[^ ]+]] = scf.for %[[K:[^ ]+]] = {{.*}} step %[[STEP]]
// CHECK-SAME: iter_args(%[[ACC:[^ ]+]] = %[[INIT]])
// CHECK:   hc.idx_apply (%[[K]] as "k")
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   hc.add %[[ACC]]
// CHECK: hc.ptr_store %[[FINAL]], %{{[^ ]+}} : f32, !hc.ptr<global, f32>
func.func @vec_reduce_unroll_1d(%dst: !hc.ptr<global, f32>,
                                %src: !hc.ptr<global, f32>) {
  %k = hc.idx_apply () : () -> !hc.idx<"32">
  hc.generic
      iter (reduction k = %k : !hc.idx<"32">)
      ins (%src at [#hc.expr<"k">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"0">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%sv: f32, %dv: f32):
    %s = hc.add %dv, %sv : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// Mixed parallel + reduction with bounds 4 and 32, row-major offset.
// Total budget caps prod(p_i) at 32, so the search picks partition
// `(4, 8)` — i fully unrolled as a 4-wide outs vector, j chunked
// into 4 inner-loop iterations of 8 unrolled lanes each. The
// `iter_args` flat layout carries 4 accumulators (one per parallel
// lane), and the inner body runs 4 * 8 = 32 sequential clones in
// declaration order, threading each parallel lane's accumulator
// independently.
// CHECK-LABEL: func.func @vec_mixed_par_red
// CHECK: %[[STEPI:[^ ]+]] = arith.constant 4 : index
// CHECK: scf.parallel (%[[I:[^)]+]]) = {{.*}} step (%[[STEPI]])
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<4xf32>
// CHECK:   %[[STEPJ:[^ ]+]] = arith.constant 8 : index
// CHECK:   scf.for %[[J:[^ ]+]] = {{.*}} step %[[STEPJ]]
// CHECK-SAME: iter_args(%[[A0:[^ ]+]] = %{{[^,]+}}, %[[A1:[^ ]+]] = %{{[^,]+}}, %[[A2:[^ ]+]] = %{{[^,]+}}, %[[A3:[^ ]+]] = %{{[^)]+}})
// CHECK:     hc.idx_apply (%[[I]] as "i", %[[J]] as "j")
// CHECK:     hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<8xf32>
// CHECK:     scf.yield %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}}
// CHECK:   vector.from_elements {{.*}} : vector<4xf32>
// CHECK:   hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : vector<4xf32>, !hc.ptr<global, f32>
func.func @vec_mixed_par_red(%dst: !hc.ptr<global, f32>,
                             %src: !hc.ptr<global, f32>) {
  %m = hc.idx_apply () : () -> !hc.idx<"4">
  %n = hc.idx_apply () : () -> !hc.idx<"32">
  hc.generic
      iter (parallel i = %m : !hc.idx<"4">,
            reduction j = %n : !hc.idx<"32">)
      ins (%src at [#hc.expr<"i*32 + j">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%sv: f32, %dv: f32):
    %s = hc.add %dv, %sv : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return
}

// -----

// Plain `index` bounds defeat the divisibility probe — `probeDivisible`
// returns false when the bound's SSA type is not `!hc.idx<expr>`, so
// every per-axis factor > 1 gets filtered and the search lands on the
// trivial partition `(1, ...)`. Emission collapses to the scalar
// baseline verbatim, with no `vector.*` ops anywhere.
// CHECK-LABEL: func.func @scalar_fallback_index_bound
// CHECK: scf.parallel (%[[I:[^)]+]])
// CHECK-NOT: vector.
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> f32
// CHECK:   hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : f32, !hc.ptr<global, f32>
func.func @scalar_fallback_index_bound(%n: index,
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

// Collective dispatch: outs is a `!hc.ptr<workgroup, T>` (LDS-staged
// tile), every iter is parallel, and the op sits inside `gpu.launch`.
// The pass picks the chunk-and-publish shape instead of `scf.parallel`
// — every thread of the wave processes a strided subset of the
// linearised iter space, with an in-range gate at the trailing
// partial chunk and a closing `gpu.barrier` so the populated tile is
// visible to downstream readers. The body still runs once per
// in-range iteration (no partition unroll on this path: collective
// dispatch is inherently per-element, the merge analyzer can't claim
// anything useful when every lane owns a different element).
// CHECK-LABEL: func.func @collective_lds_population
// CHECK: gpu.launch blocks
// CHECK-SAME: threads({{[^,]+}}, %[[TY:[^,]+]], %[[TZ:[^)]+]])
// CHECK: %[[TZBY:.+]] = arith.muli %[[TZ]], %{{.+}} : index
// CHECK: %[[TZBYTY:.+]] = arith.addi %[[TZBY]], %[[TY]] : index
// CHECK: %[[ROWSPAN:.+]] = arith.muli %[[TZBYTY]], %{{.+}} : index
// CHECK: %[[LIN_TID:.+]] = arith.addi %[[ROWSPAN]], %{{.+}} : index
// CHECK: %[[BXBY:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK: %[[WG_SIZE:.+]] = arith.muli %[[BXBY]], %{{.+}} : index
// CHECK: arith.muli {{.+}} : index
// CHECK: %[[TOTAL:.+]] = arith.muli {{.+}} : index
// CHECK: %[[CHUNKS:.+]] = arith.ceildivui %[[TOTAL]], %[[WG_SIZE]] : index
// CHECK: scf.for %[[C:[^=]+]] = %{{.+}} to %[[CHUNKS]] step
// CHECK: %[[OFF:.+]] = arith.muli %[[C]], %[[WG_SIZE]] : index
// CHECK: %[[LIN:.+]] = arith.addi %[[OFF]], %[[LIN_TID]] : index
// CHECK: %[[INR:.+]] = arith.cmpi ult, %[[LIN]], %[[TOTAL]] : index
// CHECK: scf.if %[[INR]] {
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: hc.ptr_load %{{.+}} : !hc.ptr<global, f32> -> f32
// CHECK: hc.ptr_load %{{.+}} : !hc.ptr<workgroup, f32> -> f32
// CHECK: hc.ptr_store %{{.+}}, %{{.+}} : f32, !hc.ptr<workgroup, f32>
// CHECK: }
// CHECK: gpu.barrier
// CHECK-NOT: hc.generic
// CHECK-NOT: scf.parallel
func.func @collective_lds_population(%src: !hc.ptr<global, f32>,
                                     %lds: !hc.ptr<workgroup, f32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    %m = hc.idx_apply () : () -> !hc.idx<"8">
    %n = hc.idx_apply () : () -> !hc.idx<"16">
    hc.generic
        iter (parallel i = %m : !hc.idx<"8">,
              parallel j = %n : !hc.idx<"16">)
        ins (%src at [#hc.expr<"i*16 + j">] : !hc.ptr<global, f32>)
        outs (%lds at [#hc.expr<"i*16 + j">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%sv: f32, %dv: f32):
      hc.yield %sv : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Workgroup-shared outs but no enclosing `gpu.launch`: the pass has
// no thread-id source to chunk against, so the collective path
// rejects the op and the partition-aware emitter handles it as the
// per-lane scf.parallel shape would. The cooperative-staging shape
// the pipeline ultimately wants comes from running this after the
// kernel was wrapped in `gpu.launch`; running stand-alone is the
// hc-opt smoke-test slot.
// CHECK-LABEL: func.func @workgroup_outs_no_launch_falls_through
// CHECK-NOT: scf.for
// CHECK: scf.parallel
// CHECK-NOT: hc.generic
// CHECK-NOT: gpu.barrier
func.func @workgroup_outs_no_launch_falls_through(
    %src: !hc.ptr<global, f32>, %lds: !hc.ptr<workgroup, f32>) {
  %m = hc.idx_apply () : () -> !hc.idx<"8">
  hc.generic
      iter (parallel i = %m : !hc.idx<"8">)
      ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
      -> () {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield %sv : f32
  }
  return
}

// -----

// Global-only outs (no workgroup ptr in the outs list) inside a
// `gpu.launch`: collective dispatch isn't appropriate — running
// every lane on every element of a global-shared output races, the
// same way it does without a launch. The partition-aware emitter
// still produces the per-lane `scf.parallel` shape.
// CHECK-LABEL: func.func @global_outs_in_launch_falls_through
// CHECK: gpu.launch
// CHECK: scf.parallel
// CHECK-NOT: gpu.barrier
// CHECK-NOT: hc.generic
func.func @global_outs_in_launch_falls_through(%src: !hc.ptr<global, f32>,
                                               %dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    %m = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %m : !hc.idx<"8">)
        ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%sv: f32, %dv: f32):
      hc.yield %sv : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Workgroup-shared outs with a reduction iter: collective dispatch
// requires every iter to be parallel — cross-iter accumulation
// across threads is a cross-thread reduction the chunk-and-publish
// shape doesn't model. Reduction generics fall through to the
// partition-aware emitter even when the outs is in workgroup AS;
// the per-thread accumulator path runs locally and the user's
// upstream code is responsible for the cross-thread synchronization.
// CHECK-LABEL: func.func @workgroup_outs_with_reduction_falls_through
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: scf.reduce
// CHECK-NOT: gpu.barrier
// CHECK-NOT: hc.generic
func.func @workgroup_outs_with_reduction_falls_through(
    %src: !hc.ptr<global, f32>, %lds: !hc.ptr<workgroup, f32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    %m = hc.idx_apply () : () -> !hc.idx<"8">
    %n = hc.idx_apply () : () -> !hc.idx<"16">
    hc.generic
        iter (parallel i = %m : !hc.idx<"8">,
              reduction j = %n : !hc.idx<"16">)
        ins (%src at [#hc.expr<"i*16 + j">] : !hc.ptr<global, f32>)
        outs (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%sv: f32, %dv: f32):
      %s = hc.add %dv, %sv : (f32, f32) -> f32
      hc.yield %s : f32
    }
    gpu.terminator
  }
  return
}
