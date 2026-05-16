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

// 2D parallel iter with a non-injective source offset: `i_1 + j`
// references only `i_1` (and ambient kernel-scope `j`), never `i_0`.
// That's the post-flatten shape of a `hc.vload` against the per-lane
// fragment layout in `doc/layouts.md` — flatten composes
// `index_syms = [i, j, lane]` against the layout's `offset = i1`,
// substitutes `i_1` for `j`'s position, and drops the broadcast axis
// entirely. After lower-generic, the per-iter `hc.idx_apply` for the
// source load lists exactly `{i_1, j}` and *not* `i_0`, so the load
// offset is constant across i_0 — every i_0 iteration reads the
// same scalar. The dst offset `i_1 + B*i_0` is the identity tile
// layout and lists both iter syms; the stored value is the same
// scalar repeated across i_0. That repetition is the implicit
// broadcast the `doc/layouts.md` "Non-injective layouts" section
// documents — no explicit gather op, just the iter sym that the
// composed offset declines to mention.
// CHECK-LABEL: func.func @noninjective_broadcast_2d
// CHECK: scf.parallel (%[[II0:[^,]+]], %[[II1:[^)]+]])
// CHECK:   hc.idx_apply (%{{[^ ]+}} as "B", %[[II0]] as "i_0", %[[II1]] as "i_1")
// CHECK-SAME: -> !hc.idx<"i_1 + B*i_0">
// CHECK:   %[[PD0:[^ ]+]] = hc.ptr_offset
// CHECK:   hc.ptr_load %[[PD0]]
// CHECK:   hc.idx_apply (%[[II1]] as "i_1", %{{[^ ]+}} as "j")
// CHECK-SAME: -> !hc.idx<"i_1 + j">
// CHECK:   %[[PS:[^ ]+]] = hc.ptr_offset
// CHECK:   %[[V:[^ ]+]] = hc.ptr_load %[[PS]]
// CHECK:   hc.idx_apply (%{{[^ ]+}} as "B", %[[II0]] as "i_0", %[[II1]] as "i_1")
// CHECK-SAME: -> !hc.idx<"i_1 + B*i_0">
// CHECK:   %[[PD1:[^ ]+]] = hc.ptr_offset
// CHECK:   hc.ptr_store %[[V]], %[[PD1]]
// CHECK-NOT: hc.generic
func.func @noninjective_broadcast_2d(
    %a: !hc.idx<"A">, %b: !hc.idx<"B">, %j: !hc.idx<"j">,
    %src: !hc.ptr<global, f32>,
    %dst: !hc.ptr<global, f32>) {
  hc.generic
      iter (parallel i_0 = %a : !hc.idx<"A">,
            parallel i_1 = %b : !hc.idx<"B">)
      ins (%src at [#hc.expr<"i_1 + j">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"i_1 + B*i_0">] : !hc.ptr<global, f32>)
      ambient (%b as "B" : !hc.idx<"B">, %j as "j" : !hc.idx<"j">)
      -> () {
  ^bb0(%v: f32, %d: f32):
    hc.yield %v : f32
  }
  return
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

// 2D identity layout (offset `i*32 + j`, j contig). The merge
// analyzer finds a 32-wide contig run on the j-axis and picks
// partition `(1, 32)` — i steps by 1 in the parallel loop, j steps
// by 32, and every lane folds into one vector load / store of width 32.
// CHECK-LABEL: func.func @vec_contig_2d
// CHECK: %[[STEPI:[^ ]+]] = arith.constant 1 : index
// CHECK: %[[STEPJ:[^ ]+]] = arith.constant 32 : index
// CHECK: scf.parallel (%[[I:[^,]+]], %[[J:[^)]+]]) = {{.*}} step (%[[STEPI]], %[[STEPJ]])
// CHECK:   hc.idx_apply (%[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: !hc.idx<"32*i + j">
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<32xf32>
// CHECK:   hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : vector<32xf32>, !hc.ptr<global, f32>
func.func @vec_contig_2d(%a: !hc.ptr<global, f32>, %c: !hc.ptr<global, f32>) {
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

// Mixed parallel + reduction with bounds 4 and 32, identity offset.
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

// Collective dispatch: outs is a `!hc.ptr<workgroup, T>` (workgroup-staged
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

// -----

// `bare_tensor` outs backed by `hc.alloc workgroup` via a UCC bridge
// inside `gpu.launch`: the outs carrier type is `bare_tensor` (the
// shape view), the underlying storage is an LDS allocation, and the
// post-conversion pipeline planted a `ptr<workgroup> → bare_tensor`
// UCC from the producing alloc. The dispatch routes to the
// collective shape and plants a fresh `bare_tensor → ptr<workgroup>`
// UCC at the access site; together with the upstream one they form
// a foldable pair that canonicalize collapses back to the alloc, so
// the per-thread stores end up hitting the real LDS storage. The
// generic's `bare_tensor` result RAUWs to the original outs SSA so
// the downstream UCC back to `ptr<workgroup>` folds the same way
// instead of dangling against the erased generic.
// CHECK-LABEL: func.func @collective_bare_tensor_ucc_lds
// CHECK: gpu.launch
// CHECK: %[[ALLOC:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
// CHECK: %[[TILE:.+]] = builtin.unrealized_conversion_cast %[[ALLOC]]
// CHECK-SAME: !hc.ptr<workgroup, f32> to !hc.bare_tensor<f32, ["256"]>
// CHECK: builtin.unrealized_conversion_cast %[[TILE]]
// CHECK-SAME: !hc.bare_tensor<f32, ["256"]> to !hc.ptr<workgroup, f32>
// CHECK: scf.for
// CHECK: scf.if
// CHECK: hc.ptr_load %{{.+}} : !hc.ptr<global, f32> -> f32
// CHECK: hc.ptr_load %{{.+}} : !hc.ptr<workgroup, f32> -> f32
// CHECK: hc.ptr_store %{{.+}}, %{{.+}} : f32, !hc.ptr<workgroup, f32>
// CHECK: gpu.barrier
// CHECK: builtin.unrealized_conversion_cast %[[TILE]]
// CHECK-SAME: !hc.bare_tensor<f32, ["256"]> to !hc.ptr<workgroup, f32>
// CHECK: hc.ptr_load %{{.+}} : !hc.ptr<workgroup, f32> -> f32
// CHECK-NOT: hc.generic
// CHECK-NOT: vector.from_elements
func.func @collective_bare_tensor_ucc_lds(%src: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    %lds = hc.alloc count = %c256 : index -> !hc.ptr<workgroup, f32>
    %tile = builtin.unrealized_conversion_cast %lds
        : !hc.ptr<workgroup, f32> to !hc.bare_tensor<f32, ["256"]>
    %m = hc.idx_apply () : () -> !hc.idx<"16">
    %n = hc.idx_apply () : () -> !hc.idx<"16">
    %r = hc.generic
        iter (parallel i_0 = %m : !hc.idx<"16">,
              parallel i_1 = %n : !hc.idx<"16">)
        ins (%src at [#hc.expr<"16*i_0 + i_1">] : !hc.ptr<global, f32>)
        outs (%tile at [#hc.expr<"16*i_0 + i_1">]
              : !hc.bare_tensor<f32, ["256"]>)
        -> (!hc.bare_tensor<f32, ["256"]>) {
    ^bb0(%sv: f32, %dv: f32):
      hc.yield %sv : f32
    }
    %back = builtin.unrealized_conversion_cast %r
        : !hc.bare_tensor<f32, ["256"]> to !hc.ptr<workgroup, f32>
    %addr = hc.ptr_offset %back, %c1
        : (!hc.ptr<workgroup, f32>, index) -> !hc.ptr<workgroup, f32>
    %v = hc.ptr_load %addr : !hc.ptr<workgroup, f32> -> f32
    gpu.terminator
  }
  return
}

// -----

// Value-typed out: ptr ins + bare_vector out, single parallel iter
// with a constant bound. The value-outs lowering unrolls the
// parallel sweep at compile time (no `scf.parallel`), composes
// per-lane finals through a single `vector.from_elements`, and
// UCC's the result back to `!hc.bare_vector<...>` for the original
// consumer. Contig-group analysis on the ins side collapses the
// 8 unit-stride loads into one `vector<8xf32>` `hc.ptr_load`.
// CHECK-LABEL: func.func @value_out_1d_ptr_in
// CHECK-NOT: scf.parallel
// CHECK: hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f32> -> vector<8xf32>
// CHECK-NOT: hc.ptr_store
// CHECK: %[[VEC:[^ ]+]] = vector.from_elements
// CHECK-SAME: vector<8xf32>
// CHECK: %[[CAST:[^ ]+]] = builtin.unrealized_conversion_cast %[[VEC]]
// CHECK-SAME: vector<8xf32> to !hc.bare_vector<f32, ["8"]>
// CHECK: return %[[CAST]]
// CHECK-NOT: hc.generic
func.func @value_out_1d_ptr_in(%src: !hc.ptr<global, f32>,
                               %init: !hc.bare_vector<f32, ["8"]>)
    -> !hc.bare_vector<f32, ["8"]> {
  %n = hc.idx_apply () : () -> !hc.idx<"8">
  %r = hc.generic
      iter (parallel i = %n : !hc.idx<"8">)
      ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%init at [#hc.expr<"i">] : !hc.bare_vector<f32, ["8"]>)
      -> (!hc.bare_vector<f32, ["8"]>) {
  ^bb0(%sv: f32, %iv: f32):
    %s = hc.add %iv, %sv : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_vector<f32, ["8"]>
}

// -----

// Value-typed out: 2D parallel iter (16x16, both constant-bound)
// with identity outs offset `16*i_0 + i_1` — the WMMA fragment-load
// shape. Slot = parLane under identity-layout decomposition, so
// `vector.from_elements` sees finals in identity order. Contig
// analysis on the ins (unit-stride inner axis) collapses to one
// `vector<256xf16>` load.
// CHECK-LABEL: func.func @value_out_2d_fragment_load
// CHECK-NOT: scf.parallel
// CHECK: hc.ptr_load %{{[^ ]+}} : !hc.ptr<global, f16> -> vector<256xf16>
// CHECK: vector.from_elements
// CHECK-SAME: vector<256xf16>
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: vector<256xf16> to !hc.bare_tensor<f16, ["256"]>
// CHECK-NOT: hc.generic
func.func @value_out_2d_fragment_load(
    %src: !hc.ptr<global, f16>,
    %init: !hc.bare_tensor<f16, ["256"]>)
    -> !hc.bare_tensor<f16, ["256"]> {
  %m = hc.idx_apply () : () -> !hc.idx<"16">
  %n = hc.idx_apply () : () -> !hc.idx<"16">
  %r = hc.generic
      iter (parallel i_0 = %m : !hc.idx<"16">,
            parallel i_1 = %n : !hc.idx<"16">)
      ins (%src at [#hc.expr<"16*i_0 + i_1">] : !hc.ptr<global, f16>)
      outs (%init at [#hc.expr<"16*i_0 + i_1">]
            : !hc.bare_tensor<f16, ["256"]>)
      -> (!hc.bare_tensor<f16, ["256"]>) {
  ^bb0(%sv: f16, %iv: f16):
    hc.yield %sv : f16
  }
  return %r : !hc.bare_tensor<f16, ["256"]>
}

// -----

// Value-typed ins, ptr-typed out, unpredicated yield. The fully-
// unrolled emitter UCC's the bare_vector carrier to `vector<8xf32>`,
// extracts per parLane at the slot the offset evaluates to (here
// `i` itself, so slot == parLane), and stores each lane through the
// ptr out. The contig analyzer collapses the 8 unit-stride stores
// into one `vector<8xf32>` store via `vector.from_elements`. No
// `scf.parallel` because the parallel sweep is compile-time-
// unrolled.
// CHECK-LABEL: func.func @value_in_1d_ptr_out
// CHECK-NOT: scf.parallel
// CHECK: builtin.unrealized_conversion_cast %{{[^ ]+}} : !hc.bare_vector<f32, ["8"]> to vector<8xf32>
// CHECK: vector.extract %{{[^ ]+}}[0]
// CHECK: vector.extract %{{[^ ]+}}[7]
// CHECK: vector.from_elements
// CHECK-SAME: vector<8xf32>
// CHECK: hc.ptr_store %{{[^ ]+}}, %{{[^ ]+}} : vector<8xf32>, !hc.ptr<global, f32>
// CHECK-NOT: hc.generic
func.func @value_in_1d_ptr_out(%vec: !hc.bare_vector<f32, ["8"]>,
                               %dst: !hc.ptr<global, f32>) {
  %n = hc.idx_apply () : () -> !hc.idx<"8">
  hc.generic
      iter (parallel i = %n : !hc.idx<"8">)
      ins (%vec at [#hc.expr<"i">] : !hc.bare_vector<f32, ["8"]>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%v: f32, %init: f32):
    hc.yield %v : f32
  }
  return
}

// -----

// WMMA-writeback shape: value-typed value ins + value-typed mask ins
// + ptr out + `hc.yield_predicated`. Per-lane materialization:
// UCC value carrier to `vector<8xf32>`, UCC mask carrier to
// `vector<8xi1>` (the `!hc.pred` element is mapped to `i1` for the
// arith.select boundary), `arith.select(mask, val, init)` per lane,
// then per-lane store. The init feeds the masked-out lanes so they
// preserve the dst contents the per-lane init-load just published.
// CHECK-LABEL: func.func @value_in_wmma_writeback
// CHECK-NOT: scf.parallel
// CHECK: builtin.unrealized_conversion_cast %{{[^ ]+}} : !hc.bare_vector<f32, ["8"]> to vector<8xf32>
// CHECK: builtin.unrealized_conversion_cast %{{[^ ]+}} : !hc.bare_vector<!hc.pred, ["8"]> to vector<8xi1>
// CHECK: hc.ptr_load
// CHECK: arith.select
// CHECK: hc.ptr_store
// CHECK-NOT: hc.generic
func.func @value_in_wmma_writeback(%vec: !hc.bare_vector<f32, ["8"]>,
                                   %pred: !hc.bare_vector<!hc.pred, ["8"]>,
                                   %dst: !hc.ptr<global, f32>) {
  %m = hc.idx_apply () : () -> !hc.idx<"8">
  %n = hc.idx_apply () : () -> !hc.idx<"1">
  hc.generic
      iter (parallel i_0 = %m : !hc.idx<"8">,
            parallel i_1 = %n : !hc.idx<"1">)
      ins (%vec at [#hc.expr<"i_0 + i_1">] : !hc.bare_vector<f32, ["8"]>,
           %pred at [#hc.expr<"i_0 + i_1">] : !hc.bare_vector<!hc.pred, ["8"]>)
      outs (%dst at [#hc.expr<"i_0 + i_1">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%v: f32, %m_arg: !hc.pred, %init: f32):
    hc.yield_predicated %v mask %m_arg : (f32), (!hc.pred)
  }
  return
}

// -----

// Body-side iter-sym binding: the value-outs unroll surfaces each
// iter sym's compile-time lane value as an explicit `(%const as
// "i_0")` binding on every body `hc.idx_apply` / `hc.pred_apply`
// whose expression references the iter sym without listing it. The
// motivating consumer is the mask emitter — body computes
// `(lo + step * i_0) < D0` without a prior pass needing to know the
// concrete value of `i_0`. Here a 4-lane parallel sweep over a body
// that pred-applies `i_0 < 2` on the iter sym alone, then yields a
// per-lane predicated mask. After lowering, each lane's
// `hc.pred_apply` should bind `i_0` to the per-lane constant
// index — `0`, `1`, `2`, `3` in declaration order. The unpinned-
// pred UCC just rounds the pinned `hc.pred_apply` result into the
// `!hc.pred` shape `hc.yield_predicated`'s mask slot wants.
// CHECK-LABEL: func.func @value_out_body_iter_sym
// CHECK-NOT: scf.parallel
// Each lane's body pred_apply gets augmented with the lane const as
// the "i_0" binding. The const for parLane=0 isn't checked against a
// specific SSA name because `buildZeroIterScope` plants an unrelated
// `arith.constant 0` ahead of the lane consts and FileCheck would
// otherwise bind to that one. Subsequent lanes are pinned via
// CHECK-NEXT — the lane const is the line directly above its
// `hc.pred_apply`.
// CHECK: hc.pred_apply (%{{[^ ]+}} as "i_0") : (index) -> !hc.pred
// CHECK: %[[C1:[^ ]+]] = arith.constant 1 : index
// CHECK-NEXT: hc.pred_apply (%[[C1]] as "i_0") : (index) -> !hc.pred
// CHECK: %[[C2:[^ ]+]] = arith.constant 2 : index
// CHECK-NEXT: hc.pred_apply (%[[C2]] as "i_0") : (index) -> !hc.pred
// CHECK: %[[C3:[^ ]+]] = arith.constant 3 : index
// CHECK-NEXT: hc.pred_apply (%[[C3]] as "i_0") : (index) -> !hc.pred
// CHECK: vector.from_elements
// CHECK-NOT: hc.generic
func.func @value_out_body_iter_sym(%init: !hc.bare_vector<!hc.pred, ["4"]>)
    -> !hc.bare_vector<!hc.pred, ["4"]> {
  %n = hc.idx_apply () : () -> !hc.idx<"4">
  %r = hc.generic
      iter (parallel i_0 = %n : !hc.idx<"4">)
      ins ()
      outs (%init at [#hc.expr<"i_0">] : !hc.bare_vector<!hc.pred, ["4"]>)
      -> (!hc.bare_vector<!hc.pred, ["4"]>) {
  ^bb0(%iv: !hc.pred):
    %p_pinned = hc.pred_apply () : () -> !hc.pred<"i_0 < 2">
    %p = builtin.unrealized_conversion_cast %p_pinned
        : !hc.pred<"i_0 < 2"> to !hc.pred
    hc.yield_predicated %iv mask %p : (!hc.pred), (!hc.pred)
  }
  return %r : !hc.bare_vector<!hc.pred, ["4"]>
}

// -----

// `ins` is a workgroup LDS ptr — the form `hc-lower-launch-body`
// hands us when an `hc.zeros`-init shared tile feeds an `hc.generic`.
// The existing ptr-typed access path discovers the per-lane offsets
// are contig and emits a single vector-typed `hc.ptr_load`, then
// per-lane `vector.extract` for the body clones. No bare-tensor
// middleman and no UCC walk-back. The matching launch-body LIT
// (`workgroup_bare_tensor_ins_swap`) checks the ins-side retype
// that produces this shape.
// CHECK-LABEL: func.func @value_in_workgroup_ptr_ins
// CHECK-NOT: scf.parallel
// CHECK: %[[ALLOC:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
// CHECK: hc.ptr_load %{{.+}} : !hc.ptr<workgroup, f32> -> vector<8xf32>
// CHECK: vector.extract
// CHECK-NOT: builtin.unrealized_conversion_cast %{{.+}} : !hc.bare_tensor<f32, ["8"]> to vector<8xf32>
// CHECK-NOT: hc.generic
func.func @value_in_workgroup_ptr_ins(%dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %lds = hc.alloc count = %c8 : index -> !hc.ptr<workgroup, f32>
    %n = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %n : !hc.idx<"8">)
        ins (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %init: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}
