// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-insert-workgroup-barriers`: the pass that owns
// cross-`hc.generic` synchronization on workgroup-AS storage. Sits
// between the first `hc-lower-launch-body` and `hc-lower-generic` in
// the front_to_hc schedule. Conservative shape — barrier inserted iff
// some earlier generic in the same block has touched a workgroup-AS
// root the current generic reads or writes. Stray non-generic loads
// and stores on workgroup storage are deliberately out of scope.
//
// RUN: hc-opt --hc-insert-workgroup-barriers %s --split-input-file | FileCheck %s

// Init generic followed by a consumer generic on the same LDS root:
// the pass plants `gpu.barrier` between the two so the consumer sees
// the populated tile.
// CHECK-LABEL: func.func @writer_then_reader_same_lds
// CHECK: gpu.launch
// CHECK: %[[LDS:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
// CHECK: hc.generic
// CHECK-SAME: outs (%[[LDS]] at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
// CHECK: gpu.barrier
// CHECK: hc.generic
// CHECK-SAME: ins (%[[LDS]] at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
func.func @writer_then_reader_same_lds(%dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %total = arith.constant 8 : index
    %lin_b = hc.idx_apply () : () -> !hc.idx<"8">
    %lds = hc.alloc count = %total : index -> !hc.ptr<workgroup, f32>
    %zero = arith.constant 0.0 : f32
    hc.generic
        iter (parallel lin = %lin_b : !hc.idx<"8">)
        ins ()
        outs (%lds at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%init: f32):
      hc.yield %zero : f32
    }
    %iter_b = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Two consecutive writes to the same LDS root: writer-then-writer also
// barriers (the second write may publish over the first's slot before
// every thread of the first generic's chunk loop has finished).
// CHECK-LABEL: func.func @writer_then_writer_same_lds
// CHECK: %[[LDS:.+]] = hc.alloc
// CHECK: hc.generic
// CHECK-SAME: outs (%[[LDS]] at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
// CHECK: gpu.barrier
// CHECK: hc.generic
// CHECK-SAME: outs (%[[LDS]] at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
func.func @writer_then_writer_same_lds() {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %total = arith.constant 8 : index
    %lin_b = hc.idx_apply () : () -> !hc.idx<"8">
    %lds = hc.alloc count = %total : index -> !hc.ptr<workgroup, f32>
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    hc.generic
        iter (parallel lin = %lin_b : !hc.idx<"8">)
        ins ()
        outs (%lds at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%init: f32):
      hc.yield %zero : f32
    }
    hc.generic
        iter (parallel lin = %lin_b : !hc.idx<"8">)
        ins ()
        outs (%lds at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%init: f32):
      hc.yield %one : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Two consecutive reads on the same LDS root, with no intervening
// write: pending is empty between them, no barrier inserted.
// CHECK-LABEL: func.func @reader_then_reader_no_barrier
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
// CHECK-NOT: gpu.barrier
// CHECK: hc.generic
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
func.func @reader_then_reader_no_barrier(%lds: !hc.ptr<workgroup, f32>,
                                         %d1: !hc.ptr<global, f32>,
                                         %d2: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %iter_b = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%d1 at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%d2 at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Two generics on disjoint workgroup-AS allocations: writes to LDS A
// don't seed a barrier in front of reads on LDS B.
// CHECK-LABEL: func.func @disjoint_lds_no_barrier
// CHECK: %[[A:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
// CHECK: %[[B:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
// CHECK: hc.generic
// CHECK-SAME: outs (%[[A]] at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
// CHECK-NOT: gpu.barrier
// CHECK: hc.generic
// CHECK-SAME: ins (%[[B]] at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
func.func @disjoint_lds_no_barrier(%dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %total = arith.constant 8 : index
    %lin_b = hc.idx_apply () : () -> !hc.idx<"8">
    %a = hc.alloc count = %total : index -> !hc.ptr<workgroup, f32>
    %b = hc.alloc count = %total : index -> !hc.ptr<workgroup, f32>
    %zero = arith.constant 0.0 : f32
    hc.generic
        iter (parallel lin = %lin_b : !hc.idx<"8">)
        ins ()
        outs (%a at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%init: f32):
      hc.yield %zero : f32
    }
    %iter_b = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%b at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}

// -----

// A `gpu.barrier` already present between writer and reader: the pass
// notices, clears pending, and does not plant a sibling.
// CHECK-LABEL: func.func @existing_barrier_not_doubled
// CHECK: %[[LDS:.+]] = hc.alloc
// CHECK: hc.generic
// CHECK-SAME: outs (%[[LDS]] at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
// CHECK: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: hc.generic
// CHECK-SAME: ins (%[[LDS]] at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
func.func @existing_barrier_not_doubled(%dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %total = arith.constant 8 : index
    %lin_b = hc.idx_apply () : () -> !hc.idx<"8">
    %lds = hc.alloc count = %total : index -> !hc.ptr<workgroup, f32>
    %zero = arith.constant 0.0 : f32
    hc.generic
        iter (parallel lin = %lin_b : !hc.idx<"8">)
        ins ()
        outs (%lds at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%init: f32):
      hc.yield %zero : f32
    }
    gpu.barrier
    %iter_b = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Pointer aliased through `hc.ptr_offset`: a write at offset zero and
// a read at offset N share the same `hc.alloc workgroup` storage
// root, so the pass barriers the reader. Walking back through
// `hc.ptr_offset` is how the canonicalize step's slot-into-the-tile
// shape stays connected to its alloc for the pass.
// CHECK-LABEL: func.func @ptr_offset_alias_shares_root
// CHECK: %[[LDS:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
// CHECK: %[[SLOT:.+]] = hc.ptr_offset %[[LDS]]
// CHECK: hc.generic
// CHECK-SAME: outs (%[[LDS]] at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
// CHECK: gpu.barrier
// CHECK: hc.generic
// CHECK-SAME: ins (%[[SLOT]] at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
func.func @ptr_offset_alias_shares_root(%dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %total = arith.constant 8 : index
    %lin_b = hc.idx_apply () : () -> !hc.idx<"8">
    %lds = hc.alloc count = %total : index -> !hc.ptr<workgroup, f32>
    %slot = hc.ptr_offset %lds, %c4
        : (!hc.ptr<workgroup, f32>, index) -> !hc.ptr<workgroup, f32>
    %zero = arith.constant 0.0 : f32
    hc.generic
        iter (parallel lin = %lin_b : !hc.idx<"8">)
        ins ()
        outs (%lds at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
        -> () {
    ^bb0(%init: f32):
      hc.yield %zero : f32
    }
    %four_b = hc.idx_apply () : () -> !hc.idx<"4">
    hc.generic
        iter (parallel i = %four_b : !hc.idx<"4">)
        ins (%slot at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}

// -----

// Generic outside `gpu.launch`: the pass only walks launch bodies, so
// a free-standing pair of generics on a shared LDS pointer gets no
// barrier. Running this at the hc-opt level is the smoke-test slot;
// the production pipeline always wraps kernels in `gpu.launch` before
// this pass fires.
// CHECK-LABEL: func.func @generic_outside_launch_skipped
// CHECK-NOT: gpu.barrier
func.func @generic_outside_launch_skipped(%lds: !hc.ptr<workgroup, f32>,
                                          %dst: !hc.ptr<global, f32>) {
  %iter_b = hc.idx_apply () : () -> !hc.idx<"8">
  %zero = arith.constant 0.0 : f32
  hc.generic
      iter (parallel lin = %iter_b : !hc.idx<"8">)
      ins ()
      outs (%lds at [#hc.expr<"lin">] : !hc.ptr<workgroup, f32>)
      -> () {
  ^bb0(%init: f32):
    hc.yield %zero : f32
  }
  hc.generic
      iter (parallel i = %iter_b : !hc.idx<"8">)
      ins (%lds at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%v: f32, %d: f32):
    hc.yield %v : f32
  }
  return
}

// -----

// Global-only generic pair inside `gpu.launch`: no workgroup-AS storage
// touched, no barrier emitted.
// CHECK-LABEL: func.func @global_only_generics_no_barrier
// CHECK: gpu.launch
// CHECK-NOT: gpu.barrier
func.func @global_only_generics_no_barrier(%src: !hc.ptr<global, f32>,
                                           %dst: !hc.ptr<global, f32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {
    %iter_b = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    hc.generic
        iter (parallel i = %iter_b : !hc.idx<"8">)
        ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%v: f32, %d: f32):
      hc.yield %v : f32
    }
    gpu.terminator
  }
  return
}
