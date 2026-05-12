// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Round-trip coverage for the `!hc.ptr` family. The four ops carry the
// same address-space-and-element invariant; the verifier negatives live
// in `verify-hc.mlir` next to the rest of the dialect's "wrong spelling"
// pins.
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// Surface mirrors `!llvm.ptr<3>`: required address space first as a
// bare keyword, optional element type after the comma.
// CHECK-LABEL: func.func @ptr_types(
// CHECK-SAME: %{{.*}}: !hc.ptr<workgroup, f16>
// CHECK-SAME: %{{.*}}: !hc.ptr<global, f32>
// CHECK-SAME: %{{.*}}: !hc.ptr<workgroup>
// CHECK-SAME: %{{.*}}: !hc.ptr<global>
// CHECK-SAME: %{{.*}}: !hc.ptr<private>
func.func @ptr_types(%a: !hc.ptr<workgroup, f16>,
                     %b: !hc.ptr<global, f32>,
                     %c: !hc.ptr<workgroup>,
                     %d: !hc.ptr<global>,
                     %e: !hc.ptr<private>) {
  return
}

// Allocations cover all three address spaces and both typed/opaque
// flavours so the assembly keyword emission stays exhaustive.
// CHECK-LABEL: func.func @alloc_all_spaces
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup, f16>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<global, f32>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<private, i32>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<workgroup>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<global>
func.func @alloc_all_spaces(%n: index) {
  %a = hc.alloc count = %n : index -> !hc.ptr<workgroup, f16>
  %b = hc.alloc count = %n : index -> !hc.ptr<global, f32>
  %c = hc.alloc count = %n : index -> !hc.ptr<private, i32>
  %d = hc.alloc count = %n : index -> !hc.ptr<workgroup>
  %e = hc.alloc count = %n : index -> !hc.ptr<global>
  return
}

// CHECK-LABEL: func.func @alloc_progressive
// Pre-inference IR may carry !hc.undef for both the count and the
// returned pointer; the op surface stays parseable so the front-end
// can emit it before specialization runs.
// CHECK: hc.alloc count = %{{.*}} : !hc.undef -> !hc.undef
func.func @alloc_progressive(%n: !hc.undef) {
  %p = hc.alloc count = %n : !hc.undef -> !hc.undef
  return
}

// CHECK-LABEL: func.func @ptr_offset_typed
// CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup, f16>
func.func @ptr_offset_typed(%p: !hc.ptr<workgroup, f16>,
                            %i: index) -> !hc.ptr<workgroup, f16> {
  %r = hc.ptr_offset %p, %i
      : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup, f16>
  return %r : !hc.ptr<workgroup, f16>
}

// CHECK-LABEL: func.func @ptr_offset_opaque
// CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.ptr<global>, index) -> !hc.ptr<global>
func.func @ptr_offset_opaque(%p: !hc.ptr<global>,
                             %i: index) -> !hc.ptr<global> {
  %r = hc.ptr_offset %p, %i
      : (!hc.ptr<global>, index) -> !hc.ptr<global>
  return %r : !hc.ptr<global>
}

// Pre-inference operands stay legal: `!hc.undef` on either pointer side
// or on the index escapes the verifier parity check.
// CHECK-LABEL: func.func @ptr_offset_progressive
// CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
func.func @ptr_offset_progressive(%p: !hc.undef, %i: !hc.undef) -> !hc.undef {
  %r = hc.ptr_offset %p, %i : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// CHECK-LABEL: func.func @ptr_load_store_typed
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<workgroup, f16> -> f16
// CHECK: hc.ptr_store %[[V]], %{{.*}} : f16, !hc.ptr<workgroup, f16>
func.func @ptr_load_store_typed(%p: !hc.ptr<workgroup, f16>) {
  %v = hc.ptr_load %p : !hc.ptr<workgroup, f16> -> f16
  hc.ptr_store %v, %p : f16, !hc.ptr<workgroup, f16>
  return
}

// Opaque pointers carry the element type on the load/store op instead
// of on the pointer; verifier accepts any scalar as long as either
// side is opaque.
// CHECK-LABEL: func.func @ptr_load_store_opaque
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<global> -> f32
// CHECK: hc.ptr_store %[[V]], %{{.*}} : f32, !hc.ptr<global>
func.func @ptr_load_store_opaque(%p: !hc.ptr<global>) {
  %v = hc.ptr_load %p : !hc.ptr<global> -> f32
  hc.ptr_store %v, %p : f32, !hc.ptr<global>
  return
}

// Vector form: `hc.ptr_load` / `hc.ptr_store` accept `vector<NxT>` to
// denote a contiguous N-element access starting at the pointer. The
// pointer's element type matches `T`, never the whole vector type.
// Width is unconstrained at the op; the LLVM-lowering boundary owns
// the hardware-width split. This is the surface the vec slice of
// `hc-lower-generic` emits at merged contig groups.
// CHECK-LABEL: func.func @ptr_load_store_typed_vector
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<workgroup, f16> -> vector<8xf16>
// CHECK: hc.ptr_store %[[V]], %{{.*}} : vector<8xf16>, !hc.ptr<workgroup, f16>
func.func @ptr_load_store_typed_vector(%p: !hc.ptr<workgroup, f16>) {
  %v = hc.ptr_load %p : !hc.ptr<workgroup, f16> -> vector<8xf16>
  hc.ptr_store %v, %p : vector<8xf16>, !hc.ptr<workgroup, f16>
  return
}

// Opaque pointer + vector value: parity check is skipped on the pointer
// side, the vector width and element type are carried on the op.
// CHECK-LABEL: func.func @ptr_load_store_opaque_vector
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<global> -> vector<4xf32>
// CHECK: hc.ptr_store %[[V]], %{{.*}} : vector<4xf32>, !hc.ptr<global>
func.func @ptr_load_store_opaque_vector(%p: !hc.ptr<global>) {
  %v = hc.ptr_load %p : !hc.ptr<global> -> vector<4xf32>
  hc.ptr_store %v, %p : vector<4xf32>, !hc.ptr<global>
  return
}

// Predicated forms — scalar. Mirror the unconditional pair plus an i1
// predicate operand and (for the load) a passthrough fill that the op
// returns when the predicate is false. Pointer-element-type parity is
// the same rule as the unconditional ops; predicate is i1 for scalar
// values.
// CHECK-LABEL: func.func @ptr_load_store_pred_typed_scalar
// CHECK: %[[V:.*]] = hc.ptr_load_pred %{{.*}}, %{{.*}} passthrough %{{.*}} : !hc.ptr<workgroup, f16>, i1, f16 -> f16
// CHECK: hc.ptr_store_pred %[[V]], %{{.*}}, %{{.*}} : f16, !hc.ptr<workgroup, f16>, i1
func.func @ptr_load_store_pred_typed_scalar(%p: !hc.ptr<workgroup, f16>,
                                             %pred: i1, %fill: f16) {
  %v = hc.ptr_load_pred %p, %pred passthrough %fill
      : !hc.ptr<workgroup, f16>, i1, f16 -> f16
  hc.ptr_store_pred %v, %p, %pred : f16, !hc.ptr<workgroup, f16>, i1
  return
}

// Predicated forms — vector. The predicate is `vector<Nxi1>` with N
// matching the value's vector length; this is the surface the vec
// lowering emits at merged contig groups when the body had a
// per-lane mask. Parity check runs on the vector's element type, the
// per-lane mask's lane count must match.
// CHECK-LABEL: func.func @ptr_load_store_pred_typed_vector
// CHECK: %[[V:.*]] = hc.ptr_load_pred %{{.*}}, %{{.*}} passthrough %{{.*}} : !hc.ptr<workgroup, f16>, vector<8xi1>, vector<8xf16> -> vector<8xf16>
// CHECK: hc.ptr_store_pred %[[V]], %{{.*}}, %{{.*}} : vector<8xf16>, !hc.ptr<workgroup, f16>, vector<8xi1>
func.func @ptr_load_store_pred_typed_vector(%p: !hc.ptr<workgroup, f16>,
                                             %mask: vector<8xi1>,
                                             %fill: vector<8xf16>) {
  %v = hc.ptr_load_pred %p, %mask passthrough %fill
      : !hc.ptr<workgroup, f16>, vector<8xi1>, vector<8xf16>
      -> vector<8xf16>
  hc.ptr_store_pred %v, %p, %mask
      : vector<8xf16>, !hc.ptr<workgroup, f16>, vector<8xi1>
  return
}

// Opaque pointer + predicated vector access: pointer-side parity is
// skipped, predicate-shape parity still runs on the op.
// CHECK-LABEL: func.func @ptr_load_store_pred_opaque_vector
// CHECK: %[[V:.*]] = hc.ptr_load_pred %{{.*}}, %{{.*}} passthrough %{{.*}} : !hc.ptr<global>, vector<4xi1>, vector<4xf32> -> vector<4xf32>
// CHECK: hc.ptr_store_pred %[[V]], %{{.*}}, %{{.*}} : vector<4xf32>, !hc.ptr<global>, vector<4xi1>
func.func @ptr_load_store_pred_opaque_vector(%p: !hc.ptr<global>,
                                              %mask: vector<4xi1>,
                                              %fill: vector<4xf32>) {
  %v = hc.ptr_load_pred %p, %mask passthrough %fill
      : !hc.ptr<global>, vector<4xi1>, vector<4xf32> -> vector<4xf32>
  hc.ptr_store_pred %v, %p, %mask
      : vector<4xf32>, !hc.ptr<global>, vector<4xi1>
  return
}

// `hc.predicate` is the value-side counterpart of the predicated mem
// pair: a `mask ? value : passthrough` triple whose lowering hoists the
// predicate to the producer of `$value` (turning a plain `hc.ptr_load`
// into `hc.ptr_load_pred`, an extract into `arith.select`, ...). The
// surface op itself is `Pure` and parent-agnostic so producer rewriters
// and the `hc.generic` body lowering share it. Mask shape parity mirrors
// the predicated mem ops: scalar value with `i1`, vector value with
// `vector<Nxi1>` of the same N.
// CHECK-LABEL: func.func @predicate_scalar
// CHECK: %{{.+}} = hc.predicate %{{.+}} mask %{{.+}} passthrough %{{.+}} : f32, i1
func.func @predicate_scalar(%v: f32, %m: i1, %fill: f32) -> f32 {
  %r = hc.predicate %v mask %m passthrough %fill : f32, i1
  return %r : f32
}

// Vector-form `hc.predicate`: per-lane mask, per-lane passthrough. This
// is the shape the post-merge vector lowering picks when it collapses
// a contig group of width N into a single `vector<NxT>` register.
// CHECK-LABEL: func.func @predicate_vector
// CHECK: %{{.+}} = hc.predicate %{{.+}} mask %{{.+}} passthrough %{{.+}} : vector<4xf32>, vector<4xi1>
func.func @predicate_vector(%v: vector<4xf32>, %m: vector<4xi1>,
                            %fill: vector<4xf32>) -> vector<4xf32> {
  %r = hc.predicate %v mask %m passthrough %fill
      : vector<4xf32>, vector<4xi1>
  return %r : vector<4xf32>
}

// Pre-inference `!hc.undef` on either side escapes the shape-parity
// check, same policy the predicated mem ops use. Lets frontend / early
// passes emit the op before inference has pinned the operand types.
// CHECK-LABEL: func.func @predicate_undef
// CHECK: %{{.+}} = hc.predicate %{{.+}} mask %{{.+}} passthrough %{{.+}} : !hc.undef, !hc.undef
func.func @predicate_undef(%v: !hc.undef, %m: !hc.undef, %fill: !hc.undef)
    -> !hc.undef {
  %r = hc.predicate %v mask %m passthrough %fill : !hc.undef, !hc.undef
  return %r : !hc.undef
}

// Pointer values flow through `scf.for` iter_args because `!hc.ptr` is
// a legal `HC_ValueType`; this is how the post-flatten lowering will
// hand a sliding workgroup pointer through a cooperative-copy loop.
// CHECK-LABEL: func.func @ptr_through_scf
// CHECK: scf.for %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} iter_args(%[[ARG:[^ ]+]] = %{{[^)]+}}) -> (!hc.ptr<workgroup, f16>)
// CHECK: %[[NEXT:.*]] = hc.ptr_offset %[[ARG]], %{{.*}}
// CHECK: scf.yield %[[NEXT]] : !hc.ptr<workgroup, f16>
func.func @ptr_through_scf(%p: !hc.ptr<workgroup, f16>,
                           %lb: index, %ub: index, %step: index)
    -> !hc.ptr<workgroup, f16> {
  %r = scf.for %i = %lb to %ub step %step
       iter_args(%cur = %p) -> (!hc.ptr<workgroup, f16>) {
    %next = hc.ptr_offset %cur, %step
        : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup, f16>
    scf.yield %next : !hc.ptr<workgroup, f16>
  }
  return %r : !hc.ptr<workgroup, f16>
}
