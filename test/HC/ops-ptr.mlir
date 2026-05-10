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
