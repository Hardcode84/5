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

// CHECK-LABEL: func.func @ptr_types(
// CHECK-SAME: %{{.*}}: !hc.ptr<f16, addrspace = workgroup>
// CHECK-SAME: %{{.*}}: !hc.ptr<f32, addrspace = global>
// CHECK-SAME: %{{.*}}: !hc.ptr<addrspace = workgroup>
// CHECK-SAME: %{{.*}}: !hc.ptr<addrspace = global>
// CHECK-SAME: %{{.*}}: !hc.ptr<addrspace = private>
func.func @ptr_types(%a: !hc.ptr<f16, addrspace = workgroup>,
                     %b: !hc.ptr<f32, addrspace = global>,
                     %c: !hc.ptr<addrspace = workgroup>,
                     %d: !hc.ptr<addrspace = global>,
                     %e: !hc.ptr<addrspace = private>) {
  return
}

// Allocations cover all three address spaces and both typed/opaque
// flavours so the assembly keyword emission stays exhaustive.
// CHECK-LABEL: func.func @alloc_all_spaces
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<f16, addrspace = workgroup>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<f32, addrspace = global>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<i32, addrspace = private>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<addrspace = workgroup>
// CHECK: hc.alloc count = %{{.*}} : index -> !hc.ptr<addrspace = global>
func.func @alloc_all_spaces(%n: index) {
  %a = hc.alloc count = %n : index -> !hc.ptr<f16, addrspace = workgroup>
  %b = hc.alloc count = %n : index -> !hc.ptr<f32, addrspace = global>
  %c = hc.alloc count = %n : index -> !hc.ptr<i32, addrspace = private>
  %d = hc.alloc count = %n : index -> !hc.ptr<addrspace = workgroup>
  %e = hc.alloc count = %n : index -> !hc.ptr<addrspace = global>
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
// CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.ptr<f16, addrspace = workgroup>, index) -> !hc.ptr<f16, addrspace = workgroup>
func.func @ptr_offset_typed(%p: !hc.ptr<f16, addrspace = workgroup>,
                            %i: index) -> !hc.ptr<f16, addrspace = workgroup> {
  %r = hc.ptr_offset %p, %i
      : (!hc.ptr<f16, addrspace = workgroup>, index)
        -> !hc.ptr<f16, addrspace = workgroup>
  return %r : !hc.ptr<f16, addrspace = workgroup>
}

// CHECK-LABEL: func.func @ptr_offset_opaque
// CHECK: hc.ptr_offset %{{.*}}, %{{.*}} : (!hc.ptr<addrspace = global>, index) -> !hc.ptr<addrspace = global>
func.func @ptr_offset_opaque(%p: !hc.ptr<addrspace = global>,
                             %i: index) -> !hc.ptr<addrspace = global> {
  %r = hc.ptr_offset %p, %i
      : (!hc.ptr<addrspace = global>, index) -> !hc.ptr<addrspace = global>
  return %r : !hc.ptr<addrspace = global>
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
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<f16, addrspace = workgroup> -> f16
// CHECK: hc.ptr_store %[[V]], %{{.*}} : f16, !hc.ptr<f16, addrspace = workgroup>
func.func @ptr_load_store_typed(%p: !hc.ptr<f16, addrspace = workgroup>) {
  %v = hc.ptr_load %p : !hc.ptr<f16, addrspace = workgroup> -> f16
  hc.ptr_store %v, %p : f16, !hc.ptr<f16, addrspace = workgroup>
  return
}

// Opaque pointers carry the element type on the load/store op instead
// of on the pointer; verifier accepts any scalar as long as either
// side is opaque.
// CHECK-LABEL: func.func @ptr_load_store_opaque
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<addrspace = global> -> f32
// CHECK: hc.ptr_store %[[V]], %{{.*}} : f32, !hc.ptr<addrspace = global>
func.func @ptr_load_store_opaque(%p: !hc.ptr<addrspace = global>) {
  %v = hc.ptr_load %p : !hc.ptr<addrspace = global> -> f32
  hc.ptr_store %v, %p : f32, !hc.ptr<addrspace = global>
  return
}

// Pointer values flow through `scf.for` iter_args because `!hc.ptr` is
// a legal `HC_ValueType`; this is how the post-flatten lowering will
// hand a sliding workgroup pointer through a cooperative-copy loop.
// CHECK-LABEL: func.func @ptr_through_scf
// CHECK: scf.for %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} iter_args(%[[ARG:[^ ]+]] = %{{[^)]+}}) -> (!hc.ptr<f16, addrspace = workgroup>)
// CHECK: %[[NEXT:.*]] = hc.ptr_offset %[[ARG]], %{{.*}}
// CHECK: scf.yield %[[NEXT]] : !hc.ptr<f16, addrspace = workgroup>
func.func @ptr_through_scf(%p: !hc.ptr<f16, addrspace = workgroup>,
                           %lb: index, %ub: index, %step: index)
    -> !hc.ptr<f16, addrspace = workgroup> {
  %r = scf.for %i = %lb to %ub step %step
       iter_args(%cur = %p) -> (!hc.ptr<f16, addrspace = workgroup>) {
    %next = hc.ptr_offset %cur, %step
        : (!hc.ptr<f16, addrspace = workgroup>, index)
          -> !hc.ptr<f16, addrspace = workgroup>
    scf.yield %next : !hc.ptr<f16, addrspace = workgroup>
  }
  return %r : !hc.ptr<f16, addrspace = workgroup>
}
