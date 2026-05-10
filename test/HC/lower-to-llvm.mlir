// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-to-llvm`. The pass takes whatever launch-body
// (and, eventually, lower-generic) emits in the `!hc.ptr` family and
// finishes the LLVM-dialect lowering: workgroup `hc.alloc` to a sibling
// addrspace(3) global + `llvm.mlir.addressof` at the use site, private
// `hc.alloc` to `llvm.alloca` in addrspace 5, `hc.ptr_offset` to GEP,
// `hc.ptr_load`/`hc.ptr_store` to `llvm.load`/`llvm.store`, predicated
// pairs to either `llvm.intr.masked.{load,store}` (vector) or `scf.if`
// (scalar). See `doc/layouts.md` "hc.ptr and memory ops" for the full
// contract.
//
// RUN: hc-opt --hc-lower-to-llvm %s --split-input-file | FileCheck %s

// Workgroup allocation sized by a constant `index` count: emit one
// `llvm.mlir.global` per allocation in addrspace 3 with an
// `!llvm.array<N x T>` body, then plant `llvm.mlir.addressof` at the
// use site. The `_workgroup_alloc` test exercises both an alloc and a
// follow-up `hc.ptr_offset` + `hc.ptr_store` so the GEP and store
// machinery share a fixture.
// CHECK-LABEL: llvm.mlir.global private @__hc_workgroup
// CHECK-SAME: !llvm.array<256 x f16>
// CHECK-LABEL: func.func @workgroup_alloc(
// CHECK-SAME: %[[VAL:[^:]+]]: vector<8xf16>
// CHECK-SAME: %[[IDX:[^:]+]]: index
// CHECK: %[[BASE:.+]] = llvm.mlir.addressof @__hc_workgroup{{.*}} : !llvm.ptr<3>
// CHECK: %[[OFFI64:.+]] = arith.index_castui %[[IDX]] : index to i64
// CHECK: %[[GEP:.+]] = llvm.getelementptr %[[BASE]][%[[OFFI64]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
// CHECK: llvm.store %[[VAL]], %[[GEP]] : vector<8xf16>, !llvm.ptr<3>
func.func @workgroup_alloc(%v: vector<8xf16>, %off: index) {
  %c256 = arith.constant 256 : index
  %p = hc.alloc count = %c256 : index -> !hc.ptr<workgroup, f16>
  %q = hc.ptr_offset %p, %off
      : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup, f16>
  hc.ptr_store %v, %q : vector<8xf16>, !hc.ptr<workgroup, f16>
  return
}

// -----

// Private allocation lowers to `llvm.alloca` in addrspace 5. The count
// can be runtime — `llvm.alloca` handles a dynamic count natively, so
// the LIT pins the i64 cast that bridges the `index` count without
// forcing constness. Scalar load/store covers the non-vector path.
// CHECK-LABEL: func.func @private_alloc(
// CHECK-SAME: %[[V:[^:]+]]: f32
// CHECK-SAME: %[[N:[^:)]+]]: index
// CHECK: %[[NI64:.+]] = arith.index_castui %[[N]] : index to i64
// CHECK: %[[ALLOCA:.+]] = llvm.alloca %[[NI64]] x f32 : (i64) -> !llvm.ptr<5>
// CHECK: llvm.store %[[V]], %[[ALLOCA]] : f32, !llvm.ptr<5>
// CHECK: %[[L:.+]] = llvm.load %[[ALLOCA]] : !llvm.ptr<5> -> f32
// CHECK: return %[[L]] : f32
func.func @private_alloc(%v: f32, %n: index) -> f32 {
  %p = hc.alloc count = %n : index -> !hc.ptr<private, f32>
  hc.ptr_store %v, %p : f32, !hc.ptr<private, f32>
  %r = hc.ptr_load %p : !hc.ptr<private, f32> -> f32
  return %r : f32
}

// -----

// Vector predicated load lowers to `llvm.intr.masked.load`; vector
// predicated store to `llvm.intr.masked.store`. Both take an explicit
// alignment (we pass 0 — LLVM treats that as "natural alignment for the
// type"). The passthrough is required by the op verifier and rides
// straight through.
// CHECK-LABEL: func.func @masked_vector(
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<workgroup, f32>
// CHECK-SAME: %[[M:[^:]+]]: vector<4xi1>
// CHECK-SAME: %[[PT:[^:]+]]: vector<4xf32>
// CHECK-SAME: %[[V:[^:)]+]]: vector<4xf32>
// CHECK: %[[PCAST:.+]] = builtin.unrealized_conversion_cast %[[P]] : !hc.ptr<workgroup, f32> to !llvm.ptr<3>
// CHECK: %[[L:.+]] = llvm.intr.masked.load %[[PCAST]], %[[M]], %[[PT]] {alignment = 0 : i32} : (!llvm.ptr<3>, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
// CHECK: llvm.intr.masked.store %[[V]], %[[PCAST]], %[[M]] {alignment = 0 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr<3>
// CHECK: return %[[L]] : vector<4xf32>
func.func @masked_vector(%p: !hc.ptr<workgroup, f32>, %m: vector<4xi1>,
                         %pt: vector<4xf32>, %v: vector<4xf32>)
    -> vector<4xf32> {
  %r = hc.ptr_load_pred %p, %m passthrough %pt
      : !hc.ptr<workgroup, f32>, vector<4xi1>, vector<4xf32>
        -> vector<4xf32>
  hc.ptr_store_pred %v, %p, %m
      : vector<4xf32>, !hc.ptr<workgroup, f32>, vector<4xi1>
  return %r : vector<4xf32>
}

// -----

// Scalar predicated load lowers to an `scf.if` whose then-region runs
// `llvm.load` and whose else-region yields the passthrough; scalar
// predicated store lowers to a one-sided `scf.if` (no else) around an
// `llvm.store`. The masked-intrinsic ops are vector-only, so scalar
// has to go through scf.
// CHECK-LABEL: func.func @masked_scalar(
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<workgroup, f32>
// CHECK-SAME: %[[PRED:[^:]+]]: i1
// CHECK-SAME: %[[PT:[^:]+]]: f32
// CHECK-SAME: %[[V:[^:)]+]]: f32
// CHECK: %[[PCAST:.+]] = builtin.unrealized_conversion_cast %[[P]] : !hc.ptr<workgroup, f32> to !llvm.ptr<3>
// CHECK: %[[R:.+]] = scf.if %[[PRED]] -> (f32) {
// CHECK:   %[[L:.+]] = llvm.load %[[PCAST]] : !llvm.ptr<3> -> f32
// CHECK:   scf.yield %[[L]] : f32
// CHECK: } else {
// CHECK:   scf.yield %[[PT]] : f32
// CHECK: }
// CHECK: scf.if %[[PRED]] {
// CHECK:   llvm.store %[[V]], %[[PCAST]] : f32, !llvm.ptr<3>
// CHECK: }
// CHECK: return %[[R]] : f32
func.func @masked_scalar(%p: !hc.ptr<workgroup, f32>, %pred: i1,
                         %pt: f32, %v: f32) -> f32 {
  %r = hc.ptr_load_pred %p, %pred passthrough %pt
      : !hc.ptr<workgroup, f32>, i1, f32 -> f32
  hc.ptr_store_pred %v, %p, %pred
      : f32, !hc.ptr<workgroup, f32>, i1
  return %r : f32
}
