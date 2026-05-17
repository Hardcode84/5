// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-fold-predicates -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @ptr_load_producer_scalar
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[M:[^:]+]]: i1
// CHECK-SAME: %[[F:[^)]+]]: f32
// `hc.ptr_load` collapses into `hc.ptr_load_pred` at the load's site —
// the predicate is gone, the load itself carries the mask and the
// passthrough now.
// CHECK: %[[R:.*]] = hc.ptr_load_pred %[[P]], %[[M]] passthrough %[[F]]
// CHECK-NOT: hc.predicate
// CHECK-NOT: hc.ptr_load %
// CHECK: return %[[R]]
func.func @ptr_load_producer_scalar(%p: !hc.ptr<global, f32>,
                                    %m: i1, %f: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %r = hc.predicate %v mask %m passthrough %f : f32, i1
  return %r : f32
}

// -----

// Vector predicate / vector mask: same shape, predicated vector load.
// CHECK-LABEL: func.func @ptr_load_producer_vector
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[M:[^:]+]]: vector<4xi1>
// CHECK-SAME: %[[F:[^)]+]]: vector<4xf32>
// CHECK: %[[R:.*]] = hc.ptr_load_pred %[[P]], %[[M]] passthrough %[[F]]
// CHECK-NOT: hc.predicate
// CHECK: return %[[R]]
func.func @ptr_load_producer_vector(%p: !hc.ptr<global, f32>,
                                    %m: vector<4xi1>,
                                    %f: vector<4xf32>) -> vector<4xf32> {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> vector<4xf32>
  %r = hc.predicate %v mask %m passthrough %f : vector<4xf32>, vector<4xi1>
  return %r : vector<4xf32>
}

// -----

// `vector.extract` producer: the vector load behind the extract stays
// unconditional, the predicate becomes a per-lane `arith.select` at the
// predicate site.
// CHECK-LABEL: func.func @extract_producer
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[M:[^:]+]]: i1
// CHECK-SAME: %[[F:[^)]+]]: f32
// CHECK: %[[V:.*]] = hc.ptr_load %[[P]]
// CHECK-NOT: hc.ptr_load_pred
// CHECK: %[[E:.*]] = vector.extract %[[V]][1] : f32 from vector<4xf32>
// CHECK: %[[R:.*]] = arith.select %[[M]], %[[E]], %[[F]] : f32
// CHECK-NOT: hc.predicate
// CHECK: return %[[R]]
func.func @extract_producer(%p: !hc.ptr<global, f32>,
                            %m: i1, %f: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> vector<4xf32>
  %e = vector.extract %v[1] : f32 from vector<4xf32>
  %r = hc.predicate %e mask %m passthrough %f : f32, i1
  return %r : f32
}

// -----

// Two predicates on the same load: each predicate produces its own
// clone at the load's site. The original load is dropped (no
// non-predicated uses remain).
// CHECK-LABEL: func.func @two_predicates_one_load
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[M1:[^:]+]]: i1
// CHECK-SAME: %[[F1:[^,]+]]: f32
// CHECK-SAME: %[[M2:[^:]+]]: i1
// CHECK-SAME: %[[F2:[^)]+]]: f32
// CHECK-DAG: %[[R1:.*]] = hc.ptr_load_pred %[[P]], %[[M1]] passthrough %[[F1]]
// CHECK-DAG: %[[R2:.*]] = hc.ptr_load_pred %[[P]], %[[M2]] passthrough %[[F2]]
// CHECK-NOT: hc.predicate
// CHECK-NOT: hc.ptr_load %[[P]] :
// CHECK: arith.addf %[[R1]], %[[R2]] : f32
func.func @two_predicates_one_load(%p: !hc.ptr<global, f32>,
                                   %m1: i1, %f1: f32,
                                   %m2: i1, %f2: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %r1 = hc.predicate %v mask %m1 passthrough %f1 : f32, i1
  %r2 = hc.predicate %v mask %m2 passthrough %f2 : f32, i1
  %s = arith.addf %r1, %r2 : f32
  return %s : f32
}

// -----

// Mixed: one unpredicated use + one predicate keeps the unpredicated
// load alive and clones a predicated form alongside.
// CHECK-LABEL: func.func @mixed_use
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[M:[^:]+]]: i1
// CHECK-SAME: %[[F:[^)]+]]: f32
// CHECK-DAG: %[[V:.*]] = hc.ptr_load %[[P]] :
// CHECK-DAG: %[[R:.*]] = hc.ptr_load_pred %[[P]], %[[M]] passthrough %[[F]]
// CHECK: arith.addf %[[V]], %[[R]] : f32
func.func @mixed_use(%p: !hc.ptr<global, f32>, %m: i1, %f: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %r = hc.predicate %v mask %m passthrough %f : f32, i1
  %s = arith.addf %v, %r : f32
  return %s : f32
}

// -----

// `m_One()` scalar i1: predicate drops, result is the unpredicated
// value.
// CHECK-LABEL: func.func @fold_const_true_scalar
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[F:[^)]+]]: f32
// CHECK: %[[V:.*]] = hc.ptr_load %[[P]] :
// CHECK-NOT: hc.ptr_load_pred
// CHECK-NOT: hc.predicate
// CHECK: return %[[V]]
func.func @fold_const_true_scalar(%p: !hc.ptr<global, f32>, %f: f32) -> f32 {
  %t = arith.constant true
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %r = hc.predicate %v mask %t passthrough %f : f32, i1
  return %r : f32
}

// -----

// `m_Zero()` scalar i1: predicate drops to the passthrough; the load
// is dead and gets cleaned up by the m_Zero branch's trivially-dead
// sweep ("load elided" semantics).
// CHECK-LABEL: func.func @fold_const_false_scalar
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[F:[^)]+]]: f32
// CHECK-NOT: hc.ptr_load
// CHECK-NOT: hc.predicate
// CHECK: return %[[F]]
func.func @fold_const_false_scalar(%p: !hc.ptr<global, f32>, %f: f32) -> f32 {
  %z = arith.constant false
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %r = hc.predicate %v mask %z passthrough %f : f32, i1
  return %r : f32
}

// -----

// `m_One()` vector splat: same fold path picks up the splat constant.
// CHECK-LABEL: func.func @fold_const_true_vector
// CHECK: %[[V:.*]] = hc.ptr_load %{{.*}} : !hc.ptr<global, f32> -> vector<4xf32>
// CHECK-NOT: hc.ptr_load_pred
// CHECK-NOT: hc.predicate
// CHECK: return %[[V]]
func.func @fold_const_true_vector(%p: !hc.ptr<global, f32>,
                                  %f: vector<4xf32>) -> vector<4xf32> {
  %t = arith.constant dense<true> : vector<4xi1>
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> vector<4xf32>
  %r = hc.predicate %v mask %t passthrough %f : vector<4xf32>, vector<4xi1>
  return %r : vector<4xf32>
}

// -----

// Mask defined AFTER the load in the same block: the fold pass
// hoists the pure mask chain back across the load so dominance
// holds, then collapses into `hc.ptr_load_pred`. Mirrors the
// pattern `hc-load-store-to-generic` plants in its load body
// (`pred_apply` + UCC + `predicate`) — after `hc-lower-generic`
// materialises the load, the chain lands past it and needs the
// hoist to fold.
// CHECK-LABEL: func.func @hoist_mask_chain
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[LHS:[^:]+]]: i32
// CHECK-SAME: %[[RHS:[^:]+]]: i32
// CHECK-SAME: %[[F:[^)]+]]: f32
// CHECK: %[[M:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i32
// CHECK: %[[R:.*]] = hc.ptr_load_pred %[[P]], %[[M]] passthrough %[[F]]
// CHECK-NOT: hc.predicate
// CHECK-NOT: hc.ptr_load %
// CHECK: return %[[R]]
func.func @hoist_mask_chain(%p: !hc.ptr<global, f32>,
                            %lhs: i32, %rhs: i32, %f: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %m = arith.cmpi slt, %lhs, %rhs : i32
  %r = hc.predicate %v mask %m passthrough %f : f32, i1
  return %r : f32
}

// -----

// Same shape for passthrough: pure compute defined after the load
// hoists too. `arith.negf` rides along with the cmpi.
// CHECK-LABEL: func.func @hoist_passthrough_chain
// CHECK-SAME: %[[P:[^:]+]]: !hc.ptr<global, f32>
// CHECK-SAME: %[[M:[^:]+]]: i1
// CHECK-SAME: %[[SRC:[^)]+]]: f32
// CHECK: %[[F:.*]] = arith.negf %[[SRC]] : f32
// CHECK: %[[R:.*]] = hc.ptr_load_pred %[[P]], %[[M]] passthrough %[[F]]
// CHECK-NOT: hc.predicate
// CHECK: return %[[R]]
func.func @hoist_passthrough_chain(%p: !hc.ptr<global, f32>,
                                   %m: i1, %src: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %f = arith.negf %src : f32
  %r = hc.predicate %v mask %m passthrough %f : f32, i1
  return %r : f32
}
