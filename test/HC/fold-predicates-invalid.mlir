// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative-path coverage for `-hc-fold-predicates`: producers outside
// the allow-list (or no producer at all) must surface diagnostics
// rather than silently survive.
//
// RUN: hc-opt -hc-fold-predicates -split-input-file -verify-diagnostics %s

// Block-arg producer: nothing to hoist into.
func.func @block_arg_producer(%v: f32, %m: i1, %f: f32) -> f32 {
  // expected-error @+1 {{'hc.predicate' op predicate value is a block argument; supported producers are hc.ptr_load and vector.extract}}
  %r = hc.predicate %v mask %m passthrough %f : f32, i1
  return %r : f32
}

// -----

// Arbitrary producer (arith.addf here): unsupported, diagnostic.
func.func @unsupported_producer(%a: f32, %b: f32, %m: i1, %f: f32) -> f32 {
  %s = arith.addf %a, %b : f32
  // expected-error @+1 {{'hc.predicate' op unsupported producer for predicate hoist: 'arith.addf'; supported producers are hc.ptr_load and vector.extract}}
  %r = hc.predicate %s mask %m passthrough %f : f32, i1
  return %r : f32
}

// -----

// Already-predicated load + extra `hc.predicate`: hard error, don't
// silently combine masks. Re-predicating a predicated load is a
// design smell at the source; combining is opt-in if it ever ships.
func.func @double_predicate(%p: !hc.ptr<global, f32>,
                            %m1: i1, %f1: f32,
                            %m2: i1, %f2: f32) -> f32 {
  %v = hc.ptr_load_pred %p, %m1 passthrough %f1
      : !hc.ptr<global, f32>, i1, f32 -> f32
  // expected-error @+1 {{'hc.predicate' op value is already produced by hc.ptr_load_pred; double-predicating is not supported (combine masks at the producer)}}
  %r = hc.predicate %v mask %m2 passthrough %f2 : f32, i1
  return %r : f32
}

// -----

// Mask producer with memory effects: pure-only hoist refuses to move
// a `hc.ptr_load`-produced mask back across the data load.
func.func @mask_after_load_with_side_effect(%p: !hc.ptr<global, f32>,
                                            %mp: !hc.ptr<global, i1>,
                                            %f: f32) -> f32 {
  %v = hc.ptr_load %p : !hc.ptr<global, f32> -> f32
  %m = hc.ptr_load %mp : !hc.ptr<global, i1> -> i1
  // expected-error @+1 {{'hc.predicate' op mask does not dominate the hc.ptr_load producer}}
  %r = hc.predicate %v mask %m passthrough %f : f32, i1
  return %r : f32
}
