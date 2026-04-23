// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Behavioral coverage for the `MemoryEffects` traits on `hc.assign` and
// `hc.name_load`. Roundtrip / region coverage lives in
// `test/HC/ops-name-store.mlir`; this file only asserts what the effects
// buy us under the standard MLIR infrastructure passes.
//
// `hc.name_load` carries `MemRead`: loads with no uses are trivially dead
// (MLIR's `wouldOpBeTriviallyDead` deletes read-only ops with no uses),
// but standard `-cse` skips ops with non-empty effects, so two loads of
// the same name — between or around an `hc.assign` — are never collapsed.
//
// `hc.assign` carries `MemWrite`: writes are never considered trivially
// dead even when the op has no SSA result, so an assign with no follow-up
// load still survives canonicalization.
//
// RUN: hc-opt -canonicalize -cse -split-input-file %s | FileCheck %s

// An unused `hc.name_load` is deleted by canonicalize; the preceding
// `hc.assign` stays because writes are observable.
// CHECK-LABEL: func.func @dce_unused_load
// CHECK: hc.assign "x"
// CHECK-NOT: hc.name_load
// CHECK: return
func.func @dce_unused_load(%arg0: !hc.undef) {
  hc.assign "x", %arg0 : !hc.undef
  %dead = hc.name_load "x" : !hc.undef
  return
}

// -----

// Two loads of the same name across an intervening `hc.assign` must not
// be collapsed by `-cse` — they can observe different bindings. The
// function returns both so the loads are unambiguously live.
// CHECK-LABEL: func.func @no_cse_across_assign
// CHECK: hc.name_load "x"
// CHECK: hc.assign "x"
// CHECK: hc.name_load "x"
// CHECK: return
func.func @no_cse_across_assign(%arg0: !hc.undef, %arg1: !hc.undef)
    -> (!hc.undef, !hc.undef) {
  hc.assign "x", %arg0 : !hc.undef
  %a = hc.name_load "x" : !hc.undef
  hc.assign "x", %arg1 : !hc.undef
  %b = hc.name_load "x" : !hc.undef
  return %a, %b : !hc.undef, !hc.undef
}

// -----

// An `hc.assign` with no subsequent load still survives canonicalization;
// the write is side-effecting so DCE leaves it alone. Promotion, not the
// canonicalizer, is responsible for erasing name-store ops.
// CHECK-LABEL: func.func @assign_survives
// CHECK: hc.assign "x"
// CHECK: return
func.func @assign_survives(%arg0: !hc.undef) {
  hc.assign "x", %arg0 : !hc.undef
  return
}
