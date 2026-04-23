// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Flat-body coverage for `-hc-promote-names`: the callable has no
// region-carrying ops and every `hc.name_load` is reachable from a
// plain prior `hc.assign` in the same block. `hc.for_range` / `hc.if`
// / workitem/subgroup shapes live in `promote-names-regions.mlir`;
// negative paths in `promote-names-invalid.mlir`.
//
// RUN: hc-opt -hc-promote-names -split-input-file %s | FileCheck %s

// A single assign followed by a load: the load is replaced with the
// assigned value and both name-store ops are erased.
// CHECK-LABEL: hc.func @single_assign_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: hc.return %arg0
hc.func @single_assign_load(%a: !hc.undef) -> !hc.undef {
  hc.assign "x", %a : !hc.undef
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// A later `hc.assign` shadows an earlier one; the load sees the most
// recent write, not the earliest.
// CHECK-LABEL: hc.func @overwrite_assign
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: hc.return %arg1
hc.func @overwrite_assign(%a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  hc.assign "x", %a : !hc.undef
  hc.assign "x", %b : !hc.undef
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Distinct names coexist in the same block without bleeding into each
// other; both loads resolve to their own assign's value.
// CHECK-LABEL: hc.func @distinct_names
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: %[[SUM:.*]] = hc.add %arg0, %arg1
// CHECK: hc.return %[[SUM]]
hc.func @distinct_names(%a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  hc.assign "x", %a : !hc.undef
  hc.assign "y", %b : !hc.undef
  %va = hc.name_load "x" : !hc.undef
  %vb = hc.name_load "y" : !hc.undef
  %sum = hc.add %va, %vb : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// -----

// An assign with no subsequent load is still erased — the pass's
// post-condition is "no name-store ops survive", regardless of whether
// the write had a reader. (DCE of the dangling assign is the caller's
// responsibility before promotion runs if the write was always dead.)
// CHECK-LABEL: hc.func @dangling_assign
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: hc.return
hc.func @dangling_assign(%a: !hc.undef) {
  hc.assign "x", %a : !hc.undef
  hc.return
}

// -----

// Kernels also get promoted — same algorithm, different carrier op. A
// kernel's `hc.return` is operand-less, so the only effect visible in
// the IR is that the assign disappears.
// CHECK-LABEL: hc.kernel @kernel_assign
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: hc.return
hc.kernel @kernel_assign(%a: !hc.undef) {
  hc.assign "x", %a : !hc.undef
  %v = hc.name_load "x" : !hc.undef
  hc.return
}
