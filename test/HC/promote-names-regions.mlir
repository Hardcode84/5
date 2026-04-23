// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Stage-2 coverage for `-hc-promote-names`: region-carrying ops. Each
// case below starts with a `hc.assign` / `hc.name_load`-shaped body and
// expects the pass to rebuild the region op so every Python-level name
// becomes a real SSA edge:
//   - `hc.for_range` gets an `iter_args` thread per carried name;
//   - `hc.if` gets one result per carried name, with both branches
//     agreeing on the yield set (silent-branch fallback = outer
//     snapshot);
//   - nested ops are promoted bottom-up, so an inner `hc.if` becomes a
//     value flowing into the enclosing loop's yield.
//
// RUN: hc-opt -hc-promote-names -split-input-file %s | FileCheck %s

// Classic accumulator: the `acc` name is read and written inside the
// loop body, so it becomes an iter_arg seeded from the outer
// `hc.assign "acc", %init`. The final outer `hc.name_load "acc"`
// resolves to the loop's iter_result.
// CHECK-LABEL: hc.func @for_range_accumulator
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: %[[RES:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3) :
// CHECK-SAME: (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:        %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:        hc.yield %[[SUM]]
// CHECK: hc.return %[[RES]]
hc.func @for_range_accumulator(%lo: !hc.undef, %hi: !hc.undef,
                               %step: !hc.undef,
                               %init: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %prev = hc.name_load "acc" : !hc.undef
    %next = hc.add %prev, %iv : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "acc", %next : !hc.undef
    hc.yield
  }
  %final = hc.name_load "acc" : !hc.undef
  hc.return %final : !hc.undef
}

// -----

// Symmetric `hc.if`: both branches write the same name, so no outer
// snapshot is needed — each branch's own write supplies the yield
// value, and the op gains a single result that the outer
// `hc.name_load` consumes.
// CHECK-LABEL: hc.func @if_symmetric_rebind
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[R:.*]] = hc.if %arg0 -> (!hc.undef)
// CHECK:   hc.yield %arg1
// CHECK: } else {
// CHECK:   hc.yield %arg2
// CHECK: hc.return %[[R]]
hc.func @if_symmetric_rebind(%cond: !hc.undef, %a: !hc.undef,
                             %b: !hc.undef) -> !hc.undef {
  hc.if %cond : !hc.undef {
    hc.assign "x", %a : !hc.undef
    hc.yield
  } else {
    hc.assign "x", %b : !hc.undef
    hc.yield
  }
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Asymmetric `hc.if`: only the `then` branch writes. The else branch
// yields the outer snapshot of `x` (here `%seed`), matching Python's
// fall-through semantics for names bound in one arm.
// CHECK-LABEL: hc.func @if_asymmetric_write
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[R:.*]] = hc.if %arg0 -> (!hc.undef)
// CHECK:   hc.yield %arg2
// CHECK: } else {
// CHECK:   hc.yield %arg1
// CHECK: hc.return %[[R]]
hc.func @if_asymmetric_write(%cond: !hc.undef, %seed: !hc.undef,
                             %new: !hc.undef) -> !hc.undef {
  hc.assign "x", %seed : !hc.undef
  hc.if %cond : !hc.undef {
    hc.assign "x", %new : !hc.undef
    hc.yield
  } else {
    hc.yield
  }
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// `hc.if` with no explicit else and a carried write: the pass
// synthesizes an else region whose sole purpose is to yield the outer
// snapshot so the op's single result is well-defined on both paths.
// CHECK-LABEL: hc.func @if_no_else_carried
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[R:.*]] = hc.if %arg0 -> (!hc.undef)
// CHECK:   hc.yield %arg2
// CHECK: } else {
// CHECK:   hc.yield %arg1
// CHECK: hc.return %[[R]]
hc.func @if_no_else_carried(%cond: !hc.undef, %seed: !hc.undef,
                            %new: !hc.undef) -> !hc.undef {
  hc.assign "x", %seed : !hc.undef
  hc.if %cond : !hc.undef {
    hc.assign "x", %new : !hc.undef
    hc.yield
  }
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Nested: `hc.if` inside `hc.for_range`. The inner op is promoted
// first (post-order walk), so by the time the loop is rebuilt the `if`
// already produces a value the loop's new yield can thread through.
// CHECK-LABEL: hc.func @nested_if_in_for
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.name_load
// CHECK: %[[L:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:        %[[INNER:.*]] = hc.if %arg4 -> (!hc.undef)
// CHECK:          %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:          hc.yield %[[SUM]]
// CHECK:        } else {
// CHECK:          hc.yield %[[ACC]]
// CHECK:        hc.yield %[[INNER]]
// CHECK: hc.return %[[L]]
hc.func @nested_if_in_for(%lo: !hc.undef, %hi: !hc.undef,
                          %step: !hc.undef, %init: !hc.undef,
                          %cond: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %cur = hc.name_load "acc" : !hc.undef
    hc.if %cond : !hc.undef {
      %sum = hc.add %cur, %iv : (!hc.undef, !hc.undef) -> !hc.undef
      hc.assign "acc", %sum : !hc.undef
      hc.yield
    }
    hc.yield
  }
  %final = hc.name_load "acc" : !hc.undef
  hc.return %final : !hc.undef
}

// -----

// Nested: `hc.for_range` inside `hc.if`. The else branch doesn't write
// `acc`, so it falls back to the outer snapshot (`%init`); the then
// branch threads `acc` through the inner loop and yields the loop's
// iter_result.
// CHECK-LABEL: hc.func @nested_for_in_if
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[R:.*]] = hc.if %arg4 -> (!hc.undef)
// CHECK:   %[[L:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT:   ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:          %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:          hc.yield %[[SUM]]
// CHECK:   hc.yield %[[L]]
// CHECK: } else {
// CHECK:   hc.yield %arg3
// CHECK: hc.return %[[R]]
hc.func @nested_for_in_if(%lo: !hc.undef, %hi: !hc.undef,
                          %step: !hc.undef, %init: !hc.undef,
                          %cond: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.if %cond : !hc.undef {
    hc.for_range %lo to %hi step %step
        : (!hc.undef, !hc.undef, !hc.undef) -> () {
    ^bb0(%iv: !hc.undef):
      %cur = hc.name_load "acc" : !hc.undef
      %next = hc.add %cur, %iv : (!hc.undef, !hc.undef) -> !hc.undef
      hc.assign "acc", %next : !hc.undef
      hc.yield
    }
    hc.yield
  }
  %v = hc.name_load "acc" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Read-only name inside a loop body: the body loads `limit` but never
// assigns it. The loop grows no iter_arg for `limit` — only the
// accumulator is threaded — and the body's load is rewritten to the
// outer snapshot.
// CHECK-LABEL: hc.func @for_range_readonly_name
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[R:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:        %[[SUM:.*]] = hc.add %[[ACC]], %arg4
// CHECK:        hc.yield %[[SUM]]
// CHECK: hc.return %[[R]]
hc.func @for_range_readonly_name(%lo: !hc.undef, %hi: !hc.undef,
                                 %step: !hc.undef, %init: !hc.undef,
                                 %lim: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.assign "limit", %lim : !hc.undef
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %cur = hc.name_load "acc" : !hc.undef
    %l = hc.name_load "limit" : !hc.undef
    %sum = hc.add %cur, %l : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "acc", %sum : !hc.undef
    hc.yield
  }
  %final = hc.name_load "acc" : !hc.undef
  hc.return %final : !hc.undef
}

// -----

// `hc.workitem_region` is an isolated scope: assigns inside are local
// to the region and don't leak out. With no outbound mechanism yet,
// a body that only binds internally leaves no observable trace —
// every name-store op should be erased and nothing should flow to
// the enclosing kernel.
// CHECK-LABEL: hc.kernel @workitem_isolated_local_only
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.workitem_region {
// CHECK-NEXT: }
// CHECK: hc.return
hc.kernel @workitem_isolated_local_only(%v: !hc.undef) {
  hc.workitem_region {
    hc.assign "x", %v : !hc.undef
    %cur = hc.name_load "x" : !hc.undef
    hc.assign "y", %cur : !hc.undef
  }
  hc.return
}

// -----

// `hc.subgroup_region` with a nested `hc.for_range`: the inner
// accumulator is seeded *inside* the region (isolated scope forbids
// capturing outer bindings), and its name-store plumbing promotes
// against the region's own local store. Outer kernel state is
// untouched.
// CHECK-LABEL: hc.kernel @subgroup_isolated_local_for
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.subgroup_region {
// CHECK:   hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT:   ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:          %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:          hc.yield %[[SUM]]
// CHECK: hc.return
hc.kernel @subgroup_isolated_local_for(%lo: !hc.undef, %hi: !hc.undef,
                                       %step: !hc.undef,
                                       %init: !hc.undef) {
  hc.subgroup_region {
    hc.assign "acc", %init : !hc.undef
    hc.for_range %lo to %hi step %step
        : (!hc.undef, !hc.undef, !hc.undef) -> () {
    ^bb0(%iv: !hc.undef):
      %cur = hc.name_load "acc" : !hc.undef
      %next = hc.add %cur, %iv : (!hc.undef, !hc.undef) -> !hc.undef
      hc.assign "acc", %next : !hc.undef
      hc.yield
    }
  }
  hc.return
}

// -----

// Shadowing: the same name bound in the outer scope AND inside a
// `hc.workitem_region` are independent. The outer binding remains
// untouched by the in-region assign; a subsequent outer
// `hc.name_load` sees the outer value, never the region's local one.
// CHECK-LABEL: hc.func @workitem_shadows_outer_name
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.workitem_region {
// CHECK-NEXT: }
// CHECK: hc.return %arg0
hc.func @workitem_shadows_outer_name(%outer: !hc.undef,
                                     %inner: !hc.undef) -> !hc.undef {
  hc.assign "x", %outer : !hc.undef
  hc.workitem_region {
    hc.assign "x", %inner : !hc.undef
    %shadow = hc.name_load "x" : !hc.undef
    hc.assign "used", %shadow : !hc.undef
  }
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}
