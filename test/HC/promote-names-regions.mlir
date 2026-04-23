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

// `hc.workitem_region` is a nested scope: reads capture outer
// bindings, writes shadow. A body that only writes and then reads
// its own writes needs no outer capture — local binding wins — and
// every name-store op erases cleanly, leaving an empty region.
// CHECK-LABEL: hc.kernel @workitem_nested_local_only
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.workitem_region {
// CHECK-NEXT: }
// CHECK: hc.return
hc.kernel @workitem_nested_local_only(%v: !hc.undef) {
  hc.workitem_region {
    hc.assign "x", %v : !hc.undef
    %cur = hc.name_load "x" : !hc.undef
    hc.assign "y", %cur : !hc.undef
  }
  hc.return
}

// -----

// `hc.subgroup_region` with a nested `hc.for_range`: the accumulator
// is seeded *inside* the region, so no outer capture is materialized;
// the for-range snapshot resolves against the region's own local
// store, and nothing leaks back to the enclosing kernel.
// CHECK-LABEL: hc.kernel @subgroup_nested_local_for
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.subgroup_region {
// CHECK:   hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT:   ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:          %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:          hc.yield %[[SUM]]
// CHECK: hc.return
hc.kernel @subgroup_nested_local_for(%lo: !hc.undef, %hi: !hc.undef,
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
// `hc.workitem_region`. The in-region write is local; it doesn't
// leak back to the outer name store, so the outer `hc.name_load`
// resolves to `%outer`. The in-region read sees the local write
// (which arrived first lexically), never the captured outer — so no
// outer snapshot is materialized for this body.
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

// -----

// Outer capture: `hc.name_load` inside a `hc.workitem_region` with no
// prior in-region write falls back to an outer snapshot. The pass
// materializes a lazy `hc.name_load "x"` just before the region op,
// which the outer flat sweep then resolves to the outer `%a`.
// Post-condition: no name-store ops survive anywhere.
// CHECK-LABEL: hc.kernel @workitem_captures_outer
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.workitem_region {
// CHECK-NEXT: }
// CHECK: hc.return
hc.kernel @workitem_captures_outer(%a: !hc.undef) {
  hc.assign "x", %a : !hc.undef
  hc.workitem_region {
    %v = hc.name_load "x" : !hc.undef
    hc.assign "used", %v : !hc.undef
  }
  hc.return
}

// -----

// Same capture mechanism for `hc.subgroup_region`: an in-region read
// with no prior local write resolves to the outer binding via a
// lazy outer snapshot.
// CHECK-LABEL: hc.kernel @subgroup_captures_outer
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.subgroup_region {
// CHECK-NEXT: }
// CHECK: hc.return
hc.kernel @subgroup_captures_outer(%a: !hc.undef) {
  hc.assign "x", %a : !hc.undef
  hc.subgroup_region {
    %v = hc.name_load "x" : !hc.undef
    hc.assign "used", %v : !hc.undef
  }
  hc.return
}

// -----

// Capture-then-shadow: a read of an outer-bound name followed by a
// local write followed by another read. The first read captures
// `%outer`; the local write flips the binding; the second read sees
// the local value. The outer binding stays `%outer` throughout — the
// shadowing write never leaks back.
// CHECK-LABEL: hc.func @workitem_capture_then_shadow
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.workitem_region {
// CHECK-NEXT: }
// CHECK: hc.return %arg0
hc.func @workitem_capture_then_shadow(%outer: !hc.undef,
                                      %local: !hc.undef) -> !hc.undef {
  hc.assign "x", %outer : !hc.undef
  hc.workitem_region {
    %first = hc.name_load "x" : !hc.undef
    hc.assign "captured", %first : !hc.undef
    hc.assign "x", %local : !hc.undef
    %second = hc.name_load "x" : !hc.undef
    hc.assign "local", %second : !hc.undef
  }
  %v = hc.name_load "x" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// `hc.for_range` inside a `hc.workitem_region` whose accumulator is
// bound in the enclosing scope: the for-range snap resolves against
// the region body, which in turn captures upward from the outer
// `hc.assign "acc", %init`. The whole chain threads cleanly and the
// outer binding is never rewritten (no out-of-region leak).
// CHECK-LABEL: hc.func @workitem_capturing_for
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.workitem_region {
// CHECK:   hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT:   ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:          %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:          hc.yield %[[SUM]]
// CHECK: hc.return %arg3
hc.func @workitem_capturing_for(%lo: !hc.undef, %hi: !hc.undef,
                                %step: !hc.undef,
                                %init: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.workitem_region {
    hc.for_range %lo to %hi step %step
        : (!hc.undef, !hc.undef, !hc.undef) -> () {
    ^bb0(%iv: !hc.undef):
      %cur = hc.name_load "acc" : !hc.undef
      %next = hc.add %cur, %iv : (!hc.undef, !hc.undef) -> !hc.undef
      hc.assign "acc", %next : !hc.undef
      hc.yield
    }
  }
  %final = hc.name_load "acc" : !hc.undef
  hc.return %final : !hc.undef
}
