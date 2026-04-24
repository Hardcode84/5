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

// Loop-local temporary: the body writes `t` before reading it, and
// no outer `hc.assign "t"` exists. The promoter must not emit an
// outer `hc.name_load "t"` as the iter_init (there would be no
// reaching assign for the callable-level flat sweep to resolve);
// instead, the iter_arg value is semantically dead on every iter
// (the first in-body assign overwrites it), and the op carries it
// out via a placeholder-seeded iter_result. `unrealized_conversion_cast`
// is the expected placeholder — it's MLIR's "type-correct value with
// no producer" scaffolding, and later inference passes drop it when
// the type gets pinned.
// CHECK-LABEL: hc.func @for_range_loop_local_temp
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[UNDEF:.*]] = builtin.unrealized_conversion_cast to !hc.undef
// CHECK: %[[R:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%[[UNDEF]])
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef, %{{.*}}: !hc.undef):
// CHECK:        %[[T:.*]] = hc.add %[[IV]], %[[IV]]
// CHECK:        %{{.*}} = hc.add %[[T]], %[[IV]]
// CHECK:        hc.yield %[[T]]
// CHECK: hc.return
hc.func @for_range_loop_local_temp(%lo: !hc.undef, %hi: !hc.undef,
                                   %step: !hc.undef) {
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %x = hc.add %iv, %iv : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "t", %x : !hc.undef
    %t = hc.name_load "t" : !hc.undef
    %y = hc.add %t, %iv : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield
  }
  hc.return
}

// -----

// Mixed shape: accumulator `acc` is read-before-written (needs outer
// snap), loop-local `tmp` is written-before-read (no snap — the
// assign feeds the read directly). Carried order follows the
// frontend's in-body write order (`tmp` is assigned before `acc`),
// so iter_args / iter_results / yield all run `[tmp, acc]`. The
// promoter must emit an outer `hc.name_load "acc"` but not
// `hc.name_load "tmp"`.
// CHECK-LABEL: hc.func @for_range_mixed_local_and_accumulator
// CHECK-NOT: hc.name_load "tmp"
// CHECK: %[[TMP_UNDEF:.*]] = builtin.unrealized_conversion_cast to !hc.undef
// CHECK: %[[R:.*]]:2 = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%[[TMP_UNDEF]], %arg3)
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef, %{{.*}}: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:        %[[DBL:.*]] = hc.add %[[IV]], %[[IV]]
// CHECK:        %[[NEW_ACC:.*]] = hc.add %[[ACC]], %[[DBL]]
// CHECK:        hc.yield %[[DBL]], %[[NEW_ACC]]
// CHECK: hc.return %[[R]]#1
hc.func @for_range_mixed_local_and_accumulator(
    %lo: !hc.undef, %hi: !hc.undef, %step: !hc.undef,
    %init: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %dbl = hc.add %iv, %iv : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "tmp", %dbl : !hc.undef
    %t = hc.name_load "tmp" : !hc.undef
    %cur = hc.name_load "acc" : !hc.undef
    %sum = hc.add %cur, %t : (!hc.undef, !hc.undef) -> !hc.undef
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

// -----

// "Return acc" shape on `hc.workitem_region`. The body assigns `acc`
// locally, then `hc.region_return ["acc"]` asks the pass to lift that
// binding as the op's result. Post-promotion the region carries one
// `!hc.undef` result, the terminator is `hc.yield %acc`, and the
// outer scope consumes that result through the writeback assign —
// which the flat sweep then fuses into the outer `hc.name_load "acc"`.
// CHECK-LABEL: hc.func @workitem_return_local
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.region_return
// CHECK: %[[R:.*]] = hc.workitem_region -> (!hc.undef) {
// CHECK:   %[[ADD:.*]] = hc.add %arg0, %arg1
// CHECK:   hc.yield %[[ADD]] : !hc.undef
// CHECK: }
// CHECK: hc.return %[[R]]
hc.func @workitem_return_local(%a: !hc.undef,
                               %b: !hc.undef) -> !hc.undef {
  hc.workitem_region {
    %sum = hc.add %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "acc", %sum : !hc.undef
    hc.region_return ["acc"]
  }
  %v = hc.name_load "acc" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Same shape on `hc.subgroup_region` — covers the second op kind so
// the dispatch-case hasn't bit-rotted.
// CHECK-LABEL: hc.func @subgroup_return_local
// CHECK: %[[R:.*]] = hc.subgroup_region -> (!hc.undef) {
// CHECK:   hc.yield %arg0 : !hc.undef
// CHECK: }
// CHECK: hc.return %[[R]]
hc.func @subgroup_return_local(%x: !hc.undef) -> !hc.undef {
  hc.subgroup_region {
    hc.assign "out", %x : !hc.undef
    hc.region_return ["out"]
  }
  %v = hc.name_load "out" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Multi-name return: two bindings surface as a `(!hc.undef, !hc.undef)`
// result pair. The frontend order in `hc.region_return` is preserved
// result-by-result; the writeback assigns land one per name and the
// outer reads pick up the matching result.
// CHECK-LABEL: hc.func @workitem_return_multi
// CHECK: %[[R:.*]]:2 = hc.workitem_region -> (!hc.undef, !hc.undef) {
// CHECK:   hc.yield %arg0, %arg1 : !hc.undef, !hc.undef
// CHECK: }
// CHECK: %[[SUM:.*]] = hc.add %[[R]]#0, %[[R]]#1
// CHECK: hc.return %[[SUM]]
hc.func @workitem_return_multi(%a: !hc.undef,
                               %b: !hc.undef) -> !hc.undef {
  hc.workitem_region {
    hc.assign "lhs", %a : !hc.undef
    hc.assign "rhs", %b : !hc.undef
    hc.region_return ["lhs", "rhs"]
  }
  %l = hc.name_load "lhs" : !hc.undef
  %r = hc.name_load "rhs" : !hc.undef
  %sum = hc.add %l, %r : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// -----

// Capture-in + return-out round-trip. The body reads `seed` (bound in
// the enclosing kernel) via the lazy outer snap, assigns `acc`, and
// returns it. Post-promotion:
//   - no outer snap for `seed` shows up as an `hc.name_load`
//     *outside* the region op (the flat sweep resolves it against
//     the outer `hc.assign "seed", %init`);
//   - the region op has one `!hc.undef` result;
//   - the writeback + outer load on `acc` collapses into a direct
//     use of that result.
// CHECK-LABEL: hc.func @workitem_capture_then_return
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.region_return
// CHECK: %[[R:.*]] = hc.workitem_region -> (!hc.undef) {
// CHECK:   %[[SUM:.*]] = hc.add %arg0, %arg1
// CHECK:   hc.yield %[[SUM]] : !hc.undef
// CHECK: }
// CHECK: hc.return %[[R]]
hc.func @workitem_capture_then_return(%init: !hc.undef,
                                      %delta: !hc.undef) -> !hc.undef {
  hc.assign "seed", %init : !hc.undef
  hc.workitem_region {
    %s = hc.name_load "seed" : !hc.undef
    %sum = hc.add %s, %delta : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "acc", %sum : !hc.undef
    hc.region_return ["acc"]
  }
  %v = hc.name_load "acc" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Return shape composed with an inner `hc.for_range`. The loop promotes
// first (post-order), leaving its snap/writeback assigns at the region
// body's top level; the region then scans, picks up the final `acc`
// binding (the loop's writeback), and returns it. Verifies that inner
// and outer carrying mechanisms plumb together cleanly — the region's
// result is the loop's accumulator, propagated through a single
// `hc.yield` rather than a chain of intermediate assigns.
// CHECK-LABEL: hc.func @subgroup_return_after_inner_for
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.region_return
// CHECK: %[[R:.*]] = hc.subgroup_region -> (!hc.undef) {
// CHECK:   %[[LOOP:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3)
// CHECK-NEXT:   ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:          %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:          hc.yield %[[SUM]] : !hc.undef
// CHECK:   hc.yield %[[LOOP]] : !hc.undef
// CHECK: }
// CHECK: hc.return %[[R]]
hc.func @subgroup_return_after_inner_for(%lo: !hc.undef, %hi: !hc.undef,
                                         %step: !hc.undef,
                                         %init: !hc.undef) -> !hc.undef {
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
    hc.region_return ["acc"]
  }
  %v = hc.name_load "acc" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Return shape composed with an inner `hc.if`. Both branches assign
// `acc`; the if-promotion produces an `hc.if -> (!hc.undef)` whose
// result becomes the region's `acc` binding, which in turn surfaces
// as the region's result.
// CHECK-LABEL: hc.func @workitem_return_after_inner_if
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.region_return
// CHECK: %[[R:.*]] = hc.workitem_region -> (!hc.undef) {
// CHECK:   %[[IF:.*]] = hc.if %arg0 -> (!hc.undef) : !hc.undef {
// CHECK:     hc.yield %arg1 : !hc.undef
// CHECK:   } else {
// CHECK:     hc.yield %arg2 : !hc.undef
// CHECK:   }
// CHECK:   hc.yield %[[IF]] : !hc.undef
// CHECK: }
// CHECK: hc.return %[[R]]
hc.func @workitem_return_after_inner_if(%cond: !hc.undef,
                                        %a: !hc.undef,
                                        %b: !hc.undef) -> !hc.undef {
  hc.workitem_region {
    hc.if %cond : !hc.undef {
      hc.assign "acc", %a : !hc.undef
      hc.yield
    } else {
      hc.assign "acc", %b : !hc.undef
      hc.yield
    }
    hc.region_return ["acc"]
  }
  %v = hc.name_load "acc" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Writes that aren't listed in `hc.region_return` still shadow — the
// outer `"only_outer"` binding survives the region untouched. The
// region-level `"leaked"` assign disappears (erased with its name),
// but the outer `hc.assign "only_outer"` is the reaching def for the
// outer `hc.name_load "only_outer"`.
// CHECK-LABEL: hc.func @workitem_return_shadow_other
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.region_return
// CHECK: %[[R:.*]] = hc.workitem_region -> (!hc.undef) {
// CHECK:   hc.yield %arg1 : !hc.undef
// CHECK: }
// CHECK: %[[SUM:.*]] = hc.add %arg0, %[[R]]
// CHECK: hc.return %[[SUM]]
hc.func @workitem_return_shadow_other(%outer: !hc.undef,
                                      %inner: !hc.undef) -> !hc.undef {
  hc.assign "only_outer", %outer : !hc.undef
  hc.workitem_region {
    hc.assign "leaked", %inner : !hc.undef
    hc.assign "returned", %inner : !hc.undef
    hc.region_return ["returned"]
  }
  %o = hc.name_load "only_outer" : !hc.undef
  %r = hc.name_load "returned" : !hc.undef
  %sum = hc.add %o, %r : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// -----

// `hc.region_return` names a binding the body never touches (no
// in-region `hc.assign` or `hc.name_load` for it). The scan can't see
// it, so the pass has to materialize the outer snap itself when
// wiring up the yield. Behaviourally: the region op becomes a
// pass-through for `%outer` — outer assign → snap → yield → writeback
// → outer read, with the flat sweep collapsing the whole chain.
// CHECK-LABEL: hc.func @workitem_return_passthrough
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK-NOT: hc.region_return
// CHECK: %[[R:.*]] = hc.workitem_region -> (!hc.undef) {
// CHECK:   hc.yield %arg0 : !hc.undef
// CHECK: }
// CHECK: hc.return %[[R]]
hc.func @workitem_return_passthrough(%outer: !hc.undef) -> !hc.undef {
  hc.assign "v", %outer : !hc.undef
  hc.workitem_region {
    hc.region_return ["v"]
  }
  %v = hc.name_load "v" : !hc.undef
  hc.return %v : !hc.undef
}

// -----

// IV self-bind, positive path. The frontend emits
// `hc.assign "<n>", %iv-block-arg` as the first body op (pattern
// documented on `hc.assign` in HCOps.td); promotion matches the
// pattern and folds "i" to direct uses of `%iv`. Post-promotion:
// no iter_arg for "i", in-body `hc.name_load "i"` resolves to
// `%iv`, no outer `hc.name_load "i"` survives.
// CHECK-LABEL: hc.func @for_range_iv_self_bind
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: hc.for_range %arg0 to %arg1 step %arg2 : (!hc.undef, !hc.undef, !hc.undef) -> ()
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef):
// CHECK:        hc.store %arg3[%[[IV]]], %[[IV]]
// CHECK:      }
// CHECK: hc.return
hc.func @for_range_iv_self_bind(%lo: !hc.undef, %hi: !hc.undef,
                                %step: !hc.undef, %buf: !hc.undef) {
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    hc.assign "i", %iv : !hc.undef
    %a = hc.name_load "i" : !hc.undef
    %b = hc.name_load "i" : !hc.undef
    hc.store %buf[%a], %b : (!hc.undef, !hc.undef, !hc.undef) -> ()
    hc.yield
  }
  hc.return
}

// -----

// IV self-bind composed with an independent carried name. "i" is
// loop-local (erased at the pre-scan); "acc" is a real iter_arg
// threaded through the body + yield. Flat sweep resolves the outer
// snap for "acc" to the function's `%init` param (%arg3), so the
// printed op reads `iter_args(%arg3)`.
// CHECK-LABEL: hc.func @for_range_iv_self_bind_with_acc
// CHECK-NOT: hc.name_load
// CHECK-NOT: hc.assign
// CHECK: %[[R:.*]] = hc.for_range %arg0 to %arg1 step %arg2 iter_args(%arg3) :
// CHECK-SAME: (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK-NEXT: ^bb0(%[[IV:.*]]: !hc.undef, %[[ACC:.*]]: !hc.undef):
// CHECK:        %[[SUM:.*]] = hc.add %[[ACC]], %[[IV]]
// CHECK:        hc.yield %[[SUM]]
// CHECK: hc.return %[[R]]
hc.func @for_range_iv_self_bind_with_acc(%lo: !hc.undef, %hi: !hc.undef,
                                         %step: !hc.undef,
                                         %init: !hc.undef) -> !hc.undef {
  hc.assign "acc", %init : !hc.undef
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    hc.assign "i", %iv : !hc.undef
    %cur = hc.name_load "acc" : !hc.undef
    %i = hc.name_load "i" : !hc.undef
    %next = hc.add %cur, %i : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "acc", %next : !hc.undef
    hc.yield
  }
  %final = hc.name_load "acc" : !hc.undef
  hc.return %final : !hc.undef
}
