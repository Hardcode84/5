// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Roundtrip coverage for the transient imperative name-store ops
// `hc.assign` / `hc.name_load`. These ops are emitted by
// `-convert-hc-front-to-hc` and erased by `-hc-promote-names`. This
// file only proves they parse, print, and round-trip cleanly across
// every region kind they're expected to show up in; behavioural
// coverage for their memory effects lives in
// `test/HC/name-store-effects.mlir`.
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// Top-level in a kernel body — the simplest "function-scope name store"
// pattern. Pre-inference types (`!hc.undef`) are the common case at the
// point these ops are first emitted.
// CHECK-LABEL: hc.kernel @bare_name_store
// CHECK: hc.assign "acc", %{{.*}} : !hc.undef
// CHECK: %[[ACC:.*]] = hc.name_load "acc" : !hc.undef
// CHECK: hc.return
hc.kernel @bare_name_store {
  %zero = hc.const<0 : i64> : !hc.undef
  hc.assign "acc", %zero : !hc.undef
  %acc = hc.name_load "acc" : !hc.undef
  hc.return
}

// Typed values — post-inference the store still works. Every
// `HC_ValueType` is admissible.
// CHECK-LABEL: hc.kernel @typed_name_store
// CHECK: hc.assign "count", %{{.*}} : i32
// CHECK: %{{.*}} = hc.name_load "count" : i32
// CHECK: hc.return
hc.kernel @typed_name_store {
  %c = hc.const<7 : i32> : i32
  hc.assign "count", %c : i32
  %read = hc.name_load "count" : i32
  hc.return
}

// Inside `hc.for_range` — pre-promotion shape of a loop-carried var.
// The body reads `acc` via `hc.name_load` and writes it back via
// `hc.assign`; promotion will fold both into an iter_arg + yield.
// CHECK-LABEL: hc.kernel @for_body_name_store
// CHECK: hc.for_range
// CHECK: %{{.*}} = hc.name_load "acc" : !hc.undef
// CHECK: hc.assign "acc", %{{.*}} : !hc.undef
hc.kernel @for_body_name_store {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<8 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  %init = hc.const<0 : i64> : !hc.undef
  hc.assign "acc", %init : !hc.undef
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) -> () {
    ^bb0(%iv: !hc.undef):
    %prev = hc.name_load "acc" : !hc.undef
    %next = hc.add %prev, %iv : (!hc.undef, !hc.undef) -> !hc.undef
    hc.assign "acc", %next : !hc.undef
    hc.yield
  }
  hc.return
}

// Inside `hc.if` — pre-promotion shape of a branch-dependent rebind.
// Both branches write `x`; promotion will turn this into an
// `hc.if -> (!hc.undef)` with `hc.yield %x` in each branch.
// CHECK-LABEL: hc.kernel @if_body_name_store
// CHECK: hc.if
// CHECK: hc.assign "x"
// CHECK: } else {
// CHECK: hc.assign "x"
hc.kernel @if_body_name_store {
  %cond = hc.const<1 : i64> : !hc.undef
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  hc.if %cond : !hc.undef {
    hc.assign "x", %one : !hc.undef
    hc.yield
  } else {
    hc.assign "x", %two : !hc.undef
    hc.yield
  }
  %final = hc.name_load "x" : !hc.undef
  hc.return
}

// Inside captured regions — same pattern, same expected behaviour. The
// promotion pass will turn nested-def-style `return value` idioms into
// a result-producing workitem/subgroup region.
// CHECK-LABEL: hc.kernel @region_body_name_store
// CHECK: hc.workitem_region
// CHECK: hc.assign "inner", %{{.*}} : !hc.undef
// CHECK: hc.subgroup_region
// CHECK: %{{.*}} = hc.name_load "inner" : !hc.undef
hc.kernel @region_body_name_store {
  %v = hc.const<42 : i64> : !hc.undef
  hc.workitem_region {
    hc.assign "inner", %v : !hc.undef
  }
  hc.subgroup_region {
    %read = hc.name_load "inner" : !hc.undef
  }
  hc.return
}
