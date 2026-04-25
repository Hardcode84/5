// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-front-inline %s | FileCheck %s
// RUN: hc-opt --hc-front-inline --convert-hc-front-to-hc --hc-promote-names %s | FileCheck %s --check-prefix=FLAT

// Scalar inline callee — one return operand, non-tuple: the inlined
// region lands with arity 1 and the call's use is replaced 1:1.
module {
  hc_front.func "inc" attributes {
    parameters = [{name = "x"}],
    ref = {kind = "inline", qualified_name = "pkg.inc"}
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %one = hc_front.constant <1 : i64>
    %r = hc_front.binop "Add"(%x, %one)
    hc_front.return %r
  }

  // Tuple-return callee: AST lowers `return a, b` as a single return
  // operand that is itself an `hc_front.tuple`. The inliner preserves that
  // tuple as one first-class value; tuple-unpack lowers through `hc.getitem`.
  hc_front.func "split" attributes {
    parameters = [{name = "v"}],
    ref = {kind = "inline", qualified_name = "pkg.split"}
  } {
    %v = hc_front.name "v" {ctx = "load", ref = {kind = "param"}}
    %two = hc_front.constant <2 : i64>
    %lo = hc_front.binop "Mult"(%v, %two)
    %hi = hc_front.binop "Add"(%v, %two)
    %t = hc_front.tuple (%lo, %hi)
    hc_front.return %t
  }

  // Nested inline: `outer` calls `inc` on its arg then scales. After
  // the pass, both helpers are erased and `demo` contains nested
  // `hc_front.inlined_region`s — one per call site, each a name
  // boundary keyed off its own prefix at conversion.
  hc_front.func "outer" attributes {
    parameters = [{name = "y"}],
    ref = {kind = "inline", qualified_name = "pkg.outer"}
  } {
    %y = hc_front.name "y" {ctx = "load", ref = {kind = "param"}}
    %inc = hc_front.name "inc" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.inc"}}
    %incd = hc_front.call %inc(%y)
    %two = hc_front.constant <2 : i64>
    %r = hc_front.binop "Mult"(%incd, %two)
    hc_front.return %r
  }

  hc_front.kernel "demo" attributes {parameters = [{name = "g"}]} {
    %a = hc_front.constant <3 : i64>
    %b = hc_front.constant <7 : i64>

    %inc_n = hc_front.name "inc" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.inc"}}
    %scalar = hc_front.call %inc_n(%a)
    %sn = hc_front.target_name "scalar_out"
    hc_front.assign %sn = %scalar

    %split_n = hc_front.name "split" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.split"}}
    %pair = hc_front.call %split_n(%b)
    %tn_lo = hc_front.target_name "lo"
    %tn_hi = hc_front.target_name "hi"
    %tgt = hc_front.target_tuple (%tn_lo, %tn_hi)
    hc_front.assign %tgt = %pair

    %outer_n = hc_front.name "outer" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.outer"}}
    %nested = hc_front.call %outer_n(%a)
    %on = hc_front.target_name "nested_out"
    hc_front.assign %on = %nested
  }
}

// Every `ref.kind = "inline"` marker must be gone after the pass: no
// inline helper funcs, no call sites referring to them.
// CHECK-NOT: hc_front.func "inc"
// CHECK-NOT: hc_front.func "split"
// CHECK-NOT: hc_front.func "outer"
// CHECK-NOT: ref = {{.*}}kind = "inline"

// Scalar site: single-result region replaces the call.
// CHECK: hc_front.inlined_region "inc"
// CHECK-SAME: -> (!hc_front.value)
// Tuple site: one result, because the tuple is a first-class value.
// CHECK: hc_front.inlined_region "split"
// CHECK-SAME: -> (!hc_front.value)
// Nested: outer region contains an inlined_region for `inc` too.
// CHECK: hc_front.inlined_region "outer"
// CHECK: hc_front.inlined_region "inc"

// After the full pipeline every inline site flattens into the caller.
// The alpha-renamed per-site params should be bound before the body
// runs and no `hc_front.*` op should survive.
// FLAT-NOT: hc_front
// FLAT-NOT: hc.call @inc
// FLAT-NOT: hc.call @split
// FLAT-NOT: hc.call @outer
