// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: hc.intrinsic @wave_barrier
// CHECK-SAME: scope = <"SubGroup">
// CHECK-SAME: effects = read_write
hc.intrinsic @wave_barrier scope = #hc.scope<"SubGroup">
    effects = read_write {}

// CHECK-LABEL: hc.intrinsic @subgroup_dot
// CHECK-SAME: scope = <"SubGroup">
// CHECK-NOT:  effects
hc.intrinsic @subgroup_dot scope = #hc.scope<"SubGroup"> {}

// Signature on an intrinsic declaration now shows as `(args) -> result`,
// not the legacy `: (T) -> T` form. The body is still allowed to be empty
// (pure declaration).
// CHECK-LABEL: hc.intrinsic @typed_decl
// CHECK-SAME: (%arg0: f32, %arg1: f32) -> f32 scope = <"WorkItem">
hc.intrinsic @typed_decl(%a: f32, %b: f32) -> f32
    scope = #hc.scope<"WorkItem"> {}

// CHECK-LABEL: func.func @simple_for_range
// CHECK: hc.for_range %{{.*}} to %{{.*}} step %{{.*}} : (!hc.undef, !hc.undef, !hc.undef) -> () {
// CHECK: }
func.func @simple_for_range(%lo: !hc.undef, %hi: !hc.undef, %step: !hc.undef) {
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%i: !hc.undef):
    hc.yield
  }
  return
}

// CHECK-LABEL: func.func @for_range_with_iter_args
// CHECK: hc.for_range %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}}) :
// CHECK-SAME: (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">, !hc.undef) -> !hc.undef {
// CHECK:   hc.yield %{{.*}} : !hc.undef
// CHECK: }
func.func @for_range_with_iter_args(
    %lo: !hc.idx<"0">, %hi: !hc.idx<"N">, %step: !hc.idx<"1">,
    %init: !hc.undef) -> !hc.undef {
  %acc = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">, !hc.undef) -> !hc.undef {
  ^bb0(%i: !hc.idx, %sum: !hc.undef):
    hc.yield %sum : !hc.undef
  }
  return %acc : !hc.undef
}

// CHECK-LABEL: func.func @if_without_results
// CHECK: hc.if %{{.*}} : !hc.undef {
// CHECK-NOT: else
func.func @if_without_results(%c: !hc.undef) {
  hc.if %c : !hc.undef {
  }
  return
}

// CHECK-LABEL: func.func @if_with_else_and_results
// CHECK: hc.if %{{.*}} -> (!hc.undef) : !hc.pred<"M - N < 0"> {
// CHECK:   hc.yield %{{.*}} : !hc.undef
// CHECK: } else {
// CHECK:   hc.yield %{{.*}} : !hc.undef
// CHECK: }
func.func @if_with_else_and_results(
    %c: !hc.pred<"M < N">, %a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  %x = hc.if %c -> (!hc.undef) : !hc.pred<"M < N"> {
    hc.yield %a : !hc.undef
  } else {
    hc.yield %b : !hc.undef
  }
  return %x : !hc.undef
}
