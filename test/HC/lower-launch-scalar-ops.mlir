// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-scalar-ops`: arith cleanup for the scalar HC
// ops surviving before `hc-lower-launch-body` runs. Const + int /
// float binary arith + compare go to `arith.*` through the
// launch-body type converter (`!hc.idx` -> `index`, `!hc.pred` ->
// `i1`). Control flow (`hc.for_range` / `hc.if`) and the type-
// bridge `hc.cast` stay in launch-body proper -- their body cloning
// + shaped-type converter interactions leak cross-pass UCCs when
// split.
//
// RUN: hc-opt %s --hc-lower-launch-scalar-ops --reconcile-unrealized-casts | FileCheck %s

module {
  // `hc.const` -> `arith.constant`; idx-typed result rides a UCC
  // bridge until launch-body folds it.
  // CHECK-LABEL: func.func @lower_const_and_add(
  // CHECK-SAME: %[[X:[^:]+]]: !hc.idx<"X">
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[XI:.*]] = builtin.unrealized_conversion_cast %[[X]] : !hc.idx<"X"> to index
  // CHECK: %[[SUM:.*]] = arith.addi %[[XI]], %[[ONE]] : index
  // CHECK-NOT: hc.const
  // CHECK-NOT: hc.add
  func.func @lower_const_and_add(%x: !hc.idx<"X">) -> !hc.idx<"1 + X"> {
    %one = hc.const<1 : i64> : !hc.idx<"1">
    %r = hc.add %x, %one
        : (!hc.idx<"X">, !hc.idx<"1">) -> !hc.idx<"1 + X">
    return %r : !hc.idx<"1 + X">
  }

  // Float arith lowers to the matching `arith` op.
  // CHECK-LABEL: func.func @lower_float_arith(
  // CHECK-SAME: %[[A:[^:]+]]: f32, %[[B:[^:]+]]: f32
  // CHECK: %[[S:.*]] = arith.mulf %[[A]], %[[B]] : f32
  // CHECK-NOT: hc.mul
  func.func @lower_float_arith(%a: f32, %b: f32) -> f32 {
    %r = hc.mul %a, %b : (f32, f32) -> f32
    return %r : f32
  }

  // Compare: `hc.cmp.lt` -> `arith.cmpi slt`.
  // CHECK-LABEL: func.func @lower_cmp(
  // CHECK-SAME: %[[A:[^:]+]]: !hc.idx<"A">, %[[B:[^:]+]]: !hc.idx<"B">
  // CHECK: arith.cmpi slt, %{{.+}}, %{{.+}} : index
  // CHECK-NOT: hc.cmp.lt
  func.func @lower_cmp(%a: !hc.idx<"A">, %b: !hc.idx<"B">) -> !hc.pred<"A < B"> {
    %p = hc.cmp.lt %a, %b
        : (!hc.idx<"A">, !hc.idx<"B">) -> !hc.pred<"A < B">
    return %p : !hc.pred<"A < B">
  }

  // `hc.for_range` survives the scalar pass -- launch-body owns it.
  // CHECK-LABEL: func.func @for_range_survives
  // CHECK: hc.for_range
  func.func @for_range_survives(%lo: !hc.idx<"0">, %hi: !hc.idx<"N">,
                                 %step: !hc.idx<"1">) {
    hc.for_range %lo to %hi step %step
        : (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">) {
    ^bb0(%i: !hc.idx<"$join0">):
      hc.yield
    }
    return
  }
}
