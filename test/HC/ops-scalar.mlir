// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: func.func @scalar_producers
// CHECK: hc.const<1 : i64> : !hc.undef
// CHECK: hc.const<3.140000e+00 : f32> : !hc.idx
// CHECK: hc.symbol : !hc.idx<"M">
// CHECK: hc.cast {{.*}} : !hc.undef -> !hc.idx<"M">
func.func @scalar_producers() {
  %c0 = hc.const<1 : i64> : !hc.undef
  %c1 = hc.const<3.14 : f32> : !hc.idx
  %s = hc.symbol : !hc.idx<"M">
  %casted = hc.cast %c0 : !hc.undef -> !hc.idx<"M">
  return
}

// CHECK-LABEL: func.func @arithmetic
// CHECK: hc.add {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.sub {{.*}} : (!hc.idx<"M">, !hc.idx<"N">) -> !hc.idx<"M - N">
// CHECK: hc.mul {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.div {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.mod {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.neg {{.*}} : !hc.undef -> !hc.undef
func.func @arithmetic(
    %u0: !hc.undef, %u1: !hc.undef,
    %m: !hc.idx<"M">, %n: !hc.idx<"N">) {
  %a = hc.add %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %b = hc.sub %m, %n : (!hc.idx<"M">, !hc.idx<"N">) -> !hc.idx<"M - N">
  %c = hc.mul %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %d = hc.div %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %e = hc.mod %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %f = hc.neg %u0 : !hc.undef -> !hc.undef
  return
}

// CHECK-LABEL: func.func @boolean_and_compare
// CHECK: hc.and {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.or {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.not {{.*}} : !hc.undef -> !hc.undef
// CHECK: hc.cmp.lt {{.*}} : (!hc.idx<"M">, !hc.idx<"N">) -> !hc.pred<"M - N < 0">
// CHECK: hc.cmp.le {{.*}} : (!hc.undef, !hc.undef) -> i1
// CHECK: hc.cmp.gt {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.cmp.ge {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.cmp.eq {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.cmp.ne {{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
func.func @boolean_and_compare(
    %u0: !hc.undef, %u1: !hc.undef,
    %m: !hc.idx<"M">, %n: !hc.idx<"N">) {
  %a = hc.and %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %b = hc.or %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %c = hc.not %u0 : !hc.undef -> !hc.undef
  %lt = hc.cmp.lt %m, %n
      : (!hc.idx<"M">, !hc.idx<"N">) -> !hc.pred<"M - N < 0">
  %le = hc.cmp.le %u0, %u1 : (!hc.undef, !hc.undef) -> i1
  %gt = hc.cmp.gt %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %ge = hc.cmp.ge %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %eq = hc.cmp.eq %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  %ne = hc.cmp.ne %u0, %u1 : (!hc.undef, !hc.undef) -> !hc.undef
  return
}
