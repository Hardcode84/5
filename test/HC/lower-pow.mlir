// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-pow`. The pass rewrites the structural
// `hc.pow` carrier (the front pass emits one for every Python `**`)
// into a binary-squaring chain of `hc.mul` ops, which the rest of the
// pipeline already understands. See the design on
// `include/hc/Transforms/Passes.td` and the op contract on
// `include/hc/IR/HCOps.td`.
//
// RUN: hc-opt --hc-lower-pow %s --split-input-file | FileCheck %s

// K = 2: just one `hc.mul`. The result type of the mul is sourced from
// the original `hc.pow`'s result type (here `!hc.undef`, matching the
// progressive-typing shape the front pass emits).
// CHECK-LABEL: func.func @pow_squared
// CHECK: hc.mul %arg0, %arg0 : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK-NOT: hc.mul
// CHECK-NOT: hc.pow
// CHECK: return
func.func @pow_squared(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<2 : i64> : !hc.undef
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// K = 3 (0b11): square then multiply by `x` — two muls.
// CHECK-LABEL: func.func @pow_cubed
// CHECK: %[[SQ:.*]] = hc.mul %arg0, %arg0
// CHECK: hc.mul %[[SQ]], %arg0
// CHECK-NOT: hc.mul
// CHECK-NOT: hc.pow
// CHECK: return
func.func @pow_cubed(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<3 : i64> : !hc.undef
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// K = 4 (0b100): square-then-square — still two muls. Demonstrates the
// "no bit set" branch of the bit walk (the `0` bit in `100` skips the
// extra multiply-by-x step).
// CHECK-LABEL: func.func @pow_quartic
// CHECK: %[[SQ:.*]] = hc.mul %arg0, %arg0
// CHECK: hc.mul %[[SQ]], %[[SQ]]
// CHECK-NOT: hc.mul
// CHECK-NOT: hc.pow
// CHECK: return
func.func @pow_quartic(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<4 : i64> : !hc.undef
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// K = 7 (0b111): exercises every step on the bit walk — square (^2),
// *x (^3), square (^6), *x (^7). Four muls instead of the naive six.
// CHECK-LABEL: func.func @pow_seventh
// CHECK: %[[K2:.*]] = hc.mul %arg0, %arg0
// CHECK: %[[K3:.*]] = hc.mul %[[K2]], %arg0
// CHECK: %[[K6:.*]] = hc.mul %[[K3]], %[[K3]]
// CHECK: hc.mul %[[K6]], %arg0
// CHECK-NOT: hc.mul
// CHECK-NOT: hc.pow
// CHECK: return
func.func @pow_seventh(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<7 : i64> : !hc.undef
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// K = 1: trivial identity. The pass forwards `lhs` to the result's
// uses and erases the `hc.pow` op without emitting any mul.
// CHECK-LABEL: func.func @pow_one
// CHECK-NOT: hc.mul
// CHECK-NOT: hc.pow
// CHECK: return %arg0 : !hc.undef
func.func @pow_one(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<1 : i64> : !hc.undef
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// Pow on inferred-scalar operands keeps the result type through the
// chain. The `hc.const` here carries an `IntegerAttr`, so the unfold
// fires; the resulting `hc.mul` ops inherit the same `f32` result type
// the original `hc.pow` had.
// CHECK-LABEL: func.func @pow_typed_f32
// CHECK: hc.mul %arg0, %arg0 : (f32, f32) -> f32
// CHECK-NOT: hc.pow
// CHECK: return
func.func @pow_typed_f32(%x: f32) -> f32 {
  %e = hc.const<2 : i64> : !hc.undef
  %r = hc.pow %x, %e : (f32, !hc.undef) -> f32
  return %r : f32
}
