// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: hc-opt -canonicalize -split-input-file %s | FileCheck %s

// `ConstantLike` ops must be foldable: canonicalizer's generic constant
// matchers assert on the trait otherwise.
// CHECK-LABEL: func.func @constant_like_smoke
// CHECK: %[[UNDEF:.*]] = hc.undef_value : !hc.undef
// CHECK: %[[SYM:.*]] = hc.symbol : !hc.idx<"M">
// CHECK: %[[CONST:.*]] = hc.const<7 : i32> : i32
// CHECK: return %[[UNDEF]], %[[SYM]], %[[CONST]]
func.func @constant_like_smoke() -> (!hc.undef, !hc.idx<"M">, i32) {
  %u = hc.undef_value : !hc.undef
  %s = hc.symbol : !hc.idx<"M">
  %c = hc.const<7 : i32> : i32
  return %u, %s, %c : !hc.undef, !hc.idx<"M">, i32
}
