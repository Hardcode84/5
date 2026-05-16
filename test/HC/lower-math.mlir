// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-math`. The pass rewrites `hc.builtin_call`
// (the open-ended carrier the front pass plants for every NumPy ufunc
// it forwards) into the matching upstream `math.<op>`. See the design
// on `include/hc/Transforms/Passes.td` and the op contract on
// `include/hc/IR/HCOps.td`.
//
// RUN: hc-opt --hc-lower-math %s --split-input-file --verify-diagnostics | FileCheck %s

// `numpy.sqrt` -> `math.sqrt`. The pass takes scalar/vector/tensor
// float operands uniformly because `math.sqrt` does — we don't have
// to fan-out the dispatch on operand shape.
// CHECK-LABEL: func.func @sqrt_scalar
// CHECK: %[[R:.*]] = math.sqrt %arg0 : f32
// CHECK-NOT: hc.builtin_call
// CHECK: return %[[R]]
func.func @sqrt_scalar(%x: f32) -> f32 {
  %r = hc.builtin_call "numpy.sqrt"(%x) : (f32) -> f32
  return %r : f32
}

// -----

// `numpy.exp` -> `math.exp`. Same shape as the sqrt arm; pinning a
// second name confirms the dispatch table looks at the name string
// rather than hard-wiring sqrt only.
// CHECK-LABEL: func.func @exp_scalar
// CHECK: math.exp %arg0 : f32
// CHECK-NOT: hc.builtin_call
func.func @exp_scalar(%x: f32) -> f32 {
  %r = hc.builtin_call "numpy.exp"(%x) : (f32) -> f32
  return %r : f32
}

// -----

// A non-float scalar operand is diagnosed at the carrier rather than
// silently passed to `math.sqrt` (which would reject it on its own
// type constraints downstream). Keeps the failure pointed at the
// user-visible `np.sqrt(...)` source.
func.func @sqrt_int_rejected(%x: i32) -> i32 {
  // expected-error@+1 {{only float-element operand is supported today}}
  %r = hc.builtin_call "numpy.sqrt"(%x) : (i32) -> i32
  return %r : i32
}

// -----

// Unknown name also gets a single localised diagnostic instead of
// leaking through to LLVM lowering.
func.func @unknown_name_rejected(%x: f32) -> f32 {
  // expected-error@+1 {{no lowering registered}}
  %r = hc.builtin_call "numpy.unknown"(%x) : (f32) -> f32
  return %r : f32
}
