// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Behavioural proof for the MemAlloc / Pure rule of thumb in doc/lowering.md:
// allocators carry `MemAlloc` precisely so CSE cannot merge sibling allocs
// (each call produces a fresh, independently-writable value); Pure ops fold
// normally. Without this, later patterns would collapse two `hc.vzeros`
// into one and subsequent stores would clobber each other.

// RUN: hc-opt -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @sibling_vzeros
// CHECK: %[[A:.*]] = hc.vzeros {shape = #hc.shape<["16"]>}
// CHECK: %[[B:.*]] = hc.vzeros {shape = #hc.shape<["16"]>}
// CHECK: return %[[A]], %[[B]]
func.func @sibling_vzeros()
    -> (!hc.vector<f32, ["16"]>, !hc.vector<f32, ["16"]>) {
  %0 = hc.vzeros {shape = #hc.shape<["16"]>} : !hc.vector<f32, ["16"]>
  %1 = hc.vzeros {shape = #hc.shape<["16"]>} : !hc.vector<f32, ["16"]>
  return %0, %1 : !hc.vector<f32, ["16"]>, !hc.vector<f32, ["16"]>
}

// -----

// CHECK-LABEL: func.func @sibling_zeros
// CHECK: %[[A:.*]] = hc.zeros {shape = #hc.shape<["M"]>}
// CHECK: %[[B:.*]] = hc.zeros {shape = #hc.shape<["M"]>}
// CHECK: return %[[A]], %[[B]]
func.func @sibling_zeros()
    -> (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) {
  %0 = hc.zeros {shape = #hc.shape<["M"]>} : !hc.tensor<f32, ["M"]>
  %1 = hc.zeros {shape = #hc.shape<["M"]>} : !hc.tensor<f32, ["M"]>
  return %0, %1 : !hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>
}

// -----

// Contrast case: Pure op with identical operands does collapse. If this ever
// starts failing, either `hc.add` lost `Pure` or CSE semantics changed — both
// are worth a stare.

// CHECK-LABEL: func.func @pure_add_collapses
// CHECK: %[[X:.*]] = hc.add
// CHECK: return %[[X]], %[[X]]
func.func @pure_add_collapses(%a: !hc.undef, %b: !hc.undef)
    -> (!hc.undef, !hc.undef) {
  %0 = hc.add %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
  %1 = hc.add %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
  return %0, %1 : !hc.undef, !hc.undef
}

// -----

// Effect class on `hc.func` narrows the call site. A `pure` helper gets the
// Pure effect, so two identical calls CSE away — silent proof that the
// interface is plumbed through and not just a label.

hc.func @pure_helper(%a: !hc.undef, %b: !hc.undef) -> !hc.undef
    effects = pure {
  hc.return
}

// CHECK-LABEL: func.func @pure_calls_collapse
// CHECK: %[[X:.*]] = hc.call @pure_helper
// CHECK: return %[[X]], %[[X]]
func.func @pure_calls_collapse(%a: !hc.undef, %b: !hc.undef)
    -> (!hc.undef, !hc.undef) {
  %0 = hc.call @pure_helper(%a, %b) : (!hc.undef, !hc.undef) -> !hc.undef
  %1 = hc.call @pure_helper(%a, %b) : (!hc.undef, !hc.undef) -> !hc.undef
  return %0, %1 : !hc.undef, !hc.undef
}

// -----

// No effect class → conservative read_write default → CSE must *not* merge.
// If this ever collapses, something removed the fallback in `getEffects`.

hc.func @opaque_helper(%a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  hc.return
}

// CHECK-LABEL: func.func @opaque_calls_stay
// CHECK: %[[A:.*]] = hc.call @opaque_helper
// CHECK: %[[B:.*]] = hc.call @opaque_helper
// CHECK: return %[[A]], %[[B]]
func.func @opaque_calls_stay(%a: !hc.undef, %b: !hc.undef)
    -> (!hc.undef, !hc.undef) {
  %0 = hc.call @opaque_helper(%a, %b) : (!hc.undef, !hc.undef) -> !hc.undef
  %1 = hc.call @opaque_helper(%a, %b) : (!hc.undef, !hc.undef) -> !hc.undef
  return %0, %1 : !hc.undef, !hc.undef
}
