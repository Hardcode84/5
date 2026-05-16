// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `hc-lower-launch-body` enforces the upstream contract that
// `hc-decompose-shaped-values` has already split every semantic
// `!hc.tensor` / `!hc.vector` producer/consumer into bare (data,
// mask) pairs. If a semantic carrier survives to this point, the
// pre-pass gate fires with a diagnostic that names the offending op
// and points at the upstream layer that should have decomposed it —
// instead of identity-converting the type and producing a vague
// "failed to legalize" error two stages later.
//
// RUN: hc-opt %s --hc-lower-launch-body --verify-diagnostics --split-input-file

module {
  func.func @semantic_tensor_survives(%a: f32) -> f32 {
    // expected-error@+1 {{semantic shaped type '!hc.tensor<f32, ["8"]>' survived past hc-decompose-shaped-values on result; decompose must split !hc.tensor / !hc.vector into bare (data, mask) pairs before hc-lower-launch-body runs}}
    %0 = builtin.unrealized_conversion_cast %a : f32 to !hc.tensor<f32, ["8"]>
    %1 = builtin.unrealized_conversion_cast %0 : !hc.tensor<f32, ["8"]> to f32
    return %1 : f32
  }
}

// -----

module {
  func.func @semantic_vector_survives(%a: f32) -> f32 {
    // expected-error@+1 {{semantic shaped type '!hc.vector<f32, ["8"]>' survived past hc-decompose-shaped-values on result; decompose must split !hc.tensor / !hc.vector into bare (data, mask) pairs before hc-lower-launch-body runs}}
    %0 = builtin.unrealized_conversion_cast %a : f32 to !hc.vector<f32, ["8"]>
    %1 = builtin.unrealized_conversion_cast %0 : !hc.vector<f32, ["8"]> to f32
    return %1 : f32
  }
}
