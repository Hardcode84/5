// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `hc-lower-launch-body` enforces that every `!hc.bare_tensor` /
// `!hc.bare_vector` carrier reaching this pass has an
// integer-literal shape. The launch-body converter collapses bare
// carriers to `!hc.ptr<workgroup, T>` / `!vector<...>`, both of
// which need the element count known at compile time. A bare
// carrier whose shape still references a free symbol after
// `hc-specialize-literals` would identity-convert and surface as a
// vague "failed to legalize" downstream; the pre-pass gate fires
// here with a diagnostic that names the offending op and points at
// the binding surface (`hc.compile(symbols={...})` /
// kernel-decorator `literals=`).
//
// RUN: hc-opt %s --hc-lower-launch-body --verify-diagnostics --split-input-file

module {
  func.func @bare_tensor_symbolic_dim(%a: f32) -> f32 {
    // expected-error@+1 {{bare carrier '!hc.bare_tensor<f32, ["8*H"]>' has a non-literal shape on result; hc-lower-launch-body needs every dim resolved to an integer literal to allocate the workgroup tile, so bind the free symbol(s) via hc.compile(symbols={...}) (add the symbol to the kernel decorator's `literals=` set if it isn't already)}}
    %0 = builtin.unrealized_conversion_cast %a : f32 to !hc.bare_tensor<f32, ["8*H"]>
    %1 = builtin.unrealized_conversion_cast %0 : !hc.bare_tensor<f32, ["8*H"]> to f32
    return %1 : f32
  }
}

// -----

module {
  func.func @bare_vector_symbolic_dim(%a: f32) -> f32 {
    // expected-error@+1 {{bare carrier '!hc.bare_vector<f32, ["N"]>' has a non-literal shape on result; hc-lower-launch-body needs every dim resolved to an integer literal to allocate the workgroup tile, so bind the free symbol(s) via hc.compile(symbols={...}) (add the symbol to the kernel decorator's `literals=` set if it isn't already)}}
    %0 = builtin.unrealized_conversion_cast %a : f32 to !hc.bare_vector<f32, ["N"]>
    %1 = builtin.unrealized_conversion_cast %0 : !hc.bare_vector<f32, ["N"]> to f32
    return %1 : f32
  }
}

// -----

// Mixed shape (one literal dim, one symbolic) still fails the gate
// because element count is the product and one free symbol leaves
// the product non-literal.
module {
  func.func @bare_tensor_mixed_dim(%a: f32) -> f32 {
    // expected-error@+1 {{bare carrier '!hc.bare_tensor<f32, ["8", "K"]>' has a non-literal shape on result}}
    %0 = builtin.unrealized_conversion_cast %a : f32 to !hc.bare_tensor<f32, ["8", "K"]>
    %1 = builtin.unrealized_conversion_cast %0 : !hc.bare_tensor<f32, ["8", "K"]> to f32
    return %1 : f32
  }
}
