// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-infer-generic-bounds`. The pass walks every
// `hc.generic` and rewrites `iter_bounds` operands defined by
// `hc.undef_value` to a concrete `!hc.idx<dim>` SSA value sourced
// from the operand's symbolic shape. Identity offsets only — affine
// patterns are skipped, missing bindings diagnose, conflicting
// bindings diagnose. Already-resolved bounds are left alone.
//
// RUN: hc-opt --hc-infer-generic-bounds %s --split-input-file --verify-diagnostics | FileCheck %s

// Two parallel iters, both placeholders. Each iter sym appears
// identity-style in one input axis and the matching output axis;
// the implied bounds (M and N) materialize as hc.idx values.
// CHECK-LABEL: func.func @two_parallel_identity
// CHECK: %[[M_BOUND:.+]] = hc.idx_apply () : () -> !hc.idx<"M">
// CHECK: %[[N_BOUND:.+]] = hc.idx_apply () : () -> !hc.idx<"N">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %[[M_BOUND]] : !hc.idx<"M">, parallel j = %[[N_BOUND]] : !hc.idx<"N">)
func.func @two_parallel_identity(%a: !hc.bare_tensor<f32, ["M", "N"]>,
                                 %c: !hc.bare_tensor<f32, ["M", "N"]>)
    -> !hc.bare_tensor<f32, ["M", "N"]> {
  %u0 = hc.undef_value : !hc.undef
  %u1 = hc.undef_value : !hc.undef
  %r = hc.generic
      iter (parallel i = %u0 : !hc.undef,
            parallel j = %u1 : !hc.undef)
      ins (%a at [#hc.expr<"i">, #hc.expr<"j">]
              : !hc.bare_tensor<f32, ["M", "N"]>)
      outs (%c at [#hc.expr<"i">, #hc.expr<"j">]
               : !hc.bare_tensor<f32, ["M", "N"]>)
      -> (!hc.bare_tensor<f32, ["M", "N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["M", "N"]>
}

// -----

// Mixed kinds with reduction. K binds through the input only; the
// output's parallel iters bind from both operands. Verifies the
// pass walks ins as well as outs and doesn't trip on the reduction
// label.
// CHECK-LABEL: func.func @parallel_and_reduction
// CHECK: %[[M_BOUND:.+]] = hc.idx_apply () : () -> !hc.idx<"M">
// CHECK: %[[N_BOUND:.+]] = hc.idx_apply () : () -> !hc.idx<"N">
// CHECK: %[[K_BOUND:.+]] = hc.idx_apply () : () -> !hc.idx<"K">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %[[M_BOUND]] : !hc.idx<"M">, parallel j = %[[N_BOUND]] : !hc.idx<"N">, reduction k = %[[K_BOUND]] : !hc.idx<"K">)
func.func @parallel_and_reduction(%a: !hc.bare_tensor<f16, ["M", "K"]>,
                                  %b: !hc.bare_tensor<f16, ["K", "N"]>,
                                  %c: !hc.bare_tensor<f32, ["M", "N"]>)
    -> !hc.bare_tensor<f32, ["M", "N"]> {
  %u0 = hc.undef_value : !hc.undef
  %u1 = hc.undef_value : !hc.undef
  %u2 = hc.undef_value : !hc.undef
  %r = hc.generic
      iter (parallel i = %u0 : !hc.undef,
            parallel j = %u1 : !hc.undef,
            reduction k = %u2 : !hc.undef)
      ins (%a at [#hc.expr<"i">, #hc.expr<"k">]
              : !hc.bare_tensor<f16, ["M", "K"]>,
           %b at [#hc.expr<"k">, #hc.expr<"j">]
              : !hc.bare_tensor<f16, ["K", "N"]>)
      outs (%c at [#hc.expr<"i">, #hc.expr<"j">]
               : !hc.bare_tensor<f32, ["M", "N"]>)
      -> (!hc.bare_tensor<f32, ["M", "N"]>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %ae = hc.astype %av, target = f32 : f16 -> f32
    %be = hc.astype %bv, target = f32 : f16 -> f32
    %p = hc.mul %ae, %be : (f32, f32) -> f32
    %s = hc.add %cv, %p : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["M", "N"]>
}

// -----

// A subset of bounds is concrete on entry; the pass touches only the
// undef-defined ones and leaves the rest alone. Concrete `index` value
// flows through unchanged; placeholder gets an idx-typed materialize.
// CHECK-LABEL: func.func @partial_bounds
// CHECK: %[[N_BOUND:.+]] = hc.idx_apply () : () -> !hc.idx<"N">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index, parallel j = %[[N_BOUND]] : !hc.idx<"N">)
func.func @partial_bounds(%m: index,
                          %a: !hc.bare_tensor<f32, ["M", "N"]>,
                          %c: !hc.bare_tensor<f32, ["M", "N"]>)
    -> !hc.bare_tensor<f32, ["M", "N"]> {
  %uj = hc.undef_value : !hc.undef
  %r = hc.generic
      iter (parallel i = %m : index,
            parallel j = %uj : !hc.undef)
      ins (%a at [#hc.expr<"i">, #hc.expr<"j">]
              : !hc.bare_tensor<f32, ["M", "N"]>)
      outs (%c at [#hc.expr<"i">, #hc.expr<"j">]
               : !hc.bare_tensor<f32, ["M", "N"]>)
      -> (!hc.bare_tensor<f32, ["M", "N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["M", "N"]>
}

// -----

// Fully concrete already: pass is a no-op, no `hc.idx_apply`
// shows up in the output.
// CHECK-LABEL: func.func @noop_when_concrete
// CHECK-NOT: hc.idx_apply
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index)
func.func @noop_when_concrete(%n: index,
                              %a: !hc.bare_tensor<f32, ["N"]>,
                              %c: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// -----

// Conflict: input axis 0 says i ranges over M, output axis 0 says it
// ranges over P. The pass refuses to silently pick one.
func.func @conflicting_bounds(%a: !hc.bare_tensor<f32, ["M"]>,
                              %c: !hc.bare_tensor<f32, ["P"]>)
    -> !hc.bare_tensor<f32, ["P"]> {
  %ui = hc.undef_value : !hc.undef
  // expected-error@+1 {{iter sym 'i' has conflicting implied bounds}}
  %r = hc.generic
      iter (parallel i = %ui : !hc.undef)
      ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>)
      outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["P"]>)
      -> (!hc.bare_tensor<f32, ["P"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["P"]>
}

// -----

// No identity occurrence anywhere — the offset is `i + 1` which the
// v0 matcher does not unwind. Diagnoses instead of silently leaving
// the bound as undef.
func.func @no_identity_match(%a: !hc.bare_tensor<f32, ["N"]>,
                             %c: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %ui = hc.undef_value : !hc.undef
  // expected-error@+1 {{iter sym 'i' has no identity occurrence}}
  %r = hc.generic
      iter (parallel i = %ui : !hc.undef)
      ins (%a at [#hc.expr<"i + 1">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%c at [#hc.expr<"i + 1">] : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}
