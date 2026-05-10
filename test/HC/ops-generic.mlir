// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Round-trip coverage for `hc.generic`. Verifier negatives live in
// `verify-hc.mlir` next to the rest of the dialect's "wrong spelling" pins.
//
// Per-operand `#hc.expr<...>` offsets sit inside parens (`at (#...)`) so
// the parser's generic dialect-attr path doesn't eat the literal `:`
// separator that introduces the operand type — same trick `hc.as_layout`
// uses for its structured layout payload. The expression text below
// matches ixsimpl's canonical form so the round-trip stays text-stable.
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// Plain elementwise: one parallel iter, one input, one output. Body adds
// the input to the running output value (`outs-as-init`) and yields.
// CHECK-LABEL: func.func @elementwise_add
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["N"]>)
// CHECK: ^bb0(%[[A:.+]]: f32, %[[C:.+]]: f32):
// CHECK:   %[[S:.+]] = hc.add %[[C]], %[[A]] : (f32, f32) -> f32
// CHECK:   hc.yield %[[S]] : f32
func.func @elementwise_add(%n: index,
                           %a: !hc.bare_tensor<f32, ["N"]>,
                           %c: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["N"]>)
      outs (%c at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// Matmul-flavoured: two parallel + one reduction iter, two inputs, one
// output. Output offset references only parallel iters; the reduction
// iter `k` only appears in the input offsets.
// CHECK-LABEL: func.func @matmul_like
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index, parallel j = %{{[^ ]+}} : index, reduction k = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at (#hc.expr<"k + K*i">) : !hc.bare_tensor<f16, ["K*M"]>,
// CHECK-SAME:      %{{[^ ]+}} at (#hc.expr<"j + N*k">) : !hc.bare_tensor<f16, ["K*N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at (#hc.expr<"j + N*i">) : !hc.bare_tensor<f32, ["M*N"]>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["M*N"]>)
func.func @matmul_like(%m: index, %n: index, %k: index,
                       %a: !hc.bare_tensor<f16, ["K*M"]>,
                       %b: !hc.bare_tensor<f16, ["K*N"]>,
                       %c: !hc.bare_tensor<f32, ["M*N"]>)
    -> !hc.bare_tensor<f32, ["M*N"]> {
  %r = hc.generic
      iter (parallel i = %m : index,
            parallel j = %n : index,
            reduction k = %k : index)
      ins (%a at (#hc.expr<"k + K*i">) : !hc.bare_tensor<f16, ["K*M"]>,
           %b at (#hc.expr<"j + N*k">) : !hc.bare_tensor<f16, ["K*N"]>)
      outs (%c at (#hc.expr<"j + N*i">) : !hc.bare_tensor<f32, ["M*N"]>)
      -> (!hc.bare_tensor<f32, ["M*N"]>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %ae = hc.astype %av, target = f32 : f16 -> f32
    %be = hc.astype %bv, target = f32 : f16 -> f32
    %p = hc.mul %ae, %be : (f32, f32) -> f32
    %s = hc.add %cv, %p : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["M*N"]>
}

// Multiple outputs (softmax-style: tracking running max + running sum).
// Each output gets its own offset and its own block-arg/yield slot.
// CHECK-LABEL: func.func @two_outputs
// CHECK: %{{.+}}:2 = hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index, reduction j = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at (#hc.expr<"j + N*i">) : !hc.bare_tensor<f32, ["M*N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["M"]>,
// CHECK-SAME:       %{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["M"]>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>)
func.func @two_outputs(%m: index, %n: index,
                       %x: !hc.bare_tensor<f32, ["M*N"]>,
                       %mx: !hc.bare_tensor<f32, ["M"]>,
                       %sm: !hc.bare_tensor<f32, ["M"]>)
    -> (!hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>) {
  %r:2 = hc.generic
      iter (parallel i = %m : index, reduction j = %n : index)
      ins (%x at (#hc.expr<"j + N*i">) : !hc.bare_tensor<f32, ["M*N"]>)
      outs (%mx at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["M"]>,
            %sm at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["M"]>)
      -> (!hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>) {
  ^bb0(%xv: f32, %mxv: f32, %smv: f32):
    %nm = hc.add %mxv, %xv : (f32, f32) -> f32
    %ns = hc.add %smv, %xv : (f32, f32) -> f32
    hc.yield %nm, %ns : f32, f32
  }
  return %r#0, %r#1
      : !hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>
}

// Empty `ins ()`: pure output-only fill; legal because outs-as-init
// supplies the body's read channel.
// CHECK-LABEL: func.func @fill_only
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index)
// CHECK-SAME: ins ()
// CHECK-SAME: outs (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["N"]>)
func.func @fill_only(%n: index, %dst: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %z = hc.const<0.0 : f32> : f32
  %r = hc.generic
      iter (parallel i = %n : index)
      ins ()
      outs (%dst at (#hc.expr<"i">) : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%dv: f32):
    hc.yield %z : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// Bare vector operands: the op handles either bare carrier kind without
// constraint chaining.
// CHECK-LABEL: func.func @vector_operands
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_vector<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.bare_vector<f32, ["N"]>)
func.func @vector_operands(%n: index,
                           %a: !hc.bare_vector<f32, ["N"]>,
                           %c: !hc.bare_vector<f32, ["N"]>)
    -> !hc.bare_vector<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at (#hc.expr<"i">) : !hc.bare_vector<f32, ["N"]>)
      outs (%c at (#hc.expr<"i">) : !hc.bare_vector<f32, ["N"]>)
      -> (!hc.bare_vector<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_vector<f32, ["N"]>
}

// Pre-inference flavour: `!hc.undef` operand types and block args round-
// trip without engaging the parity check, so the frontend can emit the
// op before type inference fills in the carrier types.
// CHECK-LABEL: func.func @progressive_undef
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.undef)
// CHECK-SAME: outs (%{{[^ ]+}} at (#hc.expr<"i">) : !hc.undef)
func.func @progressive_undef(%n: index, %a: !hc.undef, %c: !hc.undef)
    -> !hc.undef {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at (#hc.expr<"i">) : !hc.undef)
      outs (%c at (#hc.expr<"i">) : !hc.undef)
      -> (!hc.undef) {
  ^bb0(%av: !hc.undef, %cv: !hc.undef):
    hc.yield %cv : !hc.undef
  }
  return %r : !hc.undef
}
