// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-elementwise-to-generic`. The pass funnels the
// per-element shaped arith / cmp / astype family into `hc.generic`.
// Pure source-level rewrite — no memory effects, no codegen.
//
// RUN: hc-opt --hc-elementwise-to-generic %s --split-input-file | FileCheck %s

// Plain rank-2 add: two parallel iters with `!hc.undef` bounds (the
// later `hc-infer-generic-bounds` pass resolves them). All operands
// carry identity per-axis offsets. Body emits `hc.add` on the per-
// element block args.
// CHECK-LABEL: func.func @add_rank2_f32
// CHECK-DAG: %[[B0:.+]] = hc.undef_value : !hc.undef
// CHECK-DAG: %[[B1:.+]] = hc.undef_value : !hc.undef
// CHECK: %[[FILL:.+]] = hc.zeros shape %{{[^ ]+}} {{.*}} -> !hc.tensor<f32, ["M", "N"]>
// CHECK: %[[OUT:.+]] = hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[B0]] : !hc.undef, parallel i_1 = %[[B1]] : !hc.undef)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.tensor<f32, ["M", "N"]>,
// CHECK-SAME:      %{{[^ ]+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.tensor<f32, ["M", "N"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">, #hc.expr<"i_1">] : !hc.tensor<f32, ["M", "N"]>)
// CHECK: ^bb0(%[[A:.+]]: f32, %[[B:.+]]: f32, %{{.+}}: f32):
// CHECK:   %[[S:.+]] = hc.add %[[A]], %[[B]] : (f32, f32) -> f32
// CHECK:   hc.yield %[[S]] : f32
// CHECK-NOT: hc.add %{{.+}}, %{{.+}} : (!hc.tensor
func.func @add_rank2_f32(%a: !hc.tensor<f32, ["M", "N"]>,
                         %b: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["M", "N"]> {
  %r = hc.add %a, %b
      : (!hc.tensor<f32, ["M", "N"]>, !hc.tensor<f32, ["M", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  return %r : !hc.tensor<f32, ["M", "N"]>
}

// -----

// `hc.sub` / `hc.mul` / `hc.div` / `hc.mod` follow the same
// template — one round-trip per family is enough to confirm the
// dispatch table picks the right body op.
// CHECK-LABEL: func.func @binary_arith_family
// CHECK: hc.generic
// CHECK: ^bb0
// CHECK:   hc.sub
// CHECK: hc.generic
// CHECK: ^bb0
// CHECK:   hc.mul
// CHECK: hc.generic
// CHECK: ^bb0
// CHECK:   hc.div
// CHECK: hc.generic
// CHECK: ^bb0
// CHECK:   hc.mod
func.func @binary_arith_family(%a: !hc.tensor<f32, ["M"]>,
                               %b: !hc.tensor<f32, ["M"]>)
    -> (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>,
        !hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) {
  %s = hc.sub %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]>
  %m = hc.mul %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]>
  %d = hc.div %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]>
  %mo = hc.mod %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]>
  return %s, %m, %d, %mo : !hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>,
                            !hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>
}

// -----

// Booleans: same shape as the arith family but the body op is the
// boolean equivalent.
// CHECK-LABEL: func.func @logic_family
// CHECK: hc.generic
// CHECK:   hc.and
// CHECK: hc.generic
// CHECK:   hc.or
// CHECK: hc.generic
// CHECK:   hc.not
func.func @logic_family(%a: !hc.tensor<i1, ["M"]>,
                        %b: !hc.tensor<i1, ["M"]>)
    -> (!hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>) {
  %x = hc.and %a, %b
      : (!hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>) -> !hc.tensor<i1, ["M"]>
  %y = hc.or %a, %b
      : (!hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>) -> !hc.tensor<i1, ["M"]>
  %z = hc.not %a : !hc.tensor<i1, ["M"]> -> !hc.tensor<i1, ["M"]>
  return %x, %y, %z : !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>,
                       !hc.tensor<i1, ["M"]>
}

// -----

// Comparisons: result element type is `i1`, distinct from the input
// element type. Block args follow each operand's element type, so
// the body emits `hc.cmp.lt %f32, %f32 -> i1`.
// CHECK-LABEL: func.func @cmp_lt_f32
// CHECK: %[[FILL:.+]] = hc.zeros shape %{{[^ ]+}} {{.*}} -> !hc.tensor<i1, ["M"]>
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["M"]>,
// CHECK-SAME:      %{{[^ ]+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["M"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">] : !hc.tensor<i1, ["M"]>)
// CHECK: ^bb0(%[[A:.+]]: f32, %[[B:.+]]: f32, %{{.+}}: i1):
// CHECK:   %[[C:.+]] = hc.cmp.lt %[[A]], %[[B]] : (f32, f32) -> i1
// CHECK:   hc.yield %[[C]] : i1
func.func @cmp_lt_f32(%a: !hc.tensor<f32, ["M"]>,
                      %b: !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]> {
  %r = hc.cmp.lt %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]>
  return %r : !hc.tensor<i1, ["M"]>
}

// -----

// Sample one each from the comparison family — confirms the
// dispatch picks the right `HC_CmpOp` variant per source op.
// CHECK-LABEL: func.func @cmp_family
// CHECK: hc.generic
// CHECK:   hc.cmp.le
// CHECK: hc.generic
// CHECK:   hc.cmp.gt
// CHECK: hc.generic
// CHECK:   hc.cmp.ge
// CHECK: hc.generic
// CHECK:   hc.cmp.eq
// CHECK: hc.generic
// CHECK:   hc.cmp.ne
func.func @cmp_family(%a: !hc.tensor<f32, ["M"]>,
                      %b: !hc.tensor<f32, ["M"]>)
    -> (!hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>,
        !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>) {
  %le = hc.cmp.le %a, %b : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]>
  %gt = hc.cmp.gt %a, %b : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]>
  %ge = hc.cmp.ge %a, %b : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]>
  %eq = hc.cmp.eq %a, %b : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]>
  %ne = hc.cmp.ne %a, %b : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<i1, ["M"]>
  return %le, %gt, %ge, %eq, %ne
      : !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>,
        !hc.tensor<i1, ["M"]>, !hc.tensor<i1, ["M"]>
}

// -----

// `hc.astype`: rank stays, element type changes. Body emits a
// scalar `hc.astype` to the result element type.
// CHECK-LABEL: func.func @astype_f16_to_f32
// CHECK: %[[FILL:.+]] = hc.zeros shape %{{[^ ]+}} {{.*}} -> !hc.tensor<f32, ["M"]>
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i_0">] : !hc.tensor<f16, ["M"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">] : !hc.tensor<f32, ["M"]>)
// CHECK: ^bb0(%[[V:.+]]: f16, %{{.+}}: f32):
// CHECK:   %[[C:.+]] = hc.astype %[[V]], target = f32 : f16 -> f32
// CHECK:   hc.yield %[[C]] : f32
func.func @astype_f16_to_f32(%v: !hc.tensor<f16, ["M"]>) -> !hc.tensor<f32, ["M"]> {
  %r = hc.astype %v, target = f32
      : !hc.tensor<f16, ["M"]> -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// `hc.neg`: unary, no element type change.
// CHECK-LABEL: func.func @neg_f32
// CHECK: hc.generic
// CHECK: ^bb0(%[[V:.+]]: f32, %{{.+}}: f32):
// CHECK:   %[[N:.+]] = hc.neg %[[V]] : f32 -> f32
// CHECK:   hc.yield %[[N]] : f32
func.func @neg_f32(%v: !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]> {
  %r = hc.neg %v : !hc.tensor<f32, ["M"]> -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// Vector result picks `hc.vzeros` for the init.
// CHECK-LABEL: func.func @add_vector
// CHECK: hc.vzeros shape %{{[^ ]+}} {{.*}} -> !hc.vector<f32, ["M"]>
// CHECK: hc.generic
func.func @add_vector(%a: !hc.vector<f32, ["M"]>,
                      %b: !hc.vector<f32, ["M"]>) -> !hc.vector<f32, ["M"]> {
  %r = hc.add %a, %b
      : (!hc.vector<f32, ["M"]>, !hc.vector<f32, ["M"]>) -> !hc.vector<f32, ["M"]>
  return %r : !hc.vector<f32, ["M"]>
}

// -----

// Scalar / shaped mix is a broadcast the v0 rewrite doesn't model
// — the op stays for downstream / follow-up.
// CHECK-LABEL: func.func @broadcast_falls_through
// CHECK: hc.add %{{.+}}, %{{.+}} : (f32, !hc.tensor
// CHECK-NOT: hc.generic
func.func @broadcast_falls_through(%a: f32, %b: !hc.tensor<f32, ["M"]>)
    -> !hc.tensor<f32, ["M"]> {
  %r = hc.add %a, %b : (f32, !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// Pre-inference IR: operand and result are `!hc.undef`. The
// rewriter has no shape to source iter syms from and leaves the op
// alone for later inference / downstream diagnostics.
// CHECK-LABEL: func.func @undef_falls_through
// CHECK: hc.add %{{.+}}, %{{.+}} : (!hc.undef, !hc.undef)
// CHECK-NOT: hc.generic
func.func @undef_falls_through(%a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  %r = hc.add %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// Mismatched per-axis dim expressions: lhs is `["M"]`, rhs is
// `["N"]`. Symbolic shapes don't agree, so the rewrite bails
// rather than emit an `hc.generic` that picks one of the dims
// arbitrarily. Op stays for downstream diagnostics or a later
// broadcast-aware rewrite.
// CHECK-LABEL: func.func @mismatched_dims_falls_through
// CHECK: hc.add %{{.+}}, %{{.+}} : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["N"]>)
// CHECK-NOT: hc.generic
func.func @mismatched_dims_falls_through(%a: !hc.tensor<f32, ["M"]>,
                                          %b: !hc.tensor<f32, ["N"]>)
    -> !hc.tensor<f32, ["M"]> {
  %r = hc.add %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["N"]>) -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// The post-rewrite IR feeds straight into `hc-infer-generic-bounds`:
// every `hc.undef` placeholder bound resolves through the operand
// shapes (identity per-axis offsets), so the next pass picks up the
// concrete `!hc.idx<dim>` form without further help. Two-pass
// invocation here verifies the hand-off doesn't leave anything
// stranded.
//
// RUN: hc-opt --hc-elementwise-to-generic --hc-infer-generic-bounds %s --split-input-file | FileCheck %s --check-prefix=INFER

// CHECK-LABEL: func.func @infer_handoff
// INFER-LABEL: func.func @infer_handoff
// INFER: %{{[^ ]+}} = hc.materialize_bound_expr : !hc.idx<"M">
// INFER: hc.generic iter (parallel i_0 = %{{[^ ]+}} : !hc.idx<"M">)
// INFER-NOT: !hc.undef
func.func @infer_handoff(%a: !hc.tensor<f32, ["M"]>,
                         %b: !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]> {
  %r = hc.add %a, %b
      : (!hc.tensor<f32, ["M"]>, !hc.tensor<f32, ["M"]>) -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}
