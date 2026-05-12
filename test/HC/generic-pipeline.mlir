// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end smoke for the generic-based compute pipeline. Mirrors the
// chain `hc/schedules/front_to_hc.mlir` runs after decompose-shaped-values:
//
//   `hc-canonicalize-layouts`
//     -> `hc-shaped-compute-to-generic`
//     -> `hc-elementwise-to-generic`
//     -> `hc-load-store-to-generic`
//     -> `hc-infer-generic-bounds`
//
// Each rewriter is conservative on inputs it can't prove safe; this LIT
// pins the post-chain shape on the surface it does support — rank-2
// matmul, single-axis reduce, all-shaped per-element arith, pinned
// `!hc.idx<expr>`-indexed loads — so the schedule wiring stays
// observably wired up under one diff. Per-pass details live in the
// dedicated LITs (`shaped-compute-to-generic.mlir`,
// `elementwise-to-generic.mlir`, `load-store-to-generic.mlir`,
// `infer-generic-bounds.mlir`); the goal here is end-to-end composition.
//
// RUN: hc-opt %s --pass-pipeline='builtin.module(hc-canonicalize-layouts,hc-shaped-compute-to-generic,hc-elementwise-to-generic,hc-load-store-to-generic,hc-infer-generic-bounds)' --split-input-file | FileCheck %s
//
// Same chain plus `hc-flatten-with-layouts`. The generic-rewriter
// outputs feed straight into the flatten step — every shaped
// operand/result collapses to its 1D storage form, layout slots
// vanish, and `hc.generic`'s per-operand per-axis offset arrays
// compose through the operand's layout (or the identity-layout
// fallback when the operand has no layout) into a single 1D offset,
// matching the post-flatten 1D operand rank.
// RUN: hc-opt %s --pass-pipeline='builtin.module(hc-canonicalize-layouts,hc-shaped-compute-to-generic,hc-elementwise-to-generic,hc-load-store-to-generic,hc-infer-generic-bounds,hc-flatten-with-layouts)' --split-input-file | FileCheck %s --check-prefix=POSTFLATTEN --implicit-check-not='#hc.layout'

// Matmul + elementwise add on the result. The matmul rewriter emits one
// `hc.generic` with parallel-parallel-reduction iters and a `hc.zeros`
// fill; the add rewriter emits a second `hc.generic` with two parallel
// iters and `!hc.undef` placeholder bounds, which `hc-infer-generic-bounds`
// then resolves to the operand-shape-derived `!hc.idx<"M">` /
// `!hc.idx<"N">` SSA. After the chain there are no `hc.matmul` /
// `hc.add` ops and every iter bound is concrete.
// CHECK-LABEL: func.func @matmul_then_add
// CHECK-NOT: hc.matmul
// CHECK-NOT: hc.add %{{[^ ]+}}, %{{[^ ]+}} : (!hc.tensor
// CHECK-NOT: hc.undef_value
// CHECK: %[[M0:.+]] = hc.idx_apply () : () -> !hc.idx<"M">
// CHECK: %[[N0:.+]] = hc.idx_apply () : () -> !hc.idx<"N">
// CHECK: %[[K0:.+]] = hc.idx_apply () : () -> !hc.idx<"K">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %[[M0]] : !hc.idx<"M">, parallel j = %[[N0]] : !hc.idx<"N">, reduction k = %[[K0]] : !hc.idx<"K">)
// CHECK: ^bb0(%{{.+}}: f32, %{{.+}}: f32, %{{.+}}: f32):
// CHECK:   hc.mul
// CHECK:   hc.add
// CHECK:   hc.yield
// CHECK: %[[M1:.+]] = hc.idx_apply () : () -> !hc.idx<"M">
// CHECK: %[[N1:.+]] = hc.idx_apply () : () -> !hc.idx<"N">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[M1]] : !hc.idx<"M">, parallel i_1 = %[[N1]] : !hc.idx<"N">)
// CHECK: ^bb0(%[[L:.+]]: f32, %[[R:.+]]: f32, %{{.+}}: f32):
// CHECK:   %[[S:.+]] = hc.add %[[L]], %[[R]] : (f32, f32) -> f32
// CHECK:   hc.yield %[[S]] : f32

// Each shaped tensor arg expands 1-to-N into a flat carrier + one
// `!hc.idx<sym>` per free dim symbol from its pre-flatten shape;
// `hc.generic`'s per-axis offsets compose through the identity
// layout (the no-layout fallback) into a single 1D offset that matches the
// 1D operand rank. `[i, k]` over `[M, K]` becomes `k + K*i`, etc.
// `hc.matmul` and the bias `hc.add` are already gone after the
// rewriters above; the implicit-check banner pins that no
// `#hc.layout` survives.
// POSTFLATTEN-LABEL: func.func @matmul_then_add
// POSTFLATTEN-SAME: -> (!hc.tensor<f32, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">)
// POSTFLATTEN: hc.generic iter (parallel i = %{{.+}} : !hc.idx<"M">, parallel j = %{{.+}} : !hc.idx<"N">, reduction k = %{{.+}} : !hc.idx<"K">) ins (%{{.+}} at [#hc.expr<"k + K*i">] : !hc.tensor<f32, ["K*M"]>, %{{.+}} at [#hc.expr<"j + N*k">] : !hc.tensor<f32, ["K*N"]>) outs (%{{.+}} at [#hc.expr<"j + N*i">] : !hc.tensor<f32, ["M*N"]>)
// POSTFLATTEN: hc.generic iter (parallel i_0 = %{{.+}} : !hc.idx<"M">, parallel i_1 = %{{.+}} : !hc.idx<"N">)
func.func @matmul_then_add(%a: !hc.tensor<f32, ["M", "K"]>,
                           %b: !hc.tensor<f32, ["K", "N"]>,
                           %bias: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["M", "N"]> {
  %p = hc.matmul %a, %b
      : (!hc.tensor<f32, ["M", "K"]>, !hc.tensor<f32, ["K", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  %r = hc.add %p, %bias
      : (!hc.tensor<f32, ["M", "N"]>, !hc.tensor<f32, ["M", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  return %r : !hc.tensor<f32, ["M", "N"]>
}

// -----

// Reduce sum along axis 1 of a rank-2 float tensor. Sum-identity
// `hc.zeros` fill, one parallel iter (M) and one reduction iter (N).
// `hc-shaped-compute-to-generic` picks the per-axis iter sym names —
// `i_<k>` for parallel iters, `r` for the reduction.
// CHECK-LABEL: func.func @reduce_sum_axis1
// CHECK-NOT: hc.reduce
// CHECK: %[[M:.+]] = hc.idx_apply () : () -> !hc.idx<"M">
// CHECK: %[[N:.+]] = hc.idx_apply () : () -> !hc.idx<"N">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[M]] : !hc.idx<"M">, reduction r = %[[N]] : !hc.idx<"N">)
// CHECK: ^bb0(%[[V:.+]]: f32, %[[ACC:.+]]: f32):
// CHECK:   %[[S:.+]] = hc.add %[[ACC]], %[[V]]
// CHECK:   hc.yield %[[S]]

// Reduce already produces a 1D result type, so the output side's
// per-axis array is already a single entry; the rank-2 input
// tensor's `[i_0, r]` over `[M, N]` composes through the identity
// layout to `r + N*i_0`.
// POSTFLATTEN-LABEL: func.func @reduce_sum_axis1
// POSTFLATTEN-SAME: !hc.tensor<f32, ["M*N"]>
// POSTFLATTEN: hc.generic iter (parallel i_0 = %{{.+}} : !hc.idx<"M">, reduction r = %{{.+}} : !hc.idx<"N">) ins (%{{.+}} at [#hc.expr<"r + N*i_0">] : !hc.tensor<f32, ["M*N"]>) outs (%{{.+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["M"]>)
func.func @reduce_sum_axis1(%v: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["M"]> {
  %r = hc.reduce %v, kind = sum, axis = 1
      : !hc.tensor<f32, ["M", "N"]> -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// Pure-elementwise. Two binary arith ops feed each other; both rewrite
// into `hc.generic` with `!hc.undef` placeholder bounds, then
// `hc-infer-generic-bounds` resolves them from the operand shapes.
// `hc-elementwise-to-generic` keeps the `hc.zeros shape ...` tuple
// pinned to `!hc.undef` placeholders even after bounds inference —
// that's part of the v0 surface; the post-flatten lowering picks the
// concrete extent off the result type, not the shape tuple.
// CHECK-LABEL: func.func @elementwise_chain
// CHECK-NOT: hc.mul %{{[^ ]+}}, %{{[^ ]+}} : (!hc.tensor
// CHECK-NOT: hc.add %{{[^ ]+}}, %{{[^ ]+}} : (!hc.tensor
// CHECK-COUNT-2: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{.+}} : !hc.idx<"M">, parallel i_1 = %{{.+}} : !hc.idx<"N">)

// All three operands and the result expand 1-to-3 (flat + dim aux
// for M and N). The rank-2 `[i_0, i_1]` per-axis arrays compose
// through the identity layout to `i_1 + N*i_0` on every operand,
// matching the 1D `["M*N"]` storage form.
// POSTFLATTEN-LABEL: func.func @elementwise_chain
// POSTFLATTEN-SAME: -> (!hc.tensor<f32, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">)
// POSTFLATTEN: hc.generic iter (parallel i_0 = %{{.+}} : !hc.idx<"M">, parallel i_1 = %{{.+}} : !hc.idx<"N">) ins (%{{.+}} at [#hc.expr<"i_1 + N*i_0">] : !hc.tensor<f32, ["M*N"]>, %{{.+}} at [#hc.expr<"i_1 + N*i_0">] : !hc.tensor<f32, ["M*N"]>) outs (%{{.+}} at [#hc.expr<"i_1 + N*i_0">] : !hc.tensor<f32, ["M*N"]>)
// POSTFLATTEN: hc.generic iter (parallel i_0 = %{{.+}} : !hc.idx<"M">, parallel i_1 = %{{.+}} : !hc.idx<"N">) ins (%{{.+}} at [#hc.expr<"i_1 + N*i_0">] : !hc.tensor<f32, ["M*N"]>, %{{.+}} at [#hc.expr<"i_1 + N*i_0">] : !hc.tensor<f32, ["M*N"]>) outs (%{{.+}} at [#hc.expr<"i_1 + N*i_0">] : !hc.tensor<f32, ["M*N"]>)
func.func @elementwise_chain(%a: !hc.tensor<f32, ["M", "N"]>,
                             %b: !hc.tensor<f32, ["M", "N"]>,
                             %c: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["M", "N"]> {
  %p = hc.mul %a, %b
      : (!hc.tensor<f32, ["M", "N"]>, !hc.tensor<f32, ["M", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  %r = hc.add %p, %c
      : (!hc.tensor<f32, ["M", "N"]>, !hc.tensor<f32, ["M", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  return %r : !hc.tensor<f32, ["M", "N"]>
}

// -----

// Load from a buffer with pinned `!hc.idx<expr>` indices flows through
// `hc-load-store-to-generic`. The result tile is the `outs` init via
// `hc.zeros`; the source is read at `[i_0_expr + i_0, i_1_expr + i_1]`.
// CHECK-LABEL: func.func @tiled_load
// CHECK-NOT: hc.load %{{.+}}[
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %{{.+}} : !hc.idx<"16">, parallel i_1 = %{{.+}} : !hc.idx<"16">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"16*$WG0 + i_0">, #hc.expr<"i_1">]
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i_0">, #hc.expr<"i_1">]
// CHECK: ^bb0(%[[V:.+]]: f32, %{{.+}}: f32):
// CHECK:   hc.yield %[[V]] : f32

// Buffer args collapse to `!hc.buffer<..., ["?"]>` (host owns the
// allocation, the IR doesn't have enough symbols to name the
// extent), with dim aux idxs trailing. The buffer's `[16*$WG0+i_0,
// i_1]` access composes against the identity layout over `[M, N]`
// (this LIT skips the `hc-canonicalize-layouts` default-strided-
// layout attach for `func.func` args, so there's no explicit layout
// to substitute through) to `i_1 + N*(16*$WG0 + i_0)`. The result
// tile is statically `[16, 16]`, collapsing to flat `["256"]` with
// no trailing aux; its `[i_0, i_1]` composes to `16*i_0 + i_1`.
// POSTFLATTEN-LABEL: func.func @tiled_load
// POSTFLATTEN-SAME: !hc.buffer<f32, ["?"]>
// POSTFLATTEN-SAME: -> !hc.tensor<f32, ["256"]>
// POSTFLATTEN: hc.generic iter (parallel i_0 = %{{.+}} : !hc.idx<"16">, parallel i_1 = %{{.+}} : !hc.idx<"16">) ins (%{{.+}} at [#hc.expr<"i_1 + N*(16*$WG0 + i_0)">] : !hc.buffer<f32, ["?"]>) outs (%{{.+}} at [#hc.expr<"16*i_0 + i_1">] : !hc.tensor<f32, ["256"]>)
func.func @tiled_load(%a: !hc.buffer<f32, ["M", "N"]>,
                      %row: !hc.idx<"16*$WG0">,
                      %col: !hc.idx<"0">)
    -> !hc.tensor<f32, ["16", "16"]> {
  %sixteen = hc.const<16 : i64> : !hc.idx<"16">
  %shape = hc.tuple(%sixteen, %sixteen) : (!hc.idx<"16">, !hc.idx<"16">)
      -> tuple<!hc.idx<"16">, !hc.idx<"16">>
  %tile = hc.load %a[%row, %col], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"16*$WG0">, !hc.idx<"0">,
         tuple<!hc.idx<"16">, !hc.idx<"16">>)
        -> !hc.tensor<f32, ["16", "16"]>
  return %tile : !hc.tensor<f32, ["16", "16"]>
}
