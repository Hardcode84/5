// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-shaped-compute-to-generic`. The pass rewrites
// the source-level `hc.matmul` and `hc.reduce` ops into `hc.generic`
// + an identity fill on the result. v0 only handles rank-2 matmul
// with a uniform arith family on inputs and output, and reductions
// along a single axis with `keepdims = false` (floats fully, integer
// sum). Unsupported shapes leave the op alone so downstream
// diagnostics still fire.
//
// RUN: hc-opt --hc-shaped-compute-to-generic %s --split-input-file | FileCheck %s

// Plain f32 matmul. Iter syms `i`, `j`, `k`; ins addressed `[i,k]`
// and `[k,j]`; outs `[i,j]`. Body keeps the body in HC scalar ops
// so the round-trip stays canonical (no unintended arith mixin).
// CHECK-LABEL: func.func @matmul_f32
// CHECK-DAG: %[[M:.+]] = hc.materialize_bound_expr : !hc.idx<"M">
// CHECK-DAG: %[[N:.+]] = hc.materialize_bound_expr : !hc.idx<"N">
// CHECK-DAG: %[[K:.+]] = hc.materialize_bound_expr : !hc.idx<"K">
// CHECK: %[[SHAPE:.+]] = hc.tuple(%[[M]], %[[N]])
// CHECK: %[[FILL:.+]] = hc.zeros shape %[[SHAPE]] {{.*}} -> !hc.tensor<f32, ["M", "N"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %[[M]] : !hc.idx<"M">, parallel j = %[[N]] : !hc.idx<"N">, reduction k = %[[K]] : !hc.idx<"K">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i">, #hc.expr<"k">] : !hc.tensor<f32, ["M", "K"]>,
// CHECK-SAME:      %{{.+}} at [#hc.expr<"k">, #hc.expr<"j">] : !hc.tensor<f32, ["K", "N"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i">, #hc.expr<"j">] : !hc.tensor<f32, ["M", "N"]>)
// CHECK: ^bb0(%[[AV:.+]]: f32, %[[BV:.+]]: f32, %[[CV:.+]]: f32):
// CHECK:   %[[P:.+]] = hc.mul %[[AV]], %[[BV]]
// CHECK:   %[[S:.+]] = hc.add %[[CV]], %[[P]]
// CHECK:   hc.yield %[[S]] : f32
// CHECK-NOT: hc.matmul
func.func @matmul_f32(%a: !hc.tensor<f32, ["M", "K"]>,
                      %b: !hc.tensor<f32, ["K", "N"]>)
    -> !hc.tensor<f32, ["M", "N"]> {
  %r = hc.matmul %a, %b
      : (!hc.tensor<f32, ["M", "K"]>, !hc.tensor<f32, ["K", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  return %r : !hc.tensor<f32, ["M", "N"]>
}

// -----

// Mixed precision: f16 inputs, f32 accumulator. Body emits `hc.astype`
// up to the result element type before the multiply.
// CHECK-LABEL: func.func @matmul_f16_to_f32
// CHECK: hc.zeros shape %{{[^ ]+}} {{.*}} -> !hc.tensor<f32, ["M", "N"]>
// CHECK: hc.generic
// CHECK: ^bb0(%[[AV:.+]]: f16, %[[BV:.+]]: f16, %[[CV:.+]]: f32):
// CHECK:   %[[AE:.+]] = hc.astype %[[AV]], target = f32 : f16 -> f32
// CHECK:   %[[BE:.+]] = hc.astype %[[BV]], target = f32 : f16 -> f32
// CHECK:   %[[P:.+]] = hc.mul %[[AE]], %[[BE]]
// CHECK:   %[[S:.+]] = hc.add %[[CV]], %[[P]]
// CHECK:   hc.yield %[[S]] : f32
func.func @matmul_f16_to_f32(%a: !hc.tensor<f16, ["M", "K"]>,
                             %b: !hc.tensor<f16, ["K", "N"]>)
    -> !hc.tensor<f32, ["M", "N"]> {
  %r = hc.matmul %a, %b
      : (!hc.tensor<f16, ["M", "K"]>, !hc.tensor<f16, ["K", "N"]>)
        -> !hc.tensor<f32, ["M", "N"]>
  return %r : !hc.tensor<f32, ["M", "N"]>
}

// -----

// Integer matmul takes the same `hc.add` / `hc.mul` body — those are
// element-type-generic, the lowering picks `arith.muli` / `arith.addi`
// based on the element type.
// CHECK-LABEL: func.func @matmul_i32
// CHECK: %[[FILL:.+]] = hc.full %{{[^ ]+}}, shape %{{[^ ]+}} {{.*}} -> !hc.tensor<i32, ["M", "N"]>
// CHECK: hc.generic
// CHECK: ^bb0(%[[AV:.+]]: i32, %[[BV:.+]]: i32, %[[CV:.+]]: i32):
// CHECK:   %[[P:.+]] = hc.mul %[[AV]], %[[BV]]
// CHECK:   %[[S:.+]] = hc.add %[[CV]], %[[P]]
// CHECK:   hc.yield %[[S]] : i32
func.func @matmul_i32(%a: !hc.tensor<i32, ["M", "K"]>,
                      %b: !hc.tensor<i32, ["K", "N"]>)
    -> !hc.tensor<i32, ["M", "N"]> {
  %r = hc.matmul %a, %b
      : (!hc.tensor<i32, ["M", "K"]>, !hc.tensor<i32, ["K", "N"]>)
        -> !hc.tensor<i32, ["M", "N"]>
  return %r : !hc.tensor<i32, ["M", "N"]>
}

// -----

// Reduce sum along axis 0: parallel iter for `N`, reduction iter for
// `M`. Output rank-1 over `N`. Identity fill is `hc.zeros`.
// CHECK-LABEL: func.func @reduce_sum_axis0
// CHECK-DAG: %[[NB:.+]] = hc.materialize_bound_expr : !hc.idx<"N">
// CHECK-DAG: %[[MB:.+]] = hc.materialize_bound_expr : !hc.idx<"M">
// CHECK: %[[SH:.+]] = hc.tuple(%[[NB]])
// CHECK: %[[FILL:.+]] = hc.zeros shape %[[SH]] {{.*}} -> !hc.tensor<f32, ["N"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[NB]] : !hc.idx<"N">, reduction r = %[[MB]] : !hc.idx<"M">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"r">, #hc.expr<"i_0">] : !hc.tensor<f32, ["M", "N"]>)
// CHECK-SAME: outs (%[[FILL]] at [#hc.expr<"i_0">] : !hc.tensor<f32, ["N"]>)
// CHECK: ^bb0(%[[V:.+]]: f32, %[[ACC:.+]]: f32):
// CHECK:   %[[S:.+]] = hc.add %[[ACC]], %[[V]]
// CHECK:   hc.yield %[[S]] : f32
func.func @reduce_sum_axis0(%a: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["N"]> {
  %r = hc.reduce %a, kind = sum, axis = 0
      : !hc.tensor<f32, ["M", "N"]> -> !hc.tensor<f32, ["N"]>
  return %r : !hc.tensor<f32, ["N"]>
}

// -----

// Reduce sum along axis 1: parallel iter for `M`, reduction iter for
// `N`. Verifies the parallel-vs-reduction slot is keyed off `axis`,
// not iter-sym index.
// CHECK-LABEL: func.func @reduce_sum_axis1
// CHECK-DAG: %[[MB:.+]] = hc.materialize_bound_expr : !hc.idx<"M">
// CHECK-DAG: %[[NB:.+]] = hc.materialize_bound_expr : !hc.idx<"N">
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_0 = %[[MB]] : !hc.idx<"M">, reduction r = %[[NB]] : !hc.idx<"N">)
// CHECK-SAME: ins (%{{.+}} at [#hc.expr<"i_0">, #hc.expr<"r">] : !hc.tensor<f32, ["M", "N"]>)
// CHECK-SAME: outs (%{{.+}} at [#hc.expr<"i_0">] : !hc.tensor<f32, ["M"]>)
func.func @reduce_sum_axis1(%a: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["M"]> {
  %r = hc.reduce %a, kind = sum, axis = 1
      : !hc.tensor<f32, ["M", "N"]> -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// Reduce max on float: identity is -inf via `hc.full`, body uses
// `arith.maximumf` (no HC-level scalar max op exists yet).
// CHECK-LABEL: func.func @reduce_max_f32
// CHECK: %[[NEG_INF:.+]] = hc.const<0xFF800000 : f32> : f32
// CHECK: %[[FILL:.+]] = hc.full %[[NEG_INF]], shape %{{[^ ]+}} {{.*}} -> !hc.tensor<f32, ["N"]>
// CHECK: hc.generic
// CHECK: ^bb0(%[[V:.+]]: f32, %[[ACC:.+]]: f32):
// CHECK:   %[[S:.+]] = arith.maximumf %[[ACC]], %[[V]]
// CHECK:   hc.yield %[[S]] : f32
func.func @reduce_max_f32(%a: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["N"]> {
  %r = hc.reduce %a, kind = max, axis = 0
      : !hc.tensor<f32, ["M", "N"]> -> !hc.tensor<f32, ["N"]>
  return %r : !hc.tensor<f32, ["N"]>
}

// -----

// Reduce min on float: identity is +inf via `hc.full`, body uses
// `arith.minimumf`.
// CHECK-LABEL: func.func @reduce_min_f32
// CHECK: %[[POS_INF:.+]] = hc.const<0x7F800000 : f32> : f32
// CHECK: %[[FILL:.+]] = hc.full %[[POS_INF]], shape %{{[^ ]+}} {{.*}} -> !hc.tensor<f32, ["M"]>
// CHECK: hc.generic
// CHECK: ^bb0(%[[V:.+]]: f32, %[[ACC:.+]]: f32):
// CHECK:   %[[S:.+]] = arith.minimumf %[[ACC]], %[[V]]
// CHECK:   hc.yield %[[S]] : f32
func.func @reduce_min_f32(%a: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["M"]> {
  %r = hc.reduce %a, kind = min, axis = 1
      : !hc.tensor<f32, ["M", "N"]> -> !hc.tensor<f32, ["M"]>
  return %r : !hc.tensor<f32, ["M"]>
}

// -----

// Integer sum: identity is `hc.full <0>`, body uses `hc.add` (lowers
// to `arith.addi` for ints).
// CHECK-LABEL: func.func @reduce_sum_i32
// CHECK: %[[ZERO:.+]] = hc.const<0 : i32> : i32
// CHECK: %[[FILL:.+]] = hc.full %[[ZERO]], shape %{{[^ ]+}} {{.*}} -> !hc.tensor<i32, ["N"]>
// CHECK: hc.generic
// CHECK: ^bb0(%[[V:.+]]: i32, %[[ACC:.+]]: i32):
// CHECK:   %[[S:.+]] = hc.add %[[ACC]], %[[V]]
// CHECK:   hc.yield %[[S]] : i32
func.func @reduce_sum_i32(%a: !hc.tensor<i32, ["M", "N"]>)
    -> !hc.tensor<i32, ["N"]> {
  %r = hc.reduce %a, kind = sum, axis = 0
      : !hc.tensor<i32, ["M", "N"]> -> !hc.tensor<i32, ["N"]>
  return %r : !hc.tensor<i32, ["N"]>
}

// -----

// Pre-inference op: operand and result are `!hc.undef`. The pass has
// no shape info to emit bounds from, so it leaves the op alone for
// later inference / downstream diagnostics.
// CHECK-LABEL: func.func @noop_undef
// CHECK: hc.matmul
// CHECK-NOT: hc.generic
func.func @noop_undef(%a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  %r = hc.matmul %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// `keepdims = true` requires a different output shape (the reduced
// axis stays as size 1), which the v0 rewrite doesn't model. Op is
// left alone.
// CHECK-LABEL: func.func @noop_keepdims
// CHECK: hc.reduce
// CHECK-NOT: hc.generic
func.func @noop_keepdims(%a: !hc.tensor<f32, ["M", "N"]>)
    -> !hc.tensor<f32, ["1", "N"]> {
  %r = hc.reduce %a, kind = sum, axis = 0, keepdims = true
      : !hc.tensor<f32, ["M", "N"]> -> !hc.tensor<f32, ["1", "N"]>
  return %r : !hc.tensor<f32, ["1", "N"]>
}

// -----

// Integer max identity needs a signed-vs-unsigned slot decision and
// the matching combinator op (no `hc.max` or `arith.maxsi/maxui`
// helper wired in v0). Op is left alone.
// CHECK-LABEL: func.func @noop_int_max
// CHECK: hc.reduce
// CHECK-NOT: hc.generic
func.func @noop_int_max(%a: !hc.tensor<i32, ["M", "N"]>)
    -> !hc.tensor<i32, ["N"]> {
  %r = hc.reduce %a, kind = max, axis = 0
      : !hc.tensor<i32, ["M", "N"]> -> !hc.tensor<i32, ["N"]>
  return %r : !hc.tensor<i32, ["N"]>
}
