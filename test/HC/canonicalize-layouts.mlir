// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `hc-canonicalize-layouts`:
//   * strips explicit identity layouts from shaped Values, refreshes the
//     cached function-type attribute when entry-arg types changed,
//   * recognizes ixsimpl-equivalent identity spellings,
//   * keeps non-identity layouts (padded, col-major, params-bearing),
//   * collapses chained `hc.as_layout(hc.as_layout(...))` to the outer
//     choice and erases the dead inner op.
//
// RUN: hc-opt -split-input-file -hc-canonicalize-layouts %s | FileCheck %s

// CHECK-LABEL: @identity_tensor
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M", "N"]>
// CHECK-NOT: #hc.layout
func.func @identity_tensor(
    %a: !hc.tensor<f16, ["M", "N"],
                   #hc.layout<shape_syms = ["M", "N"],
                              index_syms = ["i", "j"],
                              params = {},
                              storage_size = #hc.expr<"M * N">,
                              offset = #hc.expr<"i * N + j">>>) {
  return
}

// -----

// Equivalent ixsimpl spelling of the identity offset / storage_size still
// strips. The store canonicalizes `j + N*i` and `M*N` at parse time, so
// the verifier matches against the same canonical handle.
// CHECK-LABEL: @identity_tensor_reordered
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M", "N"]>
// CHECK-NOT: #hc.layout
func.func @identity_tensor_reordered(
    %a: !hc.tensor<f16, ["M", "N"],
                   #hc.layout<shape_syms = ["M", "N"],
                              index_syms = ["i", "j"],
                              params = {},
                              storage_size = #hc.expr<"N * M">,
                              offset = #hc.expr<"j + i * N">>>) {
  return
}

// -----

// All five shaped types fold the same way; absent layout is the v0
// identity contract for each. Note `shape_syms` are layout-local symbol
// names that get bound positionally to the host type's shape — for the
// vector cases here, "d0" binds to the literal 8 / 16 at the use site,
// not the other way around.
// CHECK-LABEL: @identity_all_five
// CHECK-SAME: %arg0: !hc.buffer<f32, ["M"]>
// CHECK-SAME: %arg1: !hc.tensor<f16, ["M"]>
// CHECK-SAME: %arg2: !hc.vector<f32, ["8"]>
// CHECK-SAME: %arg3: !hc.bare_tensor<f16, ["M"]>
// CHECK-SAME: %arg4: !hc.bare_vector<f16, ["16"]>
func.func @identity_all_five(
    %a: !hc.buffer<f32, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>,
    %b: !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>,
    %c: !hc.vector<f32, ["8"], #hc.layout<shape_syms = ["d0"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"d0">, offset = #hc.expr<"i">>>,
    %d: !hc.bare_tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>,
    %e: !hc.bare_vector<f16, ["16"], #hc.layout<shape_syms = ["d0"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"d0">, offset = #hc.expr<"i">>>) {
  return
}

// -----

// Padded layouts (non-empty params) are preserved — they don't match
// the identity contract.
// CHECK-LABEL: @padded_preserved
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M", "K"], <
// CHECK-SAME: params = {row_stride = #hc.expr<"4 + K">}
func.func @padded_preserved(
    %a: !hc.tensor<f16, ["M", "K"],
                   #hc.layout<shape_syms = ["M", "K"],
                              index_syms = ["i", "j"],
                              params = {row_stride = #hc.expr<"K + 4">},
                              storage_size = #hc.expr<"M * row_stride">,
                              offset = #hc.expr<"i * row_stride + j">>>) {
  return
}

// -----

// Col-major layout (offset uses the first shape sym as the inner stride)
// is preserved.
// CHECK-LABEL: @col_major_preserved
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M", "N"], <
// CHECK-SAME: offset = #hc.expr<"i + M*j">
func.func @col_major_preserved(
    %a: !hc.tensor<f16, ["M", "N"],
                   #hc.layout<shape_syms = ["M", "N"],
                              index_syms = ["i", "j"],
                              params = {},
                              storage_size = #hc.expr<"M * N">,
                              offset = #hc.expr<"j * M + i">>>) {
  return
}

// -----

// Layout with a params entry that doesn't appear in offset still loses
// the identity-fold privilege: any non-empty params blocks the strip.
// CHECK-LABEL: @bound_param_preserved
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M", "N"], <
// CHECK-SAME: params = {unused = #hc.expr<"5">}
func.func @bound_param_preserved(
    %a: !hc.tensor<f16, ["M", "N"],
                   #hc.layout<shape_syms = ["M", "N"],
                              index_syms = ["i", "j"],
                              params = {unused = #hc.expr<"5">},
                              storage_size = #hc.expr<"M * N">,
                              offset = #hc.expr<"i * N + j">>>) {
  return
}

// -----

// Rank-zero identity layout (empty syms, storage_size = 1, offset = 0).
// CHECK-LABEL: @identity_rank_zero
// CHECK-SAME: %arg0: !hc.buffer<f32, []>
// CHECK-NOT: #hc.layout
func.func @identity_rank_zero(
    %a: !hc.buffer<f32, [],
                   #hc.layout<shape_syms = [], index_syms = [],
                              params = {},
                              storage_size = #hc.expr<"1">,
                              offset = #hc.expr<"0">>>) {
  return
}

// -----

// `hc.as_layout(hc.as_layout(%v, L1), L2)` collapses to
// `hc.as_layout(%v, L2)`; the inner op becomes dead and is erased.
// CHECK-LABEL: @chained_as_layout
// CHECK-NOT: hc.as_layout %{{.*}} layout = row_major
// CHECK: %[[R:.*]] = hc.as_layout %arg0, layout = col_major
// CHECK: return %[[R]]
func.func @chained_as_layout(%v: !hc.undef) -> !hc.undef {
  %x = hc.as_layout %v, layout = row_major : !hc.undef -> !hc.undef
  %y = hc.as_layout %x, layout = col_major : !hc.undef -> !hc.undef
  return %y : !hc.undef
}

// -----

// Three-deep chain collapses to the outer op via worklist.
// CHECK-LABEL: @triple_as_layout
// CHECK-COUNT-1: hc.as_layout
// CHECK-NOT: hc.as_layout
func.func @triple_as_layout(%v: !hc.undef) -> !hc.undef {
  %a = hc.as_layout %v, layout = col_major : !hc.undef -> !hc.undef
  %b = hc.as_layout %a, layout = row_major : !hc.undef -> !hc.undef
  %c = hc.as_layout %b, layout = col_major : !hc.undef -> !hc.undef
  return %c : !hc.undef
}

// -----

// A single (uncollapsed) `hc.as_layout` is left alone.
// CHECK-LABEL: @single_as_layout
// CHECK: hc.as_layout %arg0, layout = col_major
func.func @single_as_layout(%v: !hc.undef) -> !hc.undef {
  %x = hc.as_layout %v, layout = col_major : !hc.undef -> !hc.undef
  return %x : !hc.undef
}

// -----

// Identity layouts inside tuples / function results also strip.
// CHECK-LABEL: @identity_in_tuple
// CHECK-SAME: %arg0: tuple<!hc.tensor<f16, ["M"]>, !hc.buffer<f32, ["N"]>>
func.func @identity_in_tuple(
    %a: tuple<
          !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>,
          !hc.buffer<f32, ["N"], #hc.layout<shape_syms = ["N"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"N">, offset = #hc.expr<"i">>>
        >) {
  return
}

// -----

// Control flow: an `scf.for` carrying an identity-layout iter_arg
// strips on every type surface — the func entry block arg, the loop
// init operand, the loop result, the body's induction-var-adjacent
// block arg, and the `scf.yield` operand. Drives the upstream
// `populateSCFStructuralTypeConversionsAndLegality` patterns so we
// inherit their clone-and-replace correctness instead of poking
// types in place.
// CHECK-LABEL: @scf_for_identity_iter_arg
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M"]>
// CHECK-SAME: -> !hc.tensor<f16, ["M"]>
// CHECK: %[[R:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A:.*]] = %arg0) -> (!hc.tensor<f16, ["M"]>)
// CHECK: scf.yield %[[A]] : !hc.tensor<f16, ["M"]>
// CHECK: return %[[R]] : !hc.tensor<f16, ["M"]>
func.func @scf_for_identity_iter_arg(
    %t: !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>,
    %lb: index, %ub: index, %step: index)
    -> !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %t)
      -> (!hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>) {
    scf.yield %acc : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
  }
  return %r : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
}

// -----

// Non-identity iter_arg layout (col-major) is preserved on every
// surface — the strip never fires, the loop signature stays
// layout-bearing end to end.
// CHECK-LABEL: @scf_for_nonidentity_iter_arg_preserved
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M", "N"], <{{.*}}offset = #hc.expr<"i + M*j">{{.*}}>>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!hc.tensor<f16, ["M", "N"], <{{.*}}offset = #hc.expr<"i + M*j">{{.*}}>>)
func.func @scf_for_nonidentity_iter_arg_preserved(
    %t: !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["M", "N"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"M * N">, offset = #hc.expr<"j * M + i">>>,
    %lb: index, %ub: index, %step: index)
    -> !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["M", "N"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"M * N">, offset = #hc.expr<"j * M + i">>> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %t)
      -> (!hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["M", "N"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"M * N">, offset = #hc.expr<"j * M + i">>>) {
    scf.yield %acc : !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["M", "N"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"M * N">, offset = #hc.expr<"j * M + i">>>
  }
  return %r : !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["M", "N"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"M * N">, offset = #hc.expr<"j * M + i">>>
}

// -----

// `scf.if` returning an identity-layout tensor: result type strips,
// both then / else regions yield the layout-less form. Confirms the
// strip flows through region terminators in the multi-branch case.
// CHECK-LABEL: @scf_if_identity_result
// CHECK-SAME: %arg1: !hc.tensor<f16, ["M"]>
// CHECK: scf.if %arg0 -> (!hc.tensor<f16, ["M"]>)
// CHECK: scf.yield %arg1 : !hc.tensor<f16, ["M"]>
// CHECK: scf.yield %arg1 : !hc.tensor<f16, ["M"]>
func.func @scf_if_identity_result(
    %cond: i1,
    %t: !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>)
    -> !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>> {
  %r = scf.if %cond
      -> (!hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>) {
    scf.yield %t : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
  } else {
    scf.yield %t : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
  }
  return %r : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
}

// -----

// Nested `scf.if` inside `scf.for` — both ops carry identity layouts
// on their result / iter_arg surface. Confirms the conversion driver
// keeps walking inside-out instead of stalling at the outer op.
// CHECK-LABEL: @scf_nested_if_in_for
// CHECK-SAME: %arg1: !hc.tensor<f16, ["M"]>
// CHECK: %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %arg1) -> (!hc.tensor<f16, ["M"]>)
// CHECK: %[[INNER:.*]] = scf.if {{.*}} -> (!hc.tensor<f16, ["M"]>)
// CHECK: scf.yield %[[ACC]] : !hc.tensor<f16, ["M"]>
// CHECK: scf.yield %[[ACC]] : !hc.tensor<f16, ["M"]>
// CHECK: scf.yield %[[INNER]] : !hc.tensor<f16, ["M"]>
// CHECK: return %[[OUTER]] : !hc.tensor<f16, ["M"]>
func.func @scf_nested_if_in_for(
    %cond: i1,
    %t: !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>,
    %lb: index, %ub: index, %step: index)
    -> !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %t)
      -> (!hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>) {
    %inner = scf.if %cond
        -> (!hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>) {
      scf.yield %acc : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
    } else {
      scf.yield %acc : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
    }
    scf.yield %inner : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
  }
  return %r : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
}

// -----

// `func.call` between two functions whose signatures carry identity
// layouts: caller and callee strip in lockstep, the `func.call`
// operand and result types follow.
// CHECK-LABEL: func.func private @callee
// CHECK-SAME: (!hc.tensor<f16, ["M"]>) -> !hc.tensor<f16, ["M"]>
// CHECK-LABEL: @caller
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M"]>
// CHECK: %[[R:.*]] = call @callee(%arg0) : (!hc.tensor<f16, ["M"]>) -> !hc.tensor<f16, ["M"]>
// CHECK: return %[[R]] : !hc.tensor<f16, ["M"]>
func.func private @callee(
    !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>)
    -> !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
func.func @caller(
    %t: !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>)
    -> !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>> {
  %r = func.call @callee(%t)
      : (!hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>)
      -> !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
  return %r : !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["M"], index_syms = ["i"], params = {}, storage_size = #hc.expr<"M">, offset = #hc.expr<"i">>>
}
