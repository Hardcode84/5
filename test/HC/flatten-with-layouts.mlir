// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `hc-flatten-with-layouts`:
//   * collapses every shaped type to its 1D form. Tensors / vectors
//     (semantic and bare) get a concrete `storage_size_expr` from
//     the layout's `storage_size` after binding `shape_syms` to the
//     original shape entries — or from the dimension product when
//     the type sits on the implicit identity-row-major contract.
//     Buffers collapse to `[?]` (`#hc.dyn` sentinel) because the
//     host owns the allocation and the in-IR symbol set doesn't
//     have enough to name the storage extent.
//   * strips every `#hc.layout` slot off `SymbolicallyShapedTypeInterface`
//     types, including the non-identity ones `hc-canonicalize-layouts`
//     deliberately leaves alone (col-major, padded, params-bearing,
//     default strided buffer args).
//   * folds `hc.as_layout` ops to their operand once both endpoints
//     are converted (the relabel becomes cosmetic).
//   * propagates the collapse through tuple types and the upstream
//     func / scf / call signature populators.
//
// Op-surface contract: per-axis offset arrays on `hc.generic` and
// multi-index lists on `hc.load` / `hc.store` / `hc.vload` /
// `hc.buffer_view` are NOT rewritten — downstream fusion /
// vectorization needs the per-axis structure even after the operand
// type collapses to 1D. The verifier on `hc.generic` accepts that
// post-flatten shape (entries-per-operand stays at the original
// logical rank).
//
// Post-flatten invariant from `doc/layouts.md`: *no `#hc.layout`
// survives on any shaped type*. The implicit-check below pins that
// for the entire file.
//
// RUN: hc-opt -split-input-file -hc-flatten-with-layouts %s \
// RUN:   | FileCheck %s --implicit-check-not='#hc.layout' --implicit-check-not='hc.as_layout'

// CHECK-LABEL: @strided_buffer_arg
// CHECK-SAME: %arg0: !hc.buffer<f16, ["?"]>
// Default fully-strided np/torch layout the frontend pins on every
// buffer argument. Buffer shape collapses to the `[?]` sentinel
// because the host owns the allocation; the layout slot drops.
func.func @strided_buffer_arg(
    %a: !hc.buffer<f16, ["M", "K"],
                   #hc.layout<shape_syms = ["d0", "d1"],
                              index_syms = ["i0", "i1"],
                              params = {},
                              storage_size = #hc.expr<"0">,
                              offset = #hc.expr<"$STRIDE_0_a*i0 + $STRIDE_1_a*i1">>>) {
  return
}

// -----

// All five shaped types lose their layout under flatten regardless of
// whether the layout was identity or padded / col-major. Tensors and
// vectors (semantic and bare) collapse to a single-entry shape using
// the layout's `storage_size` after binding to the original shape;
// buffers collapse to `[?]` because their storage extent is
// host-owned.
// CHECK-LABEL: @all_five_non_identity
// CHECK-SAME: %arg0: !hc.buffer<f32, ["?"]>
// CHECK-SAME: %arg1: !hc.tensor<f16, ["K*M"]>
// CHECK-SAME: %arg2: !hc.vector<f32, ["8"]>
// CHECK-SAME: %arg3: !hc.bare_tensor<f16, ["M*(K + pad)"]>
// CHECK-SAME: %arg4: !hc.bare_vector<f16, ["16"]>
func.func @all_five_non_identity(
    %a: !hc.buffer<f32, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>,
    %b: !hc.tensor<f16, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 + d0 * i1">>>,
    %c: !hc.vector<f32, ["8"], #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"d0">, offset = #hc.expr<"d0 - 1 - i0">>>,
    %d: !hc.bare_tensor<f16, ["M", "K"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {pad = #hc.expr<"2">}, storage_size = #hc.expr<"d0 * (d1 + pad)">, offset = #hc.expr<"i0 * (d1 + pad) + i1">>>,
    %e: !hc.bare_vector<f16, ["16"], #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"d0">, offset = #hc.expr<"d0 - 1 - i0">>>) {
  return
}

// -----

// Layout-less identity row-major collapse: a 2D `!hc.tensor` with no
// explicit layout flattens to its dimension product. Same answer the
// canonicalize pass would have built had the layout been written out.
// CHECK-LABEL: @identity_no_layout_collapses
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK-SAME: %arg1: !hc.bare_tensor<f32, ["A*B*C"]>
func.func @identity_no_layout_collapses(
    %a: !hc.tensor<f16, ["M", "N"]>,
    %b: !hc.bare_tensor<f32, ["A", "B", "C"]>) {
  return
}

// -----

// `hc.as_layout` collapses to its operand once both endpoints route
// through the converter. Implicit-check above already pins the
// absence of any surviving op.
// CHECK-LABEL: @as_layout_structured_collapses
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK: return %arg0 : !hc.tensor<f16, ["M*N"]>
func.func @as_layout_structured_collapses(
    %t: !hc.tensor<f16, ["M", "N"],
                   #hc.layout<shape_syms = ["d0", "d1"],
                              index_syms = ["i0", "i1"],
                              params = {},
                              storage_size = #hc.expr<"d0 * d1">,
                              offset = #hc.expr<"i0 + d0 * i1">>>)
    -> !hc.tensor<f16, ["M", "N"],
                  #hc.layout<shape_syms = ["d0", "d1"],
                             index_syms = ["i0", "i1"],
                             params = {},
                             storage_size = #hc.expr<"d0 * d1">,
                             offset = #hc.expr<"i0 * d1 + i1">>> {
  %r = hc.as_layout %t,
       layout = (#hc.layout<shape_syms = ["d0", "d1"],
                            index_syms = ["i0", "i1"],
                            params = {},
                            storage_size = #hc.expr<"d0 * d1">,
                            offset = #hc.expr<"i0 * d1 + i1">>)
       : !hc.tensor<f16, ["M", "N"],
                    #hc.layout<shape_syms = ["d0", "d1"],
                               index_syms = ["i0", "i1"],
                               params = {},
                               storage_size = #hc.expr<"d0 * d1">,
                               offset = #hc.expr<"i0 + d0 * i1">>>
       -> !hc.tensor<f16, ["M", "N"],
                     #hc.layout<shape_syms = ["d0", "d1"],
                                index_syms = ["i0", "i1"],
                                params = {},
                                storage_size = #hc.expr<"d0 * d1">,
                                offset = #hc.expr<"i0 * d1 + i1">>>
  return %r : !hc.tensor<f16, ["M", "N"],
                         #hc.layout<shape_syms = ["d0", "d1"],
                                    index_syms = ["i0", "i1"],
                                    params = {},
                                    storage_size = #hc.expr<"d0 * d1">,
                                    offset = #hc.expr<"i0 * d1 + i1">>>
}

// -----

// `hc.as_layout` with the legacy named-enum form also drops, even
// though the result type isn't layout-bearing — the op is purely a
// relabel post-flatten. The 2D `!hc.tensor` collapses to its
// dimension product on both sides.
// CHECK-LABEL: @as_layout_named_collapses
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK: return %arg0 : !hc.tensor<f16, ["M*N"]>
func.func @as_layout_named_collapses(
    %t: !hc.tensor<f16, ["M", "N"]>) -> !hc.tensor<f16, ["M", "N"]> {
  %r = hc.as_layout %t, layout = col_major
      : !hc.tensor<f16, ["M", "N"]> -> !hc.tensor<f16, ["M", "N"]>
  return %r : !hc.tensor<f16, ["M", "N"]>
}

// -----

// 1D layout-less IR is a fixed point for non-buffers: the converter
// reports the tensor type legal-on-arrival and skips the rebuild.
// Buffers always collapse to the `[?]` sentinel even when the input
// already has a single named dim — the storage extent is host-owned
// regardless of how the surface IR spells it.
// CHECK-LABEL: @already_flat
// CHECK-SAME: %arg0: !hc.buffer<f16, ["?"]>
// CHECK-SAME: %arg1: !hc.tensor<f16, ["M"]>
// CHECK-NEXT: return
// CHECK-NOT: unrealized_conversion_cast
func.func @already_flat(
    %a: !hc.buffer<f16, ["M"]>,
    %b: !hc.tensor<f16, ["M"]>) {
  return
}

// -----

// Layout buried inside a tuple still flattens on every leg. The
// 1D tensor collapses to its single-entry shape (storage = d0 = M);
// the buffer collapses to the `[?]` sentinel.
// CHECK-LABEL: @layout_in_tuple
// CHECK-SAME: %arg0: tuple<!hc.tensor<f16, ["M"]>, !hc.buffer<f32, ["?"]>>
func.func @layout_in_tuple(
    %a: tuple<
          !hc.tensor<f16, ["M"], #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"d0">, offset = #hc.expr<"d0 - 1 - i0">>>,
          !hc.buffer<f32, ["N"], #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_x*i0">>>
        >) {
  return
}

// -----

// Control flow: an `scf.for` carrying a non-identity (col-major)
// layout-bearing iter_arg flattens on every type surface. Same
// upstream populators the canonicalize pass uses, exercised against
// the full 1D-collapse boundary.
// CHECK-LABEL: @scf_for_collapse
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK-SAME: -> !hc.tensor<f16, ["M*N"]>
// CHECK: %[[R:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A:.*]] = %arg0) -> (!hc.tensor<f16, ["M*N"]>)
// CHECK: scf.yield %[[A]] : !hc.tensor<f16, ["M*N"]>
// CHECK: return %[[R]] : !hc.tensor<f16, ["M*N"]>
func.func @scf_for_collapse(
    %t: !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 + d0 * i1">>>,
    %lb: index, %ub: index, %step: index)
    -> !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 + d0 * i1">>> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %t)
      -> (!hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 + d0 * i1">>>) {
    scf.yield %acc : !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 + d0 * i1">>>
  }
  return %r : !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 + d0 * i1">>>
}

// -----

// `func.call` between two functions whose signatures carry padded
// layouts: caller and callee collapse in lockstep, `func.call` operand
// and result types follow. The padded `row_stride` parameter survives
// in the storage expression because it's a free symbol of the layout.
// CHECK-LABEL: func.func private @padded_callee
// CHECK-SAME: (!hc.tensor<f16, ["M*row_stride"]>) -> !hc.tensor<f16, ["M*row_stride"]>
// CHECK-LABEL: @padded_caller
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*row_stride"]>
// CHECK: %[[R:.*]] = call @padded_callee(%arg0) : (!hc.tensor<f16, ["M*row_stride"]>) -> !hc.tensor<f16, ["M*row_stride"]>
// CHECK: return %[[R]] : !hc.tensor<f16, ["M*row_stride"]>
func.func private @padded_callee(
    !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>)
    -> !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>
func.func @padded_caller(
    %t: !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>)
    -> !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>> {
  %r = func.call @padded_callee(%t)
      : (!hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>)
      -> !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>
  return %r : !hc.tensor<f16, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {row_stride = #hc.expr<"4 + d1">}, storage_size = #hc.expr<"d0 * row_stride">, offset = #hc.expr<"i0 * row_stride + i1">>>
}

// -----

// Op-surface contract: `hc.generic` keeps its per-axis offset arrays
// at the original logical rank even after the operand types collapse
// to 1D. The verifier accepts the entries-per-operand stays at the
// pre-flatten rank — the `[i, j]` offset array on a `bare_tensor`
// that just became `["M*N"]` is the post-flatten shape.
// CHECK-LABEL: @generic_keeps_per_axis_offsets
// CHECK-SAME: %[[A:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK-SAME: %[[C:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK: hc.generic
// CHECK-SAME: ins (%[[A]] at [#hc.expr<"i">, #hc.expr<"j">] : !hc.bare_tensor<f32, ["M*N"]>)
// CHECK-SAME: outs (%[[C]] at [#hc.expr<"i">, #hc.expr<"j">] : !hc.bare_tensor<f32, ["M*N"]>)
func.func @generic_keeps_per_axis_offsets(
    %m: index, %n: index,
    %a: !hc.bare_tensor<f32, ["M", "N"]>,
    %c: !hc.bare_tensor<f32, ["M", "N"]>)
    -> !hc.bare_tensor<f32, ["M", "N"]> {
  %r = hc.generic
      iter (parallel i = %m : index, parallel j = %n : index)
      ins (%a at [#hc.expr<"i">, #hc.expr<"j">]
              : !hc.bare_tensor<f32, ["M", "N"]>)
      outs (%c at [#hc.expr<"i">, #hc.expr<"j">]
               : !hc.bare_tensor<f32, ["M", "N"]>)
      -> (!hc.bare_tensor<f32, ["M", "N"]>) {
  ^bb0(%av: f32, %cv: f32):
    hc.yield %av : f32
  }
  return %r : !hc.bare_tensor<f32, ["M", "N"]>
}

// -----

// Layouted load: `hc.load` with a 2D index list and a row-major
// `#hc.layout` collapses to a single `hc.idx_apply` of
// the substituted offset, then a one-index `hc.load`. ixsimpl
// canonicalizes `i0 * d1 + i1` with `d0->M, d1->N, i0->i, i1->j` to
// `j + N*i`.
// CHECK-LABEL: @load_with_layout_composes
// CHECK-SAME: %[[B:[^:]+]]: !hc.buffer<f32, ["?"]>
// CHECK: %[[OFF:.*]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"j + N*i">
// CHECK: hc.load %[[B]][%[[OFF]]], shape %{{[^ ]+}} : (!hc.buffer<f32, ["?"]>, !hc.idx<"j + N*i">, tuple<!hc.idx<"M">, !hc.idx<"N">>) -> !hc.bare_tensor<f32, ["M*N"]>
func.func @load_with_layout_composes(
    %buf: !hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
    %i: !hc.idx<"i">, %j: !hc.idx<"j">,
    %m: !hc.idx<"M">, %n: !hc.idx<"N">) {
  %shape = hc.tuple(%m, %n)
      : (!hc.idx<"M">, !hc.idx<"N">) -> tuple<!hc.idx<"M">, !hc.idx<"N">>
  %t = hc.load %buf[%i, %j], shape %shape
      : (!hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
         !hc.idx<"i">, !hc.idx<"j">,
         tuple<!hc.idx<"M">, !hc.idx<"N">>)
        -> !hc.bare_tensor<f32, ["M", "N"]>
  return
}

// -----

// Layout-less identity row-major composition: a 2D `hc.vload` on a
// layout-less `!hc.bare_tensor` falls back to the
// `i_0 * (d_1 * ... * d_{n-1}) + ... + i_{n-1}` formula. ixsimpl
// canonicalizes the resulting `0 + i*N + j` to `j + N*i`.
// CHECK-LABEL: @vload_identity_row_major
// CHECK-SAME: %[[T:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK: %[[OFF:.*]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"j + N*i">
// CHECK: hc.vload %[[T]][%[[OFF]]], shape %{{[^ ]+}} : (!hc.bare_tensor<f32, ["M*N"]>, !hc.idx<"j + N*i">, tuple<!hc.idx<"M">, !hc.idx<"N">>) -> !hc.bare_vector<f32, ["M*N"]>
func.func @vload_identity_row_major(
    %t: !hc.bare_tensor<f32, ["M", "N"]>,
    %i: !hc.idx<"i">, %j: !hc.idx<"j">,
    %m: !hc.idx<"M">, %n: !hc.idx<"N">) {
  %shape = hc.tuple(%m, %n)
      : (!hc.idx<"M">, !hc.idx<"N">) -> tuple<!hc.idx<"M">, !hc.idx<"N">>
  %v = hc.vload %t[%i, %j], shape %shape
      : (!hc.bare_tensor<f32, ["M", "N"]>, !hc.idx<"i">, !hc.idx<"j">,
         tuple<!hc.idx<"M">, !hc.idx<"N">>)
        -> !hc.bare_vector<f32, ["M", "N"]>
  return
}

// -----

// Slice operands bind to the slice's lower bound. Here axis 1 is a
// `[0:8:1]` slice — the substitution sets `i1->0` and the composed
// offset reduces to `N*row`.
// CHECK-LABEL: @load_with_slice_lower_bound
// CHECK: %[[OFF:.*]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"N*row">
// CHECK: hc.load %{{[^[]+}}[%[[OFF]]], shape %{{[^ ]+}} : (!hc.buffer<f32, ["?"]>, !hc.idx<"N*row">, tuple<!hc.idx<"N">>) -> !hc.bare_tensor<f32, ["8"]>
func.func @load_with_slice_lower_bound(
    %buf: !hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
    %row: !hc.idx<"row">,
    %n: !hc.idx<"N">,
    %step: !hc.idx<"1">) {
  %col_lo = hc.const<0 : i64> : !hc.idx<"0">
  %col_hi = hc.const<8 : i64> : !hc.idx<"8">
  %s = hc.slice_expr(lower = %col_lo upper = %col_hi step = %step)
      : (!hc.idx<"0">, !hc.idx<"8">, !hc.idx<"1">)
        -> !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"1">>
  %shape = hc.tuple(%n) : (!hc.idx<"N">) -> tuple<!hc.idx<"N">>
  %t = hc.load %buf[%row, %s], shape %shape
      : (!hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
         !hc.idx<"row">,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"8">, step = !hc.idx<"1">>,
         tuple<!hc.idx<"N">>)
        -> !hc.bare_tensor<f32, ["8"]>
  return
}

// -----

// Store with mask: dest collapses through the layout, source/mask
// collapse via the type converter, indices compose into a single base
// offset. The mask passes through to the `mask` operand on the
// rewritten op.
// CHECK-LABEL: @store_with_mask_composes
// CHECK-SAME: %[[B:[^:]+]]: !hc.buffer<f32, ["?"]>
// CHECK-SAME: %[[SRC:[^:]+]]: !hc.bare_tensor<f32, ["16"]>
// CHECK-SAME: %[[MASK:[^:]+]]: !hc.bare_tensor<!hc.pred, ["16"]>
// CHECK: %[[OFF:.*]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"j + N*i">
// CHECK: hc.store %[[B]][%[[OFF]]], %[[SRC]], mask %[[MASK]]
func.func @store_with_mask_composes(
    %buf: !hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
    %i: !hc.idx<"i">, %j: !hc.idx<"j">,
    %src: !hc.bare_tensor<f32, ["4", "4"]>,
    %mask: !hc.bare_tensor<!hc.pred, ["4", "4"]>) {
  hc.store %buf[%i, %j], %src, mask %mask
      : (!hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
         !hc.idx<"i">, !hc.idx<"j">,
         !hc.bare_tensor<f32, ["4", "4"]>,
         !hc.bare_tensor<!hc.pred, ["4", "4"]>) -> ()
  return
}

// -----

// `hc.load_mask` rides the same composition path as the data load —
// same multi-index addressing surface, same per-access offset
// rewrite. Verified separately because it gets emitted next to
// `hc.load` post-decompose and would diverge silently if missed.
// CHECK-LABEL: @load_mask_composes
// CHECK: %[[OFF:.*]] = hc.idx_apply() {symbols = []} : () -> !hc.idx<"j + N*i">
// CHECK: hc.load_mask %{{[^[]+}}[%[[OFF]]], shape %{{[^ ]+}} : (!hc.buffer<f32, ["?"]>, !hc.idx<"j + N*i">, tuple<!hc.idx<"M">, !hc.idx<"N">>) -> !hc.bare_tensor<!hc.pred, ["M*N"]>
func.func @load_mask_composes(
    %buf: !hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
    %i: !hc.idx<"i">, %j: !hc.idx<"j">,
    %m: !hc.idx<"M">, %n: !hc.idx<"N">) {
  %shape = hc.tuple(%m, %n)
      : (!hc.idx<"M">, !hc.idx<"N">) -> tuple<!hc.idx<"M">, !hc.idx<"N">>
  %mask = hc.load_mask %buf[%i, %j], shape %shape
      : (!hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
         !hc.idx<"i">, !hc.idx<"j">,
         tuple<!hc.idx<"M">, !hc.idx<"N">>)
        -> !hc.bare_tensor<!hc.pred, ["M", "N"]>
  return
}

// -----

// Indices that aren't pinned (raw `index` value here) leave the access
// op alone — the rewrite needs a symbolic name to bind to the layout's
// `index_sym`. The op stays on the multi-index surface and the
// generic retype patches the operand type so the IR remains
// well-formed at the type level.
// CHECK-LABEL: @load_unbound_index_falls_through
// CHECK-SAME: %[[B:[^:]+]]: !hc.buffer<f32, ["?"]>
// CHECK-NOT: hc.idx_apply
// CHECK: hc.load %[[B]][%{{[^,]+}}, %{{[^]]+}}], shape %{{[^ ]+}} : (!hc.buffer<f32, ["?"]>, index, index, tuple<!hc.idx<"M">, !hc.idx<"N">>) -> !hc.bare_tensor<f32, ["M*N"]>
func.func @load_unbound_index_falls_through(
    %buf: !hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
    %i: index, %j: index,
    %m: !hc.idx<"M">, %n: !hc.idx<"N">) {
  %shape = hc.tuple(%m, %n)
      : (!hc.idx<"M">, !hc.idx<"N">) -> tuple<!hc.idx<"M">, !hc.idx<"N">>
  %t = hc.load %buf[%i, %j], shape %shape
      : (!hc.buffer<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
         index, index,
         tuple<!hc.idx<"M">, !hc.idx<"N">>)
        -> !hc.bare_tensor<f32, ["M", "N"]>
  return
}
