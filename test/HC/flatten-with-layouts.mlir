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
//   * **expands every shaped value 1-to-N**: alongside the flat
//     carrier the converter emits one `!hc.idx<sym>` SSA value for
//     each free symbol the pre-flatten type implicitly carries —
//     dim names from the operand's shape, layout's free syms from
//     `offset` / `storage_size` / `params`, minus `index_syms`. Names
//     are sorted lexicographically so the trailing position pins the
//     symbol it carries. The expansion is what surfaces dynamic dim
//     and stride values as ordinary SSA, instead of leaking them
//     through ambient resolution at the launch boundary.
//   * folds `hc.as_layout` ops to their operand once both endpoints
//     are converted (the relabel becomes cosmetic).
//   * propagates the collapse through tuple types and the upstream
//     func / scf / call signature populators.
//
// Op-surface contract: per-axis offset arrays on `hc.generic` compose
// through the operand's `#hc.layout` offset expression (or identity
// row-major when the operand has no layout) into a single 1D offset
// matching the post-flatten 1D operand. Multi-index lists on
// `hc.load` / `hc.store` / `hc.vload` collapse the same way through
// the per-access `Compose*Offsets` patterns. `hc.buffer_view` composes
// rank-N subscripts into a single rank-1 slice on the flat carrier:
// pure-scalar identity views forward the source through, single-slice
// strided views land one `hc.slice_expr` with `lower` / `upper` /
// `step` scaled by the original axis's row-major stride. Multi-slice
// and rank-mismatched views (frontend rank-up patterns) fall through
// to the catch-all retyper; downstream lowering owns those cases.
//
// Post-flatten invariant from `doc/layouts.md`: *no `#hc.layout`
// survives on any shaped type*. The implicit-check below pins that
// for the entire file.
//
// RUN: hc-opt -split-input-file -hc-flatten-with-layouts %s \
// RUN:   | FileCheck %s --implicit-check-not='#hc.layout' --implicit-check-not='hc.as_layout'

// CHECK-LABEL: @strided_buffer_arg
// CHECK-SAME: %arg0: !hc.buffer<f16, ["?"]>
// CHECK-SAME: %arg1: !hc.idx<"$STRIDE_0_a">
// CHECK-SAME: %arg2: !hc.idx<"$STRIDE_1_a">
// CHECK-SAME: %arg3: !hc.idx<"K">
// CHECK-SAME: %arg4: !hc.idx<"M">
// Default fully-strided np/torch layout the frontend pins on every
// buffer argument. Buffer shape collapses to the `[?]` sentinel
// because the host owns the allocation; the layout slot drops.
// Trailing aux values surface the host-bound shape syms (`M`, `K`)
// and the per-axis stride params (`$STRIDE_0_a`, `$STRIDE_1_a`) the
// layout's `offset` references.
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

// Same signature on an `hc.kernel` instead of a `func.func` so the
// kernel-specific signature populator (`ConvertHCSymbolSignatureOp`)
// fires alongside the buffer's 1-to-N expansion. Aux idx slots get
// pinned back to their parent buffer's post-flatten arg index via
// the `hc.flatten_aux_args` attribute the populator attaches; the
// host-wrapper lowering (`hc-lower-kernels-to-gpu-launch`) reads
// it to skip allocating Python ABI slots for the auxes and to
// resolve each aux through `_get_dim` / `_get_stride` on the parent
// buffer's PyObject. `kind` distinguishes the accessor; `axis`
// picks the axis in the parent's pre-flatten shape (stride symbols
// encode the axis in their name, dim symbols by position in the
// pre-flatten shape).
// CHECK-LABEL: hc.kernel @strided_buffer_kernel
// CHECK-SAME: %arg0: !hc.group
// CHECK-SAME: %arg1: !hc.buffer<f16, ["?"]>
// CHECK-SAME: %arg2: !hc.idx<"$STRIDE_0_a">
// CHECK-SAME: %arg3: !hc.idx<"$STRIDE_1_a">
// CHECK-SAME: %arg4: !hc.idx<"K">
// CHECK-SAME: %arg5: !hc.idx<"M">
// CHECK-SAME: hc.flatten_aux_args
// CHECK-SAME: "2" = {aux_of = 1 : i64, axis = 0 : i64, kind = "stride"}
// CHECK-SAME: "3" = {aux_of = 1 : i64, axis = 1 : i64, kind = "stride"}
// CHECK-SAME: "4" = {aux_of = 1 : i64, axis = 1 : i64, kind = "dim"}
// CHECK-SAME: "5" = {aux_of = 1 : i64, axis = 0 : i64, kind = "dim"}
hc.kernel @strided_buffer_kernel(
    %group: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*K)"]>, group_shape = #hc.shape<["32", "1"]>>,
    %a: !hc.buffer<f16, ["M", "K"],
                   #hc.layout<shape_syms = ["d0", "d1"],
                              index_syms = ["i0", "i1"],
                              params = {},
                              storage_size = #hc.expr<"0">,
                              offset = #hc.expr<"$STRIDE_0_a*i0 + $STRIDE_1_a*i1">>>)
    attributes {
      work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*K)"]>,
      group_shape = #hc.shape<["32", "1"]>,
      bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "M", "K", "$STRIDE_0_a", "$STRIDE_1_a"]
    } {
  hc.return
}

// -----

// All five shaped types lose their layout under flatten regardless of
// whether the layout was identity or padded / col-major. Tensors and
// vectors (semantic and bare) collapse to a single-entry shape using
// the layout's `storage_size` after binding to the original shape;
// buffers collapse to `[?]` because their storage extent is
// host-owned. Each shaped operand expands 1-to-N: the buffer / tensor
// / bare_tensor operands trail their dim and `params` syms (sorted),
// while the constant-shape vectors (`vector<f32, ["8"]>`,
// `bare_vector<f16, ["16"]>`) carry no aux because their storage
// has no free symbol.
// CHECK-LABEL: @all_five_non_identity
// CHECK-SAME: %arg0: !hc.buffer<f32, ["?"]>
// CHECK-SAME: %arg1: !hc.idx<"K">, %arg2: !hc.idx<"M">, %arg3: !hc.idx<"row_stride">
// CHECK-SAME: %arg4: !hc.tensor<f16, ["K*M"]>
// CHECK-SAME: %arg5: !hc.idx<"K">, %arg6: !hc.idx<"M">
// CHECK-SAME: %arg7: !hc.vector<f32, ["8"]>
// CHECK-SAME: %arg8: !hc.bare_tensor<f16, ["M*(K + pad)"]>
// CHECK-SAME: %arg9: !hc.idx<"K">, %arg10: !hc.idx<"M">, %arg11: !hc.idx<"pad">
// CHECK-SAME: %arg12: !hc.bare_vector<f16, ["16"]>
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
// Trailing aux for each shaped arg surface the dim names that the
// pre-flatten shape pinned positionally.
// CHECK-LABEL: @identity_no_layout_collapses
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK-SAME: %arg1: !hc.idx<"M">, %arg2: !hc.idx<"N">
// CHECK-SAME: %arg3: !hc.bare_tensor<f32, ["A*B*C"]>
// CHECK-SAME: %arg4: !hc.idx<"A">, %arg5: !hc.idx<"B">, %arg6: !hc.idx<"C">
func.func @identity_no_layout_collapses(
    %a: !hc.tensor<f16, ["M", "N"]>,
    %b: !hc.bare_tensor<f32, ["A", "B", "C"]>) {
  return
}

// -----

// `hc.as_layout` collapses to its operand once both endpoints route
// through the converter. Implicit-check above already pins the
// absence of any surviving op. The trailing dim aux propagates from
// the operand's expansion to the result's expansion in lockstep.
// CHECK-LABEL: @as_layout_structured_collapses
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK-SAME: %arg1: !hc.idx<"M">, %arg2: !hc.idx<"N">
// CHECK: return %arg0, %arg1, %arg2 : !hc.tensor<f16, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">
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
// dimension product on both sides; the dim aux propagates through.
// CHECK-LABEL: @as_layout_named_collapses
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK-SAME: %arg1: !hc.idx<"M">, %arg2: !hc.idx<"N">
// CHECK: return %arg0, %arg1, %arg2 : !hc.tensor<f16, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">
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
// regardless of how the surface IR spells it. The buffer expansion
// surfaces its dim aux because its source shape was named (`M`); the
// already-flat tensor is 1-to-1 (no aux) because expanding it would
// loop the converter forever — the storage-size expression carries
// the dim names instead.
// CHECK-LABEL: @already_flat
// CHECK-SAME: %arg0: !hc.buffer<f16, ["?"]>
// CHECK-SAME: %arg1: !hc.idx<"M">
// CHECK-SAME: %arg2: !hc.tensor<f16, ["M"]>
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
// the buffer collapses to the `[?]` sentinel. Each shaped element of
// the tuple expands 1-to-N inline — the tuple's element list grows
// to include each shaped element's flat carrier followed by its dim
// / stride aux, in the same lex-sorted order as a top-level arg.
// CHECK-LABEL: @layout_in_tuple
// CHECK-SAME: %arg0: tuple<!hc.tensor<f16, ["M"]>, !hc.idx<"M">, !hc.buffer<f32, ["?"]>, !hc.idx<"$STRIDE_0_x">, !hc.idx<"N">>
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
// the full 1D-collapse boundary. The dim aux rides through every
// `iter_args` slot in lockstep — `scf::populateSCFStructuralTypeConversionsAndLegality`
// handles the 1-to-N expansion of carried values.
// CHECK-LABEL: @scf_for_collapse
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*N"]>
// CHECK-SAME: %arg1: !hc.idx<"M">, %arg2: !hc.idx<"N">
// CHECK-SAME: -> (!hc.tensor<f16, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">)
// CHECK: %[[R:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A:.*]] = %arg0, %[[B:.*]] = %arg1, %[[C:.*]] = %arg2)
// CHECK-SAME: -> (!hc.tensor<f16, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">)
// CHECK: scf.yield %[[A]], %[[B]], %[[C]] : !hc.tensor<f16, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">
// CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2 : !hc.tensor<f16, ["M*N"]>, !hc.idx<"M">, !hc.idx<"N">
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
// in the storage expression because it's a free symbol of the layout,
// and the 1-to-N expansion surfaces it as an explicit aux operand
// alongside the dim names — the call boundary forwards every value.
// CHECK-LABEL: func.func private @padded_callee
// CHECK-SAME: (!hc.tensor<f16, ["M*row_stride"]>, !hc.idx<"M">, !hc.idx<"N">, !hc.idx<"row_stride">)
// CHECK-SAME: -> (!hc.tensor<f16, ["M*row_stride"]>, !hc.idx<"M">, !hc.idx<"N">, !hc.idx<"row_stride">)
// CHECK-LABEL: @padded_caller
// CHECK-SAME: %arg0: !hc.tensor<f16, ["M*row_stride"]>
// CHECK-SAME: %arg1: !hc.idx<"M">, %arg2: !hc.idx<"N">, %arg3: !hc.idx<"row_stride">
// CHECK: %[[R:.*]]:4 = call @padded_callee(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: : (!hc.tensor<f16, ["M*row_stride"]>, !hc.idx<"M">, !hc.idx<"N">, !hc.idx<"row_stride">)
// CHECK-SAME: -> (!hc.tensor<f16, ["M*row_stride"]>, !hc.idx<"M">, !hc.idx<"N">, !hc.idx<"row_stride">)
// CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2, %[[R]]#3
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

// Op-surface contract: `hc.generic` per-operand per-axis offset arrays
// compose through the operand's layout (or identity row-major for
// layout-less types) into a single 1D offset on the post-flatten 1D
// operand. `[i, j]` over `bare_tensor<f32, ["M","N"]>` (no layout) goes
// through identity row-major — `i*N + j`, ixsimpl-canonicalized to
// `j + N*i`.
// CHECK-LABEL: @generic_composes_offsets_default_row_major
// CHECK-SAME: %[[A:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK-SAME: !hc.idx<"M">, %{{[^:]+}}: !hc.idx<"N">,
// CHECK-SAME: %[[C:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK: hc.generic
// CHECK-SAME: ins (%[[A]] at [#hc.expr<"j + N*i">] : !hc.bare_tensor<f32, ["M*N"]>)
// CHECK-SAME: outs (%[[C]] at [#hc.expr<"j + N*i">] : !hc.bare_tensor<f32, ["M*N"]>)
func.func @generic_composes_offsets_default_row_major(
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

// With an explicit layout the composer substitutes both `index_syms`
// (positional with the per-axis exprs) and `shape_syms` (positional
// with the operand dims) into `layout.offset` through ixsimpl. The
// rank-2 row-major `i0 * d1 + i1` over `[M, N]` indexed `[i, j]`
// canonicalizes to `j + N*i` — same destination as the layout-less
// case, but reached via the substitution path rather than the
// identity-row-major fallback.
// CHECK-LABEL: @generic_composes_offsets_layout
// CHECK-SAME: %[[B:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK: hc.generic
// CHECK-SAME: ins (%[[B]] at [#hc.expr<"j + N*i">] : !hc.bare_tensor<f32, ["M*N"]>)
func.func @generic_composes_offsets_layout(
    %m: index, %n: index,
    %a: !hc.bare_tensor<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>,
    %c: !hc.bare_tensor<f32, ["M", "N"]>)
    -> !hc.bare_tensor<f32, ["M", "N"]> {
  %r = hc.generic
      iter (parallel i = %m : index, parallel j = %n : index)
      ins (%a at [#hc.expr<"i">, #hc.expr<"j">]
              : !hc.bare_tensor<f32, ["M", "N"], #hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0 * d1">, offset = #hc.expr<"i0 * d1 + i1">>>)
      outs (%c at [#hc.expr<"i">, #hc.expr<"j">]
               : !hc.bare_tensor<f32, ["M", "N"]>)
      -> (!hc.bare_tensor<f32, ["M", "N"]>) {
  ^bb0(%av: f32, %cv: f32):
    hc.yield %av : f32
  }
  return %r : !hc.bare_tensor<f32, ["M", "N"]>
}

// -----

// Ptr-typed operands ride on a single 1D address by op contract — the
// per-axis array is already length 1 pre-flatten. The composer leaves
// it untouched while the operand passes through the converter.
// CHECK-LABEL: @generic_ptr_out_passthrough
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"j + N*i">] : !hc.bare_tensor<f32, ["M*N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.ptr<global, f32>)
func.func @generic_ptr_out_passthrough(
    %m: index, %n: index,
    %a: !hc.bare_tensor<f32, ["M", "N"]>,
    %dst: !hc.ptr<global, f32>) {
  hc.generic
      iter (parallel i = %m : index, parallel j = %n : index)
      ins (%a at [#hc.expr<"i">, #hc.expr<"j">]
              : !hc.bare_tensor<f32, ["M", "N"]>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%av: f32, %dv: f32):
    hc.yield %av : f32
  }
  return
}

// -----

// Layouted load: `hc.load` with a 2D index list and a row-major
// `#hc.layout` collapses to a single `hc.idx_apply` of
// the substituted offset, then a one-index `hc.load`. ixsimpl
// canonicalizes `i0 * d1 + i1` with `d0->M, d1->N, i0->i, i1->j` to
// `j + N*i`. The `idx_apply` lists every free symbol explicitly: the
// dim aux from the buffer's expansion (`%argN as "N"`) and the
// idx-typed access operands (`%argI as "i"`, `%argJ as "j"`).
// CHECK-LABEL: @load_with_layout_composes
// CHECK-SAME: %[[B:[^:]+]]: !hc.buffer<f32, ["?"]>
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[I:[^:]+]]: !hc.idx<"i">, %[[J:[^:]+]]: !hc.idx<"j">
// CHECK: %[[OFF:.*]] = hc.idx_apply (%[[BN]] as "N", %[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: : (!hc.idx<"N">, !hc.idx<"i">, !hc.idx<"j">) -> !hc.idx<"j + N*i">
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
// canonicalizes the resulting `0 + i*N + j` to `j + N*i`. Same
// 1-to-N expansion shape as `@load_with_layout_composes`: the
// source tensor's dim aux supplies the `N` binding for the offset.
// CHECK-LABEL: @vload_identity_row_major
// CHECK-SAME: %[[T:[^:]+]]: !hc.bare_tensor<f32, ["M*N"]>
// CHECK-SAME: %[[TM:[^:]+]]: !hc.idx<"M">, %[[TN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[I:[^:]+]]: !hc.idx<"i">, %[[J:[^:]+]]: !hc.idx<"j">
// CHECK: %[[OFF:.*]] = hc.idx_apply (%[[TN]] as "N", %[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: : (!hc.idx<"N">, !hc.idx<"i">, !hc.idx<"j">) -> !hc.idx<"j + N*i">
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
// offset reduces to `N*row`. Both free symbols (`N` from the
// buffer's expansion, `row` from the idx-typed access operand) bind
// explicitly in the apply.
// CHECK-LABEL: @load_with_slice_lower_bound
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[ROW:[^:]+]]: !hc.idx<"row">
// CHECK: %[[OFF:.*]] = hc.idx_apply (%[[BN]] as "N", %[[ROW]] as "row")
// CHECK-SAME: : (!hc.idx<"N">, !hc.idx<"row">) -> !hc.idx<"N*row">
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
// rewritten op. Constant-shape source / mask have no aux because
// their dim list is constant.
// CHECK-LABEL: @store_with_mask_composes
// CHECK-SAME: %[[B:[^:]+]]: !hc.buffer<f32, ["?"]>
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[I:[^:]+]]: !hc.idx<"i">, %[[J:[^:]+]]: !hc.idx<"j">
// CHECK-SAME: %[[SRC:[^:]+]]: !hc.bare_tensor<f32, ["16"]>
// CHECK-SAME: %[[MASK:[^:]+]]: !hc.bare_tensor<!hc.pred, ["16"]>
// CHECK: %[[OFF:.*]] = hc.idx_apply (%[[BN]] as "N", %[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: : (!hc.idx<"N">, !hc.idx<"i">, !hc.idx<"j">) -> !hc.idx<"j + N*i">
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
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[I:[^:]+]]: !hc.idx<"i">, %[[J:[^:]+]]: !hc.idx<"j">
// CHECK: %[[OFF:.*]] = hc.idx_apply (%[[BN]] as "N", %[[I]] as "i", %[[J]] as "j")
// CHECK-SAME: : (!hc.idx<"N">, !hc.idx<"i">, !hc.idx<"j">) -> !hc.idx<"j + N*i">
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
// well-formed at the type level. The buffer's dim aux still surfaces
// at the function boundary even though the access doesn't consume
// them.
// CHECK-LABEL: @load_unbound_index_falls_through
// CHECK-SAME: %[[B:[^:]+]]: !hc.buffer<f32, ["?"]>
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
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

// -----

// `hc.buffer_view` identity: rank-2 source where one axis is a unit
// dim, indexed by `slice + idx<"0">`. Post-flatten the source carrier
// collapses to ["8"] and the view covers the same 8 elements, so the
// rewrite forwards the flat source through unchanged. The original
// op drops out of the IR.
// CHECK-LABEL: @buffer_view_identity_forwards_source
// CHECK-SAME: %[[SRC:[^:]+]]: !hc.bare_vector<f32, ["8"]>
// CHECK-NOT: hc.buffer_view
// CHECK: return %[[SRC]]
func.func @buffer_view_identity_forwards_source(
    %src: !hc.bare_vector<f32, ["8", "1"]>)
    -> !hc.bare_vector<f32, ["8"]> {
  %z = hc.const<0 : i64> : !hc.idx<"0">
  %s = hc.slice_expr() : () -> !hc.slice
  %v = hc.buffer_view %src[%s, %z]
      : (!hc.bare_vector<f32, ["8", "1"]>, !hc.slice, !hc.idx<"0">)
        -> !hc.bare_vector<f32, ["8"]>
  return %v : !hc.bare_vector<f32, ["8"]>
}

// -----

// `hc.buffer_view` strided slice: rank-2 source with `(scalar_row,
// full_col_slice)` over `bare_tensor<f16, ["M", "N"]>`. Post-flatten
// the source carrier collapses to a single dim sized `M*N`; the row
// scalar contributes `row * N` to the flat base offset (`N` is the
// row-major stride at axis 0), and the column slice maps to a step-1
// slice of length `N` starting there. The rebuilt view sits on the
// flat carrier with one slice subscript whose lower / upper / step
// were composed through ixsimpl.
// CHECK-LABEL: @buffer_view_row_then_full_col_composes
// CHECK-SAME: %[[SRC:[^:]+]]: !hc.bare_tensor<f16, ["M*N"]>
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[ROW:[^:]+]]: !hc.idx<"r">
// CHECK: %[[LO:.*]] = hc.idx_apply (%[[BN]] as "N", %[[ROW]] as "r")
// CHECK-SAME: -> !hc.idx<"N*r">
// CHECK: %[[HI:.*]] = hc.idx_apply (%[[BN]] as "N", %[[ROW]] as "r")
// CHECK-SAME: -> !hc.idx<"N + N*r">
// CHECK: %[[STEP:.*]] = hc.idx_apply () : () -> !hc.idx<"1">
// CHECK: %[[SL:.*]] = hc.slice_expr(lower = %[[LO]] upper = %[[HI]] step = %[[STEP]])
// CHECK: hc.buffer_view %[[SRC]][%[[SL]]]
// CHECK-SAME: : (!hc.bare_tensor<f16, ["M*N"]>,
// CHECK-SAME: -> !hc.bare_tensor<f16, ["N"]>
func.func @buffer_view_row_then_full_col_composes(
    %src: !hc.bare_tensor<f16, ["M", "N"]>,
    %row: !hc.idx<"r">) -> !hc.bare_tensor<f16, ["N"]> {
  %s = hc.slice_expr() : () -> !hc.slice
  %v = hc.buffer_view %src[%row, %s]
      : (!hc.bare_tensor<f16, ["M", "N"]>, !hc.idx<"r">, !hc.slice)
        -> !hc.bare_tensor<f16, ["N"]>
  return %v : !hc.bare_tensor<f16, ["N"]>
}

// -----

// `hc.buffer_view` strided slice column-of-2D: full row slice + scalar
// column. The column scalar contributes `col * 1` (innermost axis row-
// major stride is 1) to the flat base offset; the row slice becomes a
// step-`N` slice over the flat carrier — each successive row sits `N`
// elements apart. The rebuilt rank-1 view is the column-as-strided-1D
// pattern the WMMA recipe relies on.
// CHECK-LABEL: @buffer_view_full_row_then_col_composes
// CHECK-SAME: %[[SRC:[^:]+]]: !hc.bare_tensor<f16, ["M*N"]>
// CHECK-SAME: %[[BM:[^:]+]]: !hc.idx<"M">, %[[BN:[^:]+]]: !hc.idx<"N">
// CHECK-SAME: %[[COL:[^:]+]]: !hc.idx<"c">
// CHECK: %[[LO:.*]] = hc.idx_apply (%[[COL]] as "c") : (!hc.idx<"c">) -> !hc.idx<"c">
// CHECK: %[[HI:.*]] = hc.idx_apply (%[[BM]] as "M", %[[BN]] as "N", %[[COL]] as "c")
// CHECK-SAME: -> !hc.idx<"c + M*N">
// CHECK: %[[STEP:.*]] = hc.idx_apply (%[[BN]] as "N") : (!hc.idx<"N">) -> !hc.idx<"N">
// CHECK: %[[SL:.*]] = hc.slice_expr(lower = %[[LO]] upper = %[[HI]] step = %[[STEP]])
// CHECK: hc.buffer_view %[[SRC]][%[[SL]]]
// CHECK-SAME: : (!hc.bare_tensor<f16, ["M*N"]>,
// CHECK-SAME: -> !hc.bare_tensor<f16, ["M"]>
func.func @buffer_view_full_row_then_col_composes(
    %src: !hc.bare_tensor<f16, ["M", "N"]>,
    %col: !hc.idx<"c">) -> !hc.bare_tensor<f16, ["M"]> {
  %s = hc.slice_expr() : () -> !hc.slice
  %v = hc.buffer_view %src[%s, %col]
      : (!hc.bare_tensor<f16, ["M", "N"]>, !hc.slice, !hc.idx<"c">)
        -> !hc.bare_tensor<f16, ["M"]>
  return %v : !hc.bare_tensor<f16, ["M"]>
}

// -----

// Rank-mismatched buffer_view (frontend rank-up: rank-1 source carrier
// indexed by three subscripts) falls through to the catch-all retyper.
// The view stays in the IR with its original subscripts on the now-
// flat carrier; downstream lowering owns the rewrite.
// CHECK-LABEL: @buffer_view_rank_up_falls_through
// CHECK-SAME: %[[SRC:[^:]+]]: !hc.bare_vector<f32, ["8"]>
// CHECK: hc.buffer_view %[[SRC]][%{{[^,]+}}, %{{[^,]+}}, %{{[^]]+}}]
// CHECK-SAME: : (!hc.bare_vector<f32, ["8"]>, !hc.slice, !hc.idx<"lane">, !hc.idx<"0">) -> !hc.bare_vector<f32, ["8"]>
func.func @buffer_view_rank_up_falls_through(
    %src: !hc.bare_vector<f32, ["8"]>,
    %lane: !hc.idx<"lane">) -> !hc.bare_vector<f32, ["8"]> {
  %z = hc.const<0 : i64> : !hc.idx<"0">
  %s = hc.slice_expr() : () -> !hc.slice
  %v = hc.buffer_view %src[%s, %lane, %z]
      : (!hc.bare_vector<f32, ["8"]>, !hc.slice, !hc.idx<"lane">, !hc.idx<"0">)
        -> !hc.bare_vector<f32, ["8"]>
  return %v : !hc.bare_vector<f32, ["8"]>
}
