// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: func.func @buffer_queries
// CHECK: hc.buffer_dim %{{.*}}, axis = 1 : !hc.undef -> !hc.undef
// All eight `hc.slice_expr` operand combinations round-trip independently
// so the `AttrSizedOperandSegments` wiring between operand presence and
// printed keywords is exercised exhaustively. The printer keeps a literal
// space before each optional keyword, so `hc.slice_expr(upper = ...)`
// round-trips as `hc.slice_expr( upper = ...)` — allow the leading space
// in each CHECK.
// CHECK: hc.slice_expr() : () -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}}) : (!hc.undef) -> !hc.undef
// CHECK: hc.slice_expr({{ *}}upper = %{{.*}}) : (!hc.undef) -> !hc.undef
// CHECK: hc.slice_expr({{ *}}step = %{{.*}}) : (!hc.undef) -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}} upper = %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}} step = %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.slice_expr({{ *}}upper = %{{.*}} step = %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}} upper = %{{.*}} step = %{{.*}}) : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}, %{{.*}}] : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}] : (!hc.tensor<f32, ["M", "N"]>, !hc.undef) -> !hc.undef
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}] : (!hc.vector<f32, ["8"]>, !hc.undef) -> !hc.undef
func.func @buffer_queries(%buf: !hc.undef, %lo: !hc.undef, %hi: !hc.undef,
                          %st: !hc.undef,
                          %tensor: !hc.tensor<f32, ["M", "N"]>,
                          %vector: !hc.vector<f32, ["8"]>) {
  %d        = hc.buffer_dim %buf, axis = 1 : !hc.undef -> !hc.undef
  %sl_none  = hc.slice_expr() : () -> !hc.undef
  %sl_lo    = hc.slice_expr(lower = %lo) : (!hc.undef) -> !hc.undef
  %sl_hi    = hc.slice_expr(upper = %hi) : (!hc.undef) -> !hc.undef
  %sl_st    = hc.slice_expr(step = %st)  : (!hc.undef) -> !hc.undef
  %sl_lo_hi = hc.slice_expr(lower = %lo upper = %hi)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %sl_lo_st = hc.slice_expr(lower = %lo step = %st)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %sl_hi_st = hc.slice_expr(upper = %hi step = %st)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %sl_all   = hc.slice_expr(lower = %lo upper = %hi step = %st)
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  %bv  = hc.buffer_view %buf[%sl_lo_hi, %sl_lo]
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  %tv  = hc.buffer_view %tensor[%lo]
      : (!hc.tensor<f32, ["M", "N"]>, !hc.undef) -> !hc.undef
  %vv  = hc.buffer_view %vector[%lo]
      : (!hc.vector<f32, ["8"]>, !hc.undef) -> !hc.undef
  return
}

// Symbol-user verification wants a matching `hc.func` for every `hc.call` and
// a matching `hc.intrinsic` for every `hc.call_intrinsic`; declare the
// callees up front so the round-trip is self-contained.
hc.intrinsic @wmma scope = #hc.scope<"SubGroup"> {}
hc.func @helper {
  hc.return
}

// CHECK-LABEL: func.func @data_movement
// CHECK: %[[SHAPE_MK:.*]] = hc.tuple
// CHECK: %[[SHAPE_K:.*]] = hc.tuple
// CHECK: hc.load %{{.*}}[%{{.*}}, %{{.*}}], shape %[[SHAPE_MK]]
// CHECK-SAME: (!hc.undef, !hc.undef, !hc.undef, tuple<!hc.undef, !hc.undef>) -> !hc.undef
// CHECK: hc.vload %{{.*}}[%{{.*}}], shape %[[SHAPE_K]]
// CHECK-SAME: (!hc.undef, !hc.undef, tuple<!hc.undef>) -> !hc.undef
// CHECK: hc.store %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}
// CHECK: hc.vec %{{.*}} : !hc.undef -> !hc.undef
// CHECK: %[[INACTIVE:.*]] = hc.const<0.000000e+00 : f32> : f32
// CHECK: hc.with_inactive %{{.*}}, %[[INACTIVE]] : (!hc.undef, f32) -> !hc.undef
// CHECK: hc.as_layout %{{.*}}, layout = (#hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0*d1">, offset = #hc.expr<"i1 + d1*i0">>) : !hc.undef -> !hc.undef
// CHECK: hc.as_layout %{{.*}}, layout = (#hc.layout<shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"d0*d1">, offset = #hc.expr<"i0 + d0*i1">>) : !hc.undef -> !hc.undef
func.func @data_movement(%buf: !hc.undef, %i: !hc.undef, %j: !hc.undef,
                         %v: !hc.undef) {
  %m = hc.const<"M"> : !hc.undef
  %k = hc.const<"K"> : !hc.undef
  %shape_mk = hc.tuple(%m, %k)
      : (!hc.undef, !hc.undef) -> tuple<!hc.undef, !hc.undef>
  %shape_k = hc.tuple(%k) : (!hc.undef) -> tuple<!hc.undef>
  %t = hc.load %buf[%i, %j], shape %shape_mk
      : (!hc.undef, !hc.undef, !hc.undef, tuple<!hc.undef, !hc.undef>) -> !hc.undef
  %vec = hc.vload %buf[%i], shape %shape_k
      : (!hc.undef, !hc.undef, tuple<!hc.undef>) -> !hc.undef
  hc.store %buf[%i, %j], %v
      : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> ()
  %vec2 = hc.vec %t : !hc.undef -> !hc.undef
  %inactive = hc.const<0.0 : f32> : f32
  %masked = hc.with_inactive %v, %inactive
      : (!hc.undef, f32) -> !hc.undef
  // Two distinct 2D layouts: the first has `i0` step the inner axis,
  // the second has `i1` step the inner axis. Same shape, different
  // addressing — that's what `hc.as_layout` is for.
  %reshaped = hc.as_layout %v, layout = (#hc.layout<
    shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {},
    storage_size = #hc.expr<"d0*d1">, offset = #hc.expr<"i0*d1 + i1">
  >) : !hc.undef -> !hc.undef
  %transposed = hc.as_layout %v, layout = (#hc.layout<
    shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {},
    storage_size = #hc.expr<"d0*d1">, offset = #hc.expr<"i0 + i1*d0">
  >) : !hc.undef -> !hc.undef
  return
}

// CHECK-LABEL: func.func @as_layout_structured
// Structured spellings are wrapped in `(...)` to disambiguate from the
// trailing `: type` on the op.
// CHECK: hc.as_layout %{{.*}}, layout = (#hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">>) : !hc.undef -> !hc.undef
func.func @as_layout_structured(%v: !hc.undef) -> !hc.undef {
  %r = hc.as_layout %v, layout = (#hc.layout<
    shape_syms = ["d0"], index_syms = ["i0"], params = {},
    storage_size = #hc.expr<"0">, offset = #hc.expr<"i0">
  >) : !hc.undef -> !hc.undef
  return %r : !hc.undef
}

// Shape-changing `hc.as_layout`: a 1-D bare carrier reinterpreted as
// a 4-D layout-bearing view. The verifier accepts the pair because
// the operand's storage_size (`2*L*M*N` from the explicit dim product
// on a layout-less bare tensor) equals the result layout's
// storage_size `b*m*n*l` after binding `b -> 2, m -> M, n -> N, l ->
// L`. ixsimpl hash-cons gives that comparison structural meaning, so
// the two cannot disagree on operand ordering or factoring.
//
// `lds_4d[buf_idx]` and `lds_4d[buf_idx, im_slice, in_slice, il_slice]`
// in the multibuffered LDS pattern feed off this reinterpretation —
// see `doc/layouts.md` "Free symbols in layout offsets".
// CHECK-LABEL: func.func @as_layout_shape_change_storage_match
// CHECK: hc.as_layout %{{[^,]+}}, layout = (#hc.layout<shape_syms = ["b", "m", "n", "l"]
// CHECK-SAME: : !hc.bare_tensor<f32, ["2*L*M*N"]>
// CHECK-SAME: -> !hc.tensor<f32, ["2", "M", "N", "L"]
func.func @as_layout_shape_change_storage_match(
    %lds_1d: !hc.bare_tensor<f32, ["2*M*N*L"]>)
    -> !hc.tensor<f32, ["2", "M", "N", "L"],
                  #hc.layout<shape_syms = ["b", "m", "n", "l"],
                             index_syms = ["ib", "im", "in", "il"],
                             params = {},
                             storage_size = #hc.expr<"b*m*n*l">,
                             offset = #hc.expr<"((ib*m + im)*n + in)*l + il">>> {
  %v = hc.as_layout %lds_1d,
       layout = (#hc.layout<shape_syms = ["b", "m", "n", "l"],
                            index_syms = ["ib", "im", "in", "il"],
                            params = {},
                            storage_size = #hc.expr<"b*m*n*l">,
                            offset = #hc.expr<"((ib*m + im)*n + in)*l + il">>)
       : !hc.bare_tensor<f32, ["2*M*N*L"]>
         -> !hc.tensor<f32, ["2", "M", "N", "L"],
                       #hc.layout<shape_syms = ["b", "m", "n", "l"],
                                  index_syms = ["ib", "im", "in", "il"],
                                  params = {},
                                  storage_size = #hc.expr<"b*m*n*l">,
                                  offset = #hc.expr<"((ib*m + im)*n + in)*l + il">>>
  return %v : !hc.tensor<f32, ["2", "M", "N", "L"],
                          #hc.layout<shape_syms = ["b", "m", "n", "l"],
                                     index_syms = ["ib", "im", "in", "il"],
                                     params = {},
                                     storage_size = #hc.expr<"b*m*n*l">,
                                     offset = #hc.expr<"((ib*m + im)*n + in)*l + il">>>
}

// `hc.as_layout` on a `!hc.buffer` operand accepts a non-injective
// layout whose `storage_size` does not equal the product of the
// buffer's named dims. A buffer's named dims are a proxy for its
// addressable extent — the actual allocation lives behind the
// pointer and isn't known at IR time. The bounds-clip happens at
// runtime against the real flat span (gather/scatter path), so a
// declared `storage_size = K` against `product(M, K, LANE)` is a
// legitimate broadcast reinterpretation, not a verifier-time error.
// The value-semantic (tensor / vector / bare) path stays strict —
// see `verify-hc.mlir` for the negative pin.
// CHECK-LABEL: func.func @as_layout_buffer_noninjective
// CHECK: hc.as_layout %{{.*}} : !hc.buffer<f32, ["M", "K", "LANE"]>
// CHECK-SAME: -> !hc.buffer<f32, ["M", "K", "LANE"], <{{.*}}storage_size = #hc.expr<"k">{{.*}}>>
func.func @as_layout_buffer_noninjective(
    %buf: !hc.buffer<f32, ["M", "K", "LANE"]>)
    -> !hc.buffer<f32, ["M", "K", "LANE"],
                  #hc.layout<shape_syms = ["m", "k", "l"],
                             index_syms = ["i", "j", "z"],
                             params = {},
                             storage_size = #hc.expr<"k">,
                             offset = #hc.expr<"j">>> {
  %v = hc.as_layout %buf,
       layout = (#hc.layout<shape_syms = ["m", "k", "l"],
                            index_syms = ["i", "j", "z"],
                            params = {},
                            storage_size = #hc.expr<"k">,
                            offset = #hc.expr<"j">>)
       : !hc.buffer<f32, ["M", "K", "LANE"]>
         -> !hc.buffer<f32, ["M", "K", "LANE"],
                       #hc.layout<shape_syms = ["m", "k", "l"],
                                  index_syms = ["i", "j", "z"],
                                  params = {},
                                  storage_size = #hc.expr<"k">,
                                  offset = #hc.expr<"j">>>
  return %v : !hc.buffer<f32, ["M", "K", "LANE"],
                          #hc.layout<shape_syms = ["m", "k", "l"],
                                     index_syms = ["i", "j", "z"],
                                     params = {},
                                     storage_size = #hc.expr<"k">,
                                     offset = #hc.expr<"j">>>
}

// `hc.as_layout` on a `!hc.buffer` operand can also change rank /
// shape via the optional `shape=` SSA operand: a 2-D buffer whose
// `(WMMA_M, WMMA_N)` extent is the source-side layout's allocation
// proxy can be reinterpreted as a 2-D `(WAVE_LANES,
// WMMA_ACC_FRAGMENT)` layout-bearing view whose flat storage_size
// still resolves to `WMMA_M*WMMA_N`. The declared shape rides on the
// shape= operand because the layout's `shape_syms` don't bind to the
// operand's named dims — pointer storage and reinterpreted extent
// are independent under buffer roots, and the verifier short-circuits
// the storage_size structural check for the same reason.
// CHECK-LABEL: func.func @as_layout_buffer_shape_change
// CHECK: hc.as_layout %{{.*}}, layout = (#hc.layout<{{.*}}offset = #hc.expr<"32*fi + lane">>), shape = %{{.*}} : tuple<!hc.idx<"WAVE_LANES">, !hc.idx<"WMMA_ACC_FRAGMENT">>
// CHECK-SAME: : !hc.buffer<f32, ["WMMA_M", "WMMA_N"]>
// CHECK-SAME: -> !hc.buffer<f32, ["WAVE_LANES", "WMMA_ACC_FRAGMENT"]
func.func @as_layout_buffer_shape_change(
    %buf: !hc.buffer<f32, ["WMMA_M", "WMMA_N"]>,
    %shape: tuple<!hc.idx<"WAVE_LANES">, !hc.idx<"WMMA_ACC_FRAGMENT">>)
    -> !hc.buffer<f32, ["WAVE_LANES", "WMMA_ACC_FRAGMENT"],
                  #hc.layout<shape_syms = ["wl", "waf"],
                             index_syms = ["lane", "fi"],
                             params = {},
                             storage_size = #hc.expr<"WMMA_M*WMMA_N">,
                             offset = #hc.expr<"lane + fi * 32">>> {
  %v = hc.as_layout %buf,
       layout = (#hc.layout<shape_syms = ["wl", "waf"],
                            index_syms = ["lane", "fi"],
                            params = {},
                            storage_size = #hc.expr<"WMMA_M*WMMA_N">,
                            offset = #hc.expr<"lane + fi * 32">>),
       shape = %shape : tuple<!hc.idx<"WAVE_LANES">, !hc.idx<"WMMA_ACC_FRAGMENT">>
       : !hc.buffer<f32, ["WMMA_M", "WMMA_N"]>
         -> !hc.buffer<f32, ["WAVE_LANES", "WMMA_ACC_FRAGMENT"],
                       #hc.layout<shape_syms = ["wl", "waf"],
                                  index_syms = ["lane", "fi"],
                                  params = {},
                                  storage_size = #hc.expr<"WMMA_M*WMMA_N">,
                                  offset = #hc.expr<"lane + fi * 32">>>
  return %v : !hc.buffer<f32, ["WAVE_LANES", "WMMA_ACC_FRAGMENT"],
                          #hc.layout<shape_syms = ["wl", "waf"],
                                     index_syms = ["lane", "fi"],
                                     params = {},
                                     storage_size = #hc.expr<"WMMA_M*WMMA_N">,
                                     offset = #hc.expr<"lane + fi * 32">>>
}

// CHECK-LABEL: func.func @allocators
// CHECK: hc.vzeros shape %{{.*}} : (tuple<!hc.undef, !hc.undef>) -> !hc.vector<f32, ["16", "16"]>
// CHECK: hc.vones shape %{{.*}} : (tuple<!hc.undef>) -> !hc.vector<i1, ["16"]>
// CHECK: hc.vfull %{{.*}}, shape %{{.*}} : (!hc.undef, tuple<!hc.undef>) -> !hc.vector<f32, ["16"]>
// CHECK: hc.zeros shape %{{.*}} : (tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
// CHECK: hc.ones shape %{{.*}} : (tuple<!hc.undef>) -> !hc.tensor<i1, ["M"]>
// CHECK: hc.full %{{.*}}, shape %{{.*}} : (!hc.undef, tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
// CHECK: hc.empty shape %{{.*}} : (tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
func.func @allocators(%s: !hc.undef) {
  %d16 = hc.const<16 : i64> : !hc.undef
  %dm = hc.const<"M"> : !hc.undef
  %shape_16_16 = hc.tuple(%d16, %d16)
      : (!hc.undef, !hc.undef) -> tuple<!hc.undef, !hc.undef>
  %shape_16 = hc.tuple(%d16) : (!hc.undef) -> tuple<!hc.undef>
  %shape_m = hc.tuple(%dm) : (!hc.undef) -> tuple<!hc.undef>
  %vz = hc.vzeros shape %shape_16_16
      : (tuple<!hc.undef, !hc.undef>) -> !hc.vector<f32, ["16", "16"]>
  %vo = hc.vones  shape %shape_16 : (tuple<!hc.undef>) -> !hc.vector<i1, ["16"]>
  %vf = hc.vfull  %s, shape %shape_16
      : (!hc.undef, tuple<!hc.undef>) -> !hc.vector<f32, ["16"]>
  %tz = hc.zeros  shape %shape_m : (tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
  %to = hc.ones   shape %shape_m : (tuple<!hc.undef>) -> !hc.tensor<i1, ["M"]>
  %tf = hc.full   %s, shape %shape_m
      : (!hc.undef, tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
  %te = hc.empty  shape %shape_m : (tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
  return
}

// CHECK-LABEL: func.func @reductions_and_matmul
// CHECK: hc.matmul %{{.*}}, %{{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.reduce %{{.*}}, kind = sum, axis = 0 : !hc.undef -> !hc.undef
// CHECK: hc.reduce %{{.*}}, kind = max, axis = 1, keepdims = true : !hc.undef -> !hc.undef
// CHECK: hc.reduce %{{.*}}, kind = min, axis = 0 : !hc.undef -> !hc.undef
// CHECK: hc.astype %{{.*}}, target = f32 : !hc.undef -> !hc.undef
func.func @reductions_and_matmul(%a: !hc.undef, %b: !hc.undef) {
  %m = hc.matmul %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
  %s = hc.reduce %a, kind = sum, axis = 0 : !hc.undef -> !hc.undef
  %mx = hc.reduce %a, kind = max, axis = 1, keepdims = true
      : !hc.undef -> !hc.undef
  %mn = hc.reduce %a, kind = min, axis = 0 : !hc.undef -> !hc.undef
  %cast = hc.astype %a, target = f32 : !hc.undef -> !hc.undef
  return
}

// CHECK-LABEL: func.func @calls
// CHECK: hc.call @helper(%{{.*}}, %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.call_intrinsic @wmma(%{{.*}}, %{{.*}}) {wave_size = 32 : i64}
// CHECK-SAME: (!hc.undef, !hc.undef) -> !hc.undef
func.func @calls(%x: !hc.undef, %y: !hc.undef) {
  %a = hc.call @helper(%x, %y) : (!hc.undef, !hc.undef) -> !hc.undef
  %b = hc.call_intrinsic @wmma(%x, %y) {wave_size = 32 : i64}
      : (!hc.undef, !hc.undef) -> !hc.undef
  return
}

// Declared signatures on `hc.func` / `hc.intrinsic` make the signature
// visible to `verifySymbolUses`; call sites get arity/type parity checked.
// `!hc.undef` on either side escapes the parity check (progressive typing).

// CHECK: hc.func @typed_helper(%arg0: i32, %arg1: i32) -> i32
hc.func @typed_helper(%a: i32, %b: i32) -> i32 {
  hc.return %a : i32
}

// CHECK: hc.func @callable_attrs(%arg0: i32, %arg1: i32) -> i32 attributes {arg_attrs = [{}, {}], res_attrs = [{}]}
hc.func @callable_attrs(%a: i32, %b: i32) -> i32 attributes {
  arg_attrs = [{}, {}],
  res_attrs = [{}]
} {
  hc.return %a : i32
}

// CHECK: hc.intrinsic @typed_intrinsic(%arg0: f32) -> f32 scope = <"WorkItem"> parameters = ["x"]
hc.intrinsic @typed_intrinsic(%x: f32) -> f32
    scope = #hc.scope<"WorkItem"> parameters = ["x"] {}

// CHECK-LABEL: func.func @typed_calls
// CHECK: hc.call @typed_helper(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
// CHECK: hc.call @typed_helper(%{{.*}}, %{{.*}}) : (!hc.undef, i32) -> !hc.undef
// CHECK: hc.call @callable_attrs(%{{.*}}, %{{.*}}) {arg_attrs = [{}, {}], res_attrs = [{}]} : (i32, i32) -> i32
// CHECK: hc.call_intrinsic @typed_intrinsic(%{{.*}}) : (f32) -> f32
func.func @typed_calls(%a: i32, %b: i32, %c: f32, %u: !hc.undef) {
  %r1 = hc.call @typed_helper(%a, %b) : (i32, i32) -> i32
  %r2 = hc.call @typed_helper(%u, %b) : (!hc.undef, i32) -> !hc.undef
  %r_attrs = hc.call @callable_attrs(%a, %b) {
      arg_attrs = [{}, {}], res_attrs = [{}]
    } : (i32, i32) -> i32
  %r3 = hc.call_intrinsic @typed_intrinsic(%c) : (f32) -> f32
  return
}

// `const_kwargs` on the declaration are a whitelist of names every call site
// must carry as specialization attributes. Extra attributes stay allowed.

// CHECK: hc.intrinsic @wave_like(%arg0: !hc.undef) -> !hc.undef scope = <"SubGroup"> const_kwargs = ["wave_size"] parameters = ["x", "wave_size"] keyword_only = ["wave_size"]
hc.intrinsic @wave_like(%x: !hc.undef) -> !hc.undef
    scope = #hc.scope<"SubGroup">
    const_kwargs = ["wave_size"]
    parameters = ["x", "wave_size"]
    keyword_only = ["wave_size"] {}

// CHECK-LABEL: func.func @kwarg_calls
// CHECK: hc.call_intrinsic @wave_like(%{{.*}}) {wave_size = 32 : i64}
func.func @kwarg_calls(%x: !hc.undef) {
  %r = hc.call_intrinsic @wave_like(%x) {wave_size = 32 : i64}
      : (!hc.undef) -> !hc.undef
  return
}
