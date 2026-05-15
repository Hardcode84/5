// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-infer-types -split-input-file %s | FileCheck %s --implicit-check-not=__hc_join_tmp

// CHECK-LABEL: hc.func @scalar_index_arithmetic
// CHECK: hc.const<2 : i64> : !hc.idx<"2">
// CHECK: hc.const<3 : i64> : !hc.idx<"3">
// CHECK: hc.const<7 : i64> : !hc.idx<"7">
// CHECK: hc.add {{.*}} -> !hc.idx<"5">
// CHECK: hc.cmp.lt {{.*}} -> !hc.pred<"True">
// CHECK: hc.slice_expr{{.*}} -> !hc.slice<lower = !hc.idx<"5">, upper = !hc.idx<"7">>
hc.func @scalar_index_arithmetic {
  %two = hc.const<2 : i64> : !hc.undef
  %three = hc.const<3 : i64> : !hc.undef
  %seven = hc.const<7 : i64> : !hc.undef
  %sum = hc.add %two, %three : (!hc.undef, !hc.undef) -> !hc.undef
  %cmp = hc.cmp.lt %sum, %seven : (!hc.undef, !hc.undef) -> !hc.undef
  %slice = hc.slice_expr(lower = %sum upper = %seven)
      : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @slice_region_branch_result
// CHECK: hc.if {{.*}} -> (!hc.slice<lower = !hc.idx>) : i1
// CHECK: hc.yield {{.*}} : !hc.slice<lower = !hc.idx<"1">>
// CHECK: hc.yield {{.*}} : !hc.slice<lower = !hc.idx<"2">>
hc.func @slice_region_branch_result(%cond: i1) {
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.slice_expr(lower = %one) : (!hc.undef) -> !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.slice_expr(lower = %two) : (!hc.undef) -> !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @slice_expression_conflict_renumbers_join_symbol
// CHECK: ^bb0(%[[IV:.*]]: !hc.idx<"[[JOIN:\$join[0-9]+]]">):
// CHECK: hc.slice_expr{{.*}} -> !hc.slice<lower = !hc.idx<"[[JOIN]]">, upper = !hc.idx<"16 + [[JOIN]]">>
hc.func @slice_expression_conflict_renumbers_join_symbol {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  %sixteen = hc.const<16 : i64> : !hc.undef
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) {
  ^bb0(%iv: !hc.undef):
    %upper = hc.add %iv, %sixteen : (!hc.undef, !hc.undef) -> !hc.undef
    %slice = hc.slice_expr(lower = %iv upper = %upper)
        : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @for_range_iv
// CHECK: hc.for_range
// CHECK: ^bb0(%arg0: !hc.idx<"$join{{[0-9]+}}">):
// CHECK: hc.const<1 : i64> : !hc.idx<"1">
// CHECK: hc.add %arg0, {{.*}} : (!hc.idx<"$join{{[0-9]+}}">, !hc.idx<"1">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @for_range_iv {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) {
  ^bb0(%iv: !hc.undef):
    %one = hc.const<1 : i64> : !hc.undef
    %next = hc.add %iv, %one : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @region_branch_results
// CHECK: hc.for_range
// CHECK-SAME: -> (!hc.idx<"$join{{[0-9]+}}">)
// CHECK: ^bb0(%{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">, %{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">):
// CHECK: hc.yield {{.*}} : !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
// CHECK: hc.if {{.*}} -> (!hc.idx<"$join{{[0-9]+}}">) : i1
hc.func @region_branch_results(%cond: i1) {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  %init = hc.const<0 : i64> : !hc.undef
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
  ^bb0(%iv: !hc.undef, %acc: !hc.undef):
    %one = hc.const<1 : i64> : !hc.undef
    %next = hc.add %acc, %one : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %next : !hc.undef
  }
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.const<1 : i64> : !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.const<2 : i64> : !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @tuple_region_branch_result
// CHECK: hc.if {{.*}} -> (tuple<!hc.idx<"$join{{[0-9]+}}">, f32>) : i1
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"1">, f32>
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"2">, f32>
hc.func @tuple_region_branch_result(%cond: i1, %value: f32) {
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.tuple(%one, %value) : (!hc.undef, f32) -> !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.tuple(%two, %value) : (!hc.undef, f32) -> !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @tuple_multi_index_conflicts
// CHECK: hc.if {{.*}} -> (tuple<!hc.idx<"$join0">, !hc.idx<"$join1">>) : i1
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"1">, !hc.idx<"10">>
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"2">, !hc.idx<"20">>
hc.func @tuple_multi_index_conflicts(%cond: i1) {
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %ten = hc.const<10 : i64> : !hc.undef
  %twenty = hc.const<20 : i64> : !hc.undef
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.tuple(%one, %ten) : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.tuple(%two, %twenty) : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @loads
// CHECK: hc.buffer_dim {{.*}} -> !hc.idx<"N">
// CHECK: hc.load {{.*}} -> !hc.tensor<f32, ["4", "8"]>
// CHECK: hc.vec {{.*}} -> !hc.vector<f32, ["4", "8"]>
hc.func @loads(%buf: !hc.buffer<f32, ["M", "N"]>, %i: !hc.idx<"0">,
               %j: !hc.idx<"1">) -> (!hc.undef, !hc.undef, !hc.undef) {
  %dim = hc.buffer_dim %buf, axis = 1
      : !hc.buffer<f32, ["M", "N"]> -> !hc.undef
  %four = hc.const<4 : i64> : !hc.undef
  %eight = hc.const<8 : i64> : !hc.undef
  %shape = hc.tuple(%four, %eight)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %t = hc.load %buf[%i, %j], shape %shape
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"0">, !hc.idx<"1">, !hc.undef)
        -> !hc.undef
  %v = hc.vec %t : !hc.undef -> !hc.undef
  hc.return %dim, %t, %v : !hc.undef, !hc.undef, !hc.undef
}

// -----

// Optional `layout` attribute on `hc.vload`: inference reads it and
// bakes it into the result type. This is the inference half of the
// FrontToHC `layout=` kwarg path — FrontToHC stamps the attribute on
// the producer, inference produces a layout-bearing result type.
// The non-injective storage_size here (`m*k` < `product(16, 16, 32)`)
// is what the old post-emit `hc.as_layout` overlay would have
// rejected at the verifier; attribute-on-producer sidesteps that by
// never producing a bare → layout-bearing transition.
//
// CHECK-LABEL: hc.func @vload_layout_attr
// CHECK: hc.vload {{.*}} -> !hc.vector<f32, ["16", "16", "32"], <{{.*}}storage_size = #hc.expr<"k*m">, offset = #hc.expr<"j + i*k">>>
hc.func @vload_layout_attr(%buf: !hc.buffer<f32, ["M"]>,
                           %z: !hc.idx<"0">) -> !hc.undef {
  %m = hc.const<16 : i64> : !hc.undef
  %k = hc.const<16 : i64> : !hc.undef
  %lane = hc.const<32 : i64> : !hc.undef
  %shape = hc.tuple(%m, %k, %lane)
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  %v = hc.vload %buf[%z], shape %shape
      {layout = #hc.layout<shape_syms = ["m", "k", "lane"],
                           index_syms = ["i", "j", "l"], params = {},
                           storage_size = #hc.expr<"m*k">,
                           offset = #hc.expr<"j + i*k">>}
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.undef) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// `hc.vzeros` allocator path: element type from `dtype`, shape from the
// tuple operand, layout from the op attribute.
//
// CHECK-LABEL: hc.func @vzeros_layout_attr
// CHECK: hc.vzeros {{.*}} -> !hc.vector<f32, ["8", "4"], <{{.*}}storage_size = #hc.expr<"m">, offset = #hc.expr<"i">>>
hc.func @vzeros_layout_attr() -> !hc.undef {
  %m = hc.const<8 : i64> : !hc.undef
  %n = hc.const<4 : i64> : !hc.undef
  %shape = hc.tuple(%m, %n) : (!hc.undef, !hc.undef) -> !hc.undef
  %z = hc.vzeros shape %shape
      {dtype = f32,
       layout = #hc.layout<shape_syms = ["m", "n"], index_syms = ["i", "j"],
                           params = {}, storage_size = #hc.expr<"m">,
                           offset = #hc.expr<"i">>}
      : (!hc.undef) -> !hc.undef
  hc.return %z : !hc.undef
}

// -----

// `hc.vload` without a `layout` attribute: result type is bare (no
// layout), matching the pre-existing default path.
//
// CHECK-LABEL: hc.func @vload_no_layout_attr
// CHECK: hc.vload {{.*}} -> !hc.vector<f32, ["16"]>
hc.func @vload_no_layout_attr(%buf: !hc.buffer<f32, ["M"]>,
                              %z: !hc.idx<"0">) -> !hc.undef {
  %m = hc.const<16 : i64> : !hc.undef
  %shape = hc.tuple(%m) : (!hc.undef) -> !hc.undef
  %v = hc.vload %buf[%z], shape %shape
      : (!hc.buffer<f32, ["M"]>, !hc.idx<"0">, !hc.undef) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// `hc.strip_layout` drops the layout slot but preserves the
// carrier flavor: bare_vector operand re-types as a layout-less
// bare_vector; a (semantic) vector operand re-types as a
// layout-less vector. Operand below carries a non-injective layout
// (storage_size = "m" > product = "m*k") describing a per-lane
// projection over wave-wide storage; inference propagates the
// operand's shape + element type and clears the layout.
//
// CHECK-LABEL: hc.func @strip_layout_infer_bare
// CHECK: hc.strip_layout {{.*}} -> !hc.bare_vector<f32, ["8", "4"]>
hc.func @strip_layout_infer_bare(
    %v: !hc.bare_vector<f32, ["8", "4"],
      #hc.layout<shape_syms = ["m", "k"], index_syms = ["i", "j"],
                 params = {}, storage_size = #hc.expr<"m">,
                 offset = #hc.expr<"i">>>) -> !hc.undef {
  %r = hc.strip_layout %v
      : !hc.bare_vector<f32, ["8", "4"],
          #hc.layout<shape_syms = ["m", "k"], index_syms = ["i", "j"],
                     params = {}, storage_size = #hc.expr<"m">,
                     offset = #hc.expr<"i">>>
      -> !hc.undef
  hc.return %r : !hc.undef
}

// -----

// CHECK-LABEL: hc.func @strip_layout_infer_vector
// CHECK: hc.strip_layout {{.*}} -> !hc.vector<f32, ["8"]>
hc.func @strip_layout_infer_vector(
    %v: !hc.vector<f32, ["8"],
      #hc.layout<shape_syms = ["m"], index_syms = ["i"],
                 params = {}, storage_size = #hc.expr<"256">,
                 offset = #hc.expr<"i * 16">>>) -> !hc.undef {
  %r = hc.strip_layout %v
      : !hc.vector<f32, ["8"],
          #hc.layout<shape_syms = ["m"], index_syms = ["i"],
                     params = {}, storage_size = #hc.expr<"256">,
                     offset = #hc.expr<"i * 16">>>
      -> !hc.undef
  hc.return %r : !hc.undef
}

// -----

// CHECK-LABEL: hc.func @buffer_views
// CHECK: hc.buffer_view {{.*}} -> !hc.buffer<f32, ["4", "N"]>
// CHECK: hc.buffer_dim {{.*}} -> !hc.idx<"4">
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f16, ["16"]>
// CHECK: hc.vec {{.*}} -> !hc.vector<f16, ["16"]>
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f16, ["8"]>
hc.func @buffer_views(%buf: !hc.buffer<f32, ["M", "N"]>,
                      %tensor: !hc.tensor<f16, ["16", "16"]>,
                      %idx: !hc.idx<"3">)
    -> (!hc.undef, !hc.undef, !hc.undef, !hc.undef) {
  %zero = hc.const<0 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %four = hc.const<4 : i64> : !hc.undef
  %sixteen = hc.const<16 : i64> : !hc.undef
  %head = hc.slice_expr(lower = %zero upper = %four)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %full = hc.slice_expr() : () -> !hc.undef
  %even_rows = hc.slice_expr(lower = %zero upper = %sixteen step = %two)
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  %buf_view = hc.buffer_view %buf[%head]
      : (!hc.buffer<f32, ["M", "N"]>, !hc.undef) -> !hc.undef
  %dim = hc.buffer_dim %buf_view, axis = 0
      : !hc.undef -> !hc.undef
  %tensor_view = hc.buffer_view %tensor[%idx, %full]
      : (!hc.tensor<f16, ["16", "16"]>, !hc.idx<"3">, !hc.undef)
        -> !hc.undef
  %vec = hc.vec %tensor_view : !hc.undef -> !hc.undef
  %stepped = hc.buffer_view %tensor[%even_rows, %idx]
      : (!hc.tensor<f16, ["16", "16"]>, !hc.undef, !hc.idx<"3">)
        -> !hc.undef
  hc.return %dim, %tensor_view, %vec, %stepped
      : !hc.undef, !hc.undef, !hc.undef, !hc.undef
}

// -----

// CHECK-LABEL: hc.func @buffer_view_layout_multibuf
// Slicing axis 0 of a 4-D layout-bearing tensor with a scalar
// `!hc.idx<"buf_idx">` drops the `b` shape sym and the `ib` index
// sym from the residual layout, substitutes them into the surviving
// `offset` and `storage_size` expressions (`ib` -> `buf_idx`,
// `b` -> the operand's actual `BUF` dim expr), and keeps the M / N /
// LANE slots intact. `buf_idx` survives the composition as a free
// symbol — the lowering pipeline binds it from the kernel scope per
// `doc/layouts.md` "Free symbols in layout offsets". This is the
// substrate the multi-buffered LDS example needs: a single 4-D
// layout description that fans out into per-buffer residual views.
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f32, ["M", "N", "LANE"], <shape_syms = ["m", "n", "l"], index_syms = ["im", "in", "il"], params = {}, storage_size = #hc.expr<"BUF*l*m*n">, offset = #hc.expr<"il + l*(in + n*(im + buf_idx*m))">>>
hc.func @buffer_view_layout_multibuf(
    %lds: !hc.tensor<f32, ["BUF", "M", "N", "LANE"],
                     #hc.layout<shape_syms = ["b", "m", "n", "l"],
                                index_syms = ["ib", "im", "in", "il"],
                                params = {},
                                storage_size = #hc.expr<"b*m*n*l">,
                                offset = #hc.expr<"((ib*m + im)*n + in)*l + il">>>,
    %buf_idx: !hc.idx<"buf_idx">) -> !hc.undef {
  %v = hc.buffer_view %lds[%buf_idx]
      : (!hc.tensor<f32, ["BUF", "M", "N", "LANE"],
                    #hc.layout<shape_syms = ["b", "m", "n", "l"],
                               index_syms = ["ib", "im", "in", "il"],
                               params = {},
                               storage_size = #hc.expr<"b*m*n*l">,
                               offset = #hc.expr<"((ib*m + im)*n + in)*l + il">>>,
         !hc.idx<"buf_idx">) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// CHECK-LABEL: hc.func @buffer_view_layout_mixed_scalar_slice
// Scalar on axis 0 (BUF), `[0:M]` slice on axis 1 (M), implicit
// pass-through on axis 2 (N). The scalar substitutes — `ib` and `b`
// are dropped from the residual slot lists and folded into the
// offset / storage_size — while the slice keeps its slot in place
// and leaves the layout indexing unchanged. `M` flows back into the
// residual storage_size as the dropped shape sym's source dim,
// just like the all-scalar case.
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f32, ["M", "N"], <shape_syms = ["m", "n"], index_syms = ["im", "in"], params = {}, storage_size = #hc.expr<"BUF*m*n">, offset = #hc.expr<"in + n*(im + buf_idx*m)">>>
hc.func @buffer_view_layout_mixed_scalar_slice(
    %lds: !hc.tensor<f32, ["BUF", "M", "N"],
                     #hc.layout<shape_syms = ["b", "m", "n"],
                                index_syms = ["ib", "im", "in"],
                                params = {},
                                storage_size = #hc.expr<"b*m*n">,
                                offset = #hc.expr<"(ib*m + im)*n + in">>>,
    %buf_idx: !hc.idx<"buf_idx">,
    %row_slice: !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"M">>)
    -> !hc.undef {
  %v = hc.buffer_view %lds[%buf_idx, %row_slice]
      : (!hc.tensor<f32, ["BUF", "M", "N"],
                    #hc.layout<shape_syms = ["b", "m", "n"],
                               index_syms = ["ib", "im", "in"],
                               params = {},
                               storage_size = #hc.expr<"b*m*n">,
                               offset = #hc.expr<"(ib*m + im)*n + in">>>,
         !hc.idx<"buf_idx">,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"M">>) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// CHECK-LABEL: hc.func @buffer_view_layout_all_slice
// No scalar indices: every layout slot survives, `offset` and
// `storage_size` are unchanged. Pins the trivial-slice composition
// contract (`lower = 0`, step defaulting to 1, sliced extent ==
// operand dim) so the non-trivial-slice rebinding below doesn't
// accidentally regress this baseline.
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f32, ["M", "N"], <shape_syms = ["m", "n"], index_syms = ["im", "in"], params = {}, storage_size = #hc.expr<"m*n">, offset = #hc.expr<"in + im*n">>>
hc.func @buffer_view_layout_all_slice(
    %lds: !hc.tensor<f32, ["M", "N"],
                     #hc.layout<shape_syms = ["m", "n"],
                                index_syms = ["im", "in"],
                                params = {},
                                storage_size = #hc.expr<"m*n">,
                                offset = #hc.expr<"im*n + in">>>,
    %row_slice: !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"M">>)
    -> !hc.undef {
  %v = hc.buffer_view %lds[%row_slice]
      : (!hc.tensor<f32, ["M", "N"],
                    #hc.layout<shape_syms = ["m", "n"],
                               index_syms = ["im", "in"],
                               params = {},
                               storage_size = #hc.expr<"m*n">,
                               offset = #hc.expr<"im*n + in">>>,
         !hc.slice<lower = !hc.idx<"0">, upper = !hc.idx<"M">>) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Non-trivial slice with `lower = row0`, `step = 2`, plus a scalar on
// the trailing axis. The slice axis stays in the residual rank, but
// `composeBufferViewLayout` substitutes the layout's `index_syms[0]`
// ("im") with `row0 + 2 * im` and `shape_syms[0]` ("m") with the
// operand's actual `M` dim — without the latter the residual
// `storage_size` would bind `m` to the sliced extent and lose the
// physical-storage equivalence that the flatten pass identity
// branch keys off of. The scalar axis (axis 1) substitutes "in" with
// `col0` and "n" with `N`, exactly like the trivial-slice +
// scalar mix below.
// CHECK-LABEL: hc.func @buffer_view_layout_strided_slice
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f32, ["8"], <shape_syms = ["m"], index_syms = ["im"], params = {}, storage_size = #hc.expr<"M*N">, offset = #hc.expr<"col0 + N*(2*im + row0)">>>
hc.func @buffer_view_layout_strided_slice(
    %lds: !hc.tensor<f32, ["M", "N"],
                     #hc.layout<shape_syms = ["m", "n"],
                                index_syms = ["im", "in"],
                                params = {},
                                storage_size = #hc.expr<"m*n">,
                                offset = #hc.expr<"im*n + in">>>,
    %row0: !hc.idx<"row0">,
    %col0: !hc.idx<"col0">) -> !hc.undef {
  %sixteen = hc.const<16 : i64> : !hc.idx<"16">
  %two = hc.const<2 : i64> : !hc.idx<"2">
  %row_stop = hc.add %row0, %sixteen
      : (!hc.idx<"row0">, !hc.idx<"16">) -> !hc.idx<"row0 + 16">
  %strided = hc.slice_expr(lower = %row0 upper = %row_stop step = %two)
      : (!hc.idx<"row0">, !hc.idx<"row0 + 16">, !hc.idx<"2">) -> !hc.undef
  %v = hc.buffer_view %lds[%strided, %col0]
      : (!hc.tensor<f32, ["M", "N"],
                    #hc.layout<shape_syms = ["m", "n"],
                               index_syms = ["im", "in"],
                               params = {},
                               storage_size = #hc.expr<"m*n">,
                               offset = #hc.expr<"im*n + in">>>,
         !hc.undef, !hc.idx<"col0">) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// Non-trivial slice with `lower = lo`, no explicit `step` (default 1).
// Shape rebind fires because the sliced extent (`hi - lo`) doesn't
// structurally match the operand's `M` dim; the index rebind also
// fires because `lower` isn't 0. The trailing axis is an implicit
// pass-through (more axes than subscript entries) — the layout
// keeps `n` / `in` in the residual without substitution because
// pass-through axes carry no slice info.
// CHECK-LABEL: hc.func @buffer_view_layout_lower_slice
// CHECK: hc.buffer_view {{.*}} -> !hc.tensor<f32, ["hi - lo", "N"], <shape_syms = ["m", "n"], index_syms = ["im", "in"], params = {}, storage_size = #hc.expr<"M*n">, offset = #hc.expr<"in + n*(im + lo)">>>
hc.func @buffer_view_layout_lower_slice(
    %lds: !hc.tensor<f32, ["M", "N"],
                     #hc.layout<shape_syms = ["m", "n"],
                                index_syms = ["im", "in"],
                                params = {},
                                storage_size = #hc.expr<"m*n">,
                                offset = #hc.expr<"im*n + in">>>,
    %row_slice: !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">>)
    -> !hc.undef {
  %v = hc.buffer_view %lds[%row_slice]
      : (!hc.tensor<f32, ["M", "N"],
                    #hc.layout<shape_syms = ["m", "n"],
                               index_syms = ["im", "in"],
                               params = {},
                               storage_size = #hc.expr<"m*n">,
                               offset = #hc.expr<"im*n + in">>>,
         !hc.slice<lower = !hc.idx<"lo">, upper = !hc.idx<"hi">>) -> !hc.undef
  hc.return %v : !hc.undef
}

// -----

// CHECK-LABEL: hc.func @vector_views
// CHECK: hc.buffer_view {{.*}} -> f32
// CHECK: hc.buffer_view {{.*}} -> !hc.vector<f32, ["4"]>
// CHECK: hc.buffer_view {{.*}} -> !hc.vector<f32, ["8"]>
// CHECK: hc.buffer_view {{.*}} -> f32
// CHECK: hc.getitem {{.*}} -> f32
hc.func @vector_views(%vector: !hc.vector<f32, ["8"]>,
                      %idx: !hc.idx<"3">)
    -> (!hc.undef, !hc.undef, !hc.undef, !hc.undef, !hc.undef) {
  %zero = hc.const<0 : i64> : !hc.undef
  %four = hc.const<4 : i64> : !hc.undef
  %lane = hc.const<7 : i64> : !hc.undef
  %head = hc.slice_expr(lower = %zero upper = %four)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %full = hc.slice_expr() : () -> !hc.undef
  %element = hc.buffer_view %vector[%idx]
      : (!hc.vector<f32, ["8"]>, !hc.idx<"3">) -> !hc.undef
  %fragment = hc.buffer_view %vector[%head]
      : (!hc.vector<f32, ["8"]>, !hc.undef) -> !hc.undef
  %collective_fragment = hc.buffer_view %vector[%full, %lane, %full]
      : (!hc.vector<f32, ["8"]>, !hc.undef, !hc.undef, !hc.undef)
        -> !hc.undef
  %collective_element = hc.buffer_view %vector[%idx, %lane]
      : (!hc.vector<f32, ["8"]>, !hc.idx<"3">, !hc.undef) -> !hc.undef
  %item = hc.getitem %vector[%idx]
      : (!hc.vector<f32, ["8"]>, !hc.idx<"3">) -> !hc.undef
  hc.return %element, %fragment, %collective_fragment, %collective_element, %item
      : !hc.undef, !hc.undef, !hc.undef, !hc.undef, !hc.undef
}

// -----

// CHECK-LABEL: hc.func @allocators
// CHECK: hc.vzeros shape {{.*}}, dtype = f32 {{.*}} -> !hc.vector<f32, ["4", "8"]>
// CHECK: hc.vfull {{.*}} -> !hc.vector<f32, ["4", "8"]>
// CHECK: hc.zeros shape {{.*}}, dtype = f32 {{.*}} -> !hc.tensor<f32, ["4", "8"]>
// CHECK: hc.full {{.*}} -> !hc.tensor<f32, ["4", "8"]>
hc.func @allocators(%fill: f32) -> (!hc.undef, !hc.undef, !hc.undef, !hc.undef) {
  %four = hc.const<4 : i64> : !hc.undef
  %eight = hc.const<8 : i64> : !hc.undef
  %shape = hc.tuple(%four, %eight)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %zero_vec = hc.vzeros shape %shape, dtype = f32
      : (!hc.undef) -> !hc.undef
  %vec = hc.vfull %fill, shape %shape : (f32, !hc.undef) -> !hc.undef
  %zero_tensor = hc.zeros shape %shape, dtype = f32
      : (!hc.undef) -> !hc.undef
  %tensor = hc.full %fill, shape %shape : (f32, !hc.undef) -> !hc.undef
  hc.return %zero_vec, %vec, %zero_tensor, %tensor
      : !hc.undef, !hc.undef, !hc.undef, !hc.undef
}

// -----

// CHECK-LABEL: hc.func @mixed_idx_and_builtin_stays_unknown
// CHECK: hc.add {{.*}} : (!hc.idx<"M">, i64) -> !hc.undef
// CHECK: hc.cmp.lt {{.*}} : (!hc.idx<"M">, i64) -> !hc.undef
hc.func @mixed_idx_and_builtin_stays_unknown(%idx: !hc.idx<"M">, %n: i64) {
  %sum = hc.add %idx, %n : (!hc.idx<"M">, i64) -> !hc.undef
  %cmp = hc.cmp.lt %idx, %n : (!hc.idx<"M">, i64) -> !hc.undef
  hc.return
}

// -----

// `hc.pow` is the structural carrier for Python `**`; `-hc-lower-pow`
// unfolds it before this pass in the canonical schedule, but the
// inference still needs to be permissive for partial pipelines. Two
// arms: a matching-scalar-type case folds to the scalar; an idx-vs-
// idx case stays `!hc.undef` because there's no symbolic `Pow`
// primitive in ixsimpl (the canonical pipeline relies on the unfold
// to produce the mul chain the idx inference does handle).
// CHECK-LABEL: hc.func @pow_inference
// CHECK: hc.pow {{.*}} : (f32, f32) -> f32
// CHECK: hc.pow {{.*}} : (!hc.idx<"M">, !hc.idx<"2">) -> !hc.undef
hc.func @pow_inference(%x: f32, %e: f32, %m: !hc.idx<"M">) {
  %a = hc.pow %x, %e : (f32, f32) -> !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %b = hc.pow %m, %two : (!hc.idx<"M">, !hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @untyped_string_const_stays_unknown
// CHECK: hc.const<"gfx11"> : !hc.undef
hc.func @untyped_string_const_stays_unknown {
  %arch = hc.const<"gfx11"> : !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @interprocedural_callee
// CHECK-SAME: (%{{[^:]+}}: !hc.idx<"2">) -> !hc.idx<"3">
// CHECK: hc.add {{.*}} -> !hc.idx<"3">
hc.func @interprocedural_callee(%x: !hc.undef) -> !hc.undef {
  %one = hc.const<1 : i64> : !hc.undef
  %sum = hc.add %x, %one : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// CHECK-LABEL: hc.func @interprocedural_caller
// CHECK: hc.call @interprocedural_callee(%{{.*}}) : (!hc.idx<"2">) -> !hc.idx<"3">
hc.func @interprocedural_caller {
  %two = hc.const<2 : i64> : !hc.undef
  %result = hc.call @interprocedural_callee(%two) : (!hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.intrinsic @typed_vector_intrinsic
// CHECK-SAME: (%{{[^:]+}}: !hc.vector<f16, ["16"]>) -> !hc.vector<f32, ["8"]>
hc.intrinsic @typed_vector_intrinsic(%frag: !hc.vector<f16, ["16"]>)
    -> !hc.vector<f32, ["8"]>
    scope = #hc.scope<"WorkItem"> parameters = ["frag"] {}

// CHECK-LABEL: hc.func @intrinsic_contract_caller
// CHECK: hc.call_intrinsic @typed_vector_intrinsic(%{{[^)]*}}) : (!hc.vector<f16, ["16"]>) -> !hc.vector<f32, ["8"]>
hc.func @intrinsic_contract_caller(%frag: !hc.vector<f16, ["16"]>) {
  %result = hc.call_intrinsic @typed_vector_intrinsic(%frag)
      : (!hc.vector<f16, ["16"]>) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @joined_interprocedural_callee
// CHECK-SAME: (%{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
// CHECK: hc.add {{.*}} -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @joined_interprocedural_callee(%x: !hc.undef) -> !hc.undef {
  %one = hc.const<1 : i64> : !hc.undef
  %sum = hc.add %x, %one : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// CHECK-LABEL: hc.func @joined_interprocedural_caller
// CHECK: hc.call @joined_interprocedural_callee(%{{.*}}) : (!hc.idx<"2">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
// CHECK: hc.call @joined_interprocedural_callee(%{{.*}}) : (!hc.idx<"5">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @joined_interprocedural_caller {
  %two = hc.const<2 : i64> : !hc.undef
  %five = hc.const<5 : i64> : !hc.undef
  %a = hc.call @joined_interprocedural_callee(%two)
      : (!hc.undef) -> !hc.undef
  %b = hc.call @joined_interprocedural_callee(%five)
      : (!hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @workitem_tail_region_callee
// CHECK-SAME: -> !hc.idx<"0">
// CHECK: %[[REGION:.*]] = hc.workitem_region captures = ["group"] -> (!hc.idx<"0">)
// CHECK: hc.yield {{.*}} : !hc.idx<"0">
// CHECK: hc.return %[[REGION]] : !hc.idx<"0">
hc.func @workitem_tail_region_callee(%group: !hc.undef) -> !hc.undef {
  %region = hc.workitem_region captures = ["group"] -> (!hc.undef) {
  ^bb0(%wi: !hc.undef):
    %seed = hc.const<0 : i64> : !hc.undef
    hc.yield %seed : !hc.undef
  }
  hc.return %region : !hc.undef
}

// CHECK-LABEL: hc.func @subgroup_tail_region_callee
// CHECK-SAME: -> !hc.idx<"1">
// CHECK: %[[REGION:.*]] = hc.subgroup_region captures = ["group"] -> (!hc.idx<"1">)
// CHECK: hc.yield {{.*}} : !hc.idx<"1">
// CHECK: hc.return %[[REGION]] : !hc.idx<"1">
hc.func @subgroup_tail_region_callee(%group: !hc.undef) -> !hc.undef {
  %region = hc.subgroup_region captures = ["group"] -> (!hc.undef) {
  ^bb0(%sg: !hc.undef):
    %seed = hc.const<1 : i64> : !hc.undef
    hc.yield %seed : !hc.undef
  }
  hc.return %region : !hc.undef
}

// CHECK-LABEL: hc.func @workitem_region_vector_lifts_collective_suffix
// CHECK: %[[REGION:.*]] = hc.workitem_region -> (!hc.vector<f32, ["8", "32", "1"]>)
// CHECK: hc.yield {{.*}} : !hc.vector<f32, ["8"]>
// CHECK: hc.return %[[REGION]] : !hc.vector<f32, ["8", "32", "1"]>
hc.func @workitem_region_vector_lifts_collective_suffix(
    %local: !hc.vector<f32, ["8"]>) -> !hc.undef {
  %region = hc.workitem_region -> (!hc.undef) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.vector<f32, ["8"]>
  }
  hc.return %region : !hc.undef
}

// CHECK-LABEL: hc.func @workitem_region_scalar_lifts_collective_suffix
// CHECK: %[[REGION:.*]] = hc.workitem_region -> (!hc.vector<i32, ["32", "1"]>)
// CHECK: hc.yield {{.*}} : i32
// CHECK: hc.return %[[REGION]] : !hc.vector<i32, ["32", "1"]>
hc.func @workitem_region_scalar_lifts_collective_suffix(%value: i32)
    -> !hc.undef {
  %region = hc.workitem_region -> (!hc.undef) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %value : i32
  }
  hc.return %region : !hc.undef
}

// CHECK-LABEL: hc.func @subgroup_region_vector_lifts_collective_suffix
// CHECK: %[[REGION:.*]] = hc.subgroup_region -> (!hc.vector<f32, ["4", "2"]>)
// CHECK: hc.yield {{.*}} : !hc.vector<f32, ["4"]>
// CHECK: hc.return %[[REGION]] : !hc.vector<f32, ["4", "2"]>
hc.func @subgroup_region_vector_lifts_collective_suffix(
    %local: !hc.vector<f32, ["4"]>) -> !hc.undef {
  %region = hc.subgroup_region -> (!hc.undef) {
  ^bb0(%sg: !hc.subgroup<group_shape = #hc.shape<["64"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.vector<f32, ["4"]>
  }
  hc.return %region : !hc.undef
}

// CHECK-LABEL: hc.func @distributed_vector_indexes_back_to_local_fragment
// CHECK: hc.buffer_view {{.*}} -> !hc.vector<f32, ["8"]>
hc.func @distributed_vector_indexes_back_to_local_fragment(
    %acc: !hc.vector<f32, ["8", "32", "1"]>,
    %lane: !hc.idx<"7">,
    %zero: !hc.idx<"0">) -> !hc.undef {
  %full = hc.slice_expr() : () -> !hc.undef
  %local = hc.buffer_view %acc[%full, %lane, %zero]
      : (!hc.vector<f32, ["8", "32", "1"]>, !hc.undef, !hc.idx<"7">,
         !hc.idx<"0">) -> !hc.undef
  hc.return %local : !hc.undef
}

// CHECK-LABEL: hc.func @call_result_iter_arg_keeps_loop_body_live
// CHECK: %[[INIT:.*]] = hc.call @workitem_tail_region_callee
// CHECK: hc.for_range {{.*}} iter_args(%[[INIT]])
// CHECK: ^bb0(%{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">, %{{[^:]+}}: !hc.idx<"0">):
// CHECK: hc.const<16 : i64> : !hc.idx<"16">
// CHECK: hc.add {{.*}} : (!hc.idx<"$join{{[0-9]+}}">, !hc.idx<"16">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @call_result_iter_arg_keeps_loop_body_live(%group: !hc.undef) {
  %init = hc.call @workitem_tail_region_callee(%group)
      : (!hc.undef) -> !hc.undef
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<64 : i64> : !hc.undef
  %step = hc.const<16 : i64> : !hc.undef
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
  ^bb0(%iv: !hc.undef, %carried: !hc.undef):
    %tile = hc.const<16 : i64> : !hc.undef
    %next = hc.add %iv, %tile : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %carried : !hc.undef
  }
  hc.return
}
