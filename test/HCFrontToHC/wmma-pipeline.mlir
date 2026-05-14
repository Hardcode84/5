// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Snapshot the shape-first gfx11 WMMA example at the mechanical
// hc_front -> hc boundary described in doc/lowering.md. The RUN line
// starts from the Python example, so this locks together resolver output,
// inline-helper folding, front-to-HC conversion, and HC verifier round-trip.
// RUN: %python -m examples.amdgpu_gfx11_wmma_matmul --dump-front-ir \
// RUN:   | hc-opt --hc-front-fold-region-defs --hc-front-inline --convert-hc-front-to-hc --hc-promote-names --canonicalize --cse \
// RUN:   | hc-opt \
// RUN:   | FileCheck %s --implicit-check-not=hc_front. --implicit-check-not=@_tile_origin --implicit-check-not=@_lane_a_row --implicit-check-not=@_lane_column

// CHECK: module {
// CHECK-NEXT: hc.kernel @tiled_gfx11_wmma_matmul
// Buffer args carry the default fully-strided layout the frontend
// pins on every buffer parameter. Per-axis `$STRIDE_<i>_<argname>`
// symbols join `bound_symbols` next to the shape symbols their
// owning arg introduced — `M` / `K` are seen first via `a`, then
// `a`'s strides; `N` next via `b`, then `b`'s strides; `c` only
// contributes new strides because `M` / `N` are already bound.
// CHECK-SAME: (%[[GROUP:arg[0-9]+]]: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, %[[A:arg[0-9]+]]: !hc.buffer<f16, ["M", "K"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0 + $STRIDE_1_a*i1">>>, %[[B:arg[0-9]+]]: !hc.buffer<f16, ["K", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_b*i0 + $STRIDE_1_b*i1">>>, %[[C:arg[0-9]+]]: !hc.buffer<f32, ["M", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_c*i0 + $STRIDE_1_c*i1">>>)
// CHECK-SAME: bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "$SG0", "$SG1", "$WGS0", "$WGS1", "$WO0", "$WO1", "$WS0", "$WS1", "$GSZ0", "$WV0", "M", "K", "$STRIDE_0_a", "$STRIDE_1_a", "N", "$STRIDE_0_b", "$STRIDE_1_b", "$STRIDE_0_c", "$STRIDE_1_c"]
// CHECK-SAME: group_shape = #hc.shape<["32", "1"]>
// CHECK-SAME: subgroup_size = 32 : i32
// CHECK-SAME: work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>
// CHECK: hc.mul
// CHECK: hc.mul
// CHECK: %[[ORIGIN:[^ ]+]] = hc.tuple{{.*}} -> tuple<!hc.undef, !hc.undef>
// CHECK: %[[ROW0:[^ ]+]] = hc.getitem %[[ORIGIN]]
// CHECK: %[[COL0:[^ ]+]] = hc.getitem %[[ORIGIN]]
// `init_wmma_acc` carries (group, c, row0, col0) so it can stamp the
// per-element output-validity mask onto the seed accumulator at
// construction time -- see the bounds-aware accumulator notes in the
// example kernel.
// CHECK: %[[ACC0:[^ ]+]] = hc.call @init_wmma_acc(%[[GROUP]], %[[C]], %[[ROW0]], %[[COL0]]) : (!hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, !hc.buffer<f32, ["M", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_c*i0 + $STRIDE_1_c*i1">>>, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: %[[AK:[^ ]+]] = hc.buffer_dim %[[A]], axis = 1 : !hc.buffer<f16, ["M", "K"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0 + $STRIDE_1_a*i1">>> -> !hc.undef
// CHECK: %[[ACC_FINAL:[^ ]+]]:3 = hc.for_range {{.*}} to %[[AK]] step {{.*}} iter_args({{.*}}, {{.*}}, %[[ACC0]]) : {{.*}} -> (!hc.undef, !hc.undef, !hc.undef) {
// CHECK: ^bb0(%[[K0:arg[0-9]+]]: !hc.undef,
// CHECK: %[[A_ROW:[^ ]+]] = hc.slice_expr
// CHECK: %[[K_SLICE:[^ ]+]] = hc.slice_expr
// CHECK: hc.load %[[A]][%[[A_ROW]], %[[K_SLICE]]], shape %{{.*}} : (!hc.buffer<f16, ["M", "K"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0 + $STRIDE_1_a*i1">>>, !hc.undef, !hc.undef, {{.*}}) -> !hc.undef
// CHECK: %[[B_COL:[^ ]+]] = hc.slice_expr
// CHECK: hc.load %[[B]][%[[K_SLICE]], %[[B_COL]]], shape %{{.*}} : (!hc.buffer<f16, ["K", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_b*i0 + $STRIDE_1_b*i1">>>, !hc.undef, !hc.undef, {{.*}}) -> !hc.undef
// CHECK: hc.call @issue_wmma_tile(%[[GROUP]], {{.*}}) : (!hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.yield {{.*}} : !hc.undef, !hc.undef, !hc.undef
// CHECK: hc.call @store_wmma_tile(%[[GROUP]], %[[C]], %[[ROW0]], %[[COL0]], %[[ACC_FINAL]]#2) : (!hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, !hc.buffer<f32, ["M", "N"], <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_c*i0 + $STRIDE_1_c*i1">>>, !hc.undef, !hc.undef, !hc.undef) -> ()

// `init_wmma_acc` reads the per-lane output slice of `c` (`hc.vload` on a
// strided `hc.buffer_view`) so the accumulator seed inherits the
// bounds-aware mask the vload's clip-and-pad rule produces. The mask
// rides through every k-tile iteration and gates the eventual
// `group.store`, keeping right/bottom-edge tiles from writing OOB.
// CHECK-LABEL: hc.func @init_wmma_acc
// CHECK-SAME: (%{{.*}}: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, %{{.*}}: !hc.undef, %{{.*}}: !hc.undef, %{{.*}}: !hc.undef) -> !hc.undef
// CHECK-SAME: attributes {scope = #hc.scope<"WorkGroup">}
// CHECK: %{{.*}} = hc.workitem_region captures = ["col0", "group", "c", "row0"] -> (!hc.undef)
// CHECK: hc.vload {{.*}} : ({{.*}}) -> !hc.undef
// CHECK: hc.buffer_view
// CHECK: hc.yield {{.*}} : !hc.undef
// CHECK: hc.return {{.*}} : !hc.undef

// CHECK-LABEL: hc.func @issue_wmma_tile
// CHECK-SAME: attributes {scope = #hc.scope<"WorkGroup">}
// CHECK: %{{.*}} = hc.workitem_region captures = ["a_tile", "b_tile", "group", "acc"] -> (!hc.undef)
// CHECK: hc.local_id {{.*}} : (!hc.workitem<group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
// CHECK: hc.getitem {{.*}} : (tuple<{{.*}}>, !hc.undef) -> !hc.undef
// CHECK: hc.call @load_wmma_a_fragment
// CHECK: hc.call @load_wmma_b_fragment
// CHECK: hc.buffer_view {{.*}} : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.call_intrinsic @wmma_gfx11
// CHECK-SAME: {arch = "gfx11", wave_size = 32 : i64}
// CHECK-SAME: -> !hc.undef
// CHECK: hc.yield {{.*}} : !hc.undef
// CHECK: hc.return {{.*}} : !hc.undef

// CHECK-LABEL: hc.func @store_wmma_tile
// CHECK-NOT: -> !hc.undef
// CHECK-SAME: attributes {scope = #hc.scope<"WorkGroup">}
// CHECK: hc.workitem_region captures =
// CHECK: hc.slice_expr(lower =
// CHECK: hc.store {{.*}} : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> ()

// CHECK-LABEL: hc.func @load_wmma_a_fragment
// CHECK-SAME: (%{{.*}}: !hc.workitem<group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>
// CHECK-SAME: attributes {scope = #hc.scope<"WorkItem">}
// CHECK: hc.local_id {{.*}} : (!hc.workitem<group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
// CHECK: hc.buffer_view
// CHECK: hc.vec {{.*}} : !hc.undef -> !hc.undef
// CHECK: hc.with_inactive {{.*}}, %{{.*}} : (!hc.undef, !hc.undef) -> !hc.undef

// CHECK-LABEL: hc.func @load_wmma_b_fragment
// CHECK-SAME: (%{{.*}}: !hc.workitem<group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>
// CHECK-SAME: attributes {scope = #hc.scope<"WorkItem">}
// CHECK: hc.local_id {{.*}} : (!hc.workitem<group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
// CHECK: hc.buffer_view
// CHECK: hc.vec {{.*}} : !hc.undef -> !hc.undef
// CHECK: hc.with_inactive {{.*}}, %{{.*}} : (!hc.undef, !hc.undef) -> !hc.undef

// CHECK-LABEL: hc.intrinsic @wmma_gfx11
// CHECK-SAME: (%{{.*}}: !hc.undef, %{{.*}}: !hc.tensor<f16, ["16", "16"]>, %{{.*}}: !hc.tensor<f16, ["16", "16"]>, %{{.*}}: !hc.vector<f16, ["16"]>, %{{.*}}: !hc.vector<f16, ["16"]>, %{{.*}}: !hc.vector<f32, ["8"]>, %{{.*}}: !hc.idx) -> !hc.vector<f32, ["8"]>
// CHECK-SAME: scope = <"WorkItem">
// CHECK-SAME: effects = pure
// CHECK-SAME: const_kwargs = ["arch", "wave_size"]
// CHECK-SAME: parameters = ["group", "a_tile", "b_tile", "a_frag", "b_frag", "acc_frag", "lane", "wave_size", "arch"]
// CHECK-SAME: keyword_only = ["lane", "wave_size", "arch"]
// CHECK-NEXT: }

// Target lowerings ride along as real `transform.named_sequence` ops in a
// sibling top-level module. The interpreter pass walks them by symbol;
// keeping the recipe IR first-class means the verifier checks structure
// instead of trusting an opaque string.
// CHECK-LABEL: module @__hc_intrinsic_lowerings__
// CHECK-SAME: attributes {transform.with_named_sequence}
// CHECK: transform.named_sequence @__hc_lower_wmma_gfx11_amdgpu_gfx11
// CHECK-SAME: hc.target = "amdgpu-gfx11"
// CHECK: transform.hc.match_intrinsic_call
// CHECK-SAME: @wmma_gfx11
// CHECK-SAME: target = "amdgpu-gfx11"
// Recipe-level pre-checks for the gfx11 dispatch invariants: target
// dispatch already filters by `hc.target`, so these guard against a kernel
// that accidentally calls `wmma_gfx11` with a drifted `arch`/`wave_size`.
// CHECK: transform.hc.require_intrinsic_attr
// CHECK-SAME: expected = "gfx11"
// CHECK-SAME: name = "arch"
// CHECK: transform.hc.require_intrinsic_attr
// CHECK-SAME: expected = 32 : i64
// CHECK-SAME: name = "wave_size"
// `amdgpu.wmma` consumes plain upstream `vector<NxF>` types, while the
// post-`hc-lower-launch-body` call site presents `!hc.bare_vector` operands
// and result. The recipe plants two literal types (one per fragment shape)
// plus three operand-side casts and one result-side cast that bridge bare
// ↔ upstream via `unrealized_conversion_cast`; these casts pair with the
// UCCs the launch-body pass already plants on either side of the call,
// and post-rewrite `--canonicalize` collapses the chains to identity.
// The const_type / cast_value ops appear interleaved because the recipe
// builder emits each cast right after the literal type it consumes, with
// dedup folding repeated `vector<16xf16>` references onto a single
// `constant_type` op.
// CHECK: transform.hc.constant_type vector<16xf16>
// CHECK: transform.hc.cast_value
// CHECK: transform.hc.cast_value
// CHECK: transform.hc.constant_type vector<8xf32>
// CHECK: transform.hc.cast_value
// CHECK: transform.hc.create_op "amdgpu.wmma"
// `amdgpu.wmma` rejects unknown attributes (`arch`/`wave_size` ride on
// the call site purely for dispatch), and its `m`/`n`/`k` slots are
// declared as `i32` — pin both so a future recipe drift would surface
// here instead of waiting for the upstream verifier to reject the freshly
// created op.
// CHECK-SAME: dynamic_attrs []()
// CHECK-SAME: static_attrs = {k = 16 : i32, m = 16 : i32, n = 16 : i32}
// CHECK: transform.hc.cast_value
// CHECK: transform.hc.replace_intrinsic_call
