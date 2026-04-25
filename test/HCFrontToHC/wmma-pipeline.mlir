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
// RUN:   | FileCheck %s --implicit-check-not=hc_front. --implicit-check-not=@_tile_origin --implicit-check-not=@_lane_a_row --implicit-check-not=@_lane_column --implicit-check-not=@_lane_output_row_slice_args

// CHECK: module {
// CHECK-NEXT: hc.kernel @tiled_gfx11_wmma_matmul
// CHECK-SAME: (%[[GROUP:arg[0-9]+]]: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, %[[A:arg[0-9]+]]: !hc.buffer<!hc.undef, ["M", "K"]>, %[[B:arg[0-9]+]]: !hc.buffer<!hc.undef, ["K", "N"]>, %[[C:arg[0-9]+]]: !hc.buffer<!hc.undef, ["M", "N"]>)
// CHECK-SAME: group_shape = #hc.shape<["32", "1"]>
// CHECK-SAME: subgroup_size = 32 : i32
// CHECK-SAME: work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>
// CHECK: hc.mul
// CHECK: hc.mul
// CHECK: %[[ORIGIN:[^ ]+]] = hc.tuple{{.*}} -> tuple<!hc.undef, !hc.undef>
// CHECK: %[[ROW0:[^ ]+]] = hc.getitem %[[ORIGIN]]
// CHECK: %[[COL0:[^ ]+]] = hc.getitem %[[ORIGIN]]
// CHECK: %[[ACC0:[^ ]+]] = hc.call @init_wmma_acc(%[[GROUP]]) : (!hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>) -> !hc.undef
// CHECK: %[[AK:[^ ]+]] = hc.buffer_dim %[[A]], axis = 1 : !hc.buffer<!hc.undef, ["M", "K"]> -> !hc.undef
// CHECK: %[[ACC_FINAL:[^ ]+]]:3 = hc.for_range {{.*}} to %[[AK]] step {{.*}} iter_args({{.*}}, {{.*}}, %[[ACC0]]) : {{.*}} -> (!hc.undef, !hc.undef, !hc.undef) {
// CHECK: ^bb0(%[[K0:arg[0-9]+]]: !hc.undef,
// CHECK: %[[A_ROW:[^ ]+]] = hc.slice_expr
// CHECK: %[[K_SLICE:[^ ]+]] = hc.slice_expr
// CHECK: hc.load %[[A]][%[[A_ROW]], %[[K_SLICE]]], shape %{{.*}} : (!hc.buffer<!hc.undef, ["M", "K"]>, !hc.undef, !hc.undef, {{.*}}) -> !hc.undef
// CHECK: %[[B_COL:[^ ]+]] = hc.slice_expr
// CHECK: hc.load %[[B]][%[[K_SLICE]], %[[B_COL]]], shape %{{.*}} : (!hc.buffer<!hc.undef, ["K", "N"]>, !hc.undef, !hc.undef, {{.*}}) -> !hc.undef
// CHECK: hc.call @issue_wmma_tile(%[[GROUP]], {{.*}}) : (!hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.yield {{.*}} : !hc.undef, !hc.undef, !hc.undef
// CHECK: hc.call @store_wmma_tile(%[[GROUP]], %[[C]], %[[ROW0]], %[[COL0]], %[[ACC_FINAL]]#2) : (!hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>, !hc.buffer<!hc.undef, ["M", "N"]>, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef

// CHECK-LABEL: hc.func @init_wmma_acc
// CHECK-SAME: (%{{.*}}: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>, group_shape = #hc.shape<["32", "1"]>, subgroup_size = #hc.expr<"32">>) -> !hc.undef
// CHECK-SAME: attributes {scope = #hc.scope<"WorkGroup">}
// CHECK: %{{.*}} = hc.workitem_region captures = ["group"] -> (!hc.undef)
// CHECK: hc.vzeros shape %{{.*}} : ({{.*}}) -> !hc.undef
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
// CHECK-SAME: -> !hc.undef
// CHECK-SAME: scope = <"WorkItem">
// CHECK-SAME: effects = pure
// CHECK-SAME: const_kwargs = ["arch", "wave_size"]
// CHECK-SAME: parameters = ["group", "a_tile", "b_tile", "a_frag", "b_frag", "acc_frag", "lane", "wave_size", "arch"]
// CHECK-SAME: keyword_only = ["lane", "wave_size", "arch"]
// CHECK-NEXT: }
