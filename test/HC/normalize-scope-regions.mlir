// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-normalize-scope-regions -split-input-file %s | FileCheck %s

// CHECK-LABEL: hc.func @flatten_resultless_workitem
// CHECK-NOT: hc.workitem_region
// CHECK: %[[LANE:.*]] = hc.idx_apply () : () -> !hc.idx<"$WI0">
// CHECK: hc.buffer_view %{{.*}}[%[[LANE]]]
// CHECK: hc.return
hc.func @flatten_resultless_workitem(%tile: !hc.bare_tensor<f32, ["32"]>) {
  hc.workitem_region {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    %lane = hc.idx_apply () : () -> !hc.idx<"$WI0">
    %item = hc.buffer_view %tile[%lane]
        : (!hc.bare_tensor<f32, ["32"]>, !hc.idx<"$WI0">) -> f32
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @erase_empty_yield
// CHECK-NOT: hc.workitem_region
// CHECK: hc.return
hc.func @erase_empty_yield {
  hc.workitem_region {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @project_workitem_result
// CHECK-SAME: %[[LOCAL:.*]]: !hc.bare_vector<f32, ["8"]>
// CHECK-SAME: -> !hc.bare_vector<f32, ["8"]>
// CHECK-NOT: hc.workitem_region
// CHECK: hc.return %[[LOCAL]] : !hc.bare_vector<f32, ["8"]>
hc.func @project_workitem_result(%local: !hc.bare_vector<f32, ["8"]>)
    -> !hc.bare_vector<f32, ["8", "32"]> {
  %region = hc.workitem_region -> (!hc.bare_vector<f32, ["8", "32"]>) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.bare_vector<f32, ["8"]>
  }
  hc.return %region : !hc.bare_vector<f32, ["8", "32"]>
}

// -----

// CHECK-LABEL: hc.func @drop_workitem_token_call_arg
// CHECK-SAME: %[[VALUE:[^:]+]]: i32
// CHECK-SAME: {
// CHECK: hc.call @workitem_helper(%[[VALUE]]) : (i32) -> i32
hc.func @drop_workitem_token_call_arg(
    %wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                      subgroup_size = #hc.expr<"32">>,
    %value: i32) {
  %result = hc.call @workitem_helper(%wi, %value)
      : (!hc.workitem<group_shape = #hc.shape<["32"]>,
                      subgroup_size = #hc.expr<"32">>, i32) -> i32
  hc.return
}

// CHECK-LABEL: hc.func @workitem_helper
// CHECK-SAME: (%[[VALUE:.*]]: i32) -> i32
hc.func @workitem_helper(
    %wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                      subgroup_size = #hc.expr<"32">>,
    %value: i32) -> i32 attributes {scope = #hc.scope<"WorkItem">} {
  hc.return %value : i32
}

// -----

// Post-flatten the rank-N suffix has collapsed into the 1D storage
// product (`["256"]` = `8 * 32`), so the rank-N `dropWorkitemSuffix`
// walk is a no-op on every type. The anchored-lift retyper recovers
// the lane-local form via the `hc.workitem_region` result, then the
// inline replaces the region with its yielded value.
// CHECK-LABEL: hc.func @postflatten_project_workitem_result
// CHECK-SAME: %[[LOCAL:.*]]: !hc.bare_vector<f32, ["8"]>
// CHECK-SAME: -> !hc.bare_vector<f32, ["8"]>
// CHECK-NOT: hc.workitem_region
// CHECK: hc.return %[[LOCAL]] : !hc.bare_vector<f32, ["8"]>
hc.func @postflatten_project_workitem_result(
    %local: !hc.bare_vector<f32, ["8"]>) -> !hc.bare_vector<f32, ["256"]> {
  %region = hc.workitem_region -> (!hc.bare_vector<f32, ["256"]>) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.bare_vector<f32, ["8"]>
  }
  hc.return %region : !hc.bare_vector<f32, ["256"]>
}

// -----

// Workitem-region result threaded as `hc.for_range` iter-arg
// post-flatten: anchor at the region result, propagate the lane-local
// target across the iter-init operand, iter_result slot, body block
// arg, and the body's yield. The loop body re-yields the block arg
// unchanged so every threaded slot ends up at lane-local storage.
// CHECK-LABEL: hc.func @postflatten_for_range_iter_arg
// CHECK-SAME: %[[LOCAL:[^:]+]]: !hc.bare_vector<f32, ["8"]>
// CHECK-NOT: hc.workitem_region
// CHECK: %[[LOOP:.*]] = hc.for_range
// CHECK-SAME: iter_args(%[[LOCAL]])
// CHECK-SAME: -> (!hc.bare_vector<f32, ["8"]>)
// CHECK: ^bb0(%{{.*}}: !hc.idx, %[[ARG:[^:]+]]: !hc.bare_vector<f32, ["8"]>):
// CHECK: hc.yield %[[ARG]] : !hc.bare_vector<f32, ["8"]>
hc.func @postflatten_for_range_iter_arg(
    %local: !hc.bare_vector<f32, ["8"]>,
    %lo: !hc.idx<"0">, %hi: !hc.idx<"N">, %step: !hc.idx<"1">) {
  %init = hc.workitem_region -> (!hc.bare_vector<f32, ["256"]>) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %local : !hc.bare_vector<f32, ["8"]>
  }
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">)
      -> (!hc.bare_vector<f32, ["256"]>) {
  ^bb0(%i: !hc.idx, %arg: !hc.bare_vector<f32, ["256"]>):
    hc.yield %arg : !hc.bare_vector<f32, ["256"]>
  }
  hc.return
}

// -----

// Workgroup-shared bare-tensor whose 1D storage matches the lifted
// vector by divisibility (`["256"]` divides cleanly by `32 * 1`). It
// never appears as a `hc.workitem_region` result, so the anchored
// retyper leaves it untouched: the function arg type and the post-pass
// `hc.return` operand both stay at the workgroup-shared bare carrier
// regardless of any divisibility against the suffix. The unused `%wi`
// scope token is the launch-metadata anchor; `dropUnusedScopeTokenArgs`
// peels it off after the suffix-strip walk runs, exactly as it does
// for every other token-bearing helper in this file.
// CHECK-LABEL: hc.func @postflatten_workgroup_tile_unchanged
// CHECK-SAME: %[[TILE:[^:]+]]: !hc.bare_tensor<f16, ["256"]>
// CHECK-SAME: -> !hc.bare_tensor<f16, ["256"]>
// CHECK: hc.return %[[TILE]] : !hc.bare_tensor<f16, ["256"]>
hc.func @postflatten_workgroup_tile_unchanged(
    %wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                      subgroup_size = #hc.expr<"32">>,
    %tile: !hc.bare_tensor<f16, ["256"]>)
    -> !hc.bare_tensor<f16, ["256"]>
    attributes {scope = #hc.scope<"WorkItem">} {
  hc.return %tile : !hc.bare_tensor<f16, ["256"]>
}
