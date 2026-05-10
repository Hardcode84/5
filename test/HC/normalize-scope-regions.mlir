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
