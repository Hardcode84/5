// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-normalize-scope-regions -split-input-file %s | FileCheck %s

// CHECK-LABEL: hc.func @flatten_resultless_workitem
// CHECK-NOT: hc.workitem_region
// CHECK: %[[LANE:.*]] = hc.materialize_bound_expr : !hc.idx<"$WI0">
// CHECK: hc.buffer_view %{{.*}}[%[[LANE]]]
// CHECK: hc.return
hc.func @flatten_resultless_workitem(%tile: !hc.bare_tensor<f32, ["32"]>) {
  hc.workitem_region {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    %lane = hc.materialize_bound_expr : !hc.idx<"$WI0">
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
