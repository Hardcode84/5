// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-materialize-bound-exprs -split-input-file %s | FileCheck %s

// CHECK-LABEL: hc.func @launch_expr_chain
// CHECK: %[[WG0:.*]] = hc.materialize_bound_expr : !hc.idx<"$WG0">
// CHECK: %[[C16:.*]] = hc.materialize_bound_expr : !hc.idx<"16">
// CHECK: hc.mul %[[WG0]], %[[C16]]
// CHECK-SAME: -> !hc.idx<"16*$WG0">
// CHECK: %[[ROW:.*]] = hc.materialize_bound_expr : !hc.idx<"16*$WG0">
// CHECK: hc.return %[[ROW]]
hc.func @launch_expr_chain(%group: !hc.group<work_shape = #hc.shape<["M"]>,
                                             group_shape = #hc.shape<["32"]>,
                                             subgroup_size = #hc.expr<"32">>)
    -> !hc.idx<"16*$WG0"> {
  %gid = hc.group_id %group
      : (!hc.group<work_shape = #hc.shape<["M"]>,
                   group_shape = #hc.shape<["32"]>,
                   subgroup_size = #hc.expr<"32">>)
        -> !hc.idx<"$WG0">
  %sixteen = hc.const<16 : i64> : !hc.idx<"16">
  %row = hc.mul %gid, %sixteen
      : (!hc.idx<"$WG0">, !hc.idx<"16">) -> !hc.idx<"16*$WG0">
  hc.return %row : !hc.idx<"16*$WG0">
}

// -----

// CHECK-LABEL: hc.func @bound_predicate
// CHECK: %[[PRED:.*]] = hc.materialize_bound_expr : !hc.pred<"-32 + $WI0 < 0">
// CHECK: hc.return %[[PRED]]
hc.func @bound_predicate(%pred: !hc.pred<"$WI0 - 32 < 0">)
    -> !hc.pred<"$WI0 - 32 < 0"> {
  hc.return %pred : !hc.pred<"$WI0 - 32 < 0">
}

// -----

// CHECK-LABEL: hc.func @user_symbol_stays_symbolic
// CHECK-NOT: hc.materialize_bound_expr
// CHECK: hc.return %arg0 : !hc.idx<"M">
hc.func @user_symbol_stays_symbolic(%m: !hc.idx<"M">) -> !hc.idx<"M"> {
  hc.return %m : !hc.idx<"M">
}

// -----

// CHECK-LABEL: hc.kernel @declared_kernel_bound_symbol
// CHECK-SAME: bound_symbols = ["$WG0"]
// CHECK: %[[WG0:.*]] = hc.materialize_bound_expr : !hc.idx<"$WG0">
// CHECK: hc.add %[[WG0]], %[[WG0]]
hc.kernel @declared_kernel_bound_symbol(%gid: !hc.idx<"$WG0">)
    attributes {bound_symbols = ["$WG0"]} {
  %sum = hc.add %gid, %gid : (!hc.idx<"$WG0">, !hc.idx<"$WG0">) -> !hc.idx<"2*$WG0">
  hc.return
}
