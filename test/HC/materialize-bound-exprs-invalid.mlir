// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-materialize-bound-exprs -split-input-file -verify-diagnostics %s

hc.func @live_workitem_geometry_is_rejected(
    %wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                      subgroup_size = #hc.expr<"32">>) -> !hc.idx {
  // expected-error @+1 {{still has live results depending on a workitem/subgroup scope token after bound-expression materialization}}
  %lid = hc.local_id %wi
      : (!hc.workitem<group_shape = #hc.shape<["32"]>,
                      subgroup_size = #hc.expr<"32">>) -> !hc.idx
  hc.return %lid : !hc.idx
}

// -----

hc.func @live_subgroup_geometry_is_rejected(
    %sg: !hc.subgroup<group_shape = #hc.shape<["64"]>,
                      subgroup_size = #hc.expr<"32">>) -> !hc.idx {
  // expected-error @+1 {{still has live results depending on a workitem/subgroup scope token after bound-expression materialization}}
  %sid = hc.subgroup_id %sg
      : (!hc.subgroup<group_shape = #hc.shape<["64"]>,
                      subgroup_size = #hc.expr<"32">>) -> !hc.idx
  hc.return %sid : !hc.idx
}
