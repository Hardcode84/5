// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-normalize-scope-regions -split-input-file -verify-diagnostics %s

hc.func @live_workitem_token_is_rejected {
  // expected-error @+1 {{cannot flatten workitem region while its scope token is still live}}
  hc.workitem_region {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    %lane = hc.local_id %wi
        : (!hc.workitem<group_shape = #hc.shape<["32"]>,
                        subgroup_size = #hc.expr<"32">>) -> !hc.idx<"$WI0">
  }
  hc.return
}

// -----

hc.func @result_producing_workitem_is_rejected(%v: !hc.bare_vector<f32, ["8"]>) {
  // expected-error @+1 {{result-producing workitem region normalization requires distributed value projection support}}
  %region = hc.workitem_region -> (!hc.bare_vector<f32, ["8", "32"]>) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %v : !hc.bare_vector<f32, ["8"]>
  }
  hc.return
}

// -----

hc.func @subgroup_is_rejected {
  // expected-error @+1 {{subgroup region normalization is not supported yet}}
  hc.subgroup_region {
  ^bb0(%sg: !hc.subgroup<group_shape = #hc.shape<["64"]>,
                         subgroup_size = #hc.expr<"32">>):
  }
  hc.return
}
