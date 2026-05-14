// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-distribute-wave-layouts`. The pass converts
// wave-cooperative layout-bearing carriers (e.g. the AMD gfx11 WMMA
// accumulator's `<offset = 32*fi + lane>`) into per-lane peers,
// dropping the lane axis from carrier shapes / iter spaces and
// substituting the lane index sym by `$WI0` in surviving offset
// expressions. See the pass description in
// `include/hc/Transforms/Passes.td` and the rationale in
// `doc/layouts.md`.
//
// RUN: hc-opt --hc-distribute-wave-layouts %s --split-input-file | FileCheck %s

// Canonical case: a producer `hc.generic` whose outs is a
// wave-distributable carrier (offset = 32*fi + lane, storage_size =
// 256, first shape dim = wave_size = 32). The rewrite drops the lane
// iter axis, substitutes `lane := $WI0` in ins offsets, drops the
// leading entry of outs/ins offsets that referenced the lane iter,
// and retypes the producer / its outs init to the per-lane bare form.
// The downstream `hc.buffer_view %wave[$WI0, :]` projection collapses
// to a forward; the downstream `hc.strip_layout` becomes a no-op
// because the strip's input already matches its layout-less result.
//
// CHECK-LABEL: hc.kernel @wave_distribute_producer
// CHECK: hc.workitem_region
// CHECK: %[[VZ:.+]] = hc.vzeros shape
// CHECK-SAME: -> !hc.bare_vector<f32, ["8"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_1 =
// CHECK-SAME: ins (%arg1 at [#hc.expr<"16*$WG0 + $WI0">, #hc.expr<"16*$WG1 + i_1">]
// CHECK-SAME: outs (%[[VZ]] at [#hc.expr<"i_1">]
// CHECK-SAME: -> (!hc.bare_vector<f32, ["8"]>)
// CHECK-NOT: hc.buffer_view
// CHECK-NOT: hc.bare_vector<f32, ["32", "8"]
// CHECK-NOT: storage_size = #hc.expr<"256">
hc.kernel @wave_distribute_producer(
    %arg0: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>,
                     group_shape = #hc.shape<["32", "1"]>,
                     subgroup_size = #hc.expr<"32">>,
    %arg1: !hc.buffer<f32, ["M", "N"],
      <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"],
       params = {}, storage_size = #hc.expr<"0">,
       offset = #hc.expr<"$STRIDE_0_c*i0 + $STRIDE_1_c*i1">>>)
    attributes {group_shape = #hc.shape<["32", "1"]>,
                subgroup_size = 32 : i32,
                work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>} {
  hc.workitem_region captures = [] {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                          subgroup_size = #hc.expr<"32">>):
    %wi0 = hc.idx_apply () : () -> !hc.idx<"$WI0">
    %wave_dim = hc.idx_apply () : () -> !hc.idx<"32">
    %frag_dim = hc.idx_apply () : () -> !hc.idx<"8">
    %shape = hc.tuple(%wave_dim, %frag_dim)
        : (!hc.idx<"32">, !hc.idx<"8">) -> tuple<!hc.idx<"32">, !hc.idx<"8">>
    %vz = hc.vzeros shape %shape
        : (tuple<!hc.idx<"32">, !hc.idx<"8">>)
          -> !hc.bare_vector<f32, ["32", "8"],
              <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
               params = {}, storage_size = #hc.expr<"256">,
               offset = #hc.expr<"32*fi + lane">>>
    %gen = hc.generic
      iter (parallel i_0 = %wave_dim : !hc.idx<"32">,
            parallel i_1 = %frag_dim : !hc.idx<"8">)
      ins (%arg1 at [#hc.expr<"16*$WG0 + i_0">,
                     #hc.expr<"16*$WG1 + i_1">]
           : !hc.buffer<f32, ["M", "N"],
              <shape_syms = ["d0", "d1"], index_syms = ["i0", "i1"],
               params = {}, storage_size = #hc.expr<"0">,
               offset = #hc.expr<"$STRIDE_0_c*i0 + $STRIDE_1_c*i1">>>)
      outs (%vz at [#hc.expr<"i_0">, #hc.expr<"i_1">]
            : !hc.bare_vector<f32, ["32", "8"],
                <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
                 params = {}, storage_size = #hc.expr<"256">,
                 offset = #hc.expr<"32*fi + lane">>>)
      -> (!hc.bare_vector<f32, ["32", "8"],
            <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
             params = {}, storage_size = #hc.expr<"256">,
             offset = #hc.expr<"32*fi + lane">>>) {
    ^bb0(%a: f32, %b: f32):
      hc.yield %a : f32
    }
    %open = hc.slice_expr() : () -> !hc.slice
    %view = hc.buffer_view %gen[%wi0, %open]
        : (!hc.bare_vector<f32, ["32", "8"],
              <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
               params = {}, storage_size = #hc.expr<"256">,
               offset = #hc.expr<"32*fi + lane">>>,
           !hc.idx<"$WI0">, !hc.slice)
          -> !hc.bare_vector<f32, ["8"],
              <shape_syms = ["fc"], index_syms = ["fi"],
               params = {}, storage_size = #hc.expr<"256">,
               offset = #hc.expr<"$WI0 + 32*fi">>>
  }
  hc.return
}

// -----

// Wave-distributable generic whose body materialises a
// `hc.pred_apply` referencing the lane iter sym (`i_0`) as an
// ambient binding through its result-type predicate. The producer
// rewrite drops `i_0` from the surrounding generic's iter list, so
// the body-walker must substitute every `i_0` leaf in nested
// `hc.pred_apply` / `hc.idx_apply` carriers with the wave sym
// (`$WI0`, bound by the enclosing workitem region) — otherwise the
// inner ops would dangle on the now-unbound `i_0` and downstream
// passes would fail to legalise them.
//
// CHECK-LABEL: hc.kernel @wave_distribute_body_substitution
// CHECK: hc.workitem_region
// CHECK: %[[VZ:.+]] = hc.vzeros shape
// CHECK-SAME: -> !hc.bare_vector<!hc.pred, ["8"]>
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i_1 =
// CHECK-SAME: outs (%[[VZ]] at [#hc.expr<"i_1">]
// CHECK: hc.pred_apply () : () -> !hc.pred<"16*$WG0 + $WI0 - M < 0 & 16*$WG1 - N + i_1 < 0">
// CHECK-NOT: i_0 < 0
hc.kernel @wave_distribute_body_substitution(
    %arg0: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>,
                     group_shape = #hc.shape<["32", "1"]>,
                     subgroup_size = #hc.expr<"32">>)
    attributes {group_shape = #hc.shape<["32", "1"]>,
                subgroup_size = 32 : i32,
                work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>} {
  hc.workitem_region captures = [] {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32", "1"]>,
                          subgroup_size = #hc.expr<"32">>):
    %wave_dim = hc.idx_apply () : () -> !hc.idx<"32">
    %frag_dim = hc.idx_apply () : () -> !hc.idx<"8">
    %shape = hc.tuple(%wave_dim, %frag_dim)
        : (!hc.idx<"32">, !hc.idx<"8">) -> tuple<!hc.idx<"32">, !hc.idx<"8">>
    %vz = hc.vzeros shape %shape
        : (tuple<!hc.idx<"32">, !hc.idx<"8">>)
          -> !hc.bare_vector<!hc.pred, ["32", "8"],
              <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
               params = {}, storage_size = #hc.expr<"256">,
               offset = #hc.expr<"32*fi + lane">>>
    %gen = hc.generic
      iter (parallel i_0 = %wave_dim : !hc.idx<"32">,
            parallel i_1 = %frag_dim : !hc.idx<"8">)
      ins ()
      outs (%vz at [#hc.expr<"i_0">, #hc.expr<"i_1">]
            : !hc.bare_vector<!hc.pred, ["32", "8"],
                <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
                 params = {}, storage_size = #hc.expr<"256">,
                 offset = #hc.expr<"32*fi + lane">>>)
      -> (!hc.bare_vector<!hc.pred, ["32", "8"],
            <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
             params = {}, storage_size = #hc.expr<"256">,
             offset = #hc.expr<"32*fi + lane">>>) {
    ^bb0(%a: !hc.pred):
      %p = hc.pred_apply () : ()
          -> !hc.pred<"16*$WG0 - M + i_0 < 0 & 16*$WG1 - N + i_1 < 0">
      %cast = builtin.unrealized_conversion_cast %p
          : !hc.pred<"16*$WG0 - M + i_0 < 0 & 16*$WG1 - N + i_1 < 0">
          to !hc.pred
      hc.yield %cast : !hc.pred
    }
  }
  hc.return
}

// -----

// Pass is a no-op for a kernel that has no wave-distributable
// layouts: the first shape dim doesn't match the launch's
// subgroup_size, so `tryFactorWaveLayout` bails and the producer
// stays unchanged.
//
// CHECK-LABEL: hc.kernel @noop_when_no_wave_layout
// CHECK: hc.workitem_region
// CHECK: hc.vzeros
// CHECK-SAME: -> !hc.bare_vector<f32, ["32", "8"], <{{.*}}storage_size = #hc.expr<"256">
hc.kernel @noop_when_no_wave_layout(
    %arg0: !hc.group<work_shape = #hc.shape<["64*ceiling(1/16*M)", "ceiling(1/16*N)"]>,
                     group_shape = #hc.shape<["64", "1"]>,
                     subgroup_size = #hc.expr<"64">>)
    attributes {group_shape = #hc.shape<["64", "1"]>,
                subgroup_size = 64 : i32,
                work_shape = #hc.shape<["64*ceiling(1/16*M)", "ceiling(1/16*N)"]>} {
  hc.workitem_region captures = [] {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["64", "1"]>,
                          subgroup_size = #hc.expr<"64">>):
    %wave_dim = hc.idx_apply () : () -> !hc.idx<"32">
    %frag_dim = hc.idx_apply () : () -> !hc.idx<"8">
    %shape = hc.tuple(%wave_dim, %frag_dim)
        : (!hc.idx<"32">, !hc.idx<"8">) -> tuple<!hc.idx<"32">, !hc.idx<"8">>
    %vz = hc.vzeros shape %shape
        : (tuple<!hc.idx<"32">, !hc.idx<"8">>)
          -> !hc.bare_vector<f32, ["32", "8"],
              <shape_syms = ["lc", "fc"], index_syms = ["lane", "fi"],
               params = {}, storage_size = #hc.expr<"256">,
               offset = #hc.expr<"32*fi + lane">>>
  }
  hc.return
}
