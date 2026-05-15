// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Positive-path coverage for the two-pass pipeline. See
// `ConvertHCFrontToHC` in `include/hc/Conversion/HCFrontToHC/Passes.td`
// for the contract; every LIT in this directory exercises the two
// passes together. The second RUN round-trips through a bare `hc-opt`
// parse+print to catch any IR the conversion pass emits that the `hc`
// verifier would reject.
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names %s | hc-opt | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names --cse %s | FileCheck %s --check-prefix=CSE

// Exercises the mechanical hc_front -> hc rewrite patterns this pass
// covers: top-level kernel/func/intrinsic, return/constant/binop,
// target_name + target_tuple + target_subscript via assign, name dispatch
// (param / iv / local / constant / symbol / callee / intrinsic /
// dsl_method), attr+subscript for launch geometry and buffer.shape, the
// for_range lowering from `range(...)`, and the dsl-method calls that land
// on dedicated hc ops (vec, astype, with_inactive, store).

  // CHECK-LABEL: hc.kernel @basic
  // Buffer args carry the default fully-strided layout from the
  // frontend boundary on: per-axis `$STRIDE_<i>_<argname>` symbols
  // pinned in `bound_symbols` so the host wrapper can bind them
  // against the `_mlir_ciface_hc_get_stride` runtime helper at launch.
// CHECK-SAME: (%arg0: !hc.group<work_shape = #hc.shape<["M"]>, group_shape = #hc.shape<["32"]>, subgroup_size = #hc.expr<"32">>, %arg1: !hc.buffer<!hc.undef, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0">>>, %arg2: !hc.buffer<!hc.undef, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_b*i0">>>)
// CHECK-SAME: attributes {
// CHECK-SAME: bound_symbols = ["$WG0", "$WI0", "$SG0", "$WGS0", "$WO0", "$WS0", "$GSZ0", "$WV0", "M", "$STRIDE_0_a", "$STRIDE_0_b"]
// CHECK-SAME: group_shape = #hc.shape<["32"]>
// CHECK-SAME: literals = ["TILE"]
// CHECK-SAME: subgroup_size = 32 : i32
// CHECK-SAME: work_shape = #hc.shape<["M"]>
// CHECK-NOT: hc_front.

module {
  hc_front.kernel "basic" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    literals = ["TILE"],
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M]", kind = "buffer", name = "a", shape = ["M"]},
      {annotation = "Buffer[M]", kind = "buffer", name = "b", shape = ["M"]}
    ],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    // CHECK: %[[D:.*]] = hc.buffer_dim %arg1, axis = 0 : !hc.buffer<!hc.undef, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0">>>
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %axis = hc_front.constant<0 : i64>
    %sh = hc_front.attr %a, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %dim = hc_front.subscript %sh[%axis]
    %dim_target = hc_front.target_name "dim"
    hc_front.assign %dim_target = %dim

    // CHECK: %[[GID:.*]] = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M"]>, group_shape = #hc.shape<["32"]>, subgroup_size = #hc.expr<"32">>) -> !hc.idx<"$WG0">
    // CHECK: %[[GID_TUPLE:.*]] = hc.tuple(%[[GID]]) : (!hc.idx<"$WG0">) -> tuple<!hc.idx<"$WG0">>
    // CHECK: hc.getitem %[[GID_TUPLE]]
    %gid_attr = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid0 = hc_front.subscript %gid_attr[%axis]
    %gid_target = hc_front.target_name "row0"
    hc_front.assign %gid_target = %gid0

    // CHECK: %[[M:.*]] = hc.symbol : !hc.idx<"M">
    // CHECK: hc.add
    %sym_m = hc_front.name "M" {ctx = "load", ref = {kind = "symbol"}}
    %inc = hc_front.binop "Add"(%sym_m, %axis)
    %acc = hc_front.target_name "acc"
    hc_front.assign %acc = %inc

    // CHECK: hc.for_range {{.*}} to {{.*}} step {{.*}} : (!hc.undef, !hc.undef, !hc.undef)
    // CHECK: ^bb0(%arg{{.*}}: !hc.undef):
    hc_front.for {
      %i = hc_front.target_name "i"
    } in {
      %lo = hc_front.constant<0 : i64>
      %hi = hc_front.constant<16 : i64>
      %st = hc_front.constant<1 : i64>
      %range_fn = hc_front.name "range" {ctx = "load", ref = {builtin = "range", kind = "builtin"}}
      %range_call = hc_front.call %range_fn(%lo, %hi, %st)
    } do {
      // CHECK: hc.buffer_view %arg1[%{{.*}}] : (!hc.buffer<!hc.undef, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0">>>, !hc.undef) -> !hc.undef
      %iv = hc_front.name "i" {ctx = "load", ref = {kind = "iv"}}
      %a_ref = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
      %load = hc_front.subscript %a_ref[%iv]
      // CHECK: hc.const<2 : i64> : !hc.undef
      %c2 = hc_front.constant<2 : i64>
      // CHECK: hc.store %arg2[%{{.*}}], %{{.*}} : (!hc.buffer<!hc.undef, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_b*i0">>>, !hc.undef, !hc.undef) -> ()
      %b_ref = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
      %b_idx = hc_front.target_subscript %b_ref[%iv]
      hc_front.assign %b_idx = %c2
      // `hc.yield` is an implicit terminator on for_range; not printed.
    }

    // CHECK: hc.workitem_region captures = ["group"] {
    hc_front.workitem_region captures = ["group"] attributes {
      parameters = [{kind = "launch_context", launch_context = "workitem", name = "wi"}]
    } {
      // CHECK: %[[LIDV:.*]] = hc.local_id %{{.*}} : (!hc.workitem<group_shape = #hc.shape<["32"]>, subgroup_size = #hc.expr<"32">>) -> !hc.idx<"$WI0">
      // CHECK: %[[LID_TUPLE:.*]] = hc.tuple(%[[LIDV]]) : (!hc.idx<"$WI0">) -> tuple<!hc.idx<"$WI0">>
      // CHECK: hc.getitem %[[LID_TUPLE]]
      %wi = hc_front.name "wi" {ctx = "load", ref = {kind = "param"}}
      %lid_attr = hc_front.attr %wi, "local_id" {ref = {kind = "dsl_method", method = "local_id"}}
      %axis0 = hc_front.constant<0 : i64>
      %lid0 = hc_front.subscript %lid_attr[%axis0]
      %lane_t = hc_front.target_name "lane"
      hc_front.assign %lane_t = %lid0
    }
    // CHECK: hc.return

    hc_front.return
  }

  // CHECK-LABEL: hc.func @helper
  // CHECK-SAME: attributes {scope = #hc.scope<"WorkGroup">}
  // CHECK-NOT: hc_front.
  hc_front.func "helper" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "x"}],
    scope = "WorkGroup"
  } {
    // CHECK: hc.return
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %x
  }

  // CHECK-LABEL: hc.func @tuple_values
  // CHECK: %[[PAIR:.*]] = hc.tuple(%arg0, %arg1) : (!hc.undef, !hc.undef) -> tuple<!hc.undef, !hc.undef>
  // CHECK: %[[I0:.*]] = hc.const<0 : i64> : !hc.undef
  // CHECK: hc.getitem %[[PAIR]][%[[I0]]]
  // CHECK: hc.return %[[PAIR]] : tuple<!hc.undef, !hc.undef>
  hc_front.func "tuple_values" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "a"}, {name = "b"}],
    scope = "WorkGroup"
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    %pair = hc_front.tuple (%a, %b)
    %ta = hc_front.target_name "x"
    %tb = hc_front.target_name "y"
    %tgt = hc_front.target_tuple (%ta, %tb)
    hc_front.assign %tgt = %pair
    hc_front.return %pair
  }

  // `tail_return` is stamped by `-hc-front-fold-region-defs` for
  // `return inner()` nested-region calls. Conversion should make the control
  // flow explicit without relying on the larger WMMA fixture.
  // CHECK-LABEL: hc.func @tail_return_workitem
  // CHECK: %[[WREG:.*]] = hc.workitem_region captures = ["group"] -> (!hc.undef)
  // CHECK: hc.const<7 : i64> : !hc.undef
  // CHECK: hc.yield {{.*}} : !hc.undef
  // CHECK: hc.return %[[WREG]] : !hc.undef
  hc_front.func "tail_return_workitem" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}],
    scope = "WorkGroup"
  } {
    hc_front.workitem_region captures = ["group"] attributes {
      decorators = ["group.workitems"],
      name = "inner",
      parameters = [{kind = "launch_context", launch_context = "workitem", name = "wi"}],
      tail_return
    } {
      %seed = hc_front.constant<7 : i64>
      hc_front.return %seed
    }
  }

  // CHECK-LABEL: hc.func @tail_return_subgroup
  // CHECK: %[[SREG:.*]] = hc.subgroup_region captures = ["group"] -> (!hc.undef)
  // CHECK: hc.const<11 : i64> : !hc.undef
  // CHECK: hc.yield {{.*}} : !hc.undef
  // CHECK: hc.return %[[SREG]] : !hc.undef
  hc_front.func "tail_return_subgroup" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}],
    scope = "WorkGroup"
  } {
    hc_front.subgroup_region captures = ["group"] attributes {
      decorators = ["group.subgroups"],
      name = "wave",
      parameters = [{kind = "launch_context", launch_context = "subgroup", name = "sg"}],
      tail_return
    } {
      %seed = hc_front.constant<11 : i64>
      hc_front.return %seed
    }
  }

  // Simulator-fallback body is discarded unconditionally. The body
  // below is deliberately non-trivial so a regression into walking it
  // would emit lowered ops before the closing `}` and break CHECK-NEXT.
  // CHECK-LABEL: hc.intrinsic @intr
  // CHECK-SAME: (%arg0: !hc.undef) -> !hc.undef
  // CHECK-SAME: scope = <"WorkItem">
  // CHECK-SAME: effects = pure
  // CHECK-SAME: const_kwargs = ["arch"]
  // CHECK-SAME: parameters = ["group", "arch"]
  // CHECK-SAME: keyword_only = ["arch"]
  // CHECK-NEXT: }
  // CHECK-NOT: hc_front.
  hc_front.intrinsic "intr" attributes {
    const_kwargs = ["arch"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "group", passing = "positional"},
      {name = "arch", passing = "keyword_only"}
    ],
    scope = "WorkItem"
  } {
    %zero = hc_front.constant<0 : i64>
    %one = hc_front.constant<1 : i64>
    %sum = hc_front.binop "Add"(%zero, %one)
    hc_front.return %sum
  }

  // Target lowering recipes ride along as real `transform.named_sequence`
  // ops in a sibling top-level module. The conversion pass leaves that
  // module untouched (it's not an `hc_front.*` op), so verifying it is
  // present on the output is enough to prove the carry-through is wired.
  //
  // The body is intentionally empty here — handwritten IR may still tag
  // the intrinsic with metadata even when the matching named_sequence is
  // declared in a hand-rolled lowerings module elsewhere; we don't try to
  // generate one in basic round-trip fixtures.
  // CHECK-LABEL: hc.intrinsic @intr_with_recipes
  // CHECK-SAME: scope = <"WorkItem">
  // CHECK-NOT: lowering_recipes
  hc_front.intrinsic "intr_with_recipes" attributes {
    decorators = ["kernel.intrinsic"],
    parameters = [{name = "group", passing = "positional"}],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  // The frontend emits operand-less `hc_front.return` for Python `return None`
  // in kernels; conversion must keep the resulting `hc.return` operand-less.
  // CHECK-LABEL: hc.kernel @return_none
  // CHECK: hc.return
  // CHECK-NOT: hc.return %
  hc_front.kernel "return_none" attributes {
    parameters = [{name = "group"}],
    returns = "None"
  } {
    hc_front.return
  }

  // CHECK-LABEL: hc.kernel @typed_buffer_param
  // CHECK-SAME: (%{{.*}}: !hc.buffer<f32, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_x*i0">>>)
  hc_front.kernel "typed_buffer_param" attributes {
    parameters = [
      {dtype = "float32", kind = "buffer", name = "x", shape = ["M"]}
    ]
  } {
    hc_front.return
  }

  // Launch-geometry types are synthesized during conversion, before
  // `-hc-infer-types`, and use `$` prefixes so they cannot collide with
  // Python-level symbols.
  // CHECK-LABEL: hc.kernel @launch_geo_symbols
  // CHECK: %[[GIDV:.*]] = hc.group_id %arg0 : (!hc.group) -> !hc.idx<"$WG0">
  // CHECK: %[[GIDT:.*]] = hc.tuple(%[[GIDV]])
  // CHECK: hc.getitem %[[GIDT]]
  // CHECK: %[[LIDV:.*]]:2 = hc.local_id %arg0 : (!hc.group)
  // CHECK: %[[LIDT:.*]] = hc.tuple(%[[LIDV]]#0
  // CHECK: hc.getitem %[[LIDT]]
  // CHECK: %[[SGV:.*]] = hc.subgroup_id %arg0 : (!hc.group) -> !hc.idx<"$SG0">
  // CHECK: %[[SGT:.*]] = hc.tuple(%[[SGV]])
  // CHECK: hc.getitem %[[SGT]]
  // CHECK: %[[GSHV:.*]] = hc.group_shape %arg0 : (!hc.group) -> !hc.idx<"$WGS0">
  // CHECK: %[[GSHT:.*]] = hc.tuple(%[[GSHV]])
  // CHECK: hc.getitem %[[GSHT]]
  // CHECK: %[[WOV:.*]] = hc.work_offset %arg0 : (!hc.group) -> !hc.idx<"$WO0">
  // CHECK: %[[WOT:.*]] = hc.tuple(%[[WOV]])
  // CHECK: hc.getitem %[[WOT]]
  // CHECK: %[[WSV:.*]] = hc.work_shape %arg0 : (!hc.group) -> !hc.idx<"$WS0">
  // CHECK: %[[WST:.*]] = hc.tuple(%[[WSV]])
  // CHECK: hc.getitem %[[WST]]
  // CHECK: hc.group_size %arg0 : (!hc.group) -> !hc.idx<"$GSZ0">
  // CHECK: hc.wave_size %arg0 : (!hc.group) -> !hc.idx<"$WV0">
  hc_front.kernel "launch_geo_symbols" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>

    %gid_attr = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid0 = hc_front.subscript %gid_attr[%ax0]

    %lid_attr = hc_front.attr %grp, "local_id" {ref = {kind = "dsl_method", method = "local_id"}}
    %lid1 = hc_front.subscript %lid_attr[%ax1]

    %sg_attr = hc_front.attr %grp, "subgroup_id" {ref = {kind = "dsl_method", method = "subgroup_id"}}
    %sg0 = hc_front.subscript %sg_attr[%ax0]

    %gsh_attr = hc_front.attr %grp, "group_shape" {ref = {kind = "dsl_method", method = "group_shape"}}
    %gsh0 = hc_front.subscript %gsh_attr[%ax0]

    %wo_attr = hc_front.attr %grp, "work_offset" {ref = {kind = "dsl_method", method = "work_offset"}}
    %wo0 = hc_front.subscript %wo_attr[%ax0]

    %ws_attr = hc_front.attr %grp, "work_shape" {ref = {kind = "dsl_method", method = "work_shape"}}
    %ws0 = hc_front.subscript %ws_attr[%ax0]

    %gsz_attr = hc_front.attr %grp, "group_size" {ref = {kind = "dsl_method", method = "group_size"}}
    %gsz = hc_front.call %gsz_attr()

    %wv_attr = hc_front.attr %grp, "wave_size" {ref = {kind = "dsl_method", method = "wave_size"}}
    %wv = hc_front.call %wv_attr()

    hc_front.return
  }

  // Python-side `CurrentGroup.shape` exposes the launch-geo
  // `group_shape` getter under the short attribute alias `shape`,
  // which collides by name with buffer / tensor `.shape`. The
  // dispatch resolves `group.shape[N]` to the launch-geo tuple +
  // `hc.getitem` path (matching `group.group_id[N]` etc.) and only
  // falls back to `hc.buffer_dim` when the base isn't a launch
  // context handle — see `axis_bounds` below for the buffer side.
  // CHECK-LABEL: hc.kernel @group_shape_alias
  // CHECK: %[[GSV:.*]] = hc.group_shape %arg0 : (!hc.group) -> !hc.idx<"$WGS0">
  // CHECK: %[[GST:.*]] = hc.tuple(%[[GSV]])
  // CHECK: hc.getitem %[[GST]]
  hc_front.kernel "group_shape_alias" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %sh_attr = hc_front.attr %grp, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %sh0 = hc_front.subscript %sh_attr[%ax0]
    %t_sh = hc_front.target_name "sh0"
    hc_front.assign %t_sh = %sh0
    hc_front.return
  }

  // Explicit `kind = "launch_context"` parameter under a non-"group"
  // name (Python-side `def k(g: CurrentGroup, ...)` emits this
  // shape). Pins the kind-attr branch of `isLaunchContextFrontParam`
  // separately from the implicit "group" rule covered above.
  // CHECK-LABEL: hc.kernel @group_shape_alias_explicit_kind
  // CHECK: %[[GSV:.*]] = hc.group_shape %arg0 : (!hc.group) -> !hc.idx<"$WGS0">
  // CHECK: %[[GST:.*]] = hc.tuple(%[[GSV]])
  // CHECK: hc.getitem %[[GST]]
  hc_front.kernel "group_shape_alias_explicit_kind" attributes {
    parameters = [{kind = "launch_context", launch_context = "group", name = "g"}]
  } {
    %grp = hc_front.name "g" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %sh_attr = hc_front.attr %grp, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %sh0 = hc_front.subscript %sh_attr[%ax0]
    %t_sh = hc_front.target_name "sh0"
    hc_front.assign %t_sh = %sh0
    hc_front.return
  }

  // Local-bound alias: `g = group.shape; g[N]` should hit the same
  // launch-geo tuple + getitem path as the inline `group.shape[N]`
  // form — `trySubscriptFolds` traces the local name back to the
  // attr at its source.
  // CHECK-LABEL: hc.kernel @group_shape_alias_local
  // CHECK: %[[GSV:.*]]:2 = hc.group_shape %arg0
  // CHECK: %[[GST:.*]] = hc.tuple(%[[GSV]]#0, %[[GSV]]#1)
  // CHECK: hc.getitem %[[GST]]
  // CHECK: hc.getitem %[[GST]]
  hc_front.kernel "group_shape_alias_local" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>
    %sh_attr = hc_front.attr %grp, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %t_g = hc_front.target_name "g"
    hc_front.assign %t_g = %sh_attr
    %g = hc_front.name "g" {ctx = "load", ref = {kind = "local"}}
    %g0 = hc_front.subscript %g[%ax0]
    %g1 = hc_front.subscript %g[%ax1]
    %t_g0 = hc_front.target_name "g0"
    hc_front.assign %t_g0 = %g0
    %t_g1 = hc_front.target_name "g1"
    hc_front.assign %t_g1 = %g1
    hc_front.return
  }

  // Boundary coverage:
  //   * launch-geo at axis=31 (one below the pass-internal cap) must lower
  //     cleanly — regression guard on the launch-geo bounds check.
  //   * `a.shape[200]` must pass through as `hc.buffer_dim` with no axis
  //     cap — buffer rank is not launch-geo and has its own verifier.
  // CHECK-LABEL: hc.kernel @axis_bounds
  // The `:32` pins the required hc.local_id result arity to the largest
  // accepted static axis plus one, which is also kMaxLaunchAxis here. It is
  // not a bit-width.
  // CHECK: %{{.*}}:32 = hc.local_id %arg0
  // CHECK-SAME: !hc.idx<"$WI31">
  // CHECK: hc.tuple
  // CHECK: hc.getitem
  // CHECK: hc.buffer_dim %arg1, axis = 200 : !hc.undef
  hc_front.kernel "axis_bounds" attributes {
    parameters = [{name = "group"}, {name = "a"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %ax31 = hc_front.constant<31 : i64>
    %lid_attr = hc_front.attr %grp, "local_id" {ref = {kind = "dsl_method", method = "local_id"}}
    %lid = hc_front.subscript %lid_attr[%ax31]
    %t_lid = hc_front.target_name "t_lid"
    hc_front.assign %t_lid = %lid

    %ax200 = hc_front.constant<200 : i64>
    %sh = hc_front.attr %a, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %dim = hc_front.subscript %sh[%ax200]
    %t_dim = hc_front.target_name "t_dim"
    hc_front.assign %t_dim = %dim

    hc_front.return
  }

  // CHECK-LABEL: hc.kernel @unknown_rank_static_launch_axes
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CHECK: hc.getitem
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CHECK: hc.getitem
  // CHECK: %{{.*}}:2 = hc.local_id %arg0 : (!hc.group) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
  // CHECK: hc.getitem
  // CSE-LABEL: hc.kernel @unknown_rank_static_launch_axes
  // CSE: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CSE-NOT: hc.group_id
  // CSE: %{{.*}}:2 = hc.local_id %arg0 : (!hc.group) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
  hc_front.kernel "unknown_rank_static_launch_axes" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>
    %gid_attr0 = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid0 = hc_front.subscript %gid_attr0[%ax0]
    %gid_t0 = hc_front.target_name "gid0"
    hc_front.assign %gid_t0 = %gid0
    %gid_attr1 = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid1 = hc_front.subscript %gid_attr1[%ax1]
    %gid_t1 = hc_front.target_name "gid1"
    hc_front.assign %gid_t1 = %gid1
    %lid_attr = hc_front.attr %grp, "local_id" {ref = {kind = "dsl_method", method = "local_id"}}
    %lid_tuple = hc_front.call %lid_attr()
    %lid1 = hc_front.subscript %lid_tuple[%ax1]
    %lid_t1 = hc_front.target_name "lid1"
    hc_front.assign %lid_t1 = %lid1
    hc_front.return
  }

  // CHECK-LABEL: hc.kernel @launch_geo_full_rank_cse
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CHECK: hc.tuple
  // CHECK: hc.getitem
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CHECK: hc.tuple
  // CHECK: hc.getitem
  // CSE-LABEL: hc.kernel @launch_geo_full_rank_cse
  // CSE: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CSE: hc.tuple
  // CSE: hc.getitem
  // CSE: hc.getitem
  // CSE-NOT: hc.group_id
  hc_front.kernel "launch_geo_full_rank_cse" attributes {
    group_shape = ["16", "8"],
    parameters = [{name = "group"}, {name = "out"}],
    returns = "None",
    work_shape = ["M", "N"]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.name "out" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>
    %v0 = hc_front.constant<0 : i64>
    %v1 = hc_front.constant<1 : i64>
    %gid_attr0 = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid0 = hc_front.subscript %gid_attr0[%ax0]
    %out_idx0 = hc_front.target_subscript %out[%gid0]
    hc_front.assign %out_idx0 = %v0
    %gid_attr1 = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid1 = hc_front.subscript %gid_attr1[%ax1]
    %out_idx1 = hc_front.target_subscript %out[%gid1]
    hc_front.assign %out_idx1 = %v1
    hc_front.return
  }

  // When the kernel's launch context carries no static shape and the
  // launch-geo getter is bound to a local, the static-rank pre-walk
  // must still trace the local through to its assignment so the emitted
  // launch-geo op stays at the rank the subscripts actually demand —
  // otherwise we fall back to the conservative `kMaxLaunchAxis` cap and
  // emit a far wider op than needed.
  // CHECK-LABEL: hc.kernel @launch_geo_local_no_static_shape
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CHECK: hc.tuple
  // CHECK: hc.getitem
  // CHECK: hc.getitem
  // CSE-LABEL: hc.kernel @launch_geo_local_no_static_shape
  // CSE: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CSE-NOT: hc.group_id
  hc_front.kernel "launch_geo_local_no_static_shape" attributes {
    parameters = [{name = "group"}, {name = "out"}],
    returns = "None"
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.name "out" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>
    %v0 = hc_front.constant<0 : i64>
    %v1 = hc_front.constant<1 : i64>
    %gid_attr = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %gid_t = hc_front.target_name "gid"
    hc_front.assign %gid_t = %gid_attr
    %gid = hc_front.name "gid" {ctx = "load", ref = {kind = "local"}}
    %gid0 = hc_front.subscript %gid[%ax0]
    %slot0 = hc_front.target_subscript %out[%gid0]
    hc_front.assign %slot0 = %v0
    %gid_again = hc_front.name "gid" {ctx = "load", ref = {kind = "local"}}
    %gid1 = hc_front.subscript %gid_again[%ax1]
    %slot1 = hc_front.target_subscript %out[%gid1]
    hc_front.assign %slot1 = %v1
    hc_front.return
  }

  // Bare launch-geometry getters (no immediate subscript or call) must
  // materialize the multi-axis tuple as an SSA value so a Python local
  // can name them and downstream peeling still lowers:
  //   ```
  //   gid = group.work_offset
  //   out[gid[0]] = 0
  //   out[gid[1]] = 1
  //   ```
  // After `--hc-promote-names` the dead `hc.assign "gid"` is folded
  // away — what stays is one `hc.work_offset`, one tuple wrap, and one
  // `hc.getitem` per subscript. CSE then dedups the tuple wrap.
  // CHECK-LABEL: hc.kernel @launch_geo_bound_to_local
  // CHECK: %[[WO:.*]]:2 = hc.work_offset %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>>) -> (!hc.idx<"$WO0">, !hc.idx<"$WO1">)
  // CHECK: %[[WOT:.*]] = hc.tuple(%[[WO]]#0, %[[WO]]#1) : (!hc.idx<"$WO0">, !hc.idx<"$WO1">) -> tuple<!hc.idx<"$WO0">, !hc.idx<"$WO1">>
  // CHECK: hc.getitem %[[WOT]]
  // CHECK: hc.getitem %[[WOT]]
  // CSE-LABEL: hc.kernel @launch_geo_bound_to_local
  // CSE: %{{.*}}:2 = hc.work_offset
  // CSE-NOT: hc.work_offset
  hc_front.kernel "launch_geo_bound_to_local" attributes {
    parameters = [{name = "group"}, {name = "out"}],
    returns = "None",
    work_shape = ["M", "N"]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.name "out" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>
    %v0 = hc_front.constant<0 : i64>
    %v1 = hc_front.constant<1 : i64>
    %wo_attr = hc_front.attr %grp, "work_offset" {ref = {kind = "dsl_method", method = "work_offset"}}
    %wo_t = hc_front.target_name "wo"
    hc_front.assign %wo_t = %wo_attr
    %wo = hc_front.name "wo" {ctx = "load", ref = {kind = "local"}}
    %wo0 = hc_front.subscript %wo[%ax0]
    %out_idx0 = hc_front.target_subscript %out[%wo0]
    hc_front.assign %out_idx0 = %v0
    %wo_again = hc_front.name "wo" {ctx = "load", ref = {kind = "local"}}
    %wo1 = hc_front.subscript %wo_again[%ax1]
    %out_idx1 = hc_front.target_subscript %out[%wo1]
    hc_front.assign %out_idx1 = %v1
    hc_front.return
  }

  // Scalar launch-geometry getter bound to a local. The bare attr
  // materializes the scalar `hc.group_size` result directly (no tuple
  // wrap); reading the local back as a store index keeps the def alive
  // through `--hc-promote-names`.
  // CHECK-LABEL: hc.kernel @launch_geo_scalar_bound_to_local
  // CHECK: %[[GSZ:.*]] = hc.group_size %arg0 : (!hc.group) -> !hc.idx<"$GSZ0">
  // CHECK: hc.store %arg1[%[[GSZ]]]
  hc_front.kernel "launch_geo_scalar_bound_to_local" attributes {
    parameters = [{name = "group"}, {name = "out"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.name "out" {ctx = "load", ref = {kind = "param"}}
    %v0 = hc_front.constant<0 : i64>
    %gsz_attr = hc_front.attr %grp, "group_size" {ref = {kind = "dsl_method", method = "group_size"}}
    %gsz_t = hc_front.target_name "sz"
    hc_front.assign %gsz_t = %gsz_attr
    %sz = hc_front.name "sz" {ctx = "load", ref = {kind = "local"}}
    %slot = hc_front.target_subscript %out[%sz]
    hc_front.assign %slot = %v0
    hc_front.return
  }

  // Call-style launch-geometry getter bound to a local. Mirrors the
  // property-style case above: `group.local_id()` lowers to one
  // launch-geo op whose tuple is shared by both subscripts on the
  // local through the same `hc.getitem` peel. The kernel's
  // `group_shape` attribute drives the static rank, so the launch-geo
  // op is rank-2 (not the conservative cap).
  // CHECK-LABEL: hc.kernel @launch_geo_call_bound_to_local
  // CHECK: %[[LID:.*]]:2 = hc.local_id %arg0 : (!hc.group<group_shape = #hc.shape<["G0", "G1"]>>) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
  // CHECK: %[[LIDT:.*]] = hc.tuple(%[[LID]]#0, %[[LID]]#1) : (!hc.idx<"$WI0">, !hc.idx<"$WI1">) -> tuple<!hc.idx<"$WI0">, !hc.idx<"$WI1">>
  // CHECK: hc.getitem %[[LIDT]]
  // CHECK: hc.getitem %[[LIDT]]
  hc_front.kernel "launch_geo_call_bound_to_local" attributes {
    group_shape = ["G0", "G1"],
    parameters = [{name = "group"}, {name = "out"}],
    returns = "None"
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.name "out" {ctx = "load", ref = {kind = "param"}}
    %ax0 = hc_front.constant<0 : i64>
    %ax1 = hc_front.constant<1 : i64>
    %v0 = hc_front.constant<0 : i64>
    %v1 = hc_front.constant<1 : i64>
    %lid_attr = hc_front.attr %grp, "local_id" {ref = {kind = "dsl_method", method = "local_id"}}
    %lid_call = hc_front.call %lid_attr()
    %lid_t = hc_front.target_name "lid"
    hc_front.assign %lid_t = %lid_call
    %lid = hc_front.name "lid" {ctx = "load", ref = {kind = "local"}}
    %lid0 = hc_front.subscript %lid[%ax0]
    %slot0 = hc_front.target_subscript %out[%lid0]
    hc_front.assign %slot0 = %v0
    %lid_again = hc_front.name "lid" {ctx = "load", ref = {kind = "local"}}
    %lid1 = hc_front.subscript %lid_again[%ax1]
    %slot1 = hc_front.target_subscript %out[%lid1]
    hc_front.assign %slot1 = %v1
    hc_front.return
  }

  // CHECK-LABEL: hc.kernel @zero_rank_buffer_param
  // Rank-0 buffers still get the layout slot: empty shape_syms /
  // index_syms, literal `0` offset and storage_size — the structural
  // shell stays uniform across ranks.
  // CHECK-SAME: (%{{.*}}: !hc.buffer<!hc.undef, [], <shape_syms = [], index_syms = [], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"0">>>)
  hc_front.kernel "zero_rank_buffer_param" attributes {
    parameters = [
      {annotation = "Buffer[()]", kind = "buffer", name = "cell", shape = []}
    ],
    returns = "None"
  } {
    hc_front.return
  }

  // `x ** K` for positive integer-literal K lowers to a binary-squaring
  // chain of `hc.mul`s. The exponent must be `hc_front.constant<K : i64>`
  // with K >= 1; any other rhs is rejected (see `invalid.mlir`). Every
  // other binop kind in `emitBinop` produces a single `hc` op; this is
  // the only family that fans out.

  // K = 2: just `x * x`, one mul.
  // CHECK-LABEL: hc.func @binop_pow_squared
  // CHECK: hc.mul %arg0, %arg0 : (!hc.undef, !hc.undef) -> !hc.undef
  // CHECK-NOT: hc.mul
  // CHECK: hc.return
  hc_front.func "binop_pow_squared" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "x"}],
    scope = "WorkGroup"
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %e = hc_front.constant<2 : i64>
    %r = hc_front.binop "Pow"(%x, %e)
    hc_front.return %r
  }

  // K = 3: square then multiply by `x`, two muls.
  // CHECK-LABEL: hc.func @binop_pow_cubed
  // CHECK: %[[SQ:.*]] = hc.mul %arg0, %arg0
  // CHECK: hc.mul %[[SQ]], %arg0
  // CHECK-NOT: hc.mul
  // CHECK: hc.return
  hc_front.func "binop_pow_cubed" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "x"}],
    scope = "WorkGroup"
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %e = hc_front.constant<3 : i64>
    %r = hc_front.binop "Pow"(%x, %e)
    hc_front.return %r
  }

  // K = 4: square-then-square — still two muls, beats the naive K-1 = 3.
  // CHECK-LABEL: hc.func @binop_pow_quartic
  // CHECK: %[[SQ:.*]] = hc.mul %arg0, %arg0
  // CHECK: hc.mul %[[SQ]], %[[SQ]]
  // CHECK-NOT: hc.mul
  // CHECK: hc.return
  hc_front.func "binop_pow_quartic" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "x"}],
    scope = "WorkGroup"
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %e = hc_front.constant<4 : i64>
    %r = hc_front.binop "Pow"(%x, %e)
    hc_front.return %r
  }

  // K = 7 (0b111): exercises the full bit walk — square (^2), *x (^3),
  // square (^6), *x (^7). Four muls instead of the naive six.
  // CHECK-LABEL: hc.func @binop_pow_seventh
  // CHECK: %[[K2:.*]] = hc.mul %arg0, %arg0
  // CHECK: %[[K3:.*]] = hc.mul %[[K2]], %arg0
  // CHECK: %[[K6:.*]] = hc.mul %[[K3]], %[[K3]]
  // CHECK: hc.mul %[[K6]], %arg0
  // CHECK-NOT: hc.mul
  // CHECK: hc.return
  hc_front.func "binop_pow_seventh" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "x"}],
    scope = "WorkGroup"
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %e = hc_front.constant<7 : i64>
    %r = hc_front.binop "Pow"(%x, %e)
    hc_front.return %r
  }

  // K = 1: trivial identity, no muls at all.
  // CHECK-LABEL: hc.func @binop_pow_one
  // CHECK-NOT: hc.mul
  // CHECK: hc.return %arg0
  hc_front.func "binop_pow_one" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "x"}],
    scope = "WorkGroup"
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %e = hc_front.constant<1 : i64>
    %r = hc_front.binop "Pow"(%x, %e)
    hc_front.return %r
  }
}
