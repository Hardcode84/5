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
// CHECK-SAME: (%arg0: !hc.group<work_shape = #hc.shape<["M"]>, group_shape = #hc.shape<["32"]>, subgroup_size = 32 : i32>, %arg1: !hc.buffer<!hc.undef, ["M"]>, %arg2: !hc.buffer<!hc.undef, ["M"]>)
// CHECK-SAME: attributes {
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
    // CHECK: %[[D:.*]] = hc.buffer_dim %arg1, axis = 0 : !hc.buffer<!hc.undef, ["M"]>
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %axis = hc_front.constant<0 : i64>
    %sh = hc_front.attr %a, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %dim = hc_front.subscript %sh[%axis]
    %dim_target = hc_front.target_name "dim"
    hc_front.assign %dim_target = %dim

    // CHECK: %[[GID:.*]] = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M"]>, group_shape = #hc.shape<["32"]>, subgroup_size = 32 : i32>) -> !hc.idx<"$WG0">
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
      // CHECK: hc.buffer_view %arg1[%{{.*}}] : (!hc.buffer<!hc.undef, ["M"]>, !hc.undef) -> !hc.undef
      %iv = hc_front.name "i" {ctx = "load", ref = {kind = "iv"}}
      %a_ref = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
      %load = hc_front.subscript %a_ref[%iv]
      // CHECK: hc.const<2 : i64> : !hc.undef
      %c2 = hc_front.constant<2 : i64>
      // CHECK: hc.store %arg2[%{{.*}}], %{{.*}} : (!hc.buffer<!hc.undef, ["M"]>, !hc.undef, !hc.undef) -> ()
      %b_ref = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
      %b_idx = hc_front.target_subscript %b_ref[%iv]
      hc_front.assign %b_idx = %c2
      // `hc.yield` is an implicit terminator on for_range; not printed.
    }

    // CHECK: hc.workitem_region captures = ["group"] {
    hc_front.workitem_region captures = ["group"] attributes {
      parameters = [{name = "wi"}]
    } {
      // CHECK: %[[LIDV:.*]] = hc.local_id %{{.*}} : (!hc.undef) -> !hc.idx<"$WI0">
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
      parameters = [{name = "wi"}],
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
      parameters = [{name = "sg"}],
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
  // CHECK-SAME: (%{{.*}}: !hc.buffer<f32, ["M"]>)
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
  // CHECK: hc.group_id %arg0 : (!hc.group) -> !hc.idx<"$WG0">
  // CHECK: hc.local_id %arg0 : (!hc.group) -> (!hc.idx<"$WI0">, !hc.idx<"$WI1">)
  // CHECK: hc.subgroup_id %arg0 : (!hc.group) -> !hc.idx<"$SG0">
  // CHECK: hc.group_shape %arg0 : (!hc.group) -> !hc.idx<"$WGS0">
  // CHECK: hc.work_offset %arg0 : (!hc.group) -> !hc.idx<"$WO0">
  // CHECK: hc.work_shape %arg0 : (!hc.group) -> !hc.idx<"$WS0">
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

  // Boundary coverage:
  //   * launch-geo at axis=31 (one below the pass-internal cap) must lower
  //     cleanly — regression guard on the launch-geo bounds check.
  //   * `a.shape[200]` must pass through as `hc.buffer_dim` with no axis
  //     cap — buffer rank is not launch-geo and has its own verifier.
  // CHECK-LABEL: hc.kernel @axis_bounds
  // The `:32` pins the hc.local_id result arity to kMaxLaunchAxis (32
  // variadic results, one per potential launch axis), *not* a bit-width.
  // CHECK: %{{.*}}:32 = hc.local_id %arg0
  // CHECK-SAME: !hc.idx<"$WI31">
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

  // CHECK-LABEL: hc.kernel @launch_geo_full_rank_cse
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CHECK: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  // CSE-LABEL: hc.kernel @launch_geo_full_rank_cse
  // CSE: %{{.*}}:2 = hc.group_id %arg0 : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
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
}
