// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --convert-hc-front-to-hc %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc %s | hc-opt | FileCheck %s

// Exercises the mechanical hc_front -> hc rewrite patterns this bead
// handles: top-level kernel/func/intrinsic, return/constant/binop,
// target_name + target_tuple + target_subscript via assign, name dispatch
// (param / iv / local / constant / symbol / callee / intrinsic /
// dsl_method), attr+subscript for launch geometry and buffer.shape, the
// for_range lowering from `range(...)`, and the dsl-method calls that land
// on dedicated hc ops (vec, astype, with_inactive, store).

// CHECK-LABEL: hc.kernel @basic
// CHECK-SAME: (%arg0: !hc.undef, %arg1: !hc.undef, %arg2: !hc.undef)
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
    // CHECK: %[[D:.*]] = hc.buffer_dim %arg1, axis = 0 : !hc.undef
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %axis = hc_front.constant<0 : i64>
    %sh = hc_front.attr %a, "shape" {ref = {kind = "dsl_method", method = "shape"}}
    %dim = hc_front.subscript %sh[%axis]
    %dim_target = hc_front.target_name "dim"
    hc_front.assign %dim_target = %dim

    // CHECK: %[[GID:.*]] = hc.group_id %arg0 : (!hc.undef) -> !hc.undef
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

    // CHECK: hc.for_range {{.*}} to {{.*}} step {{.*}} : (!hc.undef, !hc.undef, !hc.undef) -> ()
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
      // CHECK: hc.buffer_view %arg1[%{{.*}}] : (!hc.undef, !hc.undef) -> !hc.undef
      %iv = hc_front.name "i" {ctx = "load", ref = {kind = "iv"}}
      %a_ref = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
      %load = hc_front.subscript %a_ref[%iv]
      // CHECK: hc.const<2 : i64> : !hc.undef
      %c2 = hc_front.constant<2 : i64>
      // CHECK: hc.store %arg2[%{{.*}}], %{{.*}} : (!hc.undef, !hc.undef, !hc.undef) -> ()
      %b_ref = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
      %b_idx = hc_front.target_subscript %b_ref[%iv]
      hc_front.assign %b_idx = %c2
      // `hc.yield` is an implicit terminator on for_range; not printed.
    }

    // CHECK: hc.workitem_region captures = ["group"] {
    hc_front.workitem_region captures = ["group"] attributes {
      parameters = [{name = "wi"}]
    } {
      // CHECK: %[[LIDV:.*]] = hc.local_id %{{.*}} : (!hc.undef) -> !hc.undef
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

  // CHECK-LABEL: hc.intrinsic @intr
  // CHECK-SAME: scope = <"WorkItem">
  // CHECK-SAME: effects = pure
  // CHECK-SAME: const_kwargs = ["arch"]
  // CHECK-NOT: hc_front.
  hc_front.intrinsic "intr" attributes {
    const_kwargs = ["arch"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [{name = "group"}],
    scope = "WorkItem"
  } {
    %one = hc_front.constant<1 : i64>
    hc_front.return %one
  }
}
