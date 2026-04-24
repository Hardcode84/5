// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Non-const keyword arguments on `hc_front.call` targeting an intrinsic
// must land as SSA operands on `hc.call_intrinsic` in the callee's
// declared parameter order, while declared `const_kwargs` are lifted
// to call-site attributes. This file is scoped to the rewrite the
// `--convert-hc-front-to-hc` pass performs; the sibling positive
// tests chain `--hc-promote-names` because they assert against the
// post-promotion IR, which would rewrite the `hc.name_load`s the
// checks below anchor on.
// RUN: hc-opt --convert-hc-front-to-hc %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc %s | hc-opt | FileCheck %s

// CHECK-LABEL: hc.intrinsic @demo_intr
// Signature filters out `const_kwargs` names — they never ride the
// operand list, only the call-site attr dict; keeping them in the
// signature would double-count the runtime arity.
// CHECK-SAME: %arg0: !hc.undef, %arg1: !hc.undef, %arg2: !hc.undef
// CHECK-SAME: const_kwargs = ["arch", "wave_size"]
// CHECK-SAME: parameters = ["group", "value", "lane", "wave_size", "arch"]
module {
  hc_front.intrinsic "demo_intr" attributes {
    const_kwargs = ["arch", "wave_size"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "group"},
      {name = "value"},
      {name = "lane"},
      {name = "wave_size"},
      {name = "arch"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  // Positionals flow straight through; `lane` is a non-const kwarg and
  // lands as an SSA operand at its declared slot; `arch` and `wave_size`
  // migrate to op attributes. The resulting operand count matches the
  // filtered signature on `hc.intrinsic @demo_intr`.
  // CHECK-LABEL: hc.func @caller
  // CHECK: %[[LANE:[^ ]+]] = hc.name_load "lane"
  // CHECK: hc.call_intrinsic @demo_intr(%[[GROUP:[^,]+]], %[[VALUE:[^,]+]], %[[LANE]])
  // CHECK-SAME: {arch = "gfx11", wave_size = 32 : i64}
  // CHECK-SAME: (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  hc_front.func "caller" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "value"}, {name = "lane"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "demo_intr" {ctx = "load", ref = {
      callee = "@demo_intr",
      const_kwargs = ["arch", "wave_size"],
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %val = hc_front.name "value" {ctx = "load", ref = {kind = "param"}}
    %lane = hc_front.name "lane" {ctx = "load", ref = {kind = "param"}}
    %kw_lane = hc_front.keyword "lane" = %lane
    %ws = hc_front.constant<32 : i64>
    %kw_ws = hc_front.keyword "wave_size" = %ws
    %arch = hc_front.constant<"gfx11">
    %kw_arch = hc_front.keyword "arch" = %arch
    %r = hc_front.call %intr(%grp, %val, %kw_lane, %kw_ws, %kw_arch)
    hc_front.return %r
  }

  // Declared-parameter order anchors the operand list, so call-site
  // keyword order (`kw_arch` before `kw_lane`) must not reshuffle it;
  // `lane` lands in the third slot because that is its position in
  // `demo_intr`'s declared parameters.
  // CHECK-LABEL: hc.func @shuffled_kwargs
  // CHECK: %[[LANE2:[^ ]+]] = hc.name_load "lane"
  // CHECK: hc.call_intrinsic @demo_intr(%[[GROUP2:[^,]+]], %[[VALUE2:[^,]+]], %[[LANE2]])
  // CHECK-SAME: {arch = "gfx11", wave_size = 32 : i64}
  hc_front.func "shuffled_kwargs" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "value"}, {name = "lane"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "demo_intr" {ctx = "load", ref = {
      callee = "@demo_intr",
      const_kwargs = ["arch", "wave_size"],
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %val = hc_front.name "value" {ctx = "load", ref = {kind = "param"}}
    %arch = hc_front.constant<"gfx11">
    %kw_arch = hc_front.keyword "arch" = %arch
    %ws = hc_front.constant<32 : i64>
    %kw_ws = hc_front.keyword "wave_size" = %ws
    %lane = hc_front.name "lane" {ctx = "load", ref = {kind = "param"}}
    %kw_lane = hc_front.keyword "lane" = %lane
    %r = hc_front.call %intr(%grp, %val, %kw_arch, %kw_ws, %kw_lane)
    hc_front.return %r
  }
// Intrinsic declarations are materialized before caller bodies are lowered,
// so source order does not decide whether call-site kwarg binding can see
// the callee's full `parameters` list.
// CHECK-LABEL: hc.func @caller_before_intrinsic
// CHECK: hc.call_intrinsic @late_intr(%[[GROUP3:[^,]+]], %[[LANE3:[^,)]+]])
// CHECK-SAME: {arch = "gfx11"}
// CHECK-LABEL: hc.intrinsic @late_intr
// CHECK-SAME: const_kwargs = ["arch"]
// CHECK-SAME: parameters = ["group", "lane", "arch"]
  hc_front.func "caller_before_intrinsic" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "lane"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "late_intr" {ctx = "load", ref = {
      callee = "@late_intr",
      const_kwargs = ["arch"],
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %lane = hc_front.name "lane" {ctx = "load", ref = {kind = "param"}}
    %kw_lane = hc_front.keyword "lane" = %lane
    %arch = hc_front.constant<"gfx11">
    %kw_arch = hc_front.keyword "arch" = %arch
    %r = hc_front.call %intr(%grp, %kw_lane, %kw_arch)
    hc_front.return %r
  }

  hc_front.intrinsic "late_intr" attributes {
    const_kwargs = ["arch"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [{name = "group"}, {name = "lane"}, {name = "arch"}],
    scope = "WorkItem"
  } {
    hc_front.return
  }
}
