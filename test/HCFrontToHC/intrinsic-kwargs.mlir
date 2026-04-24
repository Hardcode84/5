// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `hc.call_intrinsic` lowering has to thread non-const keyword arguments
// into the operand list in the callee's declared parameter order while
// leaving declared `const_kwargs` behind as attributes. The bug this
// file regresses on had non-const kwargs silently dropped, producing
// an under-arity call site that blew up in the hc verifier.
// RUN: hc-opt --convert-hc-front-to-hc %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc %s | hc-opt | FileCheck %s

// CHECK-LABEL: hc.intrinsic @demo_intr
// Signature filters out `const_kwargs` names — they never ride the
// operand list, only the call-site attr dict, so including them in
// the signature would double-count and break verify.
// CHECK-SAME: %arg0: !hc.undef, %arg1: !hc.undef, %arg2: !hc.undef
// CHECK-SAME: const_kwargs = ["arch", "wave_size"]
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

  // Positional args flow straight through; `lane` is a non-const kwarg
  // so it lands as an SSA operand at the slot its name occupies in the
  // declared parameter list; `arch` and `wave_size` migrate to op
  // attributes. The resulting operand count matches the filtered
  // `hc.intrinsic @demo_intr` signature, which is the under-arity
  // failure mode this file guards against.
  // CHECK-LABEL: hc.func @caller
  // CHECK: %[[LANE:[0-9]+]] = hc.name_load "lane"
  // CHECK: hc.call_intrinsic @demo_intr(%{{[0-9]+}}, %{{[0-9]+}}, %[[LANE]])
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

  // Ordering robustness: declared-parameter order is what anchors the
  // operand list, so call-site keyword order (`kw_arch` before
  // `kw_lane`) must NOT reshuffle the operand list — the non-const
  // kwarg `lane` still lands in the third slot to match its position
  // in `demo_intr`'s declared parameter list.
  // CHECK-LABEL: hc.func @shuffled_kwargs
  // CHECK: %[[LANE2:[0-9]+]] = hc.name_load "lane"
  // CHECK: hc.call_intrinsic @demo_intr(%{{[0-9]+}}, %{{[0-9]+}}, %[[LANE2]])
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
}
