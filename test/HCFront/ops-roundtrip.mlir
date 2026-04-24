// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK: module {
// CHECK: hc_front.kernel "kernel_demo"
// CHECK-SAME: decorators = ["kernel"]
// CHECK-SAME: group_shape = ["32", "1"]
// CHECK-SAME: literals = ["WAVE_LANES", "WMMA_M"]
// CHECK-SAME: parameters =
// CHECK-SAME: {annotation = "Buffer[M, K]", kind = "buffer", name = "a", shape = ["M", "K"]}
// CHECK-SAME: {dtype = "int", kind = "scalar", name = "x"}
// CHECK-SAME: returns = "None"
// CHECK-SAME: subgroup_size = 32 : i32
// CHECK-SAME: work_shape = ["32*ceiling(1/16*M)", "ceiling(1/16*N)"]
// CHECK: hc_front.constant<0 : i64>
// CHECK: hc_front.constant<1 : i64>
// CHECK: hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
// CHECK: hc_front.name "k"
// CHECK-SAME: ref = {kind = "iv"}
// CHECK: hc_front.name "tmp"
// CHECK-SAME: ref = {callee = "@foo", kind = "callee", scope = "WorkGroup"}
// CHECK: hc_front.target_name "lhs"
// CHECK: hc_front.assign
// CHECK: hc_front.aug_assign "Add"
// CHECK: hc_front.target_tuple
// CHECK: hc_front.target_subscript
// CHECK: hc_front.attr
// CHECK: hc_front.slice
// CHECK: has_lower = true
// CHECK: has_step = false
// CHECK: has_upper = true
// CHECK: hc_front.subscript
// CHECK: hc_front.keyword "shape" =
// CHECK: hc_front.tuple
// CHECK: hc_front.list
// CHECK: hc_front.binop "Add"
// CHECK: hc_front.unaryop "USub"
// CHECK: hc_front.compare ["Lt"]
// CHECK: hc_front.call
// CHECK: hc_front.if
// CHECK: has_orelse = true
// CHECK: hc_front.for
// CHECK: hc_front.subgroup_region captures = ["outer"]
// CHECK-SAME: name = "sg_wave"
// CHECK: hc_front.workitem_region captures = ["outer", "tmp"]
// CHECK-SAME: name = "wi_wave"
// CHECK: hc_front.func "helper"
// CHECK-SAME: decorators = ["kernel.func"]
// CHECK-SAME: scope = "WorkGroup"
// CHECK: hc_front.intrinsic "intrinsic_demo"
// CHECK-SAME: const_kwargs = ["arch", "wave_size"]
// CHECK-SAME: decorators = ["kernel.intrinsic"]
// CHECK-SAME: effects = "pure"
// CHECK-SAME: scope = "WorkItem"
// CHECK: hc_front.func "inline_helper"
// CHECK-SAME: ref = {kind = "inline"
// CHECK: hc_front.inlined_region "inline_helper"

module {
  hc_front.kernel "kernel_demo" attributes {
    decorators = ["kernel"],
    group_shape = ["32", "1"],
    literals = ["WAVE_LANES", "WMMA_M"],
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M, K]", kind = "buffer", name = "a", shape = ["M", "K"]},
      {dtype = "int", kind = "scalar", name = "x"}
    ],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["32*ceiling(1/16*M)", "ceiling(1/16*N)"]
  } {
    %c0 = hc_front.constant <0>
    %c1 = hc_front.constant <1>
    %name = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %iv = hc_front.name "k" {ctx = "load", ref = {kind = "iv"}}
    %callee = hc_front.name "tmp" {ctx = "load", ref = {kind = "callee", callee = "@foo", scope = "WorkGroup"}}
    %target = hc_front.target_name "lhs"
    hc_front.assign %target = %c1
    hc_front.aug_assign "Add" %target = %c1
    %target_tuple = hc_front.target_tuple (%target, %target)
    %target_subscript = hc_front.target_subscript %name[%c0, %c1]
    %attr = hc_front.attr %name, "shape"
    %slice = hc_front.slice (%c0, %c1) {has_lower = true, has_upper = true, has_step = false}
    %subscript = hc_front.subscript %name[%slice, %c0]
    %keyword = hc_front.keyword "shape" = %c1
    %tuple = hc_front.tuple (%name, %c1)
    %list = hc_front.list [%name, %c1]
    %binop = hc_front.binop "Add"(%name, %c1)
    %unary = hc_front.unaryop "USub"(%c1)
    %compare = hc_front.compare ["Lt"](%c0, %c1)
    %call = hc_front.call %name(%tuple, %keyword)
    hc_front.if {
      %cond = hc_front.name "pred"
    } then {
      hc_front.return %call
    } else {
      %fallback = hc_front.list []
    } {has_orelse = true}
    hc_front.for {
      %i = hc_front.target_name "i"
    } in {
      %rows = hc_front.name "rows"
    } do {
      %item = hc_front.subscript %list[%c0]
      hc_front.return %item
    }
    hc_front.subgroup_region captures = ["outer"] attributes {name = "sg_wave"} {
      %sg = hc_front.name "sg"
    }
    hc_front.workitem_region captures = ["outer", "tmp"] attributes {name = "wi_wave"} {
      %wi = hc_front.name "wi"
    }
    hc_front.return %call
  }

  hc_front.func "helper" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "x"}],
    scope = "WorkGroup"
  } {
    %x = hc_front.name "x"
    hc_front.return %x
  }

  hc_front.intrinsic "intrinsic_demo" attributes {
    const_kwargs = ["arch", "wave_size"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [{name = "group"}],
    scope = "WorkItem"
  } {
    %one = hc_front.constant <1>
    hc_front.return %one
  }

  hc_front.func "inline_helper" attributes {
    parameters = [{name = "a"}, {name = "b"}],
    ref = {kind = "inline", qualified_name = "pkg.inline_helper"}
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    hc_front.return %a, %b
  }

  hc_front.func "inline_holder" attributes {
    parameters = [{name = "x"}]
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %c = hc_front.constant <1>
    %t:2 = hc_front.inlined_region "inline_helper"(%x, %c) -> (!hc_front.value, !hc_front.value) attributes {
      parameters = [{name = "a"}, {name = "b"}]
    } {
      %aa = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
      %bb = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
      hc_front.return %aa, %bb
    }
    hc_front.return %t#0
  }
}
