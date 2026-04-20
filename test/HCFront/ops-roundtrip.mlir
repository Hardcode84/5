// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK: module {
// CHECK: hc.front.kernel "kernel_demo" attributes {decorators = ["kernel"], parameters = [{name = "group"}], returns = "None"} {
// CHECK: hc.front.constant<0 : i64> : !hc.front.value
// CHECK: hc.front.constant<1 : i64> : !hc.front.value
// CHECK: hc.front.name "x" {ctx = "load"} : !hc.front.value
// CHECK: hc.front.target_name "lhs" : !hc.front.value
// CHECK: hc.front.assign
// CHECK: hc.front.aug_assign "Add"
// CHECK: hc.front.target_tuple
// CHECK: hc.front.target_subscript
// CHECK: hc.front.attr
// CHECK: hc.front.slice
// CHECK: has_lower = true
// CHECK: has_step = false
// CHECK: has_upper = true
// CHECK: hc.front.subscript
// CHECK: hc.front.keyword "shape" =
// CHECK: hc.front.tuple
// CHECK: hc.front.list
// CHECK: hc.front.binop "Add"
// CHECK: hc.front.unaryop "USub"
// CHECK: hc.front.compare ["Lt"]
// CHECK: hc.front.call
// CHECK: hc.front.if
// CHECK: has_orelse = true
// CHECK: hc.front.for
// CHECK: hc.front.subgroup_region captures = ["outer"] {
// CHECK: hc.front.workitem_region captures = ["outer", "tmp"] {
// CHECK: hc.front.func "helper" attributes {decorators = ["kernel.func"], parameters = [{name = "group"}, {name = "x"}]} {
// CHECK: hc.front.intrinsic "intrinsic_demo" attributes {decorators = ["kernel.intrinsic"], parameters = [{name = "group"}]} {

module {
  hc.front.kernel "kernel_demo" attributes {decorators = ["kernel"], parameters = [{name = "group"}], returns = "None"} {
    %c0 = hc.front.constant <0> : !hc.front.value
    %c1 = hc.front.constant <1> : !hc.front.value
    %name = hc.front.name "x" {ctx = "load"} : !hc.front.value
    %target = hc.front.target_name "lhs" : !hc.front.value
    hc.front.assign %target = %c1 : (!hc.front.value, !hc.front.value) -> ()
    hc.front.aug_assign "Add" %target = %c1 : (!hc.front.value, !hc.front.value) -> ()
    %target_tuple = hc.front.target_tuple (%target, %target) : (!hc.front.value, !hc.front.value) -> !hc.front.value
    %target_subscript = hc.front.target_subscript %name[%c0, %c1] : (!hc.front.value, !hc.front.value, !hc.front.value) -> !hc.front.value
    %attr = hc.front.attr %name, "shape" : (!hc.front.value) -> !hc.front.value
    %slice = hc.front.slice (%c0, %c1) {has_lower = true, has_upper = true, has_step = false} : (!hc.front.value, !hc.front.value) -> !hc.front.value
    %subscript = hc.front.subscript %name[%slice, %c0] : (!hc.front.value, !hc.front.value, !hc.front.value) -> !hc.front.value
    %keyword = hc.front.keyword "shape" = %c1 : (!hc.front.value) -> !hc.front.value
    %tuple = hc.front.tuple (%name, %c1) : (!hc.front.value, !hc.front.value) -> !hc.front.value
    %list = hc.front.list [%name, %c1] : (!hc.front.value, !hc.front.value) -> !hc.front.value
    %binop = hc.front.binop "Add"(%name, %c1) : (!hc.front.value, !hc.front.value) -> !hc.front.value
    %unary = hc.front.unaryop "USub"(%c1) : (!hc.front.value) -> !hc.front.value
    %compare = hc.front.compare ["Lt"](%c0, %c1) : (!hc.front.value, !hc.front.value) -> !hc.front.value
    %call = hc.front.call %name(%tuple, %keyword) : (!hc.front.value, !hc.front.value, !hc.front.value) -> !hc.front.value
    hc.front.if {
      %cond = hc.front.name "pred" : !hc.front.value
    } then {
      hc.front.return %call : !hc.front.value
    } else {
      %fallback = hc.front.list [] : () -> !hc.front.value
    } {has_orelse = true}
    hc.front.for {
      %i = hc.front.target_name "i" : !hc.front.value
    } in {
      %rows = hc.front.name "rows" : !hc.front.value
    } do {
      %item = hc.front.subscript %list[%c0] : (!hc.front.value, !hc.front.value) -> !hc.front.value
      hc.front.return %item : !hc.front.value
    }
    hc.front.subgroup_region captures = ["outer"] {
      %sg = hc.front.name "sg" : !hc.front.value
    }
    hc.front.workitem_region captures = ["outer", "tmp"] {
      %wi = hc.front.name "wi" : !hc.front.value
    }
    hc.front.return %call : !hc.front.value
  }

  hc.front.func "helper" attributes {decorators = ["kernel.func"], parameters = [{name = "group"}, {name = "x"}]} {
    %x = hc.front.name "x" : !hc.front.value
    hc.front.return %x : !hc.front.value
  }

  hc.front.intrinsic "intrinsic_demo" attributes {decorators = ["kernel.intrinsic"], parameters = [{name = "group"}]} {
    %one = hc.front.constant <1> : !hc.front.value
    hc.front.return %one : !hc.front.value
  }
}
