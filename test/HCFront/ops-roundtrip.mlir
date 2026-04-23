// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK: module {
// CHECK: hc_front.kernel "kernel_demo" attributes {decorators = ["kernel"], parameters = [{name = "group"}], returns = "None"} {
// CHECK: hc_front.constant<0 : i64> : !hc_front.value
// CHECK: hc_front.constant<1 : i64> : !hc_front.value
// CHECK: hc_front.name "x" {ctx = "load"} : !hc_front.value
// CHECK: hc_front.target_name "lhs" : !hc_front.value
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
// CHECK: hc_front.subgroup_region captures = ["outer"] {
// CHECK: hc_front.workitem_region captures = ["outer", "tmp"] {
// CHECK: hc_front.func "helper" attributes {decorators = ["kernel.func"], parameters = [{name = "group"}, {name = "x"}]} {
// CHECK: hc_front.intrinsic "intrinsic_demo" attributes {decorators = ["kernel.intrinsic"], parameters = [{name = "group"}]} {

module {
  hc_front.kernel "kernel_demo" attributes {decorators = ["kernel"], parameters = [{name = "group"}], returns = "None"} {
    %c0 = hc_front.constant <0> : !hc_front.value
    %c1 = hc_front.constant <1> : !hc_front.value
    %name = hc_front.name "x" {ctx = "load"} : !hc_front.value
    %target = hc_front.target_name "lhs" : !hc_front.value
    hc_front.assign %target = %c1 : (!hc_front.value, !hc_front.value) -> ()
    hc_front.aug_assign "Add" %target = %c1 : (!hc_front.value, !hc_front.value) -> ()
    %target_tuple = hc_front.target_tuple (%target, %target) : (!hc_front.value, !hc_front.value) -> !hc_front.value
    %target_subscript = hc_front.target_subscript %name[%c0, %c1] : (!hc_front.value, !hc_front.value, !hc_front.value) -> !hc_front.value
    %attr = hc_front.attr %name, "shape" : (!hc_front.value) -> !hc_front.value
    %slice = hc_front.slice (%c0, %c1) {has_lower = true, has_upper = true, has_step = false} : (!hc_front.value, !hc_front.value) -> !hc_front.value
    %subscript = hc_front.subscript %name[%slice, %c0] : (!hc_front.value, !hc_front.value, !hc_front.value) -> !hc_front.value
    %keyword = hc_front.keyword "shape" = %c1 : (!hc_front.value) -> !hc_front.value
    %tuple = hc_front.tuple (%name, %c1) : (!hc_front.value, !hc_front.value) -> !hc_front.value
    %list = hc_front.list [%name, %c1] : (!hc_front.value, !hc_front.value) -> !hc_front.value
    %binop = hc_front.binop "Add"(%name, %c1) : (!hc_front.value, !hc_front.value) -> !hc_front.value
    %unary = hc_front.unaryop "USub"(%c1) : (!hc_front.value) -> !hc_front.value
    %compare = hc_front.compare ["Lt"](%c0, %c1) : (!hc_front.value, !hc_front.value) -> !hc_front.value
    %call = hc_front.call %name(%tuple, %keyword) : (!hc_front.value, !hc_front.value, !hc_front.value) -> !hc_front.value
    hc_front.if {
      %cond = hc_front.name "pred" : !hc_front.value
    } then {
      hc_front.return %call : !hc_front.value
    } else {
      %fallback = hc_front.list [] : () -> !hc_front.value
    } {has_orelse = true}
    hc_front.for {
      %i = hc_front.target_name "i" : !hc_front.value
    } in {
      %rows = hc_front.name "rows" : !hc_front.value
    } do {
      %item = hc_front.subscript %list[%c0] : (!hc_front.value, !hc_front.value) -> !hc_front.value
      hc_front.return %item : !hc_front.value
    }
    hc_front.subgroup_region captures = ["outer"] {
      %sg = hc_front.name "sg" : !hc_front.value
    }
    hc_front.workitem_region captures = ["outer", "tmp"] {
      %wi = hc_front.name "wi" : !hc_front.value
    }
    hc_front.return %call : !hc_front.value
  }

  hc_front.func "helper" attributes {decorators = ["kernel.func"], parameters = [{name = "group"}, {name = "x"}]} {
    %x = hc_front.name "x" : !hc_front.value
    hc_front.return %x : !hc_front.value
  }

  hc_front.intrinsic "intrinsic_demo" attributes {decorators = ["kernel.intrinsic"], parameters = [{name = "group"}]} {
    %one = hc_front.constant <1> : !hc_front.value
    hc_front.return %one : !hc_front.value
  }
}
