// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-inline-helpers -split-input-file %s | FileCheck %s

// CHECK-LABEL: hc.func @caller
// CHECK-NOT: hc.call
// CHECK: hc.add
// CHECK: hc.return
// CHECK-NOT: hc.func @add_one
hc.func @caller(%x: i32) -> i32 {
  %r = hc.call @add_one(%x) : (i32) -> i32
  hc.return %r : i32
}

hc.func @add_one(%x: i32) -> i32 {
  %one = hc.const<1 : i32> : i32
  %r = hc.add %x, %one : (i32, i32) -> i32
  hc.return %r : i32
}

// -----

// CHECK-LABEL: hc.func @multi_result_caller
// CHECK-NOT: hc.call
// CHECK: hc.return %{{.*}}, %{{.*}} : i32, f32
// CHECK-NOT: hc.func @pair
hc.func @multi_result_caller(%x: i32, %y: f32) -> (i32, f32) {
  %a, %b = hc.call @pair(%x, %y) : (i32, f32) -> (i32, f32)
  hc.return %a, %b : i32, f32
}

hc.func @pair(%x: i32, %y: f32) -> (i32, f32) {
  hc.return %x, %y : i32, f32
}

// -----

// CHECK-LABEL: hc.func @nested_caller
// CHECK-NOT: hc.call
// CHECK: hc.add
// CHECK: hc.mul
// CHECK-NOT: hc.func @add_two
// CHECK-NOT: hc.func @double
hc.func @nested_caller(%x: i32) -> i32 {
  %r = hc.call @add_two(%x) : (i32) -> i32
  hc.return %r : i32
}

hc.func @add_two(%x: i32) -> i32 {
  %one = hc.const<1 : i32> : i32
  %plus = hc.add %x, %one : (i32, i32) -> i32
  %r = hc.call @double(%plus) : (i32) -> i32
  hc.return %r : i32
}

hc.func @double(%x: i32) -> i32 {
  %r = hc.mul %x, %x : (i32, i32) -> i32
  hc.return %r : i32
}

// -----

// CHECK-LABEL: hc.func @side_effect_caller
// CHECK-NOT: hc.call
// CHECK: hc.store
// CHECK-NOT: hc.func @side_effect_helper
hc.func @side_effect_caller(%buf: !hc.buffer<f32, []>,
                            %v: !hc.bare_vector<f32, ["4"]>,
                            %m: !hc.bare_vector<!hc.pred, ["4"]>) {
  hc.call @side_effect_helper(%buf, %v, %m)
      : (!hc.buffer<f32, []>, !hc.bare_vector<f32, ["4"]>,
         !hc.bare_vector<!hc.pred, ["4"]>) -> ()
  hc.return
}

hc.func @side_effect_helper(%buf: !hc.buffer<f32, []>,
                            %v: !hc.bare_vector<f32, ["4"]>,
                            %m: !hc.bare_vector<!hc.pred, ["4"]>) {
  hc.store %buf[], %v, mask %m
      : (!hc.buffer<f32, []>, !hc.bare_vector<f32, ["4"]>,
         !hc.bare_vector<!hc.pred, ["4"]>) -> ()
  hc.return
}
