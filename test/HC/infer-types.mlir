// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-infer-types -split-input-file %s | FileCheck %s

// CHECK-LABEL: hc.func @scalar_index_arithmetic
// CHECK: hc.const<2 : i64> : !hc.idx<"2">
// CHECK: hc.const<3 : i64> : !hc.idx<"3">
// CHECK: hc.const<7 : i64> : !hc.idx<"7">
// CHECK: hc.add {{.*}} -> !hc.idx<"5">
// CHECK: hc.cmp.lt {{.*}} -> !hc.pred<"True">
// CHECK: hc.slice_expr{{.*}} -> !hc.slice
hc.func @scalar_index_arithmetic {
  %two = hc.const<2 : i64> : !hc.undef
  %three = hc.const<3 : i64> : !hc.undef
  %seven = hc.const<7 : i64> : !hc.undef
  %sum = hc.add %two, %three : (!hc.undef, !hc.undef) -> !hc.undef
  %cmp = hc.cmp.lt %sum, %seven : (!hc.undef, !hc.undef) -> !hc.undef
  %slice = hc.slice_expr(lower = %sum upper = %seven)
      : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @for_range_iv
// CHECK: hc.for_range
// CHECK: ^bb0(%arg0: !hc.idx):
// CHECK: hc.const<1 : i64> : !hc.idx<"1">
// CHECK: hc.add %arg0, {{.*}} : (!hc.idx, !hc.idx<"1">) -> !hc.idx
hc.func @for_range_iv {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) -> () {
  ^bb0(%iv: !hc.undef):
    %one = hc.const<1 : i64> : !hc.undef
    %next = hc.add %iv, %one : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @region_branch_results
// CHECK: hc.for_range
// CHECK-SAME: -> !hc.idx
// CHECK: ^bb0(%{{[^:]+}}: !hc.idx, %{{[^:]+}}: !hc.idx):
// CHECK: hc.yield {{.*}} : !hc.idx
// CHECK: hc.if {{.*}} -> (!hc.idx) : i1
hc.func @region_branch_results(%cond: i1) {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  %init = hc.const<0 : i64> : !hc.undef
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
  ^bb0(%iv: !hc.undef, %acc: !hc.undef):
    %one = hc.const<1 : i64> : !hc.undef
    %next = hc.add %acc, %one : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %next : !hc.undef
  }
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.const<1 : i64> : !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.const<2 : i64> : !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @loads
// CHECK: hc.buffer_dim {{.*}} -> !hc.idx<"N">
// CHECK: hc.load {{.*}} -> !hc.tensor<f32, ["4", "8"]>
// CHECK: hc.vec {{.*}} -> !hc.vector<f32, ["4", "8"]>
hc.func @loads(%buf: !hc.buffer<f32, ["M", "N"]>, %i: !hc.idx<"0">,
               %j: !hc.idx<"1">) -> (!hc.undef, !hc.undef, !hc.undef) {
  %dim = hc.buffer_dim %buf, axis = 1
      : !hc.buffer<f32, ["M", "N"]> -> !hc.undef
  %t = hc.load %buf[%i, %j] {shape = #hc.shape<["4", "8"]>}
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"0">, !hc.idx<"1">)
        -> !hc.undef
  %v = hc.vec %t : !hc.undef -> !hc.undef
  hc.return %dim, %t, %v : !hc.undef, !hc.undef, !hc.undef
}

// -----

// CHECK-LABEL: hc.func @mixed_idx_and_builtin_stays_unknown
// CHECK: hc.add {{.*}} : (!hc.idx<"M">, i64) -> !hc.undef
// CHECK: hc.cmp.lt {{.*}} : (!hc.idx<"M">, i64) -> !hc.undef
hc.func @mixed_idx_and_builtin_stays_unknown(%idx: !hc.idx<"M">, %n: i64) {
  %sum = hc.add %idx, %n : (!hc.idx<"M">, i64) -> !hc.undef
  %cmp = hc.cmp.lt %idx, %n : (!hc.idx<"M">, i64) -> !hc.undef
  hc.return
}
