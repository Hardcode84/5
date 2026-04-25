// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-infer-types -split-input-file %s | FileCheck %s --implicit-check-not=__hc_join_tmp

// CHECK-LABEL: hc.func @scalar_index_arithmetic
// CHECK: hc.const<2 : i64> : !hc.idx<"2">
// CHECK: hc.const<3 : i64> : !hc.idx<"3">
// CHECK: hc.const<7 : i64> : !hc.idx<"7">
// CHECK: hc.add {{.*}} -> !hc.idx<"5">
// CHECK: hc.cmp.lt {{.*}} -> !hc.pred<"True">
// CHECK: hc.slice_expr{{.*}} -> !hc.slice<lower = !hc.idx<"5">, upper = !hc.idx<"7">>
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

// CHECK-LABEL: hc.func @slice_region_branch_result
// CHECK: hc.if {{.*}} -> (!hc.slice<lower = !hc.idx>) : i1
// CHECK: hc.yield {{.*}} : !hc.slice<lower = !hc.idx<"1">>
// CHECK: hc.yield {{.*}} : !hc.slice<lower = !hc.idx<"2">>
hc.func @slice_region_branch_result(%cond: i1) {
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.slice_expr(lower = %one) : (!hc.undef) -> !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.slice_expr(lower = %two) : (!hc.undef) -> !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @slice_expression_conflict_renumbers_join_symbol
// CHECK: ^bb0(%[[IV:.*]]: !hc.idx<"[[JOIN:\$join[0-9]+]]">):
// CHECK: hc.slice_expr{{.*}} -> !hc.slice<lower = !hc.idx<"[[JOIN]]">, upper = !hc.idx<"16 + [[JOIN]]">>
hc.func @slice_expression_conflict_renumbers_join_symbol {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  %sixteen = hc.const<16 : i64> : !hc.undef
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) {
  ^bb0(%iv: !hc.undef):
    %upper = hc.add %iv, %sixteen : (!hc.undef, !hc.undef) -> !hc.undef
    %slice = hc.slice_expr(lower = %iv upper = %upper)
        : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @for_range_iv
// CHECK: hc.for_range
// CHECK: ^bb0(%arg0: !hc.idx<"$join{{[0-9]+}}">):
// CHECK: hc.const<1 : i64> : !hc.idx<"1">
// CHECK: hc.add %arg0, {{.*}} : (!hc.idx<"$join{{[0-9]+}}">, !hc.idx<"1">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @for_range_iv {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  hc.for_range %lo to %hi step %step : (!hc.undef, !hc.undef, !hc.undef) {
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
// CHECK-SAME: -> (!hc.idx<"$join{{[0-9]+}}">)
// CHECK: ^bb0(%{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">, %{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">):
// CHECK: hc.yield {{.*}} : !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
// CHECK: hc.if {{.*}} -> (!hc.idx<"$join{{[0-9]+}}">) : i1
hc.func @region_branch_results(%cond: i1) {
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<4 : i64> : !hc.undef
  %step = hc.const<1 : i64> : !hc.undef
  %init = hc.const<0 : i64> : !hc.undef
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
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

// CHECK-LABEL: hc.func @tuple_region_branch_result
// CHECK: hc.if {{.*}} -> (tuple<!hc.idx<"$join{{[0-9]+}}">, f32>) : i1
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"1">, f32>
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"2">, f32>
hc.func @tuple_region_branch_result(%cond: i1, %value: f32) {
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.tuple(%one, %value) : (!hc.undef, f32) -> !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.tuple(%two, %value) : (!hc.undef, f32) -> !hc.undef
    hc.yield %else : !hc.undef
  }
  hc.return
}

// -----

// CHECK-LABEL: hc.func @tuple_multi_index_conflicts
// CHECK: hc.if {{.*}} -> (tuple<!hc.idx<"$join0">, !hc.idx<"$join1">>) : i1
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"1">, !hc.idx<"10">>
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"2">, !hc.idx<"20">>
hc.func @tuple_multi_index_conflicts(%cond: i1) {
  %one = hc.const<1 : i64> : !hc.undef
  %two = hc.const<2 : i64> : !hc.undef
  %ten = hc.const<10 : i64> : !hc.undef
  %twenty = hc.const<20 : i64> : !hc.undef
  %choice = hc.if %cond -> (!hc.undef) : i1 {
    %then = hc.tuple(%one, %ten) : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %then : !hc.undef
  } else {
    %else = hc.tuple(%two, %twenty) : (!hc.undef, !hc.undef) -> !hc.undef
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

// -----

// CHECK-LABEL: hc.func @untyped_string_const_stays_unknown
// CHECK: hc.const<"gfx11"> : !hc.undef
hc.func @untyped_string_const_stays_unknown {
  %arch = hc.const<"gfx11"> : !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @interprocedural_callee
// CHECK-SAME: (%{{[^:]+}}: !hc.idx<"2">) -> !hc.idx<"3">
// CHECK: hc.add {{.*}} -> !hc.idx<"3">
hc.func @interprocedural_callee(%x: !hc.undef) -> !hc.undef {
  %one = hc.const<1 : i64> : !hc.undef
  %sum = hc.add %x, %one : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// CHECK-LABEL: hc.func @interprocedural_caller
// CHECK: hc.call @interprocedural_callee(%{{.*}}) : (!hc.idx<"2">) -> !hc.idx<"3">
hc.func @interprocedural_caller {
  %two = hc.const<2 : i64> : !hc.undef
  %result = hc.call @interprocedural_callee(%two) : (!hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @joined_interprocedural_callee
// CHECK-SAME: (%{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
// CHECK: hc.add {{.*}} -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @joined_interprocedural_callee(%x: !hc.undef) -> !hc.undef {
  %one = hc.const<1 : i64> : !hc.undef
  %sum = hc.add %x, %one : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return %sum : !hc.undef
}

// CHECK-LABEL: hc.func @joined_interprocedural_caller
// CHECK: hc.call @joined_interprocedural_callee(%{{.*}}) : (!hc.idx<"2">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
// CHECK: hc.call @joined_interprocedural_callee(%{{.*}}) : (!hc.idx<"5">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @joined_interprocedural_caller {
  %two = hc.const<2 : i64> : !hc.undef
  %five = hc.const<5 : i64> : !hc.undef
  %a = hc.call @joined_interprocedural_callee(%two)
      : (!hc.undef) -> !hc.undef
  %b = hc.call @joined_interprocedural_callee(%five)
      : (!hc.undef) -> !hc.undef
  hc.return
}

// -----

// CHECK-LABEL: hc.func @workitem_tail_region_callee
// CHECK: %[[REGION:.*]] = hc.workitem_region captures = ["group"] -> (!hc.undef)
// CHECK: hc.yield {{.*}} : !hc.idx<"0">
// CHECK: hc.return %[[REGION]] : !hc.undef
hc.func @workitem_tail_region_callee(%group: !hc.undef) -> !hc.undef {
  %region = hc.workitem_region captures = ["group"] -> (!hc.undef) {
  ^bb0(%wi: !hc.undef):
    %seed = hc.const<0 : i64> : !hc.undef
    hc.yield %seed : !hc.undef
  }
  hc.return %region : !hc.undef
}

// CHECK-LABEL: hc.func @call_result_iter_arg_keeps_loop_body_live
// CHECK: %[[INIT:.*]] = hc.call @workitem_tail_region_callee
// CHECK: hc.for_range {{.*}} iter_args(%[[INIT]])
// CHECK: ^bb0(%{{[^:]+}}: !hc.idx<"$join{{[0-9]+}}">, %{{[^:]+}}: !hc.undef):
// CHECK: hc.const<16 : i64> : !hc.idx<"16">
// CHECK: hc.add {{.*}} : (!hc.idx<"$join{{[0-9]+}}">, !hc.idx<"16">) -> !hc.idx<{{.*}}$join{{[0-9]+}}{{.*}}>
hc.func @call_result_iter_arg_keeps_loop_body_live(%group: !hc.undef) {
  %init = hc.call @workitem_tail_region_callee(%group)
      : (!hc.undef) -> !hc.undef
  %lo = hc.const<0 : i64> : !hc.undef
  %hi = hc.const<64 : i64> : !hc.undef
  %step = hc.const<16 : i64> : !hc.undef
  %loop = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
  ^bb0(%iv: !hc.undef, %carried: !hc.undef):
    %tile = hc.const<16 : i64> : !hc.undef
    %next = hc.add %iv, %tile : (!hc.undef, !hc.undef) -> !hc.undef
    hc.yield %carried : !hc.undef
  }
  hc.return
}
