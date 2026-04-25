// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s
// RUN: hc-opt --hc-infer-types %s | FileCheck %s --check-prefix=INFER

// CHECK-LABEL: hc.func @tuple_roundtrip
// CHECK: hc.tuple(%{{.*}}, %{{.*}}) : (!hc.idx<"M">, f32) -> tuple<!hc.idx<"M">, f32>
// CHECK: hc.tuple() : () -> tuple<>
hc.func @tuple_roundtrip(%idx: !hc.idx<"M">, %scalar: f32) {
  %pair = hc.tuple(%idx, %scalar)
      : (!hc.idx<"M">, f32) -> tuple<!hc.idx<"M">, f32>
  %empty = hc.tuple() : () -> tuple<>
  hc.return
}

// INFER-LABEL: hc.func @tuple_infer
// INFER: hc.tuple(%{{.*}}, %{{.*}}) : (!hc.idx<"7">, f32) -> tuple<!hc.idx<"7">, f32>
// INFER: hc.getitem {{.*}} -> !hc.idx<"7">
hc.func @tuple_infer(%scalar: f32) {
  %zero = hc.const<0 : i64> : !hc.undef
  %seven = hc.const<7 : i64> : !hc.undef
  %pair = hc.tuple(%seven, %scalar) : (!hc.undef, f32) -> !hc.undef
  %item = hc.getitem %pair[%zero] : (!hc.undef, !hc.undef) -> !hc.undef
  hc.return
}

// INFER-LABEL: hc.func @tuple_refines_existing_tuple
// INFER: hc.tuple(%{{.*}}, %{{.*}}) : (!hc.idx<"3">, !hc.idx<"4">) -> tuple<!hc.idx<"3">, !hc.idx<"4">>
hc.func @tuple_refines_existing_tuple {
  %three = hc.const<3 : i64> : !hc.undef
  %four = hc.const<4 : i64> : !hc.undef
  %pair = hc.tuple(%three, %four)
      : (!hc.undef, !hc.undef) -> tuple<!hc.undef, !hc.undef>
  hc.return
}

// CHECK-LABEL: hc.func @tuple_branch_compatibility
// CHECK: hc.workitem_region -> (tuple<!hc.undef, f32>)
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"M">, f32>
// CHECK: hc.for_range {{.*}} iter_args({{.*}}) : (!hc.undef, !hc.undef, !hc.undef) -> (tuple<!hc.undef, f32>)
hc.func @tuple_branch_compatibility(%lo: !hc.undef, %hi: !hc.undef,
                                    %step: !hc.undef,
                                    %pair: tuple<!hc.idx<"M">, f32>) {
  %region = hc.workitem_region -> (tuple<!hc.undef, f32>) {
    hc.yield %pair : tuple<!hc.idx<"M">, f32>
  }
  %loop = hc.for_range %lo to %hi step %step iter_args(%region)
      : (!hc.undef, !hc.undef, !hc.undef) -> (tuple<!hc.undef, f32>) {
  ^bb0(%iv: !hc.undef, %carried: tuple<!hc.undef, f32>):
    hc.yield %pair : tuple<!hc.idx<"M">, f32>
  }
  hc.return
}

// CHECK-LABEL: hc.func @tuple_branch_joinable_elements
// CHECK: hc.workitem_region -> (tuple<!hc.idx, !hc.pred>)
// CHECK: hc.yield {{.*}} : tuple<!hc.idx<"M">, !hc.pred<{{.*}}>>
hc.func @tuple_branch_joinable_elements(
    %pair: tuple<!hc.idx<"M">, !hc.pred<"M < N">>) {
  %region = hc.workitem_region -> (tuple<!hc.idx, !hc.pred>) {
    hc.yield %pair : tuple<!hc.idx<"M">, !hc.pred<"M < N">>
  }
  hc.return
}
