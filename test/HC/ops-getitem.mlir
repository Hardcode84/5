// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s
// RUN: hc-opt --hc-infer-types %s | FileCheck %s --check-prefix=INFER

// CHECK-LABEL: hc.func @getitem_roundtrip
// CHECK: hc.getitem %{{.*}}[%{{.*}}] : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.getitem %{{.*}}[%{{.*}}, %{{.*}}] : (!hc.buffer<!hc.undef, ["M", "N"]>, !hc.undef, !hc.slice) -> !hc.undef
hc.func @getitem_roundtrip(%source: !hc.undef,
                           %buffer: !hc.buffer<!hc.undef, ["M", "N"]>,
                           %idx: !hc.undef,
                           %slice: !hc.slice) {
  %scalar = hc.getitem %source[%idx] : (!hc.undef, !hc.undef) -> !hc.undef
  %view = hc.getitem %buffer[%idx, %slice]
      : (!hc.buffer<!hc.undef, ["M", "N"]>, !hc.undef, !hc.slice) -> !hc.undef
  hc.return
}

// INFER-LABEL: hc.func @getitem_tuple_infer
// INFER: hc.getitem {{.*}} -> !hc.idx<"B">
// INFER: hc.getitem {{.*}} -> !hc.idx<"B">
hc.func @getitem_tuple_infer(%tuple: tuple<!hc.idx<"A">, !hc.idx<"B">>,
                             %minus_one: !hc.idx<"-1">) {
  %one = hc.const<1 : i64> : !hc.undef
  %item = hc.getitem %tuple[%one]
      : (tuple<!hc.idx<"A">, !hc.idx<"B">>, !hc.undef) -> !hc.undef
  %last = hc.getitem %tuple[%minus_one]
      : (tuple<!hc.idx<"A">, !hc.idx<"B">>, !hc.idx<"-1">) -> !hc.undef
  hc.return
}

// INFER-LABEL: hc.func @getitem_buffer_infer_defers
// INFER: hc.getitem {{.*}} : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"0">) -> !hc.undef
hc.func @getitem_buffer_infer_defers(%buffer: !hc.buffer<f32, ["M", "N"]>,
                                     %idx: !hc.idx<"0">) {
  %view = hc.getitem %buffer[%idx]
      : (!hc.buffer<f32, ["M", "N"]>, !hc.idx<"0">) -> !hc.undef
  hc.return
}
