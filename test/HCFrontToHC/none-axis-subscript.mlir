// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NumPy `None` / `np.newaxis` subscript path: the frontend emits
// `hc_front.constant<"None"> {python_kind = "NoneType"}` for every
// inserted axis. `-convert-hc-front-to-hc` recognises that producer,
// drops the placeholder from `hc.buffer_view`'s operand list, and
// records the inserted positions on the op's `unit_axes` attribute so
// the residual `(slice, slice, ...)` operand combination stays inside
// what downstream legalisation accepts.
//
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file %s | hc-opt --split-input-file | FileCheck %s

// `x[:, None, :]`: one mid-axis insertion at output position 1. The
// residual operand list is the two non-None slices and `unit_axes`
// pins the unit dim at position 1.

// CHECK-LABEL: hc.kernel @insert_middle
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}, %{{.*}}] {unit_axes = array<i64: 1>}

module {
  hc_front.kernel "insert_middle" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}, {name = "x"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %s0 = hc_front.slice() {has_lower = false, has_step = false, has_upper = false}
    %none = hc_front.constant<"None"> {python_kind = "NoneType"}
    %s2 = hc_front.slice() {has_lower = false, has_step = false, has_upper = false}
    %idx = hc_front.tuple(%s0, %none, %s2)
    %view = hc_front.subscript %x[%idx]
    %tgt = hc_front.target_name "t"
    hc_front.assign %tgt = %view
    hc_front.return
  }
}

// -----

// `x[None, :, :, None, :]`: two insertions wrapping a rank-3 source.
// Output rank is 5, `unit_axes` lists positions 0 and 3 in output
// coordinates. Residual operand list carries the three real slices.

// CHECK-LABEL: hc.kernel @insert_outer_pair
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {unit_axes = array<i64: 0, 3>}

module {
  hc_front.kernel "insert_outer_pair" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}, {name = "x"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %n0 = hc_front.constant<"None"> {python_kind = "NoneType"}
    %s1 = hc_front.slice() {has_lower = false, has_step = false, has_upper = false}
    %s2 = hc_front.slice() {has_lower = false, has_step = false, has_upper = false}
    %n3 = hc_front.constant<"None"> {python_kind = "NoneType"}
    %s4 = hc_front.slice() {has_lower = false, has_step = false, has_upper = false}
    %idx = hc_front.tuple(%n0, %s1, %s2, %n3, %s4)
    %view = hc_front.subscript %x[%idx]
    %tgt = hc_front.target_name "t"
    hc_front.assign %tgt = %view
    hc_front.return
  }
}

// -----

// `x[None]` on a rank-1 source: a single None subscript wraps the
// value in a leading unit dim. Output rank = 1 (real subscripts) +
// 1 (unit axis) = 2; the residual operand list is empty.

// CHECK-LABEL: hc.kernel @insert_only
// CHECK: hc.buffer_view %{{.*}}[] {unit_axes = array<i64: 0>}

module {
  hc_front.kernel "insert_only" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}, {name = "x"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %none = hc_front.constant<"None"> {python_kind = "NoneType"}
    %view = hc_front.subscript %x[%none]
    %tgt = hc_front.target_name "t"
    hc_front.assign %tgt = %view
    hc_front.return
  }
}

// -----

// Plain subscripts unchanged: no `None`, no `unit_axes` attribute on
// the resulting buffer_view. The default rendering omits absent
// optional attrs entirely.

// CHECK-LABEL: hc.kernel @no_unit_axes
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}]
// CHECK-NOT: unit_axes

module {
  hc_front.kernel "no_unit_axes" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [{name = "group"}, {name = "x"}],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %i = hc_front.constant<0 : i64>
    %view = hc_front.subscript %x[%i]
    %tgt = hc_front.target_name "t"
    hc_front.assign %tgt = %view
    hc_front.return
  }
}
