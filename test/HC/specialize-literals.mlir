// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-specialize-literals -split-input-file %s \
// RUN:     | FileCheck %s

// Folds `K -> 4` into every shape / type / op-shape carrier that
// references the bound symbol, including the kernel's `function_type`
// (block args and declared function type must stay in lockstep).
// Consumes `literal_bindings`, leaves `literals` (the declaration) put.

// CHECK-LABEL: hc.kernel @vz
// CHECK-SAME:    !hc.buffer<f32, ["4"]>
// CHECK-SAME:    !hc.buffer<f32, ["4"]>
// CHECK-SAME:    literals = ["K"]
// CHECK-NOT:     literal_bindings
// CHECK-NOT:     !hc.idx<"K">
// CHECK-NOT:     !hc.buffer<f32, ["K"]>
// CHECK-NOT:     !hc.vector<f32, ["K"]>
// CHECK:         hc.const<0 : i64> : !hc.idx<"4">
// CHECK:         hc.vzeros shape %{{.*}} : (tuple<!hc.idx<"4">>) -> !hc.vector<f32, ["4"]>
hc.kernel @vz(%src: !hc.buffer<f32, ["K"]>, %dst: !hc.buffer<f32, ["K"]>)
    attributes {literals = ["K"], literal_bindings = {K = 4 : i64},
                work_shape = #hc.shape<["1"]>,
                group_shape = #hc.shape<["1"]>} {
  %k = hc.const<0 : i64> : !hc.idx<"K">
  %shape = hc.tuple(%k) : (!hc.idx<"K">) -> tuple<!hc.idx<"K">>
  %vz = hc.vzeros shape %shape
      : (tuple<!hc.idx<"K">>) -> !hc.vector<f32, ["K"]>
  hc.return
}

// -----

// Symbols not bound stay symbolic — partial specialization is legal,
// matching the `hc.compile(symbols=...)` contract that only fixes the
// names the caller pinned.

// CHECK-LABEL: hc.kernel @partial
// CHECK-SAME:    !hc.buffer<f32, ["4", "N"]>
// CHECK-NOT:     literal_bindings
// CHECK:         hc.const<0 : i64> : !hc.idx<"4">
// CHECK:         hc.const<1 : i64> : !hc.idx<"N">
hc.kernel @partial(%buf: !hc.buffer<f32, ["M", "N"]>)
    attributes {literals = ["M", "N"], literal_bindings = {M = 4 : i64},
                work_shape = #hc.shape<["1"]>,
                group_shape = #hc.shape<["1"]>} {
  %m = hc.const<0 : i64> : !hc.idx<"M">
  %n = hc.const<1 : i64> : !hc.idx<"N">
  hc.return
}

// -----

// Empty bindings: no-op except for dropping the (empty) attribute.

// CHECK-LABEL: hc.kernel @noop
// CHECK-SAME:    !hc.buffer<f32, ["M"]>
// CHECK-NOT:     literal_bindings
// CHECK:         hc.const<0 : i64> : !hc.idx<"M">
hc.kernel @noop(%buf: !hc.buffer<f32, ["M"]>)
    attributes {literals = ["M"], literal_bindings = {},
                work_shape = #hc.shape<["1"]>,
                group_shape = #hc.shape<["1"]>} {
  %m = hc.const<0 : i64> : !hc.idx<"M">
  hc.return
}
