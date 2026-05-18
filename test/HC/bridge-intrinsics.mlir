// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-bridge-intrinsics`: retype `hc.intrinsic` signatures and
// `hc.call_intrinsic` boundaries through the launch-body type
// converter ahead of the main launch-body lowering. `!hc.idx` ->
// `index`, `!hc.pred` -> `i1`. Bare carriers stay opaque at the
// boundary by design -- the intrinsic body owns their shape.
//
// RUN: hc-opt %s --hc-bridge-intrinsics | FileCheck %s

module {
  // Intrinsic with bare-vector + idx args: idx gets retyped to
  // `index`, bare carrier stays as `!hc.bare_vector`.
  // CHECK-LABEL: hc.intrinsic @bare_absf
  // CHECK-SAME: (%{{.*}}: !hc.bare_vector<f16, ["16"]>, %{{.*}}: index)
  // CHECK-SAME: -> !hc.bare_vector<f16, ["16"]>
  hc.intrinsic @bare_absf(%a: !hc.bare_vector<f16, ["16"]>,
                          %k: !hc.idx<"K">) -> !hc.bare_vector<f16, ["16"]>
      scope = #hc.scope<"WorkItem"> parameters = ["a", "k"] {}

  // Call boundary: idx arg casts through UCC to `index`; bare arg
  // rides through its source/target materialisation round-trip.
  // CHECK-LABEL: func.func @call_site
  // CHECK: %[[K_INDEX:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !hc.idx<"K"> to index
  // CHECK: hc.call_intrinsic @bare_absf(%{{.*}}, %[[K_INDEX]])
  // CHECK-SAME: : (!hc.bare_vector<f16, ["16"]>, index) -> !hc.bare_vector<f16, ["16"]>
  func.func @call_site(%a: !hc.bare_vector<f16, ["16"]>, %k: !hc.idx<"K">)
      -> !hc.bare_vector<f16, ["16"]> {
    %r = hc.call_intrinsic @bare_absf(%a, %k)
        : (!hc.bare_vector<f16, ["16"]>, !hc.idx<"K">)
          -> !hc.bare_vector<f16, ["16"]>
    return %r : !hc.bare_vector<f16, ["16"]>
  }
}
