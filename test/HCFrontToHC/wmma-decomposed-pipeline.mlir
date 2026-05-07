// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %python -m examples.amdgpu_gfx11_wmma_matmul --dump-front-ir \
// RUN:   | hc-opt --hc-front-fold-region-defs --hc-front-inline --convert-hc-front-to-hc --hc-promote-names --hc-infer-types --hc-materialize-bound-exprs --hc-verify-static-shapes --hc-decompose-shaped-values=strict=false --hc-inline-helpers --hc-materialize-bound-exprs --canonicalize --hc-normalize-scope-regions --canonicalize --cse \
// RUN:   | FileCheck %s --implicit-check-not='!hc.tensor<' --implicit-check-not='!hc.vector<' --implicit-check-not='hc.workitem_region' --implicit-check-not='hc.call @'

// CHECK-LABEL: hc.kernel @tiled_gfx11_wmma_matmul
// CHECK: %[[LOOP:.*]]:6 = hc.for_range
// CHECK-SAME: -> (!hc.bare_tensor<f16, ["16", "16"]>, !hc.bare_tensor<!hc.pred, ["16", "16"]>, !hc.bare_tensor<f16, ["16", "16"]>, !hc.bare_tensor<!hc.pred, ["16", "16"]>, !hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>)
// CHECK: hc.load_mask %{{.*}} : (!hc.buffer<f16, ["M", "K"]>
// CHECK-SAME: -> !hc.bare_tensor<!hc.pred, ["16", "16"]>
// CHECK: hc.load_mask %{{.*}} : (!hc.buffer<f16, ["K", "N"]>
// CHECK-SAME: -> !hc.bare_tensor<!hc.pred, ["16", "16"]>
// CHECK: hc.select
// CHECK-SAME: (!hc.bare_vector<!hc.pred, ["16"]>, !hc.bare_vector<f16, ["16"]>, f16) -> !hc.bare_vector<f16, ["16"]>
// CHECK: hc.call_intrinsic @wmma_gfx11
// CHECK-SAME: !hc.bare_tensor<f16, ["16", "16"]>
// CHECK-SAME: !hc.bare_tensor<!hc.pred, ["16", "16"]>
// CHECK-SAME: !hc.bare_vector<f16, ["16"]>
// CHECK-SAME: !hc.bare_vector<!hc.pred, ["16"]>
// CHECK-SAME: !hc.bare_vector<f32, ["8"]>
// CHECK-SAME: !hc.bare_vector<!hc.pred, ["8"]>
// CHECK-SAME: !hc.idx<"$WI0">
// CHECK-SAME: -> (!hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>)
// CHECK: hc.store {{.*}}, %{{[^,]+}}, mask %{{[^ ]+}}
// CHECK-SAME: !hc.bare_vector<f32, ["8", "1"]>, !hc.bare_vector<!hc.pred, ["8", "1"]>
// CHECK-LABEL: hc.intrinsic @wmma_gfx11
// CHECK-SAME: %{{.*}}: !hc.bare_tensor<f16, ["16", "16"]>
// CHECK-SAME: %{{.*}}: !hc.bare_tensor<!hc.pred, ["16", "16"]>
// CHECK-SAME: %{{.*}}: !hc.bare_vector<f16, ["16"]>
// CHECK-SAME: %{{.*}}: !hc.bare_vector<!hc.pred, ["16"]>
// CHECK-SAME: %{{.*}}: !hc.bare_vector<f32, ["8"]>
// CHECK-SAME: %{{.*}}: !hc.bare_vector<!hc.pred, ["8"]>
// CHECK-SAME: -> (!hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>)
// CHECK-SAME: parameters = ["group", "a_tile.data", "a_tile.mask", "b_tile.data", "b_tile.mask", "a_frag.data", "a_frag.mask", "b_frag.data", "b_frag.mask", "acc_frag.data", "acc_frag.mask", "lane", "wave_size", "arch"]
