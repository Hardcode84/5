// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end snapshot of the canonical `amdgpu-gfx11` WMMA lowering pipeline.
// Mirrors `hc/schedules/front_to_hc.mlir` as a hand-rolled `hc-opt` pass
// list so this LIT can run from any builder that has `hc-opt` in PATH and
// proves the milestones the bead checklist requires:
//   * `hc.kernel` becomes a host `func.func` with a `gpu.launch` body
//   * buffer ABI arguments become upstream `memref` types
//   * `hc.for_range` becomes `scf.for` with carried `vector`/mask iter args
//   * decomposition + launch-body lowering wipe semantic `!hc.tensor<...>` /
//     `!hc.vector<...>` containers (only the bare counterparts survive at
//     the unbridged `hc.call_intrinsic` boundary)
//   * masks reach upstream `vector<NxI1>` and feed `scf.if`-guarded stores
//
// Two checklist items are *not* asserted yet because the pipeline can't
// satisfy them today; both have follow-up beads:
//   * "the WMMA intrinsic lowers" — the recipe is wired, but
//     `hc-lower-launch-body` keeps `!hc.bare_vector` types at every
//     `hc.call_intrinsic` boundary (`convertIntrinsicBoundaryType`) and
//     `amdgpu.wmma` rejects bare types, so running
//     `-hc-interpret-intrinsic-recipes` would fail at create-op time. The
//     missing piece is a bare→upstream bridge at intrinsic call boundaries.
//   * "no `hc.*` ops remain at the complete boundary" — same blocker plus
//     the absence of a pass that DCEs unused `hc.intrinsic` declarations
//     after the last call site is gone.
//
// RUN: %python -m examples.amdgpu_gfx11_wmma_matmul --dump-front-ir \
// RUN:   | hc-opt --hc-front-fold-region-defs --hc-front-inline \
// RUN:        --convert-hc-front-to-hc --hc-promote-names --hc-infer-types \
// RUN:        --hc-materialize-bound-exprs --hc-verify-static-shapes \
// RUN:        --hc-decompose-shaped-values=strict=false --hc-inline-helpers \
// RUN:        --hc-materialize-bound-exprs --canonicalize \
// RUN:        --hc-normalize-scope-regions --canonicalize --cse \
// RUN:        --hc-lower-kernels-to-gpu-launch --hc-lower-launch-body \
// RUN:        --canonicalize --cse \
// RUN:   | FileCheck %s --implicit-check-not='hc.kernel' \
// RUN:        --implicit-check-not='!hc.tensor<' \
// RUN:        --implicit-check-not='!hc.vector<' \
// RUN:        --implicit-check-not='hc.workitem_region' \
// RUN:        --implicit-check-not='hc.subgroup_region' \
// RUN:        --implicit-check-not='hc.local_id' \
// RUN:        --implicit-check-not='hc.materialize_bound_expr' \
// RUN:        --implicit-check-not='hc.for_range'

// `hc.kernel` is gone; the kernel landed as a host `func.func` taking the
// flattened buffer-ABI arguments as upstream dynamic memrefs. The implicit
// `--implicit-check-not='hc.kernel'` above guards against any regression
// that would leave a residual `hc.kernel` symbol around.
// CHECK-LABEL: func.func @tiled_gfx11_wmma_matmul(
// CHECK-SAME: %{{[^:]+}}: memref<?x?xf16>
// CHECK-SAME: %{{[^:]+}}: memref<?x?xf16>
// CHECK-SAME: %{{[^:]+}}: memref<?x?xf32>

// The kernel body becomes a `gpu.launch`; block/thread counts come from
// `work_shape / group_shape`, not raw `work_shape` — `arith.ceildivui` /
// `arith.muli` show that the pipeline computed them.
// CHECK: arith.ceildivui
// CHECK: gpu.launch blocks(%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}})
// CHECK-SAME: threads(%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}})

// The K loop survives as a real `scf.for` carrying the accumulator data /
// mask vectors as iter args.
// CHECK: scf.for %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}}
// CHECK-SAME: iter_args(%{{[^,]+}} = %{{[^,]+}}, %{{[^)]+}} = %{{[^)]+}})
// CHECK-SAME: -> (vector<8xf32>, vector<8xi1>)

// Tile loads land as `vector.transfer_read` over the dynamic input memrefs
// staging into workgroup `memref` allocations — no `!hc.tensor<...>` /
// `!hc.vector<...>` semantic containers remain (covered by the
// `--implicit-check-not` directives above).
// CHECK: vector.transfer_read %{{[^[]+}}[%{{[^,]+}}, %{{[^]]+}}]
// CHECK-SAME: : memref<?x?xf16>, vector<16x16xf16>
// CHECK: memref.alloca() : memref<16x16xf16, #gpu.address_space<workgroup>>

// Edge masks become upstream `i1` vectors via `vector.create_mask`.
// CHECK: vector.create_mask
// CHECK-SAME: : vector<16x16xi1>

// `hc.call_intrinsic @wmma_gfx11` is the documented residual: the recipe
// matches and would lower the call to `amdgpu.wmma`, but its operand /
// result types are still `!hc.bare_vector` because of the unbridged
// boundary noted in the header. The check below pins the post-decomposition
// shape so a regression that drops the data/mask split would fail loudly.
// CHECK: hc.call_intrinsic @wmma_gfx11(%{{[^)]+}})
// CHECK-SAME: {arch = "gfx11", wave_size = 32 : i64}
// CHECK-SAME: !hc.bare_vector<f16, ["16"]>
// CHECK-SAME: !hc.bare_vector<!hc.pred, ["16"]>
// CHECK-SAME: !hc.bare_vector<f32, ["8"]>
// CHECK-SAME: !hc.bare_vector<!hc.pred, ["8"]>
// CHECK-SAME: -> (!hc.bare_vector<f32, ["8"]>, !hc.bare_vector<!hc.pred, ["8"]>)

// The masked accumulator stores show up as per-lane `scf.if` guards over
// `memref.store`, with the i1 mask coming from `vector.extract` on the
// 8-wide accumulator mask vector.
// CHECK: vector.extract %{{[^[]+}}[{{[0-9]+}}] : i1 from vector<8xi1>
// CHECK: scf.if %{{[^ ]+}} {
// CHECK: memref.store %{{[^,]+}}, %{{[^[]+}}[%{{[^,]+}}, %{{[^]]+}}]
// CHECK-SAME: : memref<?x?xf32>

// The recipe module rides on the side as a sibling
// `module @__hc_intrinsic_lowerings__`. It carries the post-decomposition
// indices (5/7/9 for the data fragments and 10 for the mask passthrough)
// and the `require_intrinsic_attr` pre-checks. This LIT pins the wiring;
// the recipe interpreter pass is exercised by
// `test/HC/interpret-intrinsic-recipes.mlir` against synthetic payload
// types so the bare-type bridging gap doesn't block the recipe-side tests.
// CHECK: module @__hc_intrinsic_lowerings__
// CHECK: transform.named_sequence @__hc_lower_wmma_gfx11_amdgpu_gfx11
// CHECK-SAME: hc.target = "amdgpu-gfx11"
// CHECK: transform.hc.match_intrinsic_call %{{[^ ]+}} @wmma_gfx11 target = "amdgpu-gfx11"
// CHECK: transform.hc.get_intrinsic_operand %{{[^ ]+}} {index = 5 : i64}
// CHECK: transform.hc.get_intrinsic_operand %{{[^ ]+}} {index = 7 : i64}
// CHECK: transform.hc.get_intrinsic_operand %{{[^ ]+}} {index = 9 : i64}
// CHECK: transform.hc.require_intrinsic_attr
// CHECK-SAME: expected = "gfx11"
// CHECK: transform.hc.require_intrinsic_attr
// CHECK-SAME: expected = 32 : i64
// CHECK: transform.hc.create_op "amdgpu.wmma"
// CHECK: transform.hc.replace_intrinsic_call
