// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end snapshot of the canonical `amdgpu-gfx11` WMMA lowering pipeline.
// Mirrors `hc/schedules/front_to_hc.mlir` as a hand-rolled `hc-opt` pass list
// so this LIT can run from any builder that has `hc-opt` in PATH and proves
// every milestone the bead checklist requires:
//   * `hc.kernel` becomes a host `func.func` with a `gpu.launch` body
//   * buffer ABI arguments become upstream `memref` types
//   * `hc.for_range` becomes `scf.for` with carried `vector` iter args
//   * decomposition + launch-body lowering wipe semantic `!hc.tensor<...>` /
//     `!hc.vector<...>` containers
//   * masks reach upstream `vector<NxI1>` and feed `arith.select` paths
//   * the WMMA intrinsic lowers to `amdgpu.wmma` after the recipe runs
//   * no `hc.*` ops survive the final canonicalize/cse pair (the intrinsic
//     decl is DCE'd by `-hc-interpret-intrinsic-recipes` once its last call
//     is rewritten, and the launch-body UCC-wrapped operand types fold away
//     when the recipe-inserted bare↔upstream casts pair with the existing
//     UCCs and canonicalize collapses the chains to identity)
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
// RUN:        --hc-interpret-intrinsic-recipes='target=amdgpu-gfx11' \
// RUN:        --canonicalize --cse \
// RUN:   | FileCheck %s --implicit-check-not='hc.' --implicit-check-not='!hc.'

// `hc.kernel` is gone; the kernel landed as a host `func.func` taking the
// flattened buffer-ABI arguments as upstream dynamic memrefs. The
// `--implicit-check-not='hc.'` guard above pins zero residual HC ops or
// types — both `hc.call_intrinsic`/`hc.intrinsic` and the `!hc.bare_vector`
// types the launch-body pass plants at the call boundary should be folded
// away by the recipe interpretation + canonicalize pair.
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

// The K loop survives as a real `scf.for` carrying the accumulator
// fragment as the iter arg. The mask channel folded away after the recipe
// forwarded `acc_frag.mask` unchanged — canonicalize discovered the
// passthrough was loop-invariant and elided it.
// CHECK: scf.for %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}}
// CHECK-SAME: iter_args(%{{[^)]+}} = %{{[^)]+}})
// CHECK-SAME: -> (vector<8xf32>)

// Tile loads land as `vector.transfer_read` over the dynamic input memrefs
// staging into workgroup `memref` allocations.
// CHECK: vector.transfer_read %{{[^[]+}}[%{{[^,]+}}, %{{[^]]+}}]
// CHECK-SAME: : memref<?x?xf16>, vector<16x16xf16>
// CHECK: memref.alloca() : memref<16x16xf16, #gpu.address_space<workgroup>>

// Edge masks become upstream `i1` vectors via `vector.create_mask`, gating
// per-lane fragment loads through `arith.select` (the in-bounds half feeds
// `amdgpu.wmma`, the out-of-bounds half is zero-padded).
// CHECK: vector.create_mask
// CHECK-SAME: : vector<16x16xi1>
// CHECK: arith.select %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}}
// CHECK-SAME: : vector<16xi1>, vector<16xf16>

// The WMMA intrinsic lowers cleanly to `amdgpu.wmma`. The bare↔upstream
// casts the recipe planted around the call boundary paired with the
// launch-body UCCs and folded to identity, so the op sits between plain
// upstream vectors with no leftover bridging machinery.
// CHECK: amdgpu.wmma 16x16x16
// CHECK-SAME: : vector<16xf16>, vector<16xf16>, vector<8xf32>

// The output stores complete the kernel. `arith.select`-fed mask folded
// to a constant-true accumulator validity once the recipe's mask
// passthrough met the existing all-true `vector<8xi1>` initializer, so
// canonicalize promoted the previously `scf.if`-guarded stores to
// unconditional `memref.store`.
// CHECK: vector.extract
// CHECK-SAME: : f32 from vector<8xf32>
// CHECK: memref.store %{{[^,]+}}, %{{[^[]+}}[%{{[^,]+}}, %{{[^]]+}}]
// CHECK-SAME: : memref<?x?xf32>

// `gpu.terminator` closes the launch and the kernel returns. The recipe
// module is gone (the interpreter erased it after applying every
// matching sequence) and the `hc.intrinsic @wmma_gfx11` decl is gone
// (the interpreter sweeps unused decls once their last call site is
// rewritten). Anything left over in either category would have been
// caught by the `--implicit-check-not='hc.'` directive above.
// CHECK: gpu.terminator
// CHECK: return
