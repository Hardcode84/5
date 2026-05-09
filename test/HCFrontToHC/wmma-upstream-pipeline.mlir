// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end snapshot of the canonical `amdgpu-gfx11` WMMA lowering pipeline.
// Mirrors the composition that `hc.compile` runs — the schedule in
// `hc/schedules/front_to_hc.mlir` plus the `_GPU_LOWERING_PIPELINE` chain in
// `hc/_pipeline.py` — as a hand-rolled `hc-opt` pass list, so this LIT can
// run from any builder that has `hc-opt` in PATH and proves every milestone
// the bead checklist requires:
//
//   * `hc.kernel` becomes a host wrapper that ends up as `llvm.func` after
//     `gpu-to-llvm` runs (the host-side memref descriptor flattening hits
//     all `func.func` in the module, which is what the executable handle
//     plumbing later wants).
//   * `gpu-kernel-outlining` relocates the launch body into a sibling
//     `gpu.module @<kernel>_kernel`, `rocdl-attach-target` stamps it,
//     and `hc-lower-gpu-to-binary` finally compiles the module to an
//     HSACO blob attached to a `gpu.binary` op.
//   * No `hc.*` ops survive at any point; all upstream conversions
//     (workgroup memref AS3, vector transfer reduction, scf->cf, gpu->rocdl,
//     amdgpu wmma intrinsic lowering) compose without leftover dialect.
//
// The HSACO blob bytes are opaque, so we strip the `bin = "..."` payload
// before FileCheck — otherwise the implicit-check-nots below would match
// substrings inside the binary (the ELF carries `.amdgpu.metadata`, etc.).
// `sed` keeps only the prefix up through `bin = "` on the gpu.binary line.
//
// RUN: %python -m examples.amdgpu_gfx11_wmma_matmul --dump-front-ir \
// RUN:   | hc-opt --pass-pipeline='builtin.module(hc-front-fold-region-defs,hc-front-inline,convert-hc-front-to-hc,hc-promote-names,hc-infer-types,hc-materialize-bound-exprs,hc-verify-static-shapes,hc-decompose-shaped-values{strict=false},hc-inline-helpers,hc-materialize-bound-exprs,canonicalize,hc-normalize-scope-regions,canonicalize,cse,hc-lower-kernels-to-gpu-launch,hc-lower-launch-body,canonicalize,cse,hc-interpret-intrinsic-recipes{target=amdgpu-gfx11},canonicalize,cse,gpu-launch-sink-index-computations,gpu-kernel-outlining,canonicalize,cse,transform-preload-library{transform-library-paths=%S/Inputs/wmma-upstream-pipeline-transforms.mlir},transform-interpreter,gpu.module(fold-memref-alias-ops),lower-affine,gpu.module(lower-affine),canonicalize,cse,convert-scf-to-cf,convert-amdgpu-to-rocdl{chipset=gfx1100},lower-affine,gpu.module(lower-affine,convert-gpu-to-rocdl{chipset=gfx1100},convert-arith-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1100},gpu-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts,canonicalize,cse,hc-lower-gpu-to-binary{lld-path=%hc_lld},symbol-dce)' \
// RUN:   | sed 's/\(bin = "\).*$/\1<HSACO>"]/' \
// RUN:   | FileCheck %s --implicit-check-not='hc.' --implicit-check-not='!hc.' --implicit-check-not='gpu.module' --implicit-check-not='amdgpu.' --implicit-check-not='vector.transfer'

// Module carries the gpu.container_module attribute so the runtime side knows
// to walk for gpu.binary ops to load. The `--implicit-check-not='gpu.module'`
// directive above pins that the binary-emission pass swept the source module.
// CHECK-LABEL: module attributes {gpu.container_module}

// Host wrapper landed as `llvm.func` after `gpu-to-llvm` flattened every
// `func.func` in the module — descriptor-passing arity gives 5 i64s per
// dynamic memref (alloc ptr, aligned ptr, offset, dim0, dim1) plus 1 i64
// for the strides we explicitly bake into the launch ABI. The
// `--implicit-check-not='hc.'` guard above pins zero residual HC ops or
// types, and `--implicit-check-not='vector.transfer'` proves every
// transfer_read/write got reduced to vector.load/store before the rocdl
// chain ran.
// CHECK-LABEL: llvm.func @tiled_gfx11_wmma_matmul(
// CHECK-SAME:    %{{[^:]+}}: !llvm.ptr
// CHECK-SAME:    %{{[^:]+}}: !llvm.ptr
// CHECK-SAME:    %{{[^:]+}}: i64

// Block/thread counts come from `work_shape / group_shape` (the
// `arith.ceildivui`/`arith.muli` chain became `llvm.udiv`/`llvm.mul`
// after gpu-to-llvm) and feed straight into `gpu.launch_func`. The
// launch dispatches into the binary stamped below.
// CHECK: llvm.udiv
// CHECK: gpu.launch_func @tiled_gfx11_wmma_matmul_kernel::@tiled_gfx11_wmma_matmul_kernel
// CHECK-SAME:    blocks in (%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}})
// CHECK-SAME:    threads in (%{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}})

// `hc-lower-gpu-to-binary` replaced the source `gpu.module` with a
// sibling `gpu.binary` carrying the rocdl target attribute and a
// `bin = "..."` HSACO blob. The blob bytes are opaque (changes with
// every llvm/lld bump) — pin only the structure: the binary is
// non-empty and the rocdl target attr survived the round trip. The
// `--implicit-check-not='gpu.module'` and `--implicit-check-not='amdgpu.'`
// directives above prove the source module is gone and no leftover
// amdgpu.wmma op leaked back out.
// CHECK: gpu.binary @tiled_gfx11_wmma_matmul_kernel
// CHECK-SAME: #rocdl.target<chip = "gfx1100">
// CHECK-SAME: bin = "
