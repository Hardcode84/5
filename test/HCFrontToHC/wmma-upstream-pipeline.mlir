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
// RUN:   | hc-opt --pass-pipeline='builtin.module(hc-front-fold-region-defs,hc-front-inline,convert-hc-front-to-hc,hc-promote-names,hc-infer-types,hc-materialize-bound-exprs,hc-verify-static-shapes,hc-decompose-shaped-values{strict=false},hc-inline-helpers,hc-materialize-bound-exprs,canonicalize,hc-normalize-scope-regions,canonicalize,cse,hc-lower-kernels-to-gpu-launch,hc-lower-launch-body,canonicalize,cse,hc-interpret-intrinsic-recipes{target=amdgpu-gfx11},canonicalize,cse,gpu-launch-sink-index-computations,gpu-kernel-outlining,canonicalize,cse,transform-preload-library{transform-library-paths=%S/Inputs/wmma-upstream-pipeline-transforms.mlir},transform-interpreter,gpu.module(fold-memref-alias-ops),lower-affine,gpu.module(lower-affine),canonicalize,cse,hc-lower-to-llvm,convert-scf-to-cf,convert-amdgpu-to-rocdl{chipset=gfx1100},lower-affine,gpu.module(lower-affine,convert-gpu-to-rocdl{chipset=gfx1100},convert-arith-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1100},gpu-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts,canonicalize,cse,hc-lower-gpu-to-binary{lld-path=%hc_lld},hc-lower-launch-func-to-runtime,symbol-dce)' \
// RUN:   | sed 's/\(@[A-Za-z0-9_]*_data[A-Za-z0-9_]* *(\)"[^"]*"/\1"<HSACO>"/' \
// RUN:   | FileCheck %s --implicit-check-not='hc.' --implicit-check-not='!hc.' --implicit-check-not='gpu.' --implicit-check-not='amdgpu.' --implicit-check-not='vector.transfer'

// Module carries the gpu.container_module attribute so the runtime side knows
// to walk for gpu.binary ops to load. The implicit-check-nots above pin that
// `hc-lower-gpu-to-binary` swept every `gpu.module` and that
// `hc-lower-launch-func-to-runtime` consumed every `gpu.binary` and
// `gpu.launch_func` (`gpu.` matches all three; the only legitimate
// remaining occurrence is the `gpu.container_module` attribute below,
// matched here).
// CHECK-LABEL: module attributes {gpu.container_module}

// Per-callsite globals planted by `hc-lower-launch-func-to-runtime`:
// the `_data` global carries the HSACO blob (sed-stripped to "<HSACO>"
// above so the test doesn't drift with every llvm/lld bump), the
// `_handle` global is a zero-init `!llvm.ptr` slot the runtime fills
// on first launch, and the kernel name lands as a NUL-terminated C
// string for `hipModuleGetFunction`.
// CHECK-DAG: llvm.mlir.global internal constant @tiled_gfx11_wmma_matmul_kernel_data{{.*}}("<HSACO>")
// CHECK-DAG: llvm.mlir.global internal constant @tiled_gfx11_wmma_matmul_kernel{{[_0-9]*}}("tiled_gfx11_wmma_matmul_kernel\00")
// CHECK-DAG: llvm.mlir.global internal @tiled_gfx11_wmma_matmul_kernel_handle{{.*}}(#llvm.zero) {{.*}} : !llvm.ptr

// Runtime helper wrappers planted by `hc-lower-kernels-to-gpu-launch` and
// then cwrapper-rewritten by `convert-func-to-llvm`: the public-name
// wrapper handles memref descriptor sret packing for `hc_get_buffer` and
// passes scalars straight through; the matching `_mlir_ciface_*` symbol
// is the one libhc_rt_helpers.so actually exports. The buffer wrapper's
// signature includes the descriptor struct as the LLVM-ABI return; the
// dim wrapper passes its i64 directly.
// CHECK-DAG: llvm.func private @hc_get_buffer({{.*}}: !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG: llvm.func @_mlir_ciface_hc_get_buffer(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func private @hc_get_dim({{.*}}: !llvm.ptr, {{.*}}: i32) -> i64
// CHECK-DAG: llvm.func @_mlir_ciface_hc_get_dim(!llvm.ptr, i32) -> i64

// Host wrapper takes a leading `!llvm.ptr` stream slot followed by one
// PyObject* (lowered to `!llvm.ptr`) per kernel argument — three buffers
// in the WMMA example — and immediately calls the helpers to materialize
// each tensor's data pointer and shape dims. The
// `--implicit-check-not='hc.'` guard above pins zero residual HC ops or
// types, and `--implicit-check-not='vector.transfer'` proves every
// transfer_read/write got reduced to vector.load/store before the rocdl
// chain ran.
// CHECK-LABEL: llvm.func @tiled_gfx11_wmma_matmul(
// CHECK-SAME:    %[[STREAM:[^:]+]]: !llvm.ptr
// CHECK-SAME:    %[[A:[^:]+]]: !llvm.ptr
// CHECK-SAME:    %[[B:[^:]+]]: !llvm.ptr
// CHECK-SAME:    %[[C:[^:]+]]: !llvm.ptr
// CHECK: llvm.call @hc_get_dim(%[[A]], %{{.*}}) : (!llvm.ptr, i32) -> i64
// CHECK: llvm.call @hc_get_buffer(%[[A]]) : (!llvm.ptr) -> !llvm.struct<{{.*}}>
// CHECK: llvm.call @hc_get_buffer(%[[B]]) : (!llvm.ptr) -> !llvm.struct<{{.*}}>
// CHECK: llvm.call @hc_get_buffer(%[[C]]) : (!llvm.ptr) -> !llvm.struct<{{.*}}>

// Block/thread counts come from `work_shape / group_shape` (the
// `arith.ceildivui`/`arith.muli` chain became `llvm.udiv`/`llvm.mul`
// after gpu-to-llvm) and feed straight through to `hc_rt_launch_kernel`.
// The launch is no longer a `gpu.launch_func` op — it's a pair of HIP
// shim calls: load the kernel module (single-flight via the handle
// slot) then launch with the packed args. Both calls receive the
// host wrapper's leading stream slot as their first argument; passing
// it through (rather than `llvm.mlir.zero`) lets the caller pin a
// launch to a specific HIP stream.
// CHECK: llvm.udiv
// CHECK: llvm.call @hc_rt_load_kernel(%[[STREAM]],
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @hc_rt_launch_kernel(%[[STREAM]],
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32) -> ()

// Runtime decls land at module scope — `FunctionCallBuilder::create`
// only mints them on first use, so they prove both calls actually
// fired (otherwise the decl would be missing entirely).
// CHECK-DAG: llvm.func @hc_rt_load_kernel(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @hc_rt_launch_kernel(!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32)
