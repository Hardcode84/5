// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end snapshot of the canonical `amdgpu-gfx11` WMMA lowering pipeline.
// Hand-rolls the composition `hc.compile` runs — the schedule in
// `hc/schedules/front_to_hc.mlir` plus the `_GPU_LOWERING_PIPELINE` chain in
// `hc/_pipeline.py` — as an `hc-opt` pass list. Driving the actual schedule
// via `transform-interpreter` from inside `hc-opt` would be more
// drift-resistant, but the interpreter's nested pass manager races on
// loading the `dlti` dialect (an LLVM ERROR), so we stick with the
// hand-rolled list; the WMMA pytest exercises the real `hc.compile`
// composition and catches drift between the two. Pins the load-bearing
// invariants of the executable lowering chain:
//
//   * `hc.kernel` becomes a host wrapper that ends up as `llvm.func` after
//     `gpu-to-llvm` finishes the host-side `func.func` → `llvm.func`
//     lowering, which is what the executable handle plumbing later wants.
//   * `gpu-kernel-outlining` relocates the launch body into a sibling
//     `gpu.module @<kernel>_kernel`, `rocdl-attach-target` stamps it,
//     and `hc-lower-gpu-to-binary` finally compiles the module to an
//     HSACO blob attached to a `gpu.binary` op.
//   * No `hc.*` ops survive at any point; all upstream conversions
//     (workgroup `!hc.ptr` → addrspace-3 LLVM globals, scf->cf, gpu->rocdl,
//     amdgpu wmma intrinsic lowering) compose without leftover dialect.
//
// The HSACO blob bytes are opaque, so we strip the `bin = "..."` payload
// before FileCheck — otherwise the implicit-check-nots below would match
// substrings inside the binary (the ELF carries `.amdgpu.metadata`, etc.).
// `sed` keeps only the prefix up through `bin = "` on the gpu.binary line.
//
// RUN: %python -m examples.amdgpu_gfx11_wmma_matmul --dump-front-ir \
// RUN:   | hc-opt --pass-pipeline='builtin.module(hc-front-fold-region-defs,hc-front-inline,convert-hc-front-to-hc,hc-promote-names,hc-infer-types,hc-materialize-bound-exprs,hc-verify-static-shapes,hc-decompose-shaped-values{strict=false},hc-inline-helpers,hc-materialize-bound-exprs,canonicalize,hc-canonicalize-layouts,hc-shaped-compute-to-generic,hc-elementwise-to-generic,hc-load-store-to-generic,hc-infer-generic-bounds,hc-normalize-scope-regions,canonicalize,cse,hc-lower-kernels-to-gpu-launch,hc-flatten-with-layouts,canonicalize,cse,hc-lower-launch-body,canonicalize,cse,hc-lower-generic,hc-fold-predicates,hc-lower-launch-body,canonicalize,cse,hc-interpret-intrinsic-recipes{target=amdgpu-gfx11},canonicalize,cse,gpu-launch-sink-index-computations,gpu-kernel-outlining,canonicalize,cse,rocdl-attach-target{chip=gfx1100 features=+wavefrontsize32},hc-lower-to-llvm,convert-scf-to-cf,convert-amdgpu-to-rocdl{chipset=gfx1100},gpu.module(convert-gpu-to-rocdl{chipset=gfx1100},convert-arith-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts),gpu-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,reconcile-unrealized-casts,hc-lower-gpu-to-binary{lld-path=%hc_lld},hc-lower-launch-func-to-runtime,symbol-dce)' \
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
// wrapper passes scalars / pointers straight through to the matching
// `_mlir_ciface_*` symbol libhc_rt_helpers.so exports. The
// `hc.ptr<global, T?>` kernel-arg ABI calls into `hc_get_ptr` and pulls
// dims/strides via the matching scalar helpers — no memref on the wire.
// CHECK-DAG: llvm.func private @hc_get_ptr({{.*}}: !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @_mlir_ciface_hc_get_ptr(!llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func private @hc_get_dim({{.*}}: !llvm.ptr, {{.*}}: i32) -> i64
// CHECK-DAG: llvm.func @_mlir_ciface_hc_get_dim(!llvm.ptr, i32) -> i64
// CHECK-DAG: llvm.func private @hc_get_stride({{.*}}: !llvm.ptr, {{.*}}: i32) -> i64
// CHECK-DAG: llvm.func @_mlir_ciface_hc_get_stride(!llvm.ptr, i32) -> i64

// Host wrapper takes a leading `!llvm.ptr` stream slot followed by one
// PyObject* (lowered to `!llvm.ptr`) per kernel argument — three buffers
// in the WMMA example — and immediately calls the helpers to materialize
// each tensor's data pointer, dims, and strides. The
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
// CHECK: llvm.call @hc_get_ptr(%[[A]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @hc_get_ptr(%[[B]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @hc_get_ptr(%[[C]]) : (!llvm.ptr) -> !llvm.ptr

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
