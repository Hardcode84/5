// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-launch-func-to-runtime`. The pass walks every
// `gpu.launch_func` and rewrites it as a pair of calls into our HIP
// shim (`hc_rt_load_kernel` + `hc_rt_launch_kernel`), with the matching
// `gpu.binary`'s HSACO blob materialised inline as an LLVM global so
// the JIT'd module is self-contained. After the walk every consumed
// `gpu.binary` is erased.
//
// Tests pin the pass surface — runtime decls minted, per-callsite
// `_data`/`_handle` globals, alloca-per-arg + array packing, both
// runtime calls in the right shape — without trying to FileCheck the
// exact SSA value numbering downstream of `gpu-to-llvm` (those numbers
// shift with every upstream bump). Failure paths live in
// lower-launch-func-to-runtime-invalid.mlir.

// RUN: hc-opt --hc-lower-launch-func-to-runtime --split-input-file %s | FileCheck %s

// Single launch, two scalar args. Validates the full pipeline:
//   * `_data` global carries the binary blob (and only the blob bytes,
//     no extra NUL — ELF is self-describing and the size travels
//     alongside the pointer in `hc_rt_load_kernel`).
//   * `_handle` global is `#llvm.zero` — runtime fills it on first
//     launch via single-flight memoization.
//   * the kernel-name global is NUL-terminated (HIP wants C strings).
//   * the runtime decls land at module scope with the right ABI.
//   * the launch is gone, the binary is gone.

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK-DAG: llvm.mlir.global internal constant @entry_data{{.*}}("opaque-blob")
// CHECK-DAG: llvm.mlir.global internal constant @entry{{.*}}("entry\00")
// CHECK-DAG: llvm.mlir.global internal @entry_handle{{.*}}(#llvm.zero) {{.*}} : !llvm.ptr
// CHECK-LABEL: llvm.func @host(
// CHECK-SAME:    %[[BUF:.*]]: !llvm.ptr, %[[N:.*]]: i64
// CHECK: %[[STREAM:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[HANDLE:.*]] = llvm.mlir.addressof @entry_handle{{.*}} : !llvm.ptr
// CHECK: %[[NAME:.*]] = llvm.mlir.addressof @entry{{.*}} : !llvm.ptr
// CHECK: %[[NAMEPTR:.*]] = llvm.getelementptr %[[NAME]]
// CHECK: %[[DATA:.*]] = llvm.mlir.addressof @entry_data{{.*}} : !llvm.ptr
// CHECK: %[[DATAPTR:.*]] = llvm.getelementptr %[[DATA]]
// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(11 : i64) : i64
// CHECK: %[[FN:.*]] = llvm.call @hc_rt_load_kernel(%[[STREAM]], %[[HANDLE]], %[[DATAPTR]], %[[SIZE]], %[[NAMEPTR]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
// CHECK: %[[SMEM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[ARGS0:.*]] = llvm.mlir.poison : !llvm.array<2 x ptr>
// CHECK: %[[BUFSLOT:.*]] = llvm.alloca %{{.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[BUF]], %[[BUFSLOT]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[ARGS1:.*]] = llvm.insertvalue %[[BUFSLOT]], %[[ARGS0]][0]
// CHECK: %[[NSLOT:.*]] = llvm.alloca %{{.*}} x i64 : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[N]], %[[NSLOT]] : i64, !llvm.ptr
// CHECK: %[[ARGS2:.*]] = llvm.insertvalue %[[NSLOT]], %[[ARGS1]][1]
// CHECK: %[[ARGSPTR:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x ptr> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[ARGS2]], %[[ARGSPTR]]
// CHECK: %[[NARGS:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: llvm.call @hc_rt_launch_kernel(%[[STREAM]], %[[FN]], %[[SMEM]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ARGSPTR]], %[[NARGS]]) : (!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32) -> ()
// CHECK: llvm.return
// CHECK-DAG: llvm.func @hc_rt_load_kernel(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @hc_rt_launch_kernel(!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32)
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: gpu.binary
module attributes {gpu.container_module} {
  llvm.func @host(%arg0: !llvm.ptr, %arg1: i64) {
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c4 = llvm.mlir.constant(4 : index) : i64
    gpu.launch_func @kernel::@entry blocks in (%c4, %c1, %c1) threads in (%c4, %c1, %c1) : i64
      args(%arg0 : !llvm.ptr, %arg1 : i64)
    llvm.return
  }
  gpu.binary @kernel [#gpu.object<#rocdl.target<chip = "gfx1100">, bin = "opaque-blob">]
}

// -----

// Two launches into different binaries, plus a launch with no operands
// (validates the alloca + insertvalue chain stays well-formed when
// `args` is empty — `LLVM::LLVMArrayType::get(ptrType, 0)` is legal
// and the loop just doesn't iterate). Each callsite gets its own
// `_data` + `_handle` pair; both source binaries are erased.

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK-DAG: llvm.mlir.global internal constant @first_data{{.*}}("first-blob")
// CHECK-DAG: llvm.mlir.global internal constant @second_data{{.*}}("second-blob")
// CHECK-DAG: llvm.mlir.global internal @first_handle{{.*}}(#llvm.zero)
// CHECK-DAG: llvm.mlir.global internal @second_handle{{.*}}(#llvm.zero)
// CHECK-LABEL: llvm.func @host_two(
// CHECK: llvm.call @hc_rt_load_kernel
// CHECK: llvm.call @hc_rt_launch_kernel
// CHECK: llvm.call @hc_rt_load_kernel
// CHECK: llvm.call @hc_rt_launch_kernel
// CHECK: llvm.return
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: gpu.binary
module attributes {gpu.container_module} {
  llvm.func @host_two() {
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c2 = llvm.mlir.constant(2 : index) : i64
    gpu.launch_func @first::@first blocks in (%c2, %c1, %c1) threads in (%c1, %c1, %c1) : i64
    gpu.launch_func @second::@second blocks in (%c1, %c1, %c1) threads in (%c2, %c1, %c1) : i64
    llvm.return
  }
  gpu.binary @first [#gpu.object<#rocdl.target<chip = "gfx1100">, bin = "first-blob">]
  gpu.binary @second [#gpu.object<#rocdl.target<chip = "gfx1100">, bin = "second-blob">]
}

// -----

// Two launches into the same kernel: each callsite mints its own
// `_handle` slot (uniquing through the SymbolTable) so they don't
// share the runtime cache. `_data` is also per-callsite — wave does
// the same, and the deduplication win isn't worth the complexity for
// the hot path (every launched kernel runs forever, so the global
// table grows once and stays bounded).

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK-DAG: llvm.mlir.global internal @entry_handle{{.*}}(#llvm.zero)
// CHECK-DAG: llvm.mlir.global internal @entry_handle_{{.*}}(#llvm.zero)
// CHECK-DAG: llvm.mlir.global internal constant @entry_data{{.*}}("blob")
// CHECK-DAG: llvm.mlir.global internal constant @entry_data_{{.*}}("blob")
// CHECK-LABEL: llvm.func @host_dup(
// CHECK: llvm.call @hc_rt_load_kernel
// CHECK: llvm.call @hc_rt_launch_kernel
// CHECK: llvm.call @hc_rt_load_kernel
// CHECK: llvm.call @hc_rt_launch_kernel
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: gpu.binary
module attributes {gpu.container_module} {
  llvm.func @host_dup() {
    %c1 = llvm.mlir.constant(1 : index) : i64
    gpu.launch_func @kernel::@entry blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) : i64
    gpu.launch_func @kernel::@entry blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) : i64
    llvm.return
  }
  gpu.binary @kernel [#gpu.object<#rocdl.target<chip = "gfx1100">, bin = "blob">]
}
