// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-emit-bench-wrapper`. The pass runs after
// `-hc-lower-launch-func-to-runtime` and, for each `llvm.func`
// containing a single `hc_rt_launch_kernel` call, mints a sibling
// `<name>_bench` that calls `hc_rt_launch_kernel_repeat` instead, with
// a trailing `i64 %n_inner` argument and an `i64` return. The two
// wrappers share per-callsite `_data` / `_handle` / kernel-name
// globals via symbol reference — the clone preserves the
// `llvm.addressof @<kernel>_handle` ops verbatim, so a load-kernel
// from either wrapper warms the same `hipFunction_t` cache slot.
//
// Tests pin the dual-wrapper shape, the symbol sharing, the
// `n_inner` plumbing, and idempotency under re-application. Functions
// without a launch call (helper unpackers, runtime decl placeholders
// surfacing alongside the wrapper) are silent skips, exercised inline
// in the first case so the no-bench assertion stays scoped to the
// same input.

// RUN: hc-opt --hc-emit-bench-wrapper --split-input-file %s | FileCheck %s

// Case 1: a launch-bearing host wrapper next to a launch-free helper
// (the unpacker stub). The pass must mint `@host_bench` but leave
// `@hc_get_int64_stub` untouched — proof of the per-function skip
// path. CHECK-LABEL fences keep the no-bench assertion scoped to the
// helper's block.

// CHECK-LABEL: llvm.func @host(
// CHECK-SAME:    %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i64
// CHECK-NOT: -> i64
// CHECK: llvm.call @hc_rt_load_kernel
// CHECK: llvm.call @hc_rt_launch_kernel(
// CHECK-NOT: hc_rt_launch_kernel_repeat
// CHECK: llvm.return
// CHECK-NOT: llvm.return %
// CHECK-LABEL: llvm.func @host_bench(
// CHECK-SAME:    %[[STREAM:[^:]+]]: !llvm.ptr, %[[BUF:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i64, %[[NINNER:[^:]+]]: i64
// CHECK-SAME:    -> i64
// CHECK: %[[HANDLE:.*]] = llvm.mlir.addressof @entry_handle
// CHECK: llvm.call @hc_rt_load_kernel(%[[STREAM]], %[[HANDLE]],
// CHECK: %[[T:.*]] = llvm.call @hc_rt_launch_kernel_repeat({{.*}}, %[[NINNER]]){{.*}}-> i64
// CHECK-NEXT: llvm.return %[[T]] : i64
// CHECK-LABEL: llvm.func @hc_get_int64_stub(
// CHECK-NOT: @hc_get_int64_stub_bench
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @hc_rt_launch_kernel_repeat(
// CHECK-SAME:    !llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32, i64
// CHECK-SAME:    -> i64
module {
  llvm.mlir.global internal @entry_data("opaque-blob") : !llvm.array<11 x i8>
  llvm.mlir.global internal @entry("entry\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal @entry_handle(#llvm.zero) : !llvm.ptr
  llvm.func @hc_rt_load_kernel(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
  llvm.func @hc_rt_launch_kernel(!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32)
  llvm.func @host(%stream: !llvm.ptr, %arg0: !llvm.ptr, %arg1: i64) {
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c2_i32 = llvm.mlir.constant(2 : i32) : i32
    %c11_i64 = llvm.mlir.constant(11 : i64) : i64
    %handle = llvm.mlir.addressof @entry_handle : !llvm.ptr
    %name = llvm.mlir.addressof @entry : !llvm.ptr
    %nameptr = llvm.getelementptr %name[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
    %data = llvm.mlir.addressof @entry_data : !llvm.ptr
    %dataptr = llvm.getelementptr %data[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
    %fn = llvm.call @hc_rt_load_kernel(%stream, %handle, %dataptr, %c11_i64, %nameptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    %poison = llvm.mlir.poison : !llvm.array<2 x ptr>
    %bufslot = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %arg0, %bufslot : !llvm.ptr, !llvm.ptr
    %args0 = llvm.insertvalue %bufslot, %poison[0] : !llvm.array<2 x ptr>
    %nslot = llvm.alloca %c1_i64 x i64 : (i64) -> !llvm.ptr
    llvm.store %arg1, %nslot : i64, !llvm.ptr
    %args1 = llvm.insertvalue %nslot, %args0[1] : !llvm.array<2 x ptr>
    %argsptr = llvm.alloca %c1_i64 x !llvm.array<2 x ptr> : (i64) -> !llvm.ptr
    llvm.store %args1, %argsptr : !llvm.array<2 x ptr>, !llvm.ptr
    llvm.call @hc_rt_launch_kernel(%stream, %fn, %c0_i32, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %argsptr, %c2_i32) : (!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @hc_get_int64_stub(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
}

// -----

// Case 2: idempotency. Running the pass twice in a row mints exactly
// one bench sibling per source wrapper — the `_bench` suffix screen
// keeps the worklist on the second invocation empty. Uses a fresh
// FileCheck prefix so the assertions don't fight Case 1's.

// RUN: hc-opt --hc-emit-bench-wrapper --hc-emit-bench-wrapper --split-input-file %s | FileCheck %s --check-prefix=IDEMP

// IDEMP-LABEL: llvm.func @host(
// IDEMP-LABEL: llvm.func @host_bench(
// IDEMP-NOT: @host_bench_bench
// IDEMP-LABEL: llvm.func @hc_rt_launch_kernel_repeat(
module {
  llvm.mlir.global internal @k_data("blob") : !llvm.array<4 x i8>
  llvm.mlir.global internal @k("k\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal @k_handle(#llvm.zero) : !llvm.ptr
  llvm.func @hc_rt_load_kernel(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
  llvm.func @hc_rt_launch_kernel(!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32)
  llvm.func @host(%stream: !llvm.ptr) {
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c4_i64 = llvm.mlir.constant(4 : i64) : i64
    %c0_args = llvm.mlir.constant(0 : i32) : i32
    %h = llvm.mlir.addressof @k_handle : !llvm.ptr
    %n = llvm.mlir.addressof @k : !llvm.ptr
    %np = llvm.getelementptr %n[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %d = llvm.mlir.addressof @k_data : !llvm.ptr
    %dp = llvm.getelementptr %d[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %fn = llvm.call @hc_rt_load_kernel(%stream, %h, %dp, %c4_i64, %np) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    %poison = llvm.mlir.poison : !llvm.array<0 x ptr>
    %ap = llvm.alloca %c1_i64 x !llvm.array<0 x ptr> : (i64) -> !llvm.ptr
    llvm.store %poison, %ap : !llvm.array<0 x ptr>, !llvm.ptr
    llvm.call @hc_rt_launch_kernel(%stream, %fn, %c0_i32, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64, %ap, %c0_args) : (!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i32) -> ()
    llvm.return
  }
}
