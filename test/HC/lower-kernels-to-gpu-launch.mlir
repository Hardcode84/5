// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: hc-opt %s --hc-lower-kernels-to-gpu-launch | FileCheck %s

// Helpers are declared once at module scope and shared across kernels. They
// carry `llvm.emit_c_interface` so `convert-func-to-llvm` later routes call
// sites through the `_mlir_ciface_*` symbols exported by libhc_rt_helpers.so.
// `hc_get_ptr` is the descriptor-free entry the buffer ABI lives on; the
// legacy `hc_get_buffer` declaration sticks around for any pre-`hc.ptr`
// consumer still in flight, but no buffer-arg lowering reaches for it now.
// CHECK-DAG: func.func private @hc_get_buffer(!llvm.ptr) -> memref<?xi8> attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_ptr(!llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_dim(!llvm.ptr, i32) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_stride(!llvm.ptr, i32) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_int64(!llvm.ptr) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_float64(!llvm.ptr) -> f64 attributes {llvm.emit_c_interface}

module {
  // Static-shape buffer. `hc_get_ptr` returns the raw `data_ptr()` and gets
  // UCC'd to `!hc.ptr<global, f32>`; dim values fall out of the static shape
  // (no `hc_get_dim` call needed); strides are still pulled at runtime via
  // `hc_get_stride` because even a static-shape kernel may be called on a
  // sliced/transposed input. The bridge from the (ptr, dim0, dim1, s0, s1)
  // tuple to `!hc.buffer<f32, ["64", "16"]>` happens INSIDE the launch
  // region so `gpu-kernel-outlining` later captures the raw values, each
  // llvm-translatable, instead of the bridged buffer (which isn't).
  // The leading `!llvm.ptr` is the stream slot — the launch-func-to-runtime
  // pass picks it up later and threads it through hc_rt_load_kernel /
  // hc_rt_launch_kernel.
  // CHECK-LABEL: func.func @static_launch(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr,
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr)
  hc.kernel @static_launch(
      %group: !hc.group<work_shape = #hc.shape<["64", "16"]>, group_shape = #hc.shape<["32", "8"]>>,
      %a: !hc.buffer<f32, ["64", "16"]>)
      attributes {
        work_shape = #hc.shape<["64", "16"]>,
        group_shape = #hc.shape<["32", "8"]>,
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1"]
      } {
    // CHECK: %[[D0:.*]] = arith.constant 64 : index
    // CHECK: %[[D1:.*]] = arith.constant 16 : index
    // CHECK: %[[RAW:.*]] = call @hc_get_ptr(%[[A]]) : (!llvm.ptr) -> !llvm.ptr
    // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : !llvm.ptr to !hc.ptr<global, f32>
    // CHECK: %[[S0_IDX:.*]] = arith.constant 0 : i32
    // CHECK: %[[S0_I64:.*]] = call @hc_get_stride(%[[A]], %[[S0_IDX]]) : (!llvm.ptr, i32) -> i64
    // CHECK: %[[S0:.*]] = arith.index_cast %[[S0_I64]] : i64 to index
    // CHECK: %[[S1_IDX:.*]] = arith.constant 1 : i32
    // CHECK: %[[S1_I64:.*]] = call @hc_get_stride(%[[A]], %[[S1_IDX]]) : (!llvm.ptr, i32) -> i64
    // CHECK: %[[S1:.*]] = arith.index_cast %[[S1_I64]] : i64 to index
    // CHECK: gpu.launch blocks
    // CHECK: unrealized_conversion_cast %[[PTR]], %[[D0]], %[[D1]], %[[S0]], %[[S1]] : !hc.ptr<global, f32>, index, index, index, index to !hc.buffer<f32, ["64", "16"]>
    // CHECK: gpu.terminator
    // CHECK: return
    hc.return
  }

  // Dynamic-shape buffer. Wrapper calls `hc_get_dim` once per first-
  // occurrence shape symbol (`M` from axis 0, `N` from axis 1), index-casts
  // the i64 to index, then `hc_get_ptr` for the data pointer and
  // `hc_get_stride` per axis. The bridging UCC at the kernel boundary
  // takes (ptr, M, N, stride_0, stride_1) → !hc.buffer<f32, ["M", "N"]>.
  // CHECK-LABEL: func.func @dynamic_launch(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr,
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr)
  hc.kernel @dynamic_launch(
      %group: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "N"]>, group_shape = #hc.shape<["32", "1"]>>,
      %a: !hc.buffer<f32, ["M", "N"]>)
      attributes {
        work_shape = #hc.shape<["32*ceiling(1/16*M)", "N"]>,
        group_shape = #hc.shape<["32", "1"]>,
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "M", "N"]
      } {
    // CHECK: %[[D0_IDX:.*]] = arith.constant 0 : i32
    // CHECK: %[[D0_I64:.*]] = call @hc_get_dim(%[[A]], %[[D0_IDX]]) : (!llvm.ptr, i32) -> i64
    // CHECK: %[[M:.*]] = arith.index_cast %[[D0_I64]] : i64 to index
    // CHECK: %[[D1_IDX:.*]] = arith.constant 1 : i32
    // CHECK: %[[D1_I64:.*]] = call @hc_get_dim(%[[A]], %[[D1_IDX]]) : (!llvm.ptr, i32) -> i64
    // CHECK: %[[N:.*]] = arith.index_cast %[[D1_I64]] : i64 to index
    // CHECK: %[[RAW:.*]] = call @hc_get_ptr(%[[A]]) : (!llvm.ptr) -> !llvm.ptr
    // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : !llvm.ptr to !hc.ptr<global, f32>
    // CHECK: %[[S0_IDX:.*]] = arith.constant 0 : i32
    // CHECK: %[[S0_I64:.*]] = call @hc_get_stride(%[[A]], %[[S0_IDX]]) : (!llvm.ptr, i32) -> i64
    // CHECK: %[[S0:.*]] = arith.index_cast %[[S0_I64]] : i64 to index
    // CHECK: %[[S1_IDX:.*]] = arith.constant 1 : i32
    // CHECK: %[[S1_I64:.*]] = call @hc_get_stride(%[[A]], %[[S1_IDX]]) : (!llvm.ptr, i32) -> i64
    // CHECK: %[[S1:.*]] = arith.index_cast %[[S1_I64]] : i64 to index
    // CHECK: arith.ceildivui %{{.*}},
    // CHECK: gpu.launch
    // CHECK-SAME: blocks(
    // CHECK-SAME: threads(
    // CHECK: unrealized_conversion_cast %[[PTR]], %[[M]], %[[N]], %[[S0]], %[[S1]] : !hc.ptr<global, f32>, index, index, index, index to !hc.buffer<f32, ["M", "N"]>
    // CHECK: gpu.terminator
    hc.return
  }
}
