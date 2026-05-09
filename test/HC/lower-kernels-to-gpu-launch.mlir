// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: hc-opt %s --hc-lower-kernels-to-gpu-launch | FileCheck %s

// Helpers are declared once at module scope and shared across kernels. They
// carry `llvm.emit_c_interface` so `convert-func-to-llvm` later routes call
// sites through the `_mlir_ciface_*` symbols exported by libhc_rt_helpers.so.
// CHECK-DAG: func.func private @hc_get_buffer(!llvm.ptr) -> memref<?xi8> attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_dim(!llvm.ptr, i32) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_int64(!llvm.ptr) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @hc_get_float64(!llvm.ptr) -> f64 attributes {llvm.emit_c_interface}

module {
  // Static-shape buffer. No symbolic dims, so the wrapper only calls
  // `hc_get_buffer` and feeds zero dynamic sizes into `memref.view`.
  // CHECK-LABEL: func.func @static_launch(
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr)
  hc.kernel @static_launch(
      %group: !hc.group<work_shape = #hc.shape<["64", "16"]>, group_shape = #hc.shape<["32", "8"]>>,
      %a: !hc.buffer<f32, ["64", "16"]>)
      attributes {
        work_shape = #hc.shape<["64", "16"]>,
        group_shape = #hc.shape<["32", "8"]>,
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1"]
      } {
    // CHECK: %[[BUF:.*]] = call @hc_get_buffer(%[[A]]) : (!llvm.ptr) -> memref<?xi8>
    // CHECK: %[[OFF:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW:.*]] = memref.view %[[BUF]][%[[OFF]]][] : memref<?xi8> to memref<64x16xf32>
    // CHECK: %[[C32:.*]] = arith.constant 32 : index
    // CHECK: %[[GX:.*]] = arith.ceildivui %{{.*}}, %[[C32]] : index
    // CHECK: %[[C8:.*]] = arith.constant 8 : index
    // CHECK: %[[GY:.*]] = arith.ceildivui %{{.*}}, %[[C8]] : index
    // CHECK: gpu.launch blocks
    // CHECK-SAME: %[[GX]]
    // CHECK-SAME: %[[GY]]
    // CHECK-SAME: threads
    // CHECK-SAME: %[[C32]]
    // CHECK-SAME: %[[C8]]
    // CHECK: unrealized_conversion_cast %[[VIEW]] : memref<64x16xf32> to !hc.buffer<f32, ["64", "16"]>
    // CHECK: gpu.terminator
    // CHECK: return
    hc.return
  }

  // Dynamic-shape buffer. Wrapper calls `hc_get_dim` once per first-occurrence
  // shape symbol (`M` from axis 0, `N` from axis 1), index-casts the i64 to
  // index, and feeds both as dynamic sizes to `memref.view`.
  // CHECK-LABEL: func.func @dynamic_launch(
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
    // CHECK: %[[BUF:.*]] = call @hc_get_buffer(%[[A]]) : (!llvm.ptr) -> memref<?xi8>
    // CHECK: %[[OFF:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW:.*]] = memref.view %[[BUF]][%[[OFF]]][%[[M]], %[[N]]] : memref<?xi8> to memref<?x?xf32>
    // CHECK: arith.ceildivui %{{.*}},
    // CHECK: gpu.launch
    // CHECK-SAME: blocks(
    // CHECK-SAME: threads(
    // CHECK: unrealized_conversion_cast %[[VIEW]] : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
    // CHECK: gpu.terminator
    hc.return
  }
}
