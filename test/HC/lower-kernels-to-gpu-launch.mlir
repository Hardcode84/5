// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s --hc-lower-kernels-to-gpu-launch | FileCheck %s

module {
  // CHECK-LABEL: func.func @static_launch(
  // CHECK-SAME: %[[A:.*]]: memref<64x16xf32>)
  hc.kernel @static_launch(
      %group: !hc.group<work_shape = #hc.shape<["64", "16"]>, group_shape = #hc.shape<["32", "8"]>>,
      %a: !hc.buffer<f32, ["64", "16"]>)
      attributes {
        work_shape = #hc.shape<["64", "16"]>,
        group_shape = #hc.shape<["32", "8"]>,
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1"]
      } {
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
    // CHECK: unrealized_conversion_cast %[[A]] : memref<64x16xf32> to !hc.buffer<f32, ["64", "16"]>
    // CHECK: gpu.terminator
    // CHECK: return
    hc.return
  }

  // CHECK-LABEL: func.func @dynamic_launch(
  // CHECK-SAME: %[[A:.*]]: memref<?x?xf32>)
  hc.kernel @dynamic_launch(
      %group: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "N"]>, group_shape = #hc.shape<["32", "1"]>>,
      %a: !hc.buffer<f32, ["M", "N"]>)
      attributes {
        work_shape = #hc.shape<["32*ceiling(1/16*M)", "N"]>,
        group_shape = #hc.shape<["32", "1"]>,
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "M", "N"]
      } {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[M:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[N:.*]] = memref.dim %[[A]], %[[C1]] : memref<?x?xf32>
    // CHECK: arith.ceildivui %{{.*}},
    // CHECK: gpu.launch
    // CHECK-SAME: blocks(
    // CHECK-SAME: threads(
    // CHECK: unrealized_conversion_cast %[[A]] : memref<?x?xf32> to !hc.buffer<f32, ["M", "N"]>
    // CHECK: gpu.terminator
    hc.return
  }
}
