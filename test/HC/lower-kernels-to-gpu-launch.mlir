// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: hc-opt %s --hc-lower-kernels-to-gpu-launch | FileCheck %s

// Helpers are declared once at module scope and shared across kernels. They
// carry `llvm.emit_c_interface` so `convert-func-to-llvm` later routes call
// sites through the `_mlir_ciface_*` symbols exported by libhc_rt_helpers.so.
// `hc_get_ptr` is the buffer ABI entry — raw `data_ptr()` lifted to
// `!llvm.ptr` and UCC'd into the kernel-arg `(ptr, dim*, stride*)` tuple.
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

  // Post-flatten kernel signature: the buffer collapses to a single `?`
  // dim and the per-axis dim/stride symbols flatten previously baked into
  // the buffer's symbolic shape ride as trailing `!hc.idx<sym>` block
  // args. `hc.flatten_aux_args` pins each trailing slot back to its
  // parent buffer arg + axis + accessor kind so the host wrapper can
  // (a) skip allocating a PyObject slot for the auxes (only one Python
  // arg per buffer), (b) resolve each aux from the parent buffer's
  // PyObject via `_get_dim` / `_get_stride`, and (c) collapse the
  // per-axis dim values into a single total-element count for the
  // bridging UCC that gpu-kernel-outlining later captures.
  //
  // CHECK-LABEL: func.func @flat_launch(
  // CHECK-SAME: %{{[^:]+}}: !llvm.ptr,
  // CHECK-SAME: %[[A:.*]]: !llvm.ptr)
  // CHECK-NOT: %{{[^,)]+}}: !llvm.ptr,
  hc.kernel @flat_launch(
      %group: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "N"]>, group_shape = #hc.shape<["32", "1"]>>,
      %a: !hc.buffer<f32, ["?"]>,
      %sa0: !hc.idx<"$STRIDE_0_a">,
      %sa1: !hc.idx<"$STRIDE_1_a">,
      %m_dim: !hc.idx<"M">,
      %n_dim: !hc.idx<"N">)
      attributes {
        work_shape = #hc.shape<["32*ceiling(1/16*M)", "N"]>,
        group_shape = #hc.shape<["32", "1"]>,
        bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "M", "N", "$STRIDE_0_a", "$STRIDE_1_a"],
        hc.flatten_aux_args = {
          "2" = {aux_of = 1 : i64, axis = 0 : i64, kind = "stride"},
          "3" = {aux_of = 1 : i64, axis = 1 : i64, kind = "stride"},
          "4" = {aux_of = 1 : i64, axis = 0 : i64, kind = "dim"},
          "5" = {aux_of = 1 : i64, axis = 1 : i64, kind = "dim"}
        }
      } {
    // Aux idx slots come up first in the host wrapper: stride probes
    // for axis 0 and 1, dim probes for axis 0 (`M`) and 1 (`N`).
    // CHECK: %[[S0_IDX:.*]] = arith.constant 0 : i32
    // CHECK: %[[S0_I64:.*]] = call @hc_get_stride(%[[A]], %[[S0_IDX]])
    // CHECK: %[[S0:.*]] = arith.index_cast %[[S0_I64]] : i64 to index
    // CHECK: %[[S1_IDX:.*]] = arith.constant 1 : i32
    // CHECK: %[[S1_I64:.*]] = call @hc_get_stride(%[[A]], %[[S1_IDX]])
    // CHECK: %[[S1:.*]] = arith.index_cast %[[S1_I64]] : i64 to index
    // CHECK: %[[M_IDX:.*]] = arith.constant 0 : i32
    // CHECK: %[[M_I64:.*]] = call @hc_get_dim(%[[A]], %[[M_IDX]])
    // CHECK: %[[M:.*]] = arith.index_cast %[[M_I64]] : i64 to index
    // CHECK: %[[N_IDX:.*]] = arith.constant 1 : i32
    // CHECK: %[[N_I64:.*]] = call @hc_get_dim(%[[A]], %[[N_IDX]])
    // CHECK: %[[N:.*]] = arith.index_cast %[[N_I64]] : i64 to index
    // Buffer pack: total = M*N, stride 1, raw ptr from `hc_get_ptr`.
    // `total` is computed before the ptr probe so the host wrapper's
    // index-cast chain stays single-pass — once the aux dims are in
    // hand the buffer's contributions are all stack-local.
    // CHECK: %[[TOTAL:.*]] = arith.muli %[[M]], %[[N]]
    // CHECK: %[[ONE:.*]] = arith.constant 1 : index
    // CHECK: %[[RAW:.*]] = call @hc_get_ptr(%[[A]])
    // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : !llvm.ptr to !hc.ptr<global, f32>
    // CHECK: gpu.launch
    // CHECK: unrealized_conversion_cast %[[PTR]], %[[TOTAL]], %[[ONE]] : !hc.ptr<global, f32>, index, index to !hc.buffer<f32, ["?"]>
    // Aux idx slot values cross the launch-region boundary as the same
    // SSA they were materialized from in host scope; the launch-body
    // lowering later picks them up via the post-flatten symbol-bind
    // UCC walk.
    // CHECK-DAG: unrealized_conversion_cast %[[M]] : index to !hc.idx<"M">
    // CHECK-DAG: unrealized_conversion_cast %[[N]] : index to !hc.idx<"N">
    // CHECK-DAG: unrealized_conversion_cast %[[S0]] : index to !hc.idx<"$STRIDE_0_a">
    // CHECK-DAG: unrealized_conversion_cast %[[S1]] : index to !hc.idx<"$STRIDE_1_a">
    // CHECK: gpu.terminator
    hc.return
  }
}
