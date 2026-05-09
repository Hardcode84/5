// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Sidecar transform schedule loaded by `wmma-upstream-pipeline.mlir`'s
// `--pass-pipeline` line. Mirrors the alloca-to-global + lower-vector-transfer
// chunk that `hc/schedules/front_to_hc.mlir` runs in production — split out
// so the LIT can drive the device-side lowering chain via a flat `hc-opt`
// pass list while still using transform-dialect for the patterns that don't
// have a registered-pass equivalent.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%m: !transform.any_op) {
    %alloca = transform.structured.match ops{["memref.alloca"]} in %m
        : (!transform.any_op) -> !transform.op<"memref.alloca">
    %get_global, %global = transform.memref.alloca_to_global %alloca
        : (!transform.op<"memref.alloca">) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %m {
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
      transform.apply_patterns.vector.transfer_to_scf full_unroll = true
    } : !transform.any_op
    transform.yield
  }
}
