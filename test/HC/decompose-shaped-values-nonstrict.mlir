// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-decompose-shaped-values=strict=false %s | FileCheck %s

// The default frontend schedule uses non-strict decomposition while helper-call,
// store, and region-boundary consumers are still being taught to preserve masks.
// Unsupported semantic shaped values should keep the module intact instead of
// surfacing strict-mode diagnostics during `hc.compile`.

// CHECK-LABEL: func.func @unsupported_boundary
// CHECK-SAME: %[[ARG:.*]]: !hc.tensor<f32, ["4"]>
// CHECK: return
func.func @unsupported_boundary(%tile: !hc.tensor<f32, ["4"]>) {
  return
}
