// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-decompose-shaped-values -split-input-file -verify-diagnostics %s

// -----

// expected-error @+1 {{failed to legalize operation 'func.func'}}
func.func @semantic_block_arg(%arg0: !hc.tensor<f32, ["4"]>) {
  return
}
