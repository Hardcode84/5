// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not hc-opt %s --hc-lower-launch-body 2>&1 | FileCheck %s

module {
  func.func @unknown_symbol() {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      // CHECK: error: 'hc.idx_apply' op failed to lower idx_apply expression
      %bad = hc.idx_apply () {symbols = []} : () -> !hc.idx<"M">
      gpu.terminator
    }
    return
  }
}
