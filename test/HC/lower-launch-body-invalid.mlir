// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not hc-opt %s --hc-lower-launch-body 2>&1 | FileCheck %s

// Unknown sym at the apply site. The lowering names the unresolved
// sym in the diagnostic — the apply already came in with no
// binding source, so `M` has nowhere to come from. Important for
// free-sym-bearing layouts: when flatten leaves a sym in the
// composed offset and lowering can't find an SSA binding, the user
// sees the name they need to bind. See `doc/layouts.md` "Free
// symbols in layout offsets".
module {
  func.func @unknown_symbol() {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      // CHECK: error: 'hc.idx_apply' op cannot lower idx_apply: free symbol 'M' has no binding in the surrounding scope
      %bad = hc.idx_apply () : () -> !hc.idx<"M">
      gpu.terminator
    }
    return
  }
}
