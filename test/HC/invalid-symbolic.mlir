// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not hc-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: invalid hc.expr text
#bad = #hc.expr<"M >= 1">
module {}

// -----

// CHECK: error: invalid hc.pred text
#bad = #hc.pred<"M + 1">
module {}

// -----

// CHECK: error: invalid hc.expr text
module {
  func.func @bad(%arg0: !hc.buffer<f32, #hc.shape<["M >= 1"]>>) {
    return
  }
}

// -----

// CHECK: error: expected #hc.shape attribute
module {
  func.func @bad(%arg0: !hc.buffer<f32, #hc.expr<"M">>) {
    return
  }
}

// -----

// CHECK: error: expected constraints to contain only #hc.pred attributes
module {
  hc.kernel "bad" requirements = #hc.constraints<[#hc.expr<"M">]> {
    hc.return
  }
}

// -----

// CHECK: error: expected #hc.scope to be one of "WorkGroup", "SubGroup", "WorkItem"
module {
  func.func @bad() attributes {scope = #hc.scope<"Lane">} {
    return
  }
}

// -----

// CHECK: error: expected #hc.effects to be one of "Pure", "Read", "Write", "ReadWrite"
module {
  func.func @bad() attributes {effects = #hc.effects<"Weird">} {
    return
  }
}

// -----

// CHECK: error: 'hc.symbol' op result must pin a symbolic expression
module {
  func.func @bad() {
    %s = hc.symbol : !hc.idx
    return
  }
}
