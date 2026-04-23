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
  hc.kernel @bad requirements = #hc.constraints<[#hc.expr<"M">]> {
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

// -----

// CHECK: error: 'hc.for_range' op expected body block to take 2 arguments
module {
  func.func @bad(%lo: !hc.undef, %hi: !hc.undef, %step: !hc.undef,
                 %init: !hc.undef) {
    %r = hc.for_range %lo to %hi step %step iter_args(%init)
        : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef {
    ^bb0(%i: !hc.undef):
      hc.yield %init : !hc.undef
    }
    return
  }
}

// -----

// CHECK: error: 'hc.if' op must provide an `else` region when producing results
module {
  func.func @bad(%c: !hc.undef, %v: !hc.undef) -> !hc.undef {
    %x = hc.if %c -> (!hc.undef) : !hc.undef {
      hc.yield %v : !hc.undef
    }
    return %x : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.if' op then yield produces 2 values, expected 1
module {
  func.func @bad(%c: !hc.undef, %v: !hc.undef) -> !hc.undef {
    %x = hc.if %c -> (!hc.undef) : !hc.undef {
      hc.yield %v, %v : !hc.undef, !hc.undef
    } else {
      hc.yield %v : !hc.undef
    }
    return %x : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.if' op then yield[0] type 'i32' does not match result[0] type 'i64'
module {
  func.func @bad(%c: !hc.undef, %a: i32, %b: i64) -> i64 {
    %x = hc.if %c -> (i64) : !hc.undef {
      hc.yield %a : i32
    } else {
      hc.yield %b : i64
    }
    return %x : i64
  }
}

// -----

// CHECK: error: 'hc.for_range' op body yield produces 0 values, expected 1
module {
  func.func @bad(%lo: !hc.undef, %hi: !hc.undef, %st: !hc.undef,
                 %init: !hc.undef) -> !hc.undef {
    %r = hc.for_range %lo to %hi step %st iter_args(%init)
        : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef {
    ^bb0(%i: !hc.undef, %acc: !hc.undef):
      hc.yield
    }
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.for_range' op iter_args[0] type 'i32' does not match body block argument type 'i64'
module {
  func.func @bad(%lo: i32, %hi: i32, %st: i32, %init: i32) -> i32 {
    %r = hc.for_range %lo to %hi step %st iter_args(%init)
        : (i32, i32, i32, i32) -> i32 {
    ^bb0(%i: i32, %acc: i64):
      hc.yield %acc : i64
    }
    return %r : i32
  }
}

// -----

// CHECK: error: 'hc.buffer_dim' op axis must be non-negative
module {
  func.func @bad(%buf: !hc.undef) -> !hc.undef {
    %d = hc.buffer_dim %buf, axis = -1 : !hc.undef -> !hc.undef
    return %d : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.reduce' op kind must be one of "sum", "max", "min", got "avg"
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.reduce %v, kind = "avg", axis = 0 : !hc.undef -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.reduce' op axis must be non-negative
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.reduce %v, kind = "sum", axis = -1 : !hc.undef -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.astype' op target type must be a builtin integer, index, or float type
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.astype %v, target = !hc.slice : !hc.undef -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// Scalar result must equal target.
// CHECK: error: 'hc.astype' op result type 'i64' does not match target type 'i32'
module {
  func.func @bad(%v: i32) -> i64 {
    %r = hc.astype %v, target = i32 : i32 -> i64
    return %r : i64
  }
}

// -----

// Tensor result element must equal target.
// CHECK: error: 'hc.astype' op result element type 'i32' does not match target type 'f32'
module {
  func.func @bad(%v: !hc.tensor<i64, ["M"]>)
      -> !hc.tensor<i32, ["M"]> {
    %r = hc.astype %v, target = f32
        : !hc.tensor<i64, ["M"]> -> !hc.tensor<i32, ["M"]>
    return %r : !hc.tensor<i32, ["M"]>
  }
}

// -----

// CHECK: error: 'hc.buffer_dim' op axis 3 is out of bounds for rank-2 buffer
module {
  func.func @bad(%buf: !hc.buffer<f32, ["M", "N"]>) -> !hc.undef {
    %d = hc.buffer_dim %buf, axis = 3
        : !hc.buffer<f32, ["M", "N"]> -> !hc.undef
    return %d : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.reduce' op axis 5 is out of bounds for rank-2 value
module {
  func.func @bad(%v: !hc.tensor<f32, ["M", "N"]>) -> !hc.undef {
    %r = hc.reduce %v, kind = "sum", axis = 5
        : !hc.tensor<f32, ["M", "N"]> -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.with_inactive' op inactive literal 0.000000e+00 : f32 does not match element type 'i32'
module {
  func.func @bad(%v: !hc.tensor<i32, ["M"]>) -> !hc.tensor<i32, ["M"]> {
    %r = hc.with_inactive %v {inactive = 0.0 : f32}
        : !hc.tensor<i32, ["M"]> -> !hc.tensor<i32, ["M"]>
    return %r : !hc.tensor<i32, ["M"]>
  }
}

// -----

// CHECK: error: 'hc.as_layout' op layout must be one of "row_major", "col_major", got "weird"
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.as_layout %v {layout = "weird"} : !hc.undef -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.load' op shape rank (2) does not match index count (1)
module {
  func.func @bad(%buf: !hc.undef, %i: !hc.undef) -> !hc.undef {
    %t = hc.load %buf[%i] {shape = #hc.shape<["M", "K"]>}
        : (!hc.undef, !hc.undef) -> !hc.undef
    return %t : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.call' op 'missing' does not reference a valid hc.func
module {
  func.func @bad(%x: !hc.undef) -> !hc.undef {
    %r = hc.call @missing(%x) : (!hc.undef) -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.call_intrinsic' op 'plain_func' does not reference a valid hc.intrinsic
module {
  hc.func @plain_func { hc.return }
  func.func @bad(%x: !hc.undef) -> !hc.undef {
    %r = hc.call_intrinsic @plain_func(%x) : (!hc.undef) -> !hc.undef
    return %r : !hc.undef
  }
}
