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

// CHECK: error: subgroup_size must be non-negative
module {
  func.func @bad(%arg0: !hc.group<subgroup_size = #hc.expr<"-1">>) {
    return
  }
}

// -----

// CHECK: error: 'hc.getitem' op expected at least one index
module {
  hc.func @bad(%arg0: !hc.undef) {
    %x = hc.getitem %arg0[] : (!hc.undef) -> !hc.undef
    hc.return
  }
}

// -----

// CHECK: error: 'hc.getitem' op expected exactly one index when the base is a tuple, got 2
module {
  hc.func @bad(%arg0: tuple<!hc.undef, !hc.undef>, %idx: !hc.undef) {
    %x = hc.getitem %arg0[%idx, %idx]
        : (tuple<!hc.undef, !hc.undef>, !hc.undef, !hc.undef) -> !hc.undef
    hc.return
  }
}

// -----

// CHECK: error: 'hc.store' op mask operand requires a bare tensor/vector source, got '!hc.tensor<f32, ["4"]>'
module {
  hc.func @bad(%buf: !hc.buffer<f32, []>,
               %src: !hc.tensor<f32, ["4"]>,
               %mask: !hc.bare_tensor<!hc.pred, ["4"]>) {
    hc.store %buf[], %src, mask %mask
        : (!hc.buffer<f32, []>, !hc.tensor<f32, ["4"]>,
           !hc.bare_tensor<!hc.pred, ["4"]>) -> ()
    hc.return
  }
}

// -----

// CHECK: error: 'hc.store' op mask type '!hc.bare_tensor<!hc.pred, ["8"]>' must match source validity type '!hc.bare_tensor<!hc.pred, ["4"]>'
module {
  hc.func @bad(%buf: !hc.buffer<f32, []>,
               %src: !hc.bare_tensor<f32, ["4"]>,
               %mask: !hc.bare_tensor<!hc.pred, ["8"]>) {
    hc.store %buf[], %src, mask %mask
        : (!hc.buffer<f32, []>, !hc.bare_tensor<f32, ["4"]>,
           !hc.bare_tensor<!hc.pred, ["8"]>) -> ()
    hc.return
  }
}

// -----

// CHECK: error: 'hc.tuple' op result tuple arity 1 does not match element count 2
module {
  hc.func @bad(%arg0: !hc.undef, %arg1: !hc.undef) {
    %x = hc.tuple(%arg0, %arg1)
        : (!hc.undef, !hc.undef) -> tuple<!hc.undef>
    hc.return
  }
}

// -----

// CHECK: error: 'hc.tuple' op element #1 type 'f32' does not match result tuple element type 'i32'
module {
  hc.func @bad(%arg0: !hc.undef, %arg1: f32) {
    %x = hc.tuple(%arg0, %arg1) : (!hc.undef, f32) -> tuple<!hc.undef, i32>
    hc.return
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

// CHECK: expected ::mlir::hc::EffectClass to be one of: pure, read, write, read_write
module {
  hc.intrinsic @bad scope = #hc.scope<"WorkItem"> effects = weird {}
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
        : (!hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
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
        : (!hc.undef, !hc.undef, !hc.undef) -> (!hc.undef) {
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
        : (i32, i32, i32) -> (i32) {
    ^bb0(%i: i32, %acc: i64):
      hc.yield %acc : i64
    }
    return %r : i32
  }
}

// -----

// CHECK: error: 'hc.group_id' op expected 2 result(s) for launch-geometry query, got 1
module {
  func.func @bad(%group: !hc.group<work_shape = #hc.shape<["M", "N"]>>) {
    %gid = hc.group_id %group
        : (!hc.group<work_shape = #hc.shape<["M", "N"]>>) -> !hc.idx<"$WG0">
    return
  }
}

// -----

// CHECK: error: 'hc.local_id' op expected 2 result(s) for launch-geometry query, got 1
module {
  func.func @bad(%wi: !hc.workitem<group_shape = #hc.shape<["16", "8"]>>) {
    %lid = hc.local_id %wi
        : (!hc.workitem<group_shape = #hc.shape<["16", "8"]>>) -> !hc.idx<"$WI0">
    return
  }
}

// -----

// CHECK: error: 'hc.subgroup_id' op expected 2 result(s) for launch-geometry query, got 1
module {
  func.func @bad(%sg: !hc.subgroup<group_shape = #hc.shape<["16", "8"]>>) {
    %sid = hc.subgroup_id %sg
        : (!hc.subgroup<group_shape = #hc.shape<["16", "8"]>>) -> !hc.idx<"$SG0">
    return
  }
}

// -----

// CHECK: error: 'hc.group_shape' op expected 2 result(s) for launch-geometry query, got 1
module {
  func.func @bad(%group: !hc.group<group_shape = #hc.shape<["16", "8"]>>) {
    %shape = hc.group_shape %group
        : (!hc.group<group_shape = #hc.shape<["16", "8"]>>) -> !hc.idx<"$WGS0">
    return
  }
}

// -----

// CHECK: error: 'hc.work_offset' op expected 2 result(s) for launch-geometry query, got 1
module {
  func.func @bad(%group: !hc.group<work_shape = #hc.shape<["M", "N"]>>) {
    %offset = hc.work_offset %group
        : (!hc.group<work_shape = #hc.shape<["M", "N"]>>) -> !hc.idx<"$WO0">
    return
  }
}

// -----

// CHECK: error: 'hc.work_shape' op expected 2 result(s) for launch-geometry query, got 1
module {
  func.func @bad(%group: !hc.group<work_shape = #hc.shape<["M", "N"]>>) {
    %shape = hc.work_shape %group
        : (!hc.group<work_shape = #hc.shape<["M", "N"]>>) -> !hc.idx<"$WS0">
    return
  }
}

// -----

// CHECK: error: 'hc.group_size' op expected 1 result(s) for launch-geometry query, got 2
module {
  func.func @bad(%group: !hc.group<group_shape = #hc.shape<["32"]>>) {
    %size:2 = hc.group_size %group
        : (!hc.group<group_shape = #hc.shape<["32"]>>) -> (!hc.idx<"$GSZ0">, !hc.idx<"$GSZ1">)
    return
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

// Reduce kind is a typed enum now; parser rejects garbage before the verifier
// is invoked, which is exactly the point of the conversion (was: StrAttr).
// CHECK: expected ::mlir::hc::ReduceKind to be one of: sum, max, min
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.reduce %v, kind = avg, axis = 0 : !hc.undef -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.reduce' op axis must be non-negative
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.reduce %v, kind = sum, axis = -1 : !hc.undef -> !hc.undef
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

// CHECK: error: 'hc.buffer_dim' op axis 0 is out of bounds for rank-0 buffer
module {
  func.func @bad(%buf: !hc.buffer<f32, []>) -> !hc.undef {
    %d = hc.buffer_dim %buf, axis = 0
        : !hc.buffer<f32, []> -> !hc.undef
    return %d : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.reduce' op axis 5 is out of bounds for rank-2 value
module {
  func.func @bad(%v: !hc.tensor<f32, ["M", "N"]>) -> !hc.undef {
    %r = hc.reduce %v, kind = sum, axis = 5
        : !hc.tensor<f32, ["M", "N"]> -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// CHECK: error: 'hc.with_inactive' op inactive value type 'f32' does not match element type 'i32'
module {
  func.func @bad(%v: !hc.tensor<i32, ["M"]>) -> !hc.tensor<i32, ["M"]> {
    %inactive = hc.const<0.0 : f32> : f32
    %r = hc.with_inactive %v, %inactive
        : (!hc.tensor<i32, ["M"]>, f32) -> !hc.tensor<i32, ["M"]>
    return %r : !hc.tensor<i32, ["M"]>
  }
}

// -----

// Layout slot accepts the named-enum keyword and the structured
// `#hc.layout<...>` attr; the custom parser rejects unknown keyword spellings
// up front with a located diagnostic listing the valid forms.
// CHECK: expected `row_major`, `col_major`, or `(#hc.layout<...>)`, got 'weird'
module {
  func.func @bad(%v: !hc.undef) -> !hc.undef {
    %r = hc.as_layout %v, layout = weird : !hc.undef -> !hc.undef
    return %r : !hc.undef
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

// -----

// Narrowed operand constraints reject non-numeric types in arithmetic.
// CHECK: 'hc.add' op operand #0 must be {{.*}}Placeholder type
// CHECK-SAME: Semantic tensor type
module {
  func.func @bad(%p: !hc.pred, %q: !hc.pred) -> !hc.undef {
    %r = hc.add %p, %q : (!hc.pred, !hc.pred) -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// Buffer-rooted ops reject `!hc.tensor` handles (no buffer in sight).
// CHECK: 'hc.buffer_dim' op operand #0 must be {{.*}}Placeholder type
// CHECK-SAME: Semantic buffer type
module {
  func.func @bad(%t: !hc.tensor<f32, ["M"]>) -> !hc.undef {
    %d = hc.buffer_dim %t, axis = 0 : !hc.tensor<f32, ["M"]> -> !hc.undef
    return %d : !hc.undef
  }
}

// -----

// Shaped ops reject scalar/idx operands — matmul on idx is meaningless.
// CHECK: 'hc.matmul' op operand #0 must be {{.*}}Placeholder type
// CHECK-SAME: Semantic tensor type
module {
  func.func @bad(%m: !hc.idx<"M">, %n: !hc.idx<"N">) -> !hc.undef {
    %r = hc.matmul %m, %n : (!hc.idx<"M">, !hc.idx<"N">) -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// Signature parity: wrong arity.
// CHECK: error: 'hc.call' op callee '@typed' expects 2 argument(s), call site provides 1
module {
  hc.func @typed(%a: i32, %b: i32) -> i32 { hc.return %a : i32 }
  func.func @bad(%a: i32) -> i32 {
    %r = hc.call @typed(%a) : (i32) -> i32
    return %r : i32
  }
}

// -----

// Signature parity: wrong arg type, both sides concrete.
// CHECK: error: 'hc.call' op arg #1 type 'f32' is incompatible with callee declaration 'i32'
module {
  hc.func @typed(%a: i32, %b: i32) -> i32 { hc.return %a : i32 }
  func.func @bad(%a: i32, %b: f32) -> i32 {
    %r = hc.call @typed(%a, %b) : (i32, f32) -> i32
    return %r : i32
  }
}

// -----

// Signature parity: wrong result type.
// CHECK: error: 'hc.call' op result #0 type 'f32' is incompatible with callee declaration 'i32'
module {
  hc.func @typed(%a: i32) -> i32 { hc.return %a : i32 }
  func.func @bad(%a: i32) -> f32 {
    %r = hc.call @typed(%a) : (i32) -> f32
    return %r : f32
  }
}

// -----

// Signature parity: same story on `hc.call_intrinsic`.
// CHECK: error: 'hc.call_intrinsic' op callee '@sized' expects 1 argument(s), call site provides 2
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @bad(%a: i32, %b: i32) -> i32 {
    %r = hc.call_intrinsic @sized(%a, %b) : (i32, i32) -> i32
    return %r : i32
  }
}

// -----

// const_kwargs whitelist: missing kwarg on the call site fails verify.
// CHECK: error: 'hc.call_intrinsic' op missing required const kwarg 'wave_size' declared by callee '@wave'
module {
  hc.intrinsic @wave(%a: !hc.undef) -> !hc.undef
      scope = #hc.scope<"SubGroup">
      const_kwargs = ["wave_size"]
      parameters = ["a", "wave_size"]
      keyword_only = ["wave_size"] {}
  func.func @bad(%a: !hc.undef) -> !hc.undef {
    %r = hc.call_intrinsic @wave(%a) : (!hc.undef) -> !hc.undef
    return %r : !hc.undef
  }
}

// -----

// When an intrinsic records full parameter order, const_kwargs must be
// drawn from that list.
// CHECK: error: 'hc.intrinsic' op const_kwargs entry 'arch' is not listed in parameters
module {
  hc.intrinsic @bad(%x: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      const_kwargs = ["arch"]
      parameters = ["x"] {}
}

// -----

// parameters is the full ordered list behind function_type and const_kwargs;
// do not attach it to a legacy declaration with no signature.
// CHECK: error: 'hc.intrinsic' op parameters requires function_type to define the runtime SSA operand signature
module {
  hc.intrinsic @bad scope = #hc.scope<"WorkItem">
      parameters = ["x"] {}
}

// -----

// const_kwargs without parameters has no declared universe to validate
// against.
// CHECK: error: 'hc.intrinsic' op const_kwargs requires parameters to declare the full intrinsic parameter order
module {
  hc.intrinsic @bad scope = #hc.scope<"WorkItem">
      const_kwargs = ["wave_size"] {}
}

// -----

// keyword_only without parameters has no declared universe to validate
// against.
// CHECK: error: 'hc.intrinsic' op keyword_only requires parameters to declare the full intrinsic parameter order
module {
  hc.intrinsic @bad scope = #hc.scope<"WorkItem">
      keyword_only = ["lane"] {}
}

// -----

// Non-empty intrinsic signatures must name their operand order.
// CHECK: error: 'hc.intrinsic' op function_type with inputs requires parameters to name the intrinsic operand order
module {
  hc.intrinsic @bad(%x: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem"> {}
}

// -----

// Duplicate names make keyword binding ambiguous.
// CHECK: error: 'hc.intrinsic' op duplicate parameter name 'x'
module {
  hc.intrinsic @bad(%x: !hc.undef, %y: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      parameters = ["x", "x"] {}
}

// -----

// Duplicate const kwargs are never useful and obscure missing-attribute
// diagnostics at call sites.
// CHECK: error: 'hc.intrinsic' op duplicate const_kwargs entry 'arch'
module {
  hc.intrinsic @bad(%x: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      const_kwargs = ["arch", "arch"]
      parameters = ["x", "arch"]
      keyword_only = ["arch"] {}
}

// -----

// keyword_only names must be strings drawn from parameters.
// CHECK: error: 'hc.intrinsic' op keyword_only entry 'lane' is not listed in parameters
module {
  hc.intrinsic @bad(%x: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      parameters = ["x"]
      keyword_only = ["lane"] {}
}

// -----

// Duplicate keyword_only entries would make call-site checks ambiguous.
// CHECK: error: 'hc.intrinsic' op duplicate keyword_only entry 'lane'
module {
  hc.intrinsic @bad(%x: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      parameters = ["x", "lane"]
      keyword_only = ["lane", "lane"] {}
}

// -----

// keyword_only names form a suffix of the full parameter list.
// CHECK: error: 'hc.intrinsic' op positional parameter 'x' cannot follow a keyword-only parameter
module {
  hc.intrinsic @bad(%x: !hc.undef, %lane: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      parameters = ["lane", "x"]
      keyword_only = ["lane"] {}
}

// -----

// const_kwargs are specialization kwargs, so they must be keyword-only.
// CHECK: error: 'hc.intrinsic' op const_kwargs entry 'arch' must be listed in keyword_only
module {
  hc.intrinsic @bad(%x: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      const_kwargs = ["arch"]
      parameters = ["x", "arch"]
      keyword_only = [] {}
}

// -----

// The function_type on an intrinsic is the runtime operand signature:
// the declared parameter list with const_kwargs filtered out.
// CHECK: error: 'hc.intrinsic' op function_type declares 2 input(s) but non-const parameters declare 1 runtime SSA operand(s)
module {
  hc.intrinsic @bad(%x: !hc.undef, %arch: !hc.undef) -> !hc.undef
      scope = #hc.scope<"WorkItem">
      const_kwargs = ["arch"]
      parameters = ["x", "arch"]
      keyword_only = ["arch"] {}
}

// -----

// Tensor allocators are workgroup scope only; sitting inside a
// subgroup_region is a scope error.
// CHECK: error: 'hc.zeros' op tensor allocator is workgroup scope only; enclosed by hc.subgroup_region which narrows the scope
hc.kernel @bad {
  hc.subgroup_region {
    %m = hc.const<"M"> : !hc.undef
    %shape = hc.tuple(%m) : (!hc.undef) -> tuple<!hc.undef>
    %z = hc.zeros shape %shape : (tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
    hc.return
  }
  hc.return
}

// -----

// And workitem_region is equally wrong.
// CHECK: error: 'hc.empty' op tensor allocator is workgroup scope only; enclosed by hc.workitem_region which narrows the scope
hc.kernel @bad {
  hc.workitem_region {
    %m = hc.const<"M"> : !hc.undef
    %shape = hc.tuple(%m) : (!hc.undef) -> tuple<!hc.undef>
    %z = hc.empty shape %shape : (tuple<!hc.undef>) -> !hc.tensor<f32, ["M"]>
    hc.return
  }
  hc.return
}

// -----

// Kernels never return values; declaring results here points at a bug in
// the frontend/legalizer, not at runtime.
// CHECK: error: 'hc.kernel' op kernel signatures must declare no results; kernels return via an operand-less `hc.return`
hc.kernel @bad(%a: i32) -> i32 {
  hc.return
}

// -----

// Signature-less `hc.func` with block args on the body means "frontend
// forgot to synthesize a function_type"; we catch it rather than let the
// op round-trip into a silently broken state.
// CHECK: error: 'hc.func' op body block takes 1 argument(s) but no function_type is declared
hc.func @orphan_arg {
^bb0(%a: i32):
  hc.return
}

// -----

// Signature carried via the attr-dict with no matching block args: the
// inline form can't trigger this (parser derives function_type from args),
// but nothing stops a builder or round-trip from attaching a stale
// function_type on a no-arg body. Verifier catches the count mismatch.
// CHECK: error: 'hc.func' op body block takes 0 argument(s) but function_type declares 2 input(s)
hc.func @bad attributes {function_type = (i32, i32) -> ()} {
  hc.return
}

// -----

// Same story on `hc.kernel`.
// CHECK: error: 'hc.kernel' op body block takes 0 argument(s) but function_type declares 1 input(s)
hc.kernel @bad attributes {function_type = (i32) -> ()} {
  hc.return
}

// -----

// Same story on `hc.intrinsic`. The scope keyword lives before the
// attr-dict so it has to show up here too.
// CHECK: error: 'hc.intrinsic' op body block takes 0 argument(s) but function_type declares 1 input(s)
hc.intrinsic @bad scope = #hc.scope<"WorkItem">
    attributes {function_type = (i32) -> ()} {}

// -----

// Per-argument type mismatch: arity matches, but the block arg type
// disagrees with the declared function_type input. Needs attr-dict form
// (the inline-signature parser derives function_type from the block args,
// so the two can't disagree).
// CHECK: error: 'hc.func' op body block argument #1 type 'f32' does not match function_type input 'i32'
hc.func @type_mismatch attributes {function_type = (i32, i32) -> ()} {
^bb0(%a: i32, %b: f32):
  hc.return
}

// -----

// `hc.return` inside a kernel must be operand-less — kernels never produce
// a value. The parity check runs whether or not a signature is declared.
// CHECK: error: 'hc.return' op `hc.return` inside `hc.kernel` must be operand-less
hc.kernel @bad {
  %c = hc.const<0 : i32> : i32
  hc.return %c : i32
}

// -----

// Kernels reject returned values even through nested scope regions —
// `hc.return` falls through `hc.subgroup_region`/`hc.workitem_region` and
// terminates the kernel.
// CHECK: error: 'hc.return' op `hc.return` inside `hc.kernel` must be operand-less
hc.kernel @bad {
  hc.subgroup_region {
    hc.workitem_region {
      %c = hc.const<0 : i32> : i32
      hc.return %c : i32
    }
  }
}

// -----

// `hc.return` in a signatured func with the wrong arity is caught against
// the enclosing function_type's result list.
// CHECK: error: 'hc.return' op returns 0 value(s) but enclosing hc.func declares 1 result(s)
hc.func @bad(%a: i32) -> i32 {
  hc.return
}

// -----

// `hc.return` in a signatured func with a concrete mismatching type.
// CHECK: error: 'hc.return' op returned value #0 type 'f32' does not match enclosing hc.func result type 'i32'
hc.func @bad(%a: i32, %b: f32) -> i32 {
  hc.return %b : f32
}

// -----

// Same parity check on `hc.intrinsic`. Intrinsics are usually body-less
// declarations, but when they do carry a body, the terminator is checked
// like `hc.func`.
// CHECK: error: 'hc.return' op returns 2 value(s) but enclosing hc.intrinsic declares 1 result(s)
hc.intrinsic @bad(%a: i32) -> i32 scope = #hc.scope<"WorkItem">
    parameters = ["a"] {
  hc.return %a, %a : i32, i32
}

// -----

// `hc.workitem_region` with declared results must terminate with
// `hc.yield`. A `hc.region_return` terminator here is contradictory:
// it's the pre-promotion form while the op is simultaneously claiming
// post-promotion shape (non-empty `$results`). Most producers hit this
// through a builder bug; the frontend emits `hc.region_return` only on
// results-less region ops.
// CHECK: error: 'hc.workitem_region' op declares results; body must terminate with `hc.yield`, got hc.region_return
hc.kernel @bad {
  hc.workitem_region -> (!hc.undef) {
    hc.region_return ["x"]
  }
  hc.return
}

// -----

// Yield arity must match declared results.
// CHECK: error: 'hc.workitem_region' op body yield produces 1 values, expected 2
hc.kernel @bad {
  %v = hc.const<1 : i64> : !hc.undef
  hc.workitem_region -> (!hc.undef, !hc.undef) {
    hc.yield %v : !hc.undef
  }
  hc.return
}

// -----

// Concrete yield type disagreement with declared result type is caught;
// the `!hc.undef` escape applies on either side so pre-inference IR
// still round-trips.
// CHECK: error: 'hc.subgroup_region' op body yield[0] type 'i32' does not match result[0] type 'i64'
hc.kernel @bad {
  %v = hc.const<1 : i32> : i32
  %r = hc.subgroup_region -> (i64) {
    hc.yield %v : i32
  }
  hc.return
}

// -----

// Same verifier runs on `hc.subgroup_region`. Arity-zero yield is a
// separate case from type-mismatch so a future divergence between
// the two region kinds fails here, not silently passes.
// CHECK: error: 'hc.subgroup_region' op body yield produces 0 values, expected 1
hc.kernel @bad {
  hc.subgroup_region -> (!hc.undef) {
    hc.yield
  }
  hc.return
}

// -----

// `hc.region_return` with a duplicate name: two entries would spawn two
// results and two writebacks for the same Python-level slot.
// CHECK: error: 'hc.region_return' op duplicate name 'acc' in `names`
hc.kernel @bad {
  %v = hc.const<1 : i64> : !hc.undef
  hc.workitem_region {
    hc.assign "acc", %v : !hc.undef
    hc.region_return ["acc", "acc"]
  }
  hc.return
}

// -----

// #hc.layout requires every keyword field; missing one is a parser-time error
// rather than a verify-time one.
// CHECK: error: #hc.layout requires shape_syms, index_syms, params, storage_size, offset
module attributes {
  test.layout = #hc.layout<shape_syms = ["W"], index_syms = ["i"], params = {},
                           storage_size = #hc.expr<"W">>
} {}

// -----

// Names must be pairwise disjoint across shape_syms, index_syms, and params:
// they share one expression-symbol namespace at substitution time.
// CHECK: error: duplicate name 'i' across #hc.layout name lists
module attributes {
  test.layout = #hc.layout<shape_syms = ["i", "H"], index_syms = ["i", "j"],
                           params = {},
                           storage_size = #hc.expr<"H">,
                           offset = #hc.expr<"i">>
} {}

// -----

// params keys live in the same namespace as shape_syms / index_syms.
// CHECK: error: duplicate name 'W' across #hc.layout name lists
module attributes {
  test.layout = #hc.layout<shape_syms = ["W"], index_syms = ["i"],
                           params = {W = #hc.expr<"4">},
                           storage_size = #hc.expr<"W">,
                           offset = #hc.expr<"i">>
} {}

// -----

// params values must be #hc.expr; verifier rejects anything else.
// CHECK: error: params value for 'row_stride' must be a #hc.expr attribute
module attributes {
  test.layout = #hc.layout<shape_syms = ["W"], index_syms = ["i"],
                           params = {row_stride = 4 : i64},
                           storage_size = #hc.expr<"W">,
                           offset = #hc.expr<"i">>
} {}

// -----

// `!hc.ptr` rejects unknown address-space keywords with the upstream
// `EnumParameter` diagnostic, which lists the legal alternatives.
// CHECK: error: expected ::mlir::hc::AddrSpace to be one of: workgroup, global, private
module {
  func.func @bad(%p: !hc.ptr<somewhere, f16>) {
    return
  }
}

// -----

// Empty `<>`: parser asks for the required address-space keyword first
// and finds the closing `>` instead.
// CHECK: error: expected valid keyword or string
module {
  func.func @bad(%p: !hc.ptr<>) {
    return
  }
}

// -----

// `hc.ptr_offset` rejects address-space mismatches; mixing `workgroup`
// and `global` would silently produce an invalid GPU pointer at
// codegen.
// CHECK: error: 'hc.ptr_offset' op address space mismatch: source workgroup vs result global
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %i: index) {
    %r = hc.ptr_offset %p, %i
        : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<global, f16>
    hc.return
  }
}

// -----

// Element-type mismatch on `hc.ptr_offset` — both pointers typed but
// disagreeing on the element. Asymmetric typed/opaque is rejected by a
// separate diagnostic below.
// CHECK: error: 'hc.ptr_offset' op element type mismatch: source 'f16' vs result 'f32'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %i: index) {
    %r = hc.ptr_offset %p, %i
        : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup, f32>
    hc.return
  }
}

// -----

// Asymmetric typed/opaque on `hc.ptr_offset`: dropping the element
// type happens once at the LLVM-lowering boundary, not in the middle
// of pointer arithmetic.
// CHECK: error: 'hc.ptr_offset' op source and result must agree on whether the pointer is typed
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %i: index) {
    %r = hc.ptr_offset %p, %i
        : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup>
    hc.return
  }
}

// -----

// `hc.ptr_load` on a typed pointer enforces result-element parity.
// CHECK: error: 'hc.ptr_load' op result element type 'f32' must match pointer element type 'f16'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>) {
    %v = hc.ptr_load %p : !hc.ptr<workgroup, f16> -> f32
    hc.return
  }
}

// -----

// `hc.ptr_store` symmetrically enforces value-element parity.
// CHECK: error: 'hc.ptr_store' op value element type 'f32' must match pointer element type 'f16'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %v: f32) {
    hc.ptr_store %v, %p : f32, !hc.ptr<workgroup, f16>
    hc.return
  }
}

// -----

// Vector load on a typed pointer: the vector's element type must match
// the pointer's element type. Width is unconstrained.
// CHECK: error: 'hc.ptr_load' op result element type 'f32' must match pointer element type 'f16'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>) {
    %v = hc.ptr_load %p : !hc.ptr<workgroup, f16> -> vector<4xf32>
    hc.return
  }
}

// -----

// Vector store symmetrically: the vector's element type, not the
// vector type itself, is checked against the pointer's element type.
// CHECK: error: 'hc.ptr_store' op value element type 'f32' must match pointer element type 'f16'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %v: vector<4xf32>) {
    hc.ptr_store %v, %p : vector<4xf32>, !hc.ptr<workgroup, f16>
    hc.return
  }
}

// -----

// Predicated load: pointer-element-type parity is reused from the
// unconditional path.
// CHECK: error: 'hc.ptr_load_pred' op result element type 'f32' must match pointer element type 'f16'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %pred: i1, %fill: f32) {
    %v = hc.ptr_load_pred %p, %pred passthrough %fill
        : !hc.ptr<workgroup, f16>, i1, f32 -> f32
    hc.return
  }
}

// -----

// Predicated load: scalar value paired with vector predicate is
// rejected — predicate shape must mirror value shape.
// CHECK: error: 'hc.ptr_load_pred' op predicate shape must match value shape
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %mask: vector<4xi1>, %fill: f16) {
    %v = hc.ptr_load_pred %p, %mask passthrough %fill
        : !hc.ptr<workgroup, f16>, vector<4xi1>, f16 -> f16
    hc.return
  }
}

// -----

// Predicated load: vector value with mismatched-length predicate is
// rejected — vector lane counts must match.
// CHECK: error: 'hc.ptr_load_pred' op predicate shape 'vector<4xi1>' must match value shape 'vector<8xf16>'
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %mask: vector<4xi1>,
               %fill: vector<8xf16>) {
    %v = hc.ptr_load_pred %p, %mask passthrough %fill
        : !hc.ptr<workgroup, f16>, vector<4xi1>, vector<8xf16>
        -> vector<8xf16>
    hc.return
  }
}

// -----

// Predicated store: vector value with scalar predicate is rejected.
// CHECK: error: 'hc.ptr_store_pred' op predicate shape must match value shape
module {
  hc.func @bad(%p: !hc.ptr<workgroup, f16>, %v: vector<4xf16>, %pred: i1) {
    hc.ptr_store_pred %v, %p, %pred
        : vector<4xf16>, !hc.ptr<workgroup, f16>, i1
    hc.return
  }
}

// -----

// `hc.generic` requires at least one iter; an empty `iter ()` clause is
// rejected at parse time before the verifier ever runs. The upstream
// `parseKeyword` diagnostic fires first when the parser hits the
// closing `)` instead of a kind keyword.
// CHECK: error: custom op 'hc.generic' expected valid keyword
module {
  func.func @bad(%c: !hc.bare_tensor<f32, ["N"]>)
      -> !hc.bare_tensor<f32, ["N"]> {
    %r = hc.generic
        iter ()
        ins ()
        outs (%c at [#hc.expr<"0">] : !hc.bare_tensor<f32, ["N"]>)
        -> (!hc.bare_tensor<f32, ["N"]>) {
    ^bb0(%cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f32, ["N"]>
  }
}

// -----

// `outs ()` is parser-time-rejected because the body has nothing to
// terminate against and the op would have no results.
// CHECK: error: custom op 'hc.generic' outs clause requires at least one entry
module {
  func.func @bad(%n: index) {
    %r = hc.generic
        iter (parallel i = %n : index)
        ins ()
        outs ()
        -> () {
      hc.yield
    }
    return
  }
}

// -----

// Bad iter kind keyword fails at parse with the listed alternatives.
// CHECK: error: custom op 'hc.generic' expected `parallel` or `reduction`, got 'sequential'
module {
  func.func @bad(%n: index, %c: !hc.bare_tensor<f32, ["N"]>)
      -> !hc.bare_tensor<f32, ["N"]> {
    %r = hc.generic
        iter (sequential i = %n : index)
        ins ()
        outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        -> (!hc.bare_tensor<f32, ["N"]>) {
    ^bb0(%cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f32, ["N"]>
  }
}

// -----

// Output offsets may reference parallel iters only — a reduction iter
// would imply writing the same slot many times without specifying a
// combinator, which the op does not model. The diagnostic names the
// offending axis on top of the operand index.
// CHECK: error: 'hc.generic' op output #0 axis 1 offset references reduction iter 'k'
module {
  func.func @bad(%m: index, %k: index,
                 %a: !hc.bare_tensor<f32, ["M"]>,
                 %c: !hc.bare_tensor<f32, ["M", "K"]>)
      -> !hc.bare_tensor<f32, ["M", "K"]> {
    %r = hc.generic
        iter (parallel i = %m : index, reduction k = %k : index)
        ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>)
        outs (%c at [#hc.expr<"i">, #hc.expr<"k">]
                 : !hc.bare_tensor<f32, ["M", "K"]>)
        -> (!hc.bare_tensor<f32, ["M", "K"]>) {
    ^bb0(%av: f32, %cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f32, ["M", "K"]>
  }
}

// -----

// Duplicate iter sym names: the symbolic engine binds them positionally
// at access time, so two iters sharing a name is ambiguous.
// CHECK: error: 'hc.generic' op duplicate iter sym 'i'
module {
  func.func @bad(%m: index, %n: index, %c: !hc.bare_tensor<f32, ["M*N"]>)
      -> !hc.bare_tensor<f32, ["M*N"]> {
    %r = hc.generic
        iter (parallel i = %m : index, parallel i = %n : index)
        ins ()
        outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M*N"]>)
        -> (!hc.bare_tensor<f32, ["M*N"]>) {
    ^bb0(%cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f32, ["M*N"]>
  }
}

// -----

// Body block arg count must equal ins + outs (one scalar per operand).
// CHECK: error: 'hc.generic' op body block takes 1 argument(s), expected 2 (one per ins/outs)
module {
  func.func @bad(%n: index,
                 %a: !hc.bare_tensor<f32, ["N"]>,
                 %c: !hc.bare_tensor<f32, ["N"]>)
      -> !hc.bare_tensor<f32, ["N"]> {
    %r = hc.generic
        iter (parallel i = %n : index)
        ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        -> (!hc.bare_tensor<f32, ["N"]>) {
    ^bb0(%cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f32, ["N"]>
  }
}

// -----

// Concrete operand element types must match the corresponding body
// argument; `!hc.undef` would escape but `f32` body arg vs `f16` element
// type doesn't.
// CHECK: error: 'hc.generic' op body argument #0 type 'f32' does not match ins #0 element type 'f16'
module {
  func.func @bad(%n: index,
                 %a: !hc.bare_tensor<f16, ["N"]>,
                 %c: !hc.bare_tensor<f32, ["N"]>)
      -> !hc.bare_tensor<f32, ["N"]> {
    %r = hc.generic
        iter (parallel i = %n : index)
        ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f16, ["N"]>)
        outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        -> (!hc.bare_tensor<f32, ["N"]>) {
    ^bb0(%av: f32, %cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f32, ["N"]>
  }
}

// -----

// Yield arity must match the number of outputs; one missing operand and
// the lowering can't tell which output got the new value.
// CHECK: error: 'hc.generic' op hc.yield arity 1 != outs count 2
module {
  func.func @bad(%n: index,
                 %c0: !hc.bare_tensor<f32, ["N"]>,
                 %c1: !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>, !hc.bare_tensor<f32, ["N"]>) {
    %r:2 = hc.generic
        iter (parallel i = %n : index)
        ins ()
        outs (%c0 at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>,
              %c1 at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        -> (!hc.bare_tensor<f32, ["N"]>, !hc.bare_tensor<f32, ["N"]>) {
    ^bb0(%c0v: f32, %c1v: f32):
      hc.yield %c0v : f32
    }
    return %r#0, %r#1
        : !hc.bare_tensor<f32, ["N"]>, !hc.bare_tensor<f32, ["N"]>
  }
}

// -----

// Result types must equal outs operand types one-to-one.
// CHECK: error: 'hc.generic' op result #0 type '!hc.bare_tensor<f16, ["N"]>' does not match outs operand type '!hc.bare_tensor<f32, ["N"]>'
module {
  func.func @bad(%n: index, %c: !hc.bare_tensor<f32, ["N"]>)
      -> !hc.bare_tensor<f16, ["N"]> {
    %r = hc.generic
        iter (parallel i = %n : index)
        ins ()
        outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        -> (!hc.bare_tensor<f16, ["N"]>) {
    ^bb0(%cv: f32):
      hc.yield %cv : f32
    }
    return %r : !hc.bare_tensor<f16, ["N"]>
  }
}

// -----

// Result count must equal the number of *value-typed* outs. A ptr-typed
// out lands its work in memory (Read+Write effect) and contributes no
// SSA result, so spelling a result for a pure-store generic is wrong.
// CHECK: error: 'hc.generic' op results count 1 != value-typed outs count 0 (ptr/buffer outs contribute no SSA result)
module {
  func.func @bad(%n: index,
                 %src: !hc.bare_tensor<f32, ["N"]>,
                 %dst: !hc.ptr<global, f32>)
      -> !hc.bare_tensor<f32, ["N"]> {
    %r = hc.generic
        iter (parallel i = %n : index)
        ins (%src at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> (!hc.bare_tensor<f32, ["N"]>) {
    ^bb0(%sv: f32, %dv: f32):
      hc.yield %sv : f32
    }
    return %r : !hc.bare_tensor<f32, ["N"]>
  }
}

// -----

// On a typed ptr out, the body block-arg / yield element type is the
// pointee, not the operand type itself. `f16` body arg vs `f32` pointee
// is the same shape of mistake as the value-out parity error above; the
// diagnostic just calls out the offending slot.
// CHECK: error: 'hc.generic' op body argument #1 type 'f16' does not match outs #0 element type 'f32'
module {
  func.func @bad(%n: index,
                 %src: !hc.bare_tensor<f32, ["N"]>,
                 %dst: !hc.ptr<global, f32>) {
    hc.generic
        iter (parallel i = %n : index)
        ins (%src at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%sv: f32, %dv: f16):
      hc.yield %dv : f16
    }
    return
  }
}

// -----

// `hc.idx_apply`'s symbols list must line up element-wise with the
// operand list — same number of entries, no missing or extra slot.
// CHECK: error: 'hc.idx_apply' op symbols list has 2 entries but the op has 1 operand(s)
module {
  func.func @bad(%i: index) {
    %off = hc.idx_apply (%i) {symbols = ["i", "K"]}
         : (index) -> !hc.idx<"i + K">
    return
  }
}

// -----

// Each entry of the symbols list must correspond to a free symbol of
// the carried expression — the binding has nothing to do otherwise.
// Listing 'q' against `!hc.idx<"i">` is a producer bug worth flagging.
// CHECK: error: 'hc.idx_apply' op symbol 'q' is not a free symbol of the carried expression / predicate
module {
  func.func @bad(%i: index) {
    %off = hc.idx_apply (%i) {symbols = ["q"]}
         : (index) -> !hc.idx<"i">
    return
  }
}

// -----

// Symbols must be unique within one op so the binding stays a function
// from name to operand. Two slots claiming the same name would race.
// CHECK: error: 'hc.idx_apply' op duplicate symbol binding for 'i'
module {
  func.func @bad(%i: index, %j: index) {
    %off = hc.idx_apply (%i, %j) {symbols = ["i", "i"]}
         : (index, index) -> !hc.idx<"i">
    return
  }
}

// -----

// Bare `!hc.idx` (no pinned expression) carries no symbols to bind, so
// the apply op has nothing to lower. The check mirrors the
// `hc.materialize_bound_expr` pin requirement.
// CHECK: error: 'hc.idx_apply' op result must pin a symbolic expression
module {
  func.func @bad(%i: index) {
    %off = hc.idx_apply (%i) {symbols = ["i"]}
         : (index) -> !hc.idx
    return
  }
}

// -----

// `hc.pred_apply` enforces the same shape rules with a predicate-side
// pin requirement on the result type.
// CHECK: error: 'hc.pred_apply' op result must pin a symbolic predicate
module {
  func.func @bad(%i: index) {
    %p = hc.pred_apply (%i) {symbols = ["i"]}
       : (index) -> !hc.pred
    return
  }
}

// -----

// The "outputs reference parallel iters only" rule applies to ptr-typed
// outs too — a reduction iter on a memory destination's offset would
// store the same address several times along the reduction without a
// combinator. Same diagnostic shape as for value-typed outs.
// CHECK: error: 'hc.generic' op output #0 axis 1 offset references reduction iter 'k'
module {
  func.func @bad(%m: index, %k: index,
                 %a: !hc.bare_tensor<f32, ["M"]>,
                 %dst: !hc.ptr<global, f32>) {
    hc.generic
        iter (parallel i = %m : index, reduction k = %k : index)
        ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>)
        outs (%dst at [#hc.expr<"i">, #hc.expr<"k">] : !hc.ptr<global, f32>)
        -> () {
    ^bb0(%av: f32, %dv: f32):
      hc.yield %av : f32
    }
    return
  }
}
