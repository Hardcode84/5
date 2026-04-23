// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: func.func @buffer_queries
// CHECK: hc.buffer_dim %{{.*}}, axis = 1 : !hc.undef -> !hc.undef
// All eight `hc.slice_expr` operand combinations round-trip independently
// so the `AttrSizedOperandSegments` wiring between operand presence and
// printed keywords is exercised exhaustively. The printer keeps a literal
// space before each optional keyword, so `hc.slice_expr(upper = ...)`
// round-trips as `hc.slice_expr( upper = ...)` — allow the leading space
// in each CHECK.
// CHECK: hc.slice_expr() : () -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}}) : (!hc.undef) -> !hc.undef
// CHECK: hc.slice_expr({{ *}}upper = %{{.*}}) : (!hc.undef) -> !hc.undef
// CHECK: hc.slice_expr({{ *}}step = %{{.*}}) : (!hc.undef) -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}} upper = %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}} step = %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.slice_expr({{ *}}upper = %{{.*}} step = %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.slice_expr(lower = %{{.*}} upper = %{{.*}} step = %{{.*}}) : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.buffer_view %{{.*}}[%{{.*}}, %{{.*}}] : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
func.func @buffer_queries(%buf: !hc.undef, %lo: !hc.undef, %hi: !hc.undef,
                          %st: !hc.undef) {
  %d        = hc.buffer_dim %buf, axis = 1 : !hc.undef -> !hc.undef
  %sl_none  = hc.slice_expr() : () -> !hc.undef
  %sl_lo    = hc.slice_expr(lower = %lo) : (!hc.undef) -> !hc.undef
  %sl_hi    = hc.slice_expr(upper = %hi) : (!hc.undef) -> !hc.undef
  %sl_st    = hc.slice_expr(step = %st)  : (!hc.undef) -> !hc.undef
  %sl_lo_hi = hc.slice_expr(lower = %lo upper = %hi)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %sl_lo_st = hc.slice_expr(lower = %lo step = %st)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %sl_hi_st = hc.slice_expr(upper = %hi step = %st)
      : (!hc.undef, !hc.undef) -> !hc.undef
  %sl_all   = hc.slice_expr(lower = %lo upper = %hi step = %st)
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  %bv  = hc.buffer_view %buf[%sl_lo_hi, %sl_lo]
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  return
}

// Symbol-user verification wants a matching `hc.func` for every `hc.call` and
// a matching `hc.intrinsic` for every `hc.call_intrinsic`; declare the
// callees up front so the round-trip is self-contained.
hc.intrinsic @wmma scope = #hc.scope<"SubGroup"> {}
hc.func @helper {
  hc.return
}

// CHECK-LABEL: func.func @data_movement
// CHECK: hc.load %{{.*}}[%{{.*}}, %{{.*}}] {shape = #hc.shape<["M", "K"]>}
// CHECK-SAME: (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.vload %{{.*}}[%{{.*}}] {shape = #hc.shape<["K"]>}
// CHECK-SAME: (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.store %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}
// CHECK: hc.vec %{{.*}} : !hc.undef -> !hc.undef
// CHECK: hc.with_inactive %{{.*}} {inactive = 0.000000e+00 : f32} : !hc.undef -> !hc.undef
// CHECK: hc.as_layout %{{.*}} {layout = "row_major"} : !hc.undef -> !hc.undef
func.func @data_movement(%buf: !hc.undef, %i: !hc.undef, %j: !hc.undef,
                         %v: !hc.undef) {
  %t = hc.load %buf[%i, %j] {shape = #hc.shape<["M", "K"]>}
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  %vec = hc.vload %buf[%i] {shape = #hc.shape<["K"]>}
      : (!hc.undef, !hc.undef) -> !hc.undef
  hc.store %buf[%i, %j], %v
      : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> ()
  %vec2 = hc.vec %t : !hc.undef -> !hc.undef
  %masked = hc.with_inactive %v {inactive = 0.0 : f32}
      : !hc.undef -> !hc.undef
  %reshaped = hc.as_layout %v {layout = "row_major"}
      : !hc.undef -> !hc.undef
  return
}

// CHECK-LABEL: func.func @allocators
// CHECK: hc.vzeros {shape = #hc.shape<["16", "16"]>} : !hc.vector<f32, ["16", "16"]>
// CHECK: hc.vones {shape = #hc.shape<["16"]>} : !hc.vector<i1, ["16"]>
// CHECK: hc.vfull %{{.*}} {shape = #hc.shape<["16"]>} : !hc.undef -> !hc.vector<f32, ["16"]>
// CHECK: hc.zeros {shape = #hc.shape<["M"]>} : !hc.tensor<f32, ["M"]>
// CHECK: hc.ones {shape = #hc.shape<["M"]>} : !hc.tensor<i1, ["M"]>
// CHECK: hc.full %{{.*}} {shape = #hc.shape<["M"]>} : !hc.undef -> !hc.tensor<f32, ["M"]>
// CHECK: hc.empty {shape = #hc.shape<["M"]>} : !hc.tensor<f32, ["M"]>
func.func @allocators(%s: !hc.undef) {
  %vz = hc.vzeros {shape = #hc.shape<["16", "16"]>}
      : !hc.vector<f32, ["16", "16"]>
  %vo = hc.vones  {shape = #hc.shape<["16"]>} : !hc.vector<i1, ["16"]>
  %vf = hc.vfull  %s {shape = #hc.shape<["16"]>}
      : !hc.undef -> !hc.vector<f32, ["16"]>
  %tz = hc.zeros  {shape = #hc.shape<["M"]>} : !hc.tensor<f32, ["M"]>
  %to = hc.ones   {shape = #hc.shape<["M"]>} : !hc.tensor<i1, ["M"]>
  %tf = hc.full   %s {shape = #hc.shape<["M"]>}
      : !hc.undef -> !hc.tensor<f32, ["M"]>
  %te = hc.empty  {shape = #hc.shape<["M"]>} : !hc.tensor<f32, ["M"]>
  return
}

// CHECK-LABEL: func.func @reductions_and_matmul
// CHECK: hc.matmul %{{.*}}, %{{.*}} : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.reduce %{{.*}}, kind = "sum", axis = 0 : !hc.undef -> !hc.undef
// CHECK: hc.reduce %{{.*}}, kind = "max", axis = 1, keepdims = true : !hc.undef -> !hc.undef
// CHECK: hc.astype %{{.*}}, target = f32 : !hc.undef -> !hc.undef
func.func @reductions_and_matmul(%a: !hc.undef, %b: !hc.undef) {
  %m = hc.matmul %a, %b : (!hc.undef, !hc.undef) -> !hc.undef
  %s = hc.reduce %a, kind = "sum", axis = 0 : !hc.undef -> !hc.undef
  %mx = hc.reduce %a, kind = "max", axis = 1, keepdims = true
      : !hc.undef -> !hc.undef
  %cast = hc.astype %a, target = f32 : !hc.undef -> !hc.undef
  return
}

// CHECK-LABEL: func.func @calls
// CHECK: hc.call @helper(%{{.*}}, %{{.*}}) : (!hc.undef, !hc.undef) -> !hc.undef
// CHECK: hc.call_intrinsic @wmma(%{{.*}}, %{{.*}}) {wave_size = 32 : i64}
// CHECK-SAME: (!hc.undef, !hc.undef) -> !hc.undef
func.func @calls(%x: !hc.undef, %y: !hc.undef) {
  %a = hc.call @helper(%x, %y) : (!hc.undef, !hc.undef) -> !hc.undef
  %b = hc.call_intrinsic @wmma(%x, %y) {wave_size = 32 : i64}
      : (!hc.undef, !hc.undef) -> !hc.undef
  return
}
