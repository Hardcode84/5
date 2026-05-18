// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-reconcile-generic-operands`: walks back through the kernel-arg
// UCC bundle to resolve `hc.generic` operands. Buffer ins / outs ->
// `!hc.ptr<global>`; LDS `!hc.bare_tensor` ins -> producer-side
// `!hc.ptr<workgroup>`. Other operand types pass through.
//
// RUN: hc-opt %s --hc-reconcile-generic-operands | FileCheck %s

module {
  // Kernel-arg buffer bundle in: the UCC bundle resolves to the
  // underlying `!hc.ptr<global, f32>`. Outs (bare_tensor) survive.
  // CHECK-LABEL: func.func @resolve_buffer_in(
  // CHECK-SAME: %[[PTR:[^:]+]]: !hc.ptr<global, f32>
  // CHECK: hc.generic
  // CHECK-SAME: ins (%[[PTR]] at [{{.*}}] : !hc.ptr<global, f32>)
  // CHECK-SAME: outs (%{{.*}} at [{{.*}}] : !hc.bare_vector<f32, ["8"]>)
  // CHECK-NOT: !hc.buffer
  func.func @resolve_buffer_in(%ptr: !hc.ptr<global, f32>,
                               %m: index, %sm: index) {
    %c1 = arith.constant 1 : index
    %eight = hc.idx_apply () : () -> !hc.idx<"8">
    %shape = hc.tuple(%eight) : (!hc.idx<"8">) -> tuple<!hc.idx<"8">>
    %tile = hc.vzeros shape %shape
        : (tuple<!hc.idx<"8">>) -> !hc.bare_vector<f32, ["8"]>
    %buf = builtin.unrealized_conversion_cast %ptr, %m, %sm
        : !hc.ptr<global, f32>, index, index
          to !hc.buffer<f32, ["M"]>
    %res = hc.generic iter (parallel i_0 = %eight : !hc.idx<"8">)
        ins (%buf at [#hc.expr<"i_0">] : !hc.buffer<f32, ["M"]>)
        outs (%tile at [#hc.expr<"i_0">] : !hc.bare_vector<f32, ["8"]>)
        -> (!hc.bare_vector<f32, ["8"]>) {
      ^bb0(%v: f32, %init: f32):
        hc.yield %v : f32
    }
    return
  }

  // LDS-backed bare_tensor ins: the swap picks up the producer's
  // `!hc.ptr<workgroup>` SSA instead of routing through the
  // bare_tensor carrier.
  // CHECK-LABEL: func.func @swap_lds_bare_tensor_in(
  // CHECK: %[[LDS:.+]] = hc.alloc count = %{{.+}} : index -> !hc.ptr<workgroup, f32>
  // CHECK: hc.generic
  // CHECK-SAME: ins (%[[LDS]] at [{{.*}}] : !hc.ptr<workgroup, f32>)
  // CHECK-NOT: !hc.bare_tensor<f32, ["8"]>
  func.func @swap_lds_bare_tensor_in(%dst: !hc.ptr<global, f32>) {
    %c8 = arith.constant 8 : index
    %lds = hc.alloc count = %c8 : index -> !hc.ptr<workgroup, f32>
    %tile = builtin.unrealized_conversion_cast %lds
        : !hc.ptr<workgroup, f32> to !hc.bare_tensor<f32, ["8"]>
    %eight = hc.idx_apply () : () -> !hc.idx<"8">
    hc.generic iter (parallel i = %eight : !hc.idx<"8">)
        ins (%tile at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["8"]>)
        outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
        -> () {
      ^bb0(%v: f32, %init: f32):
        hc.yield %v : f32
    }
    return
  }
}
