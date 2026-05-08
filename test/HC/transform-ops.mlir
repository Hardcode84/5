// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: hc-opt %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %call = transform.hc.match_intrinsic_call %root @wmma_gfx11 target = "amdgpu-gfx11" : (!transform.any_op) -> !transform.any_op
    %a = transform.hc.get_intrinsic_operand %call {name = "a_frag"} : (!transform.any_op) -> !transform.any_value
    %b = transform.hc.get_intrinsic_operand %call {index = 4 : i64} : (!transform.any_op) -> !transform.any_value
    %acc = transform.hc.get_intrinsic_operand %call {name = "acc_frag"} : (!transform.any_op) -> !transform.any_value
    %result_type = transform.hc.get_intrinsic_result_type %call {index = 0 : i64} : (!transform.any_op) -> !transform.type
    %arch = transform.hc.get_intrinsic_attr %call {name = "arch"} : (!transform.any_op) -> !transform.any_param
    %wave = transform.hc.get_intrinsic_attr %call {name = "wave_size"} : (!transform.any_op) -> !transform.any_param
    transform.hc.require_intrinsic_attr %call {expected = "gfx11", name = "arch"} : !transform.any_op
    transform.hc.require_intrinsic_attr %call {expected = 32 : i64, name = "wave_size"} : !transform.any_op
    %created = transform.hc.create_op "amdgpu.wmma" at %call (%a, %b, %acc) result_types(%result_type) dynamic_attrs ["arch", "wave_size"](%arch, %wave) static_attrs = {k = 16 : i32, m = 16 : i32, n = 16 : i32} : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_value, !transform.type, !transform.any_param, !transform.any_param) -> (!transform.any_value)
    transform.hc.replace_intrinsic_call %call with %created : (!transform.any_op, !transform.any_value) -> ()
    transform.yield
  }
}

// CHECK-LABEL: transform.named_sequence @__transform_main
// CHECK: transform.hc.match_intrinsic_call
// CHECK-SAME: @wmma_gfx11
// CHECK: transform.hc.get_intrinsic_operand
// CHECK-SAME: name = "a_frag"
// CHECK: transform.hc.get_intrinsic_result_type
// CHECK: transform.hc.get_intrinsic_attr
// CHECK-SAME: name = "arch"
// CHECK: transform.hc.require_intrinsic_attr
// CHECK-SAME: expected = "gfx11"
// CHECK-SAME: name = "arch"
// CHECK: transform.hc.require_intrinsic_attr
// CHECK-SAME: expected = 32 : i64
// CHECK-SAME: name = "wave_size"
// CHECK: transform.hc.create_op "amdgpu.wmma"
// CHECK-SAME: dynamic_attrs ["arch", "wave_size"]
// CHECK: transform.hc.replace_intrinsic_call
