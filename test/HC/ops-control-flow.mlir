// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: hc.intrinsic @wave_barrier
// CHECK-SAME: scope = <"SubGroup">
// CHECK-SAME: effects = read_write
hc.intrinsic @wave_barrier scope = #hc.scope<"SubGroup">
    effects = read_write {}

// CHECK-LABEL: hc.intrinsic @subgroup_dot
// CHECK-SAME: scope = <"SubGroup">
// CHECK-NOT: effects
hc.intrinsic @subgroup_dot scope = #hc.scope<"SubGroup"> {}

// Signature on an intrinsic declaration now shows as `(args) -> result`,
// not the legacy `: (T) -> T` form. The body is still allowed to be empty
// (pure declaration).
// CHECK-LABEL: hc.intrinsic @typed_decl
// CHECK-SAME: (%arg0: f32, %arg1: f32) -> f32 scope = <"WorkItem">
// CHECK-SAME: parameters = ["a", "b"]
hc.intrinsic @typed_decl(%a: f32, %b: f32) -> f32
    scope = #hc.scope<"WorkItem"> parameters = ["a", "b"] {}

// CHECK-LABEL: func.func @undef_value_seed
// CHECK: %{{.*}} = hc.undef_value : !hc.undef
func.func @undef_value_seed() -> !hc.undef {
  %seed = hc.undef_value : !hc.undef
  return %seed : !hc.undef
}

// CHECK-LABEL: func.func @simple_for_range
// CHECK: hc.for_range %{{.*}} to %{{.*}} step %{{.*}} : (!hc.undef, !hc.undef, !hc.undef) {
// CHECK: }
func.func @simple_for_range(%lo: !hc.undef, %hi: !hc.undef, %step: !hc.undef) {
  hc.for_range %lo to %hi step %step
      : (!hc.undef, !hc.undef, !hc.undef) {
  ^bb0(%i: !hc.undef):
    hc.yield
  }
  return
}

// CHECK-LABEL: func.func @for_range_with_iter_args
// CHECK: hc.for_range %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}}) :
// CHECK-SAME: (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">) -> (!hc.undef) {
// CHECK:   hc.yield %{{.*}} : !hc.undef
// CHECK: }
func.func @for_range_with_iter_args(
    %lo: !hc.idx<"0">, %hi: !hc.idx<"N">, %step: !hc.idx<"1">,
    %init: !hc.undef) -> !hc.undef {
  %acc = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">) -> (!hc.undef) {
  ^bb0(%i: !hc.idx, %sum: !hc.undef):
    hc.yield %sum : !hc.undef
  }
  return %acc : !hc.undef
}

// Pin the bare ↔ layout-bearing branch-compat rule for shaped types.
// `HCBranchCompatibleTypes` forces parser-level binding so iter_init and
// iter_result spell out identical types in textual form; the verifier
// then runs `areHCBranchTypesCompatible` against each pair of
// (init ↔ block_arg) and (yield ↔ iter_result). The pin here exercises
// the yield ↔ result side: yielding a layout-bearing value through a
// bare iter_result must verify, because the shaped-type join keeps the
// layout-bearing form when the two sides differ only on the layout
// slot. (The layout side declares the underlying storage; the bare
// side, by contract, claims the identity layout — strictly less
// specific.) This is the substrate hook that lets the gfx11 WMMA
// example carry a layout-driven `vload` seed through a loop whose
// body yields the bare intrinsic result.
// CHECK-LABEL: func.func @for_range_yield_layout_into_bare_result
// CHECK: hc.for_range %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}}) :
// CHECK-SAME: -> (!hc.vector<f32, ["8"]>)
// CHECK: hc.yield %{{.*}} : !hc.vector<f32, ["8"],
func.func @for_range_yield_layout_into_bare_result(
    %lo: !hc.idx<"0">, %hi: !hc.idx<"N">, %step: !hc.idx<"1">,
    %init: !hc.vector<f32, ["8"]>,
    %body: !hc.vector<f32, ["8"],
      #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {},
                 storage_size = #hc.expr<"8">,
                 offset = #hc.expr<"i0">>>) -> !hc.vector<f32, ["8"]> {
  %acc = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">) -> (!hc.vector<f32, ["8"]>) {
  ^bb0(%i: !hc.idx, %sum: !hc.vector<f32, ["8"]>):
    hc.yield %body : !hc.vector<f32, ["8"],
      #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {},
                 storage_size = #hc.expr<"8">,
                 offset = #hc.expr<"i0">>>
  }
  return %acc : !hc.vector<f32, ["8"]>
}

// Mirror image: a layout-bearing block-arg paired with a bare iter_init
// (the `init ↔ block_arg` verifier slot, distinct from the yield ↔ result
// pair pinned above). The flatten pipeline produces both shapes
// depending on which side carries the explicit layout; pinning both
// directions keeps the join's symmetry honest.
// CHECK-LABEL: func.func @for_range_init_bare_block_arg_layout
// CHECK: hc.for_range %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}}) :
// CHECK-SAME: -> (!hc.vector<f32, ["8"]>)
// CHECK: ^{{.*}}(%{{.*}}: !hc.idx, %{{.*}}: !hc.vector<f32, ["8"],
func.func @for_range_init_bare_block_arg_layout(
    %lo: !hc.idx<"0">, %hi: !hc.idx<"N">, %step: !hc.idx<"1">,
    %init: !hc.vector<f32, ["8"]>) -> !hc.vector<f32, ["8"]> {
  %acc = hc.for_range %lo to %hi step %step iter_args(%init)
      : (!hc.idx<"0">, !hc.idx<"N">, !hc.idx<"1">) -> (!hc.vector<f32, ["8"]>) {
  ^bb0(%i: !hc.idx,
       %sum: !hc.vector<f32, ["8"],
         #hc.layout<shape_syms = ["d0"], index_syms = ["i0"], params = {},
                    storage_size = #hc.expr<"8">,
                    offset = #hc.expr<"i0">>>):
    hc.yield %init : !hc.vector<f32, ["8"]>
  }
  return %acc : !hc.vector<f32, ["8"]>
}

// CHECK-LABEL: func.func @if_without_results
// CHECK: hc.if %{{.*}} : !hc.undef {
// CHECK-NOT: else
func.func @if_without_results(%c: !hc.undef) {
  hc.if %c : !hc.undef {
  }
  return
}

// CHECK-LABEL: func.func @if_with_else_and_results
// CHECK: hc.if %{{.*}} -> (!hc.undef) : !hc.pred<"M - N < 0"> {
// CHECK:   hc.yield %{{.*}} : !hc.undef
// CHECK: } else {
// CHECK:   hc.yield %{{.*}} : !hc.undef
// CHECK: }
func.func @if_with_else_and_results(
    %c: !hc.pred<"M < N">, %a: !hc.undef, %b: !hc.undef) -> !hc.undef {
  %x = hc.if %c -> (!hc.undef) : !hc.pred<"M < N"> {
    hc.yield %a : !hc.undef
  } else {
    hc.yield %b : !hc.undef
  }
  return %x : !hc.undef
}
