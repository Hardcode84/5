// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-infer-types -split-input-file -verify-diagnostics %s

hc.func @const_result_conflict {
  // expected-error @+1 {{has conflicting HC type facts for result #0}}
  %c = hc.const<1 : i64> : i64
  hc.return
}

// -----

hc.func @tuple_getitem_oob(%tuple: tuple<!hc.idx<"A">, !hc.idx<"B">>,
                           %idx: !hc.idx<"2">) {
  // expected-error @+1 {{tuple index 2 out of bounds for tuple of size 2}}
  %item = hc.getitem %tuple[%idx]
      : (tuple<!hc.idx<"A">, !hc.idx<"B">>, !hc.idx<"2">) -> !hc.undef
  hc.return
}

// -----

hc.func @tuple_getitem_multi_index_after_infer {
  %zero = hc.const<0 : i64> : !hc.undef
  %one = hc.const<1 : i64> : !hc.undef
  %pair = hc.tuple(%zero, %one) : (!hc.undef, !hc.undef) -> !hc.undef
  // expected-error @+1 {{tuple getitem expects exactly one index after inference, got 2}}
  %item = hc.getitem %pair[%zero, %one]
      : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
  hc.return
}

// -----

hc.func @tuple_getitem_symbolic_index(%tuple: tuple<!hc.idx<"A">, !hc.idx<"B">>,
                                      %idx: !hc.idx<"I">) {
  // expected-error @+1 {{tuple getitem index must be a static integer after inference, got '!hc.idx<"I">'}}
  %item = hc.getitem %tuple[%idx]
      : (tuple<!hc.idx<"A">, !hc.idx<"B">>, !hc.idx<"I">) -> !hc.undef
  hc.return
}

// -----

hc.func @getitem_invalid_base(%base: !hc.idx<"N">, %idx: !hc.idx<"0">) {
  // expected-error @+1 {{getitem base type '!hc.idx<"N">' cannot be refined; expected tuple, buffer, tensor, or vector}}
  %item = hc.getitem %base[%idx]
      : (!hc.idx<"N">, !hc.idx<"0">) -> !hc.undef
  hc.return
}

// -----

hc.func @workitem_region_rejects_tensor_return(
    %tile: !hc.tensor<f32, ["4"]>) {
  // expected-error @+1 {{collective region cannot return tensor value '!hc.tensor<f32, ["4"]>'}}
  %region = hc.workitem_region -> (!hc.undef) {
  ^bb0(%wi: !hc.workitem<group_shape = #hc.shape<["32"]>,
                         subgroup_size = #hc.expr<"32">>):
    hc.yield %tile : !hc.tensor<f32, ["4"]>
  }
  hc.return
}
