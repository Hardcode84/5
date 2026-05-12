// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative coverage for `-hc-lower-generic`. Each chunk exercises a
// specific path through `diagnoseUnsupported` so the rejection
// diagnostic stays informative for end users staring at a failing
// pipeline. Surviving `hc.generic` ops are a hard pass failure
// because the un-lowered op would otherwise flow through GPU
// outlining + ROCDL attach and trip later passes with diagnostics
// far from the source.
//
// RUN: hc-opt --hc-lower-generic --split-input-file --verify-diagnostics %s

// Bare-tensor out with a symbolic shape (`["N"]`). The shape can't
// be reduced to a compile-time lane count, so neither the value-
// outs path nor the collective path can pin the slot table. The
// gate names the offending operand.
func.func @bail_symbolic_bare_tensor_out(%n: index,
                                         %src: !hc.ptr<global, f32>,
                                         %dst: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  // expected-error @below {{cannot lower hc.generic; outs #0 has type '!hc.bare_tensor<f32, ["N"]>'; expected !hc.ptr<...> with element type or a rank-1 fixed-lane carrier with integer-literal shape}}
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%dst at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield %sv : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// -----

// Value-typed ins with an offset that references an ambient (non-
// iter) symbol. Slot evaluation would be ambient-dependent — the
// gate flags the offending sym so users know which name to scope
// into iter syms or rebind upstream.
func.func @bail_value_in_ambient_offset(
    %vec: !hc.bare_vector<f32, ["8"]>,
    %dst: !hc.ptr<global, f32>,
    %k: !hc.idx<"$K">) {
  %n = hc.idx_apply () : () -> !hc.idx<"8">
  // expected-error @below {{cannot lower hc.generic; ins #0 offset references non-iter symbol '$K' (value-typed operand needs iter-only offsets)}}
  hc.generic
      iter (parallel i = %n : !hc.idx<"8">)
      ins (%vec at [#hc.expr<"i + $K">] : !hc.bare_vector<f32, ["8"]>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      ambient (%k as "$K" : !hc.idx<"$K">)
      -> () {
  ^bb0(%v: f32, %init: f32):
    hc.yield %v : f32
  }
  return
}

// -----

// Value-typed out with a non-constant iter bound. The `index`-typed
// bound carries no `IdxType` payload to fish a literal out of, so
// the constant-bound check rejects.
func.func @bail_value_out_dynamic_bound(%n: index,
                                        %src: !hc.ptr<global, f32>,
                                        %init: !hc.bare_vector<f32, ["8"]>)
    -> !hc.bare_vector<f32, ["8"]> {
  // expected-error @below {{cannot lower hc.generic; iter #0 bound is not a compile-time non-negative integer literal (value-typed operand needs constant bound)}}
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      outs (%init at [#hc.expr<"i">] : !hc.bare_vector<f32, ["8"]>)
      -> (!hc.bare_vector<f32, ["8"]>) {
  ^bb0(%sv: f32, %iv: f32):
    hc.yield %sv : f32
  }
  return %r : !hc.bare_vector<f32, ["8"]>
}

// -----

// Value-typed out with a reduction iter. The unrolled compose
// threads every parLane through one result vector; a reduction axis
// would need cross-lane carry the boundary form doesn't model.
func.func @bail_value_out_with_reduction(%src: !hc.ptr<global, f32>,
                                         %init: !hc.bare_vector<f32, ["4"]>)
    -> !hc.bare_vector<f32, ["4"]> {
  %m = hc.idx_apply () : () -> !hc.idx<"4">
  %k = hc.idx_apply () : () -> !hc.idx<"8">
  // expected-error @below {{cannot lower hc.generic; iter #1 kind is reduction (value-typed operand needs all-parallel iters)}}
  %r = hc.generic
      iter (parallel i = %m : !hc.idx<"4">,
            reduction j = %k : !hc.idx<"8">)
      ins (%src at [#hc.expr<"8*i + j">] : !hc.ptr<global, f32>)
      outs (%init at [#hc.expr<"i">] : !hc.bare_vector<f32, ["4"]>)
      -> (!hc.bare_vector<f32, ["4"]>) {
  ^bb0(%sv: f32, %iv: f32):
    %s = hc.add %iv, %sv : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_vector<f32, ["4"]>
}
