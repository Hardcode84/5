// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Round-trip coverage for `hc.generic`. Verifier negatives live in
// `verify-hc.mlir` next to the rest of the dialect's "wrong spelling" pins.
//
// Per-operand offsets ride inside `[...]` — one `#hc.expr<...>` entry per
// operand axis, length equal to the operand's rank. The bracket terminator
// shields the trailing `: type` annotation from the generic dialect-attr
// path, which would otherwise consume the `:` as part of the attribute.
// Expression text below matches ixsimpl's canonical form so the round-trip
// stays text-stable.
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// Plain elementwise: one parallel iter, one input, one output. Body adds
// the input to the running output value (`outs-as-init`) and yields.
// CHECK-LABEL: func.func @elementwise_add
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["N"]>)
// CHECK: ^bb0(%[[A:.+]]: f32, %[[C:.+]]: f32):
// CHECK:   %[[S:.+]] = hc.add %[[C]], %[[A]] : (f32, f32) -> f32
// CHECK:   hc.yield %[[S]] : f32
func.func @elementwise_add(%n: index,
                           %a: !hc.bare_tensor<f32, ["N"]>,
                           %c: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// nD matmul: 2D inputs and output with one per-axis `#hc.expr` entry per
// operand axis. The reduction iter `k` only appears in the input axes;
// outputs reference parallel iters only. This is the canonical pre-flatten
// form of a matmul-shaped contraction.
// CHECK-LABEL: func.func @matmul_like
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index, parallel j = %{{[^ ]+}} : index, reduction k = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">, #hc.expr<"k">] : !hc.bare_tensor<f16, ["M", "K"]>,
// CHECK-SAME:      %{{[^ ]+}} at [#hc.expr<"k">, #hc.expr<"j">] : !hc.bare_tensor<f16, ["K", "N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">, #hc.expr<"j">] : !hc.bare_tensor<f32, ["M", "N"]>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["M", "N"]>)
func.func @matmul_like(%m: index, %n: index, %k: index,
                       %a: !hc.bare_tensor<f16, ["M", "K"]>,
                       %b: !hc.bare_tensor<f16, ["K", "N"]>,
                       %c: !hc.bare_tensor<f32, ["M", "N"]>)
    -> !hc.bare_tensor<f32, ["M", "N"]> {
  %r = hc.generic
      iter (parallel i = %m : index,
            parallel j = %n : index,
            reduction k = %k : index)
      ins (%a at [#hc.expr<"i">, #hc.expr<"k">]
              : !hc.bare_tensor<f16, ["M", "K"]>,
           %b at [#hc.expr<"k">, #hc.expr<"j">]
              : !hc.bare_tensor<f16, ["K", "N"]>)
      outs (%c at [#hc.expr<"i">, #hc.expr<"j">]
               : !hc.bare_tensor<f32, ["M", "N"]>)
      -> (!hc.bare_tensor<f32, ["M", "N"]>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %ae = hc.astype %av, target = f32 : f16 -> f32
    %be = hc.astype %bv, target = f32 : f16 -> f32
    %p = hc.mul %ae, %be : (f32, f32) -> f32
    %s = hc.add %cv, %p : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_tensor<f32, ["M", "N"]>
}

// Mixed ranks across operands (2D input, 1D outputs) and multiple outputs:
// each output gets its own per-axis offset array and its own block-arg /
// yield slot. Models a per-row (running max, running sum) sweep.
// CHECK-LABEL: func.func @two_outputs
// CHECK: %{{.+}}:2 = hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index, reduction j = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">, #hc.expr<"j">] : !hc.bare_tensor<f32, ["M", "N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>,
// CHECK-SAME:       %{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>)
func.func @two_outputs(%m: index, %n: index,
                       %x: !hc.bare_tensor<f32, ["M", "N"]>,
                       %mx: !hc.bare_tensor<f32, ["M"]>,
                       %sm: !hc.bare_tensor<f32, ["M"]>)
    -> (!hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>) {
  %r:2 = hc.generic
      iter (parallel i = %m : index, reduction j = %n : index)
      ins (%x at [#hc.expr<"i">, #hc.expr<"j">]
              : !hc.bare_tensor<f32, ["M", "N"]>)
      outs (%mx at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>,
            %sm at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["M"]>)
      -> (!hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>) {
  ^bb0(%xv: f32, %mxv: f32, %smv: f32):
    %nm = hc.add %mxv, %xv : (f32, f32) -> f32
    %ns = hc.add %smv, %xv : (f32, f32) -> f32
    hc.yield %nm, %ns : f32, f32
  }
  return %r#0, %r#1
      : !hc.bare_tensor<f32, ["M"]>, !hc.bare_tensor<f32, ["M"]>
}

// Empty `ins ()`: pure output-only fill; legal because outs-as-init
// supplies the body's read channel.
// CHECK-LABEL: func.func @fill_only
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index)
// CHECK-SAME: ins ()
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
func.func @fill_only(%n: index, %dst: !hc.bare_tensor<f32, ["N"]>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %z = hc.const<0.0 : f32> : f32
  %r = hc.generic
      iter (parallel i = %n : index)
      ins ()
      outs (%dst at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%dv: f32):
    hc.yield %z : f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// Bare vector operands: the op handles either bare carrier kind without
// constraint chaining.
// CHECK-LABEL: func.func @vector_operands
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_vector<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_vector<f32, ["N"]>)
func.func @vector_operands(%n: index,
                           %a: !hc.bare_vector<f32, ["N"]>,
                           %c: !hc.bare_vector<f32, ["N"]>)
    -> !hc.bare_vector<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.bare_vector<f32, ["N"]>)
      outs (%c at [#hc.expr<"i">] : !hc.bare_vector<f32, ["N"]>)
      -> (!hc.bare_vector<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32):
    %s = hc.add %cv, %av : (f32, f32) -> f32
    hc.yield %s : f32
  }
  return %r : !hc.bare_vector<f32, ["N"]>
}

// Pre-inference flavour: `!hc.undef` operand types escape both the
// element-type parity check and the per-axis rank check, so the frontend
// can emit the op before type inference fills in the carrier types.
// Empty `[]` axis array doubles as the "no rank yet" placeholder.
// CHECK-LABEL: func.func @progressive_undef
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [] : !hc.undef)
// CHECK-SAME: outs (%{{[^ ]+}} at [] : !hc.undef)
func.func @progressive_undef(%n: index, %a: !hc.undef, %c: !hc.undef)
    -> !hc.undef {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at [] : !hc.undef)
      outs (%c at [] : !hc.undef)
      -> (!hc.undef) {
  ^bb0(%av: !hc.undef, %cv: !hc.undef):
    hc.yield %cv : !hc.undef
  }
  return %r : !hc.undef
}

// Pure-store form: ptr-typed out, zero SSA results. Mirrors
// `linalg.generic` over an in-place destination — the body sources the
// carry via the implicit ptr_load on the operand and the yield routes
// through an implicit ptr_store at the operand's offset. Op-level
// memory effects (Read on src, Read+Write on dst) come from
// `getEffects`; LIT only round-trips the surface.
// CHECK-LABEL: func.func @ptr_only_store
// CHECK: hc.generic
// CHECK-SAME: iter (parallel i = %{{[^ ]+}} : index)
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.ptr<global, f32>)
// CHECK-SAME: -> ()
// CHECK: ^bb0(%[[SV:.+]]: f32, %[[DV:.+]]: f32):
// CHECK:   hc.yield %[[SV]] : f32
func.func @ptr_only_store(%n: index,
                          %src: !hc.bare_tensor<f32, ["N"]>,
                          %dst: !hc.ptr<global, f32>) {
  hc.generic
      iter (parallel i = %n : index)
      ins (%src at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield %sv : f32
  }
  return
}

// Mixed outs: one value-typed out (produces a result) and one ptr-typed
// out (no result, in-place). Result count parity is against value-typed
// outs only; the ptr slot still gets a body block-arg + a yield value
// for the implicit store.
// CHECK-LABEL: func.func @mixed_outs
// CHECK: %{{.+}} = hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>,
// CHECK-SAME:       %{{[^ ]+}} at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
// CHECK-SAME: -> (!hc.bare_tensor<f32, ["N"]>)
// CHECK: ^bb0(%[[AV:.+]]: f32, %[[CV:.+]]: f32, %[[TV:.+]]: f32):
// CHECK:   %[[R:.+]] = hc.mul %[[AV]], %[[AV]] : (f32, f32) -> f32
// CHECK:   hc.yield %[[R]], %[[R]] : f32, f32
func.func @mixed_outs(%n: index,
                      %a: !hc.bare_tensor<f32, ["N"]>,
                      %c: !hc.bare_tensor<f32, ["N"]>,
                      %trace: !hc.ptr<workgroup, f32>)
    -> !hc.bare_tensor<f32, ["N"]> {
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>,
            %trace at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32, %tv: f32):
    %sq = hc.mul %av, %av : (f32, f32) -> f32
    hc.yield %sq, %sq : f32, f32
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// Buffer-typed out is also a memory carrier (Read+Write effect on the
// operand, no SSA result). Element type comes off the buffer's
// `getElementType()` for the body-arg parity check, same as a shaped
// value-typed out — just on the in-place / pointer side of the
// polymorphism (`linalg.generic`'s memref regime, in HC clothing).
// CHECK-LABEL: func.func @buffer_out
// CHECK: hc.generic
// CHECK-SAME: ins (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.buffer<f32, ["N"]>)
// CHECK-SAME: -> ()
func.func @buffer_out(%n: index,
                      %src: !hc.bare_tensor<f32, ["N"]>,
                      %dst: !hc.buffer<f32, ["N"]>) {
  hc.generic
      iter (parallel i = %n : index)
      ins (%src at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%dst at [#hc.expr<"i">] : !hc.buffer<f32, ["N"]>)
      -> () {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield %sv : f32
  }
  return
}

// Opaque ptr (no `$elementType`) escapes the body-arg / yield element-type
// parity checks, same as `!hc.undef`. The op surface lets the body adopt
// any scalar element type and the lowering reads it off the yield.
// CHECK-LABEL: func.func @opaque_ptr_out
// CHECK: hc.generic
// CHECK-SAME: outs (%{{[^ ]+}} at [#hc.expr<"i">] : !hc.ptr<global>)
// CHECK-SAME: -> ()
func.func @opaque_ptr_out(%n: index, %dst: !hc.ptr<global>) {
  %z = hc.const<0.0 : f32> : f32
  hc.generic
      iter (parallel i = %n : index)
      ins ()
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global>)
      -> () {
  ^bb0(%dv: f32):
    hc.yield %z : f32
  }
  return
}

// Predicated terminator (scalar): one masked yield value drives the publish.
// The mask is an ordinary `i1` SSA; downstream lowering routes through a
// predicated store / select using `mask` as the gate. Always-true sites
// (no semantic guard) materialise `arith.constant true` and rely on the
// emit-time fold to collapse back to an unpredicated store.
// CHECK-LABEL: func.func @yield_predicated_scalar
// CHECK: hc.generic
// CHECK: ^bb0(%[[SV:.+]]: f32, %[[DV:.+]]: f32):
// CHECK:   hc.yield_predicated %[[SV]] mask %{{.+}} : (f32), (i1)
func.func @yield_predicated_scalar(%n: index,
                                   %src: !hc.bare_tensor<f32, ["N"]>,
                                   %dst: !hc.ptr<global, f32>) {
  %t = arith.constant true
  hc.generic
      iter (parallel i = %n : index)
      ins (%src at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
      -> () {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield_predicated %sv mask %t : (f32), (i1)
  }
  return
}

// Predicated terminator (multi-value): mask count == value count, one
// `i1` mask per yielded slot. Mixed outs (value + ptr) both ride on the
// same per-slot mask channel.
// CHECK-LABEL: func.func @yield_predicated_multi
// CHECK: hc.generic
// CHECK: hc.yield_predicated %{{.+}}, %{{.+}} mask %{{.+}}, %{{.+}} : (f32, f32), (i1, i1)
func.func @yield_predicated_multi(%n: index,
                                  %a: !hc.bare_tensor<f32, ["N"]>,
                                  %c: !hc.bare_tensor<f32, ["N"]>,
                                  %trace: !hc.ptr<workgroup, f32>) ->
    !hc.bare_tensor<f32, ["N"]> {
  %tt = arith.constant true
  %r = hc.generic
      iter (parallel i = %n : index)
      ins (%a at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>)
      outs (%c at [#hc.expr<"i">] : !hc.bare_tensor<f32, ["N"]>,
            %trace at [#hc.expr<"i">] : !hc.ptr<workgroup, f32>)
      -> (!hc.bare_tensor<f32, ["N"]>) {
  ^bb0(%av: f32, %cv: f32, %tv: f32):
    %sq = hc.mul %av, %av : (f32, f32) -> f32
    hc.yield_predicated %sq, %sq mask %tt, %tt : (f32, f32), (i1, i1)
  }
  return %r : !hc.bare_tensor<f32, ["N"]>
}

// Opaque ptr (no `$elementType`) escapes the body's per-slot parity in
// the same way the unconditional yield does, so a masked publish can ride
// any scalar yield type the producer wants. The verifier still gates the
// scalar/vector-mask shape parity at the yield site — see verify-hc.mlir
// for the negative case.
// CHECK-LABEL: func.func @yield_predicated_opaque_ptr
// CHECK: hc.yield_predicated %{{.+}} mask %{{.+}} : (i32), (i1)
func.func @yield_predicated_opaque_ptr(%v: i32, %n: index,
                                       %dst: !hc.ptr<global>) {
  %t = arith.constant true
  hc.generic
      iter (parallel i = %n : index)
      ins ()
      outs (%dst at [#hc.expr<"i">] : !hc.ptr<global>)
      -> () {
  ^bb0(%dv: i32):
    hc.yield_predicated %v mask %t : (i32), (i1)
  }
  return
}
