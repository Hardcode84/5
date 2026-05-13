// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `Buffer[..., IndexMap]` on a `@kernel` parameter: the Python
// resolver stamps a `layout = {kind = "layout", ...}` sub-dict on the
// `hc_front.kernel.parameters` entry, and `-convert-hc-front-to-hc`
// must consume it via the same `layoutAttrFromRef` path used for
// body-level layouts — not synthesize the default-strided
// `$STRIDE_<i>_<argname>` layout. Companion to `layout-kwarg.mlir`
// (body-side) and `as-layout.mlir` (explicit `as_layout` calls).
//
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names --split-input-file %s | FileCheck %s
// RUN: hc-opt --convert-hc-front-to-hc --hc-promote-names --split-input-file %s | hc-opt --split-input-file | FileCheck %s

// The captured layout's `shape_syms` / `index_syms` / `params` keys
// are layout-local (they get substituted against the buffer's
// `["M", "N"]` shape when the BufferType is instantiated), so they
// must NOT leak into the kernel's `bound_symbols`. The default-strided
// builder would have stamped a `$STRIDE_0_a` symbol on `bound_symbols`
// for the first non-group parameter; asserting that against the
// captured-layout path pins both the BufferType and the kernel-level
// symbol harvest.

// CHECK-LABEL: hc.kernel @param_with_layout
// CHECK-SAME: %arg1: !hc.buffer<!hc.undef, ["M", "N"], <shape_syms = ["w", "h"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"h*w">, offset = #hc.expr<"j + h*i">>>
// CHECK-SAME: bound_symbols = [{{.*}}"M", "N"]
// CHECK-NOT: $STRIDE_
module {
  hc_front.kernel "param_with_layout" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M, N, A_LAYOUT]",
       kind = "buffer",
       name = "a",
       shape = ["M", "N"],
       layout = {kind = "layout",
                 shape_syms = ["w", "h"],
                 index_syms = ["i", "j"],
                 params = {},
                 storage_size = #hc.expr<"h*w">,
                 offset = #hc.expr<"j + h*i">}}
    ],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    hc_front.return
  }
}

// -----

// Two parameters with two distinct captured layouts: each lands on the
// corresponding `BufferType`. Both sets of `shape_syms` / `index_syms`
// stay layout-local — the resolver guarantees one `layout` sub-dict
// per parameter, so the C++ side reads them per-parameter rather than
// sharing a pass-wide override, and neither set contributes to the
// kernel's `bound_symbols` (only the buffer-shape symbols `M`, `N` do).

// CHECK-LABEL: hc.kernel @two_params_two_layouts
// CHECK-SAME: %arg1: !hc.buffer<!hc.undef, ["M", "N"], <shape_syms = ["w", "h"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"h*w">, offset = #hc.expr<"j + h*i">>>
// CHECK-SAME: %arg2: !hc.buffer<!hc.undef, ["M", "N"], <shape_syms = ["p", "q"], index_syms = ["r", "s"], params = {}, storage_size = #hc.expr<"p*q">, offset = #hc.expr<"r + p*s">>>
// CHECK-SAME: bound_symbols = [{{.*}}"M", "N"]
// CHECK-NOT: $STRIDE_
module {
  hc_front.kernel "two_params_two_layouts" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M, N, ROW_MAJOR]",
       kind = "buffer",
       name = "a",
       shape = ["M", "N"],
       layout = {kind = "layout",
                 shape_syms = ["w", "h"],
                 index_syms = ["i", "j"],
                 params = {},
                 storage_size = #hc.expr<"h*w">,
                 offset = #hc.expr<"j + h*i">}},
      {annotation = "Buffer[M, N, COL_MAJOR]",
       kind = "buffer",
       name = "b",
       shape = ["M", "N"],
       layout = {kind = "layout",
                 shape_syms = ["p", "q"],
                 index_syms = ["r", "s"],
                 params = {},
                 storage_size = #hc.expr<"p*q">,
                 offset = #hc.expr<"r + p*s">}}
    ],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    hc_front.return
  }
}

// -----

// Mixed parameters: one buffer with captured layout, one without.
// The bare buffer must still get the default `$STRIDE_<i>_<argname>`
// layout (and its `$STRIDE_0_a` shows up in `bound_symbols` for the
// host wrapper to bind at launch); only the layout-carrying parameter
// takes the override path, so no `$STRIDE_0_b` appears.

// CHECK-LABEL: hc.kernel @mixed_param_layouts
// CHECK-SAME: %arg1: !hc.buffer<!hc.undef, ["M"], <shape_syms = ["d0"], index_syms = ["i0"], params = {}, storage_size = #hc.expr<"0">, offset = #hc.expr<"$STRIDE_0_a*i0">>>
// CHECK-SAME: %arg2: !hc.buffer<!hc.undef, ["M", "N"], <shape_syms = ["w", "h"], index_syms = ["i", "j"], params = {}, storage_size = #hc.expr<"h*w">, offset = #hc.expr<"j + h*i">>>
// CHECK-SAME: bound_symbols = [{{.*}}"M", "$STRIDE_0_a", "N"]
// CHECK-NOT: $STRIDE_0_b
module {
  hc_front.kernel "mixed_param_layouts" attributes {
    decorators = ["kernel"],
    group_shape = ["32"],
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M]", kind = "buffer", name = "a", shape = ["M"]},
      {annotation = "Buffer[M, N, ROW_MAJOR]",
       kind = "buffer",
       name = "b",
       shape = ["M", "N"],
       layout = {kind = "layout",
                 shape_syms = ["w", "h"],
                 index_syms = ["i", "j"],
                 params = {},
                 storage_size = #hc.expr<"h*w">,
                 offset = #hc.expr<"j + h*i">}}
    ],
    returns = "None",
    subgroup_size = 32 : i32,
    work_shape = ["M"]
  } {
    hc_front.return
  }
}
