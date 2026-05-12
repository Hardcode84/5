// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Default hc_front -> hc schedule.
//
// Mirrors the pass order `hc-opt` and `doc/lowering.md` document for
// the frontend stage: fold/erase region scaffolding, inline undecorated
// helpers, convert to `hc`, promote `hc.name_load` / `hc.assign` into
// SSA, infer concrete HC value types, materialize bound symbolic values, verify
// static shape carriers, split semantic shaped values into bare data/masks,
// inline helpers, fold identity layouts, funnel shaped compute / per-element
// arith / load+store into `hc.generic`, infer placeholder iter bounds,
// normalize supported scope regions, run the standard cleanup pair, wrap
// kernels in upstream GPU launches, lower launch-body scalar/control flow,
// clean up, interpret target lowering recipes (which rewrites every
// `hc.call_intrinsic` and DCEs the matching `hc.intrinsic` decls), then a
// canonicalize/cse pair to fold the recipe's bridging UCCs into identity.
//
// The generic-pipeline rewriters (`hc-shaped-compute-to-generic`,
// `hc-elementwise-to-generic`, `hc-load-store-to-generic`,
// `hc-infer-generic-bounds`) are conservative — they only fire on inputs
// they can prove safe (rank-2 matmul / reduce, per-element on shaped
// types, pinned `!hc.idx<expr>` indices on load and store). Inputs that
// don't match (slice-indexed loads, intrinsic-mediated WMMA paths, ...)
// flow through untouched and lower via the existing per-op handlers in
// `hc-lower-launch-body`. `hc-flatten-with-layouts` follows the
// generic rewriters and bounds inference: every shaped value
// collapses to its 1D bare carrier and every `hc.generic` operand /
// access op offset composes through the operand layout into a single
// 1D `#hc.expr`. `hc-lower-generic` runs after `hc-lower-launch-body`
// so any `hc.generic` whose operands are `!hc.ptr` collapses to the
// `scf.parallel` / `scf.for` loop nest.
//
// The closing chunk produces the device-side artefacts the GPU lowering
// pipeline (appended by the Python driver — see `_GPU_LOWERING_PIPELINE` in
// `hc/_pipeline.py`) needs:
//
//   * `gpu-launch-sink-index-computations` rewrites every constant index
//     consumer inside `gpu.launch` so the outliner can lift the constants into
//     the `gpu.func` body instead of promoting them to kernel arguments.
//     Sinking the constants is cheap insurance against any axis-indexed
//     consumer that would otherwise force runtime-typed dim values onto the
//     kernel-arg list — keeps the dispatch regular regardless of which
//     specific patterns ride downstream.
//
//   * `gpu-kernel-outlining` splits each `gpu.launch` into a sibling
//     `gpu.module @<kernel>_kernel` + `gpu.func` + `gpu.launch_func`.
//
//   * `rocdl-attach-target` stamps each `gpu.module` with `#rocdl.target` so
//     `convert-amdgpu-to-rocdl`, `gpu-to-llvm`, and `hc-lower-gpu-to-binary`
//     can route off it.
//
// `hc.compile` loads this via `-transform-preload-library` and runs it with
// `-transform-interpreter`; callers wanting a different order can pass
// `schedule=<path-or-text>` to override. The Python driver still appends the
// GPU lowering chain after the schedule fires, so an override only needs to
// produce `gpu.module` ops carrying `#rocdl.target` for the rest of the
// pipeline to take over.
module attributes {transform.with_named_sequence} {
  // `%m` is consumed by the registered frontend/HC passes; the verifier
  // requires the entry block argument to reflect that by omitting the
  // `{transform.readonly}` attribute. Each pass produces a fresh handle
  // that threads into the next. The final cleanups use transform dialect
  // ops directly on the post-inference handle.
  transform.named_sequence @__transform_main(%m: !transform.any_op) {
    %m1 = transform.apply_registered_pass "hc-front-fold-region-defs" to %m
        : (!transform.any_op) -> !transform.any_op
    %m2 = transform.apply_registered_pass "hc-front-inline" to %m1
        : (!transform.any_op) -> !transform.any_op
    %m3 = transform.apply_registered_pass "convert-hc-front-to-hc" to %m2
        : (!transform.any_op) -> !transform.any_op
    %m4 = transform.apply_registered_pass "hc-promote-names" to %m3
        : (!transform.any_op) -> !transform.any_op
    %m5 = transform.apply_registered_pass "hc-infer-types" to %m4
        : (!transform.any_op) -> !transform.any_op
    %m6 = transform.apply_registered_pass "hc-materialize-bound-exprs" to %m5
        : (!transform.any_op) -> !transform.any_op
    %m7 = transform.apply_registered_pass "hc-verify-static-shapes" to %m6
        : (!transform.any_op) -> !transform.any_op
    %m8 = transform.apply_registered_pass "hc-decompose-shaped-values"
        with options = { "strict" = false } to %m7
        : (!transform.any_op) -> !transform.any_op
    %m9 = transform.apply_registered_pass "hc-inline-helpers" to %m8
        : (!transform.any_op) -> !transform.any_op
    %m10 = transform.apply_registered_pass "hc-materialize-bound-exprs" to %m9
        : (!transform.any_op) -> !transform.any_op
    transform.apply_dce to %m10 : !transform.any_op
    // Layout cleanup. Identity layouts fold to absent so the post-decompose
    // shape carriers stay layout-less wherever the user didn't pin a
    // non-identity layout; `hc.as_layout` chains collapse to their outer
    // arg. Non-identity layouts (col-major, padded, default-strided buffer
    // args) survive structurally — `hc-flatten-with-layouts` is the one
    // that erases the slot wholesale.
    %m10a = transform.apply_registered_pass "hc-canonicalize-layouts" to %m10
        : (!transform.any_op) -> !transform.any_op
    // Funnel the shaped op surface into `hc.generic` so the post-flatten
    // codegen has a single op family to lower. Each rewriter is
    // conservative — `hc.matmul` / `hc.reduce` v0 wants rank-2 / single-axis
    // shapes, the per-element family wants all-shaped operands, and
    // load/store want pinned `!hc.idx<expr>` indices. Anything outside that
    // surface (slice-indexed access, mixed-rank ops, masked stores, ...)
    // stays in place and routes through the existing per-op handlers in
    // `hc-lower-launch-body`. `hc-infer-generic-bounds` then resolves any
    // `!hc.undef` iter bounds the per-element rewriter emitted from the
    // operand shapes. Order: rewriters before bounds inference so the pass
    // sees every fresh `hc.generic`.
    %m10b = transform.apply_registered_pass "hc-shaped-compute-to-generic" to %m10a
        : (!transform.any_op) -> !transform.any_op
    %m10c = transform.apply_registered_pass "hc-elementwise-to-generic" to %m10b
        : (!transform.any_op) -> !transform.any_op
    %m10d = transform.apply_registered_pass "hc-load-store-to-generic" to %m10c
        : (!transform.any_op) -> !transform.any_op
    %m10e = transform.apply_registered_pass "hc-infer-generic-bounds" to %m10d
        : (!transform.any_op) -> !transform.any_op
    %m11 = transform.apply_registered_pass "hc-normalize-scope-regions" to %m10e
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m11 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m11 : !transform.any_op
    %m12 = transform.apply_registered_pass "hc-lower-kernels-to-gpu-launch" to %m11
        : (!transform.any_op) -> !transform.any_op
    %m13 = transform.apply_registered_pass "hc-lower-launch-body" to %m12
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m13 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m13 : !transform.any_op
    // Flatten runs after `hc-lower-launch-body`. The eventual target
    // is to run it right after `hc-infer-generic-bounds`. Two pieces
    // are still missing before the move can happen: slice-indexed
    // access ops (`hc.vload`, `hc.load_mask`, `hc.store`) need to
    // route through `hc-load-store-to-generic` so flatten only sees
    // generics, and the workitem-region inlining has to land. Until
    // both ship the post-flatten launch-body still trips on the
    // multi-axis slice-indexed survivors.
    %m13b = transform.apply_registered_pass "hc-flatten-with-layouts" to %m13
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m13b {
      // `applyPartialConversion` plants `unrealized_conversion_cast`s
      // on every boundary nothing else converted; fold them away here
      // so downstream passes don't have to special-case the cast walk.
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m13b : !transform.any_op
    // Lower every `hc.generic` whose operands have already been
    // converted to `!hc.ptr` (by the launch-body pass above) and
    // composed to a single 1D offset (by the flatten pass above) to
    // an outer `scf.parallel` over the parallel iters and an inner
    // `scf.for` nest over reduction iters. The pass bails on any
    // generic that doesn't fit its v0 surface — for the WMMA path
    // that means a no-op since the recipe is dispatched via
    // `hc-interpret-intrinsic-recipes` instead of generic.
    %m13a = transform.apply_registered_pass "hc-lower-generic" to %m13b
        : (!transform.any_op) -> !transform.any_op
    // `hc.predicate` ops ride through `hc-lower-generic`'s `cloneBody` as
    // ordinary body ops — the pass doesn't touch them, the predicate
    // physically lands in the lowered `scf` body next to its (now
    // explicit) `hc.ptr_load` producer. Folding them is a separate pass
    // so the emission side stays unaware of mask shapes / producer kinds
    // and so the fold runs on every emitter that can plant a predicate,
    // not just on `hc-lower-generic`'s output.
    %m13af = transform.apply_registered_pass "hc-fold-predicates" to %m13a
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m13af {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m13af : !transform.any_op
    // The `__HC_TARGET__` placeholder is substituted by the Python
    // driver before the schedule is handed to the transform
    // interpreter: `hc.compile(target="amdgpu-gfx11")` substitutes the
    // string in, `target=None` (the default) substitutes the empty
    // string. Empty `target` runs every named sequence in the sibling
    // `__hc_intrinsic_lowerings__` module — fine while each intrinsic
    // registers at most one recipe per compile; multi-target lowerings
    // must thread an explicit `target=` through `hc.compile` so the
    // interpreter picks the right recipe instead of running them all.
    // The pass is also a no-op for kernels that never use intrinsics:
    // no lowerings module, no calls, nothing to diagnose.
    %m14 = transform.apply_registered_pass "hc-interpret-intrinsic-recipes"
        with options = { "target" = "__HC_TARGET__" } to %m13af
        : (!transform.any_op) -> !transform.any_op
    // Cleanup pair folds away every `unrealized_conversion_cast` the
    // recipe-side `transform.hc.cast_value` planted around the freshly
    // created upstream payload op. The launch-body pass left a matching
    // cast on the surrounding side of each call boundary, so the chain
    // `upstream → bare → upstream` collapses to identity here and the
    // post-interpretation IR ends up bare-type-free.
    transform.apply_patterns to %m14 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m14 : !transform.any_op
    // Sink constant index ops into `gpu.launch` so the outliner inlines them
    // into `gpu.func` instead of routing them through kernel arguments.
    // Any constant-axis dim/stride producer (`hc.buffer_dim`, the kernel-arg
    // UCC indexing) is the canonical victim — runtime-typed axes hit
    // `dynamic_stackalloc` in AMDGPU codegen, so we keep them literal.
    %m15 = transform.apply_registered_pass "gpu-launch-sink-index-computations" to %m14
        : (!transform.any_op) -> !transform.any_op
    // Outline each `gpu.launch` into `gpu.module @kernel_kernel` +
    // `gpu.func` + `gpu.launch_func`. No-op on payloads that never
    // produced a `gpu.launch` (trivial kernels), so leaving it
    // unconditional keeps the schedule shape regular for every input.
    %m16 = transform.apply_registered_pass "gpu-kernel-outlining" to %m15
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m16 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m16 : !transform.any_op
    // Stamp every freshly minted `gpu.module` with a `#rocdl.target`
    // attribute. The chip is resolved Python-side from `hc.compile`'s
    // `target=` (e.g. `amdgpu-gfx11` -> `gfx1100`); the empty
    // `target=None` case maps to a sensible default chip in
    // `_resolve_chip` so the substitution is always well-formed.
    // The downstream `convert-amdgpu-to-rocdl` and
    // `hc-lower-gpu-to-binary` (both wired by the appended GPU
    // lowering pipeline in `hc/_pipeline.py`) key off this attribute.
    // Pass is a no-op on payloads with no `gpu.module`.
    //
    // `features` carries the LLVM AMDGPU `target-features` string for
    // the chip — currently just the wavefront size. gfx10+ chips need
    // `+wavefrontsize32` for WMMA to lower to the right per-lane
    // fragment layout; without it the AMDGPU backend defaults to
    // wave64 and the matmul produces silently-wrong numerics. The
    // Python driver picks the feature string from the resolved chip
    // family (`_resolve_features` in `hc/_pipeline.py`) and substitutes
    // it in here. Empty string is the right default for chips that
    // only run wave64.
    %m17 = transform.apply_registered_pass "rocdl-attach-target"
        with options = { "chip" = "__HC_CHIP__", "features" = "__HC_FEATURES__" } to %m16
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m17 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m17 : !transform.any_op
    transform.yield
  }
}
