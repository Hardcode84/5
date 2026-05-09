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
// inline helpers, normalize supported scope regions, run the standard cleanup
// pair, wrap kernels in upstream GPU launches, lower launch-body scalar/control
// flow, clean up, interpret target lowering recipes (which rewrites every
// `hc.call_intrinsic` and DCEs the matching `hc.intrinsic` decls), then a
// canonicalize/cse pair to fold the recipe's bridging UCCs into identity. The
// closing trio splits each `gpu.launch` into a `gpu.module` + `gpu.func` +
// `gpu.launch_func`, stamps the `gpu.module` with a `#rocdl.target` so the
// downstream binary-emission pass has the chip info it needs, and runs a final
// canonicalize/cse cleanup. The actual `gpu.module` body lowering chain
// (amdgpu/scf/memref/vector → llvm-dialect inside the module) and the
// `hc-lower-gpu-to-binary` step are deliberately *not* in the default
// schedule today — they need a ROCDL-equivalent of upstream's
// `gpu-lower-to-nvvm-pipeline` composition (workgroup memref address-space
// mapping, kernel ABI, scf/vector lowering inside `gpu.module`) to land
// without breaking every non-trivial kernel. Until then this schedule stops
// at the `gpu.module` boundary; callers wanting to push further can pass
// `schedule=<path-or-text>` and own the rest themselves.
// `hc.compile` loads this via `-transform-preload-library` and runs it with
// `-transform-interpreter`; callers wanting a different order can pass
// `schedule=<path-or-text>` to override.
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
    %m11 = transform.apply_registered_pass "hc-normalize-scope-regions" to %m10
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
        with options = { "target" = "__HC_TARGET__" } to %m13
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
    // Outline each `gpu.launch` into `gpu.module @kernel_kernel` +
    // `gpu.func` + `gpu.launch_func`. No-op on payloads that never
    // produced a `gpu.launch` (trivial kernels), so leaving it
    // unconditional keeps the schedule shape regular for every input.
    %m15 = transform.apply_registered_pass "gpu-kernel-outlining" to %m14
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m15 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m15 : !transform.any_op
    // Stamp every freshly minted `gpu.module` with a `#rocdl.target`
    // attribute. The chip is resolved Python-side from `hc.compile`'s
    // `target=` (e.g. `amdgpu-gfx11` -> `gfx1100`); the empty
    // `target=None` case maps to a sensible default chip in
    // `_resolve_chip` so the substitution is always well-formed.
    // Subsequent passes — `convert-amdgpu-to-rocdl`,
    // `hc-lower-gpu-to-binary` — key off this attribute. Pass is a
    // no-op on payloads with no `gpu.module`.
    %m16 = transform.apply_registered_pass "rocdl-attach-target"
        with options = { "chip" = "__HC_CHIP__" } to %m15
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m16 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m16 : !transform.any_op
    transform.yield
  }
}
