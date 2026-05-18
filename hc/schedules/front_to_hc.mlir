// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Default hc_front -> hc schedule.
//
// Inline per-pass comments are the per-call contract; this banner is
// the phase-level outline. Eight phases:
//
//   1. Frontend cleanup. Fold region scaffolding, inline undecorated
//      helpers, convert `hc_front` to `hc`.
//   2. Literal specialisation + early lowerings. Substitute
//      `hc.compile(symbols=...)` bindings throughout reachable
//      symbolic carriers, then unfold `hc.pow` literal-int exponents
//      to mul chains so type inference sees the muls.
//   3. Type inference + symbolic materialisation. Promote
//      `hc.assign` / `hc.name_load` into SSA, run `hc-infer-types`,
//      sever bound symbolic expressions from their producer chains,
//      verify static shape carriers.
//   4. Shape decomposition + helper inlining. Split semantic
//      `!hc.tensor` / `!hc.vector` into bare data/mask pairs, inline
//      `@kernel.func` helpers, then re-sever bound expressions for the
//      launch-geometry chains the inliner brought along.
//   5. Generic funnel. Fold identity layouts, lower `hc.strip_layout`,
//      funnel shaped compute / per-element arith / `hc.builtin_call` /
//      `hc.load` / `hc.store` / `hc.load_mask` into `hc.generic`, then
//      distribute wave-cooperative layouts to per-lane peers.
//   6. Scope normalisation + launch wrap. Fold supported `hc.func` call
//      boundaries + workitem regions, then wrap each `hc.kernel` in a
//      host `func.func` containing a `gpu.launch` with kernel-arg ABI
//      `(!hc.ptr<global, T>, dim*, stride*)`.
//   7. Flatten + body lowering. Collapse every shaped carrier to its
//      1D storage form and compose every offset to a single 1D
//      `#hc.expr`. Then lower the launch body twice: once over the
//      flat IR (scalar / control flow / ptr family / masked stores),
//      then again after `hc-lower-generic` collapses `hc.generic` ops
//      over `!hc.ptr` outs into `scf.parallel` / `scf.for` nests
//      planting fresh `hc.idx_apply` ops that still need lowering. A
//      barrier-insertion pass between the launch-body and lower-generic
//      slots plants `gpu.barrier` between `hc.generic` ops that share
//      a workgroup-AS storage root.
//   8. Recipes + outlining + target stamping. Interpret target
//      intrinsic recipes (`hc-interpret-intrinsic-recipes` walks the
//      sibling `@__hc_intrinsic_lowerings__` module), sink constant
//      index computations into `gpu.launch`, outline kernels into
//      `gpu.module @<kernel>_kernel` + `gpu.func` + `gpu.launch_func`,
//      and stamp each `gpu.module` with `#rocdl.target` (chip +
//      features resolved Python-side from `target=`).
//
// Conservative-funnel-with-per-op-fallback: the generic-funnel rewriters
// only fire on inputs they can prove safe. Inputs that don't match
// (intrinsic-mediated WMMA recipes, ...) flow through untouched and
// lower via the existing per-op handlers in `hc-lower-launch-body`. The
// two paths converge cleanly at `hc-lower-launch-body` / `hc-lower-generic`.
//
// `hc.compile` loads this via `-transform-preload-library` and runs it
// with `-transform-interpreter`; callers wanting a different order can
// pass `schedule=<path-or-text>` to override. The Python driver appends
// a fixed GPU-lowering chain (`_GPU_LOWERING_PIPELINE` in
// `hc/_pipeline.py`) after the schedule fires, so an override only
// needs to produce `gpu.module` ops carrying `#rocdl.target` for the
// rest of the pipeline to take over.
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
    // Specialization is the single explicit fold point for literal
    // bindings (`hc.compile(symbols={K: 4, ...})` stamps the
    // `literal_bindings` dict on each `hc.kernel`; this pass reads
    // them and substitutes every reachable `#hc.expr`/`#hc.pred` and
    // shape / layout / shaped-type carrier with the bound integers).
    // Runs immediately after the front-to-hc handshake so every later
    // pass — type inference, static-shape verification, decomposition,
    // flatten, launch-body lowering — sees concrete dims uniformly.
    // No-op when no kernel carries `literal_bindings`, so the
    // unspecialized `hc.compile(kernel)` path stays byte-identical.
    %m3a = transform.apply_registered_pass "hc-specialize-literals" to %m3
        : (!transform.any_op) -> !transform.any_op
    // `hc.pow` (the structural carrier the front-pass emits for Python
    // `**`) doesn't have a runtime lowering yet, so unfold the
    // positive-integer-literal cases into a mul chain here. Has to
    // run after `hc-specialize-literals` (in case a `literal_bindings`
    // sym ends up as the exponent) and before `hc-infer-types` so the
    // produced muls — not the carrier op — are what the inference pass
    // refines. Unsupported shapes (non-const, float, zero, negative
    // exponents) fail this pass with a per-case diagnostic.
    %m3b = transform.apply_registered_pass "hc-lower-pow" to %m3a
        : (!transform.any_op) -> !transform.any_op
    %m4 = transform.apply_registered_pass "hc-promote-names" to %m3b
        : (!transform.any_op) -> !transform.any_op
    %m5 = transform.apply_registered_pass "hc-infer-types" to %m4
        : (!transform.any_op) -> !transform.any_op
    %m6 = transform.apply_registered_pass "hc-materialize-bound-exprs" to %m5
        : (!transform.any_op) -> !transform.any_op
    %m7 = transform.apply_registered_pass "hc-verify-static-shapes" to %m6
        : (!transform.any_op) -> !transform.any_op
    %m8 = transform.apply_registered_pass "hc-decompose-shaped-values" to %m7
        : (!transform.any_op) -> !transform.any_op
    %m9 = transform.apply_registered_pass "hc-inline-helpers" to %m8
        : (!transform.any_op) -> !transform.any_op
    // Second materialize: helper inlining re-roots launch-geometry
    // producer chains (`hc.group_id` / `hc.local_id` / `hc.work_offset`
    // and friends) in the kernel scope. The first materialize handled
    // the kernel-body chains; this one catches the freshly-inlined
    // helper-body ones.
    %m10 = transform.apply_registered_pass "hc-materialize-bound-exprs" to %m9
        : (!transform.any_op) -> !transform.any_op
    transform.apply_dce to %m10 : !transform.any_op
    // Layout cleanup. Identity layouts fold to absent so the post-decompose
    // shape carriers stay layout-less wherever the user didn't pin a
    // non-identity layout; `hc.as_layout` chains collapse to their outer
    // arg. Non-identity layouts (transposed, padded, default-strided buffer
    // args) survive structurally — `hc-flatten-with-layouts` is the one
    // that erases the slot wholesale.
    %m10a = transform.apply_registered_pass "hc-canonicalize-layouts" to %m10
        : (!transform.any_op) -> !transform.any_op
    // Materialise `hc.strip_layout` — the user-marked layout-drop
    // boundary — into the `hc.generic` surface before the rest of
    // the shape-to-generic funnel fires. Strip is structurally
    // identical to a layout-aware `hc.vload` (gather elements via
    // the source's layout offset into a bare init), so the rewrite
    // shape mirrors the load-side lowering and the post-flatten
    // pipeline only ever sees ordinary `hc.generic` ops.
    %m10s = transform.apply_registered_pass "hc-lower-strip-layout" to %m10a
        : (!transform.any_op) -> !transform.any_op
    // Funnel the shaped op surface into `hc.generic` so the post-flatten
    // codegen has a single op family to lower. Each rewriter is
    // conservative — `hc.reduce` v0 wants single-axis shapes, the
    // per-element family wants all-shaped operands, and load/store
    // want pinned `!hc.idx<expr>` indices. Anything outside that
    // surface (slice-indexed access, mixed-rank ops, masked stores, ...)
    // stays in place and routes through the existing per-op handlers
    // in `hc-lower-launch-body`. The rewriters materialise iter bounds
    // directly from operand shapes; no separate inference step.
    %m10b = transform.apply_registered_pass "hc-shaped-compute-to-generic" to %m10s
        : (!transform.any_op) -> !transform.any_op
    %m10c = transform.apply_registered_pass "hc-elementwise-to-generic" to %m10b
        : (!transform.any_op) -> !transform.any_op
    // Rewrite `hc.builtin_call` (NumPy ufuncs forwarded by the front
    // pass) to upstream `math.<op>`. Runs after
    // `hc-elementwise-to-generic` so the carrier is in scalar form
    // inside a `hc.generic` body (HC's shaped types aren't compatible
    // with the `math.*` dialect, but its scalar / vector form is).
    // No-op when the kernel uses no NumPy ufuncs.
    %m10cm = transform.apply_registered_pass "hc-lower-math" to %m10c
        : (!transform.any_op) -> !transform.any_op
    %m10d = transform.apply_registered_pass "hc-load-store-to-generic" to %m10cm
        : (!transform.any_op) -> !transform.any_op
    // Distribute wave-cooperative layout-bearing carriers (e.g. the AMD
    // gfx11 WMMA accumulator's `<offset = 32*fi + lane>` accumulator)
    // to their per-lane peers. Runs after the load/store generic funnel
    // so every layout-bearing producer is in `hc.generic` shape, and
    // before `hc-normalize-scope-regions` so the workitem-region binding
    // for the lane sym is still visible to the rewrite. No-op for
    // kernels that don't carry wave-distributable layouts.
    %m10dw = transform.apply_registered_pass "hc-distribute-wave-layouts" to %m10d
        : (!transform.any_op) -> !transform.any_op
    %m11 = transform.apply_registered_pass "hc-normalize-scope-regions" to %m10dw
        : (!transform.any_op) -> !transform.any_op
    // Scope normalisation strips supported `hc.func` call boundaries
    // and result-producing workitem regions, replumbing every user
    // through to the new SSA shape. The canon/cse pair folds the
    // freshly-exposed index chains (the binding sym chains the scope
    // rewrite cut) and CSEs the duplicate computations the scope
    // rewrite left behind so the next pass walks a clean IR.
    transform.apply_patterns to %m11 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m11 : !transform.any_op
    %m12 = transform.apply_registered_pass "hc-lower-kernels-to-gpu-launch" to %m11
        : (!transform.any_op) -> !transform.any_op
    // Flatten before launch-body so launch-body sees rank-1 buffer
    // carriers and a single composed 1D `#hc.expr` per access. The
    // post-flatten layout retyper inside flatten handles the buffer-
    // from-ptr UCC chain that `hc-lower-kernels-to-gpu-launch`
    // plants on each kernel-arg boundary, and the post-flatten
    // collective-lift verifier (extended for the storage-product
    // regime) keeps `hc.workitem_region` / `hc.subgroup_region`
    // accepting the post-flatten yield/result shape. The
    // `applyPartialConversion` step that flatten drives plants
    // boundary UCCs on every type it converted; fold them through
    // the standard cleanup pair before launch-body walks the IR.
    //
    // `hc.load_mask` doesn't need a pre-flatten snapshot here:
    // `hc-load-store-to-generic` above rewrites every load_mask to an
    // `hc.generic` whose body computes the per-axis bounds predicate
    // structurally from the slice's `lo`, `step`, and the source's
    // multi-dim shape syms. Flatten then folds the result-tile
    // offsets to 1D the same way it folds load/store offsets, and
    // launch-body lowers the body's `hc.pred_apply` via the same
    // ExprLowerer that runs on every other apply.
    %m12b = transform.apply_registered_pass "hc-flatten-with-layouts" to %m12
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m12b {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m12b : !transform.any_op
    // Preflight gate: bail if any semantic `!hc.tensor` / `!hc.vector`
    // slipped past decompose, or any bare carrier has a non-literal
    // shape. Hoisted out of launch-body so the walk fires once
    // instead of per launch-body invocation.
    %m12v = transform.apply_registered_pass "hc-verify-bare-carriers" to %m12b
        : (!transform.any_op) -> !transform.any_op
    // Bridge `hc.intrinsic` signatures + `hc.call_intrinsic` boundaries
    // to the launch-body type converter's projection ahead of the
    // main lowering. Same converter, so launch-body's dyn-legal check
    // on intrinsics passes through on already-bridged IR.
    %m12i = transform.apply_registered_pass "hc-bridge-intrinsics" to %m12v
        : (!transform.any_op) -> !transform.any_op
    %m13b = transform.apply_registered_pass "hc-lower-launch-body" to %m12i
        : (!transform.any_op) -> !transform.any_op
    // Launch-body emitted fresh upstream `arith.constant` /
    // `index_cast` / `arith.muli` chains via ExprLowerer for every
    // `hc.idx_apply`, plus a sea of intermediate values on the bare
    // side. The canon/cse pair folds the constant arithmetic and CSEs
    // the redundant per-thread index computations before
    // `hc-insert-workgroup-barriers` walks the IR (its alias rules
    // key off the storage-root SSA value, so duplicate-root noise
    // would inflate the pending set).
    transform.apply_patterns to %m13b {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m13b : !transform.any_op
    // Centralise cross-`hc.generic` synchronization on workgroup-AS
    // storage (LDS). Runs after the first `hc-lower-launch-body` so
    // every workgroup-staged fill / load / store already lives inside
    // a structured `hc.generic` rooted at `hc.alloc workgroup`, and
    // before `hc-lower-generic` so the per-axis offsets / iter syms a
    // future precise-elision pass needs are still on the op.
    // Conservative rule: a `gpu.barrier` lands before an `hc.generic`
    // iff some earlier generic in the same block has already written
    // (or is about to overwrite) a workgroup-AS storage root this
    // generic touches. Empty pending after each barrier; stray
    // non-generic loads / stores on workgroup storage are out of
    // scope by design (the contract is "workgroup writes live in
    // `hc.generic`"). No-op on payloads with no workgroup tiles.
    %m13bs = transform.apply_registered_pass "hc-insert-workgroup-barriers" to %m13b
        : (!transform.any_op) -> !transform.any_op
    // Lower every `hc.generic` whose operands are `!hc.ptr<...>`
    // (resolved by the launch-body pass above off the kernel-arg
    // UCCs) with rank-1 composed offsets (built by the flatten
    // pass above) to an outer `scf.parallel` over the parallel
    // iters and an inner `scf.for` nest over reduction iters. The
    // pass bails on any generic that doesn't fit its v0 surface —
    // for the WMMA path that means a no-op since the recipe is
    // dispatched via `hc-interpret-intrinsic-recipes` instead of
    // generic.
    //
    // Per-lane offset emission plants fresh `hc.idx_apply` ops; each
    // apply has every ambient symbol operand-bound (the bindings
    // ride on `hc.generic`'s `ambient_idxs` slot, captured by
    // `hc-flatten-with-layouts` while the kernel-arg bundle UCC
    // chain and structured-loop induction vars were still HC-typed).
    %m13a = transform.apply_registered_pass "hc-lower-generic" to %m13bs
        : (!transform.any_op) -> !transform.any_op
    // `hc.predicate` ops ride through `hc-lower-generic`'s `cloneBody` as
    // ordinary body ops -- the pass doesn't touch them, the predicate
    // physically lands in the lowered `scf` body next to its (now
    // explicit) `hc.ptr_load` producer. Folding them is a separate pass
    // so the emission side stays unaware of mask shapes / producer kinds
    // and so the fold runs on every emitter that can plant a predicate,
    // not just on `hc-lower-generic`'s output.
    %m13af = transform.apply_registered_pass "hc-fold-predicates" to %m13a
        : (!transform.any_op) -> !transform.any_op
    // Pick up the per-lane `hc.idx_apply` / `hc.pred_apply` ops
    // `hc-lower-generic` minted on its way out of the body and lower
    // them to `arith.*` / `index_cast` via the same `ExprLowerer`
    // launch-body uses. `hc-lower-apply` carries only the apply
    // patterns -- no need to re-spin the full launch-body target on
    // IR that's already past the launch-body legality bar.
    %m13ar = transform.apply_registered_pass "hc-lower-apply" to %m13af
        : (!transform.any_op) -> !transform.any_op
    // Same rationale as the post-first-launch-body pair: ExprLowerer
    // planted fresh arith / index_cast chains for every per-lane
    // `hc.idx_apply` the generic lowering minted. The canon/cse pair
    // folds the constant fragments before recipe interpretation
    // pattern-matches on the post-launch shape.
    transform.apply_patterns to %m13ar {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m13ar : !transform.any_op
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
        with options = { "target" = "__HC_TARGET__" } to %m13ar
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
    // Outliner cloned every index producer used inside the launch
    // body into the new `gpu.func` (so the kernel body is
    // self-contained), leaving the originals as dead host-side
    // values. The canon/cse pair drops the dead host clones and
    // CSEs duplicate host-side dim/stride producers consumed by the
    // freshly minted `gpu.launch_func`. Likely reducible if the
    // outliner gains a built-in cleanup; tracked separately.
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
    transform.yield
  }
}
