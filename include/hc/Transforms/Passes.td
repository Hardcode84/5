// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES
#define HC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCPromoteNames : Pass<"hc-promote-names"> {
  let summary = "Promote Python name-store ops to SSA values.";
  let description = [{
    Replaces `hc.assign` / `hc.name_load` with structured SSA carriers,
    rebuilding region-carrying ops so each Python-level name becomes a
    real SSA value or iter_arg / result / block-arg of the enclosing
    region op. This pass is the second half of the pipeline; the
    pipeline contract — what the placeholders are, who is allowed to
    look at them, what "supported invocation" means — lives on
    `ConvertHCFrontToHC` in `include/hc/Conversion/HCFrontToHC/Passes.td`.

    Runs bottom-up on every function region tree (`hc.kernel` / `hc.func`
    / `hc.intrinsic`). Region ops that tag themselves with
    `NameStoreRegionOpInterface` participate:

      - `hc.for_range`  : carried names become iter_args + yielded
                          values. A leading `hc.assign "<n>", %iv`
                          (the frontend's way of publishing the IV
                          under its Python-source name) is recognized
                          as a tautological self-bind, kept loop-local,
                          and folded to direct uses of the block arg.
      - `hc.if`         : carried names become op results + two-branch
                          yields; branches must agree on the write set.
      - `hc.workitem_region` / `hc.subgroup_region` : Python-style
                          nested scopes. Reads inside capture outer
                          bindings (lazy outer snapshot); writes
                          shadow without leaking. Out-bound
                          propagation is opt-in via an explicit
                          `hc.region_return <names>` terminator: the
                          promoter rebuilds the op with one result
                          per returned name and pushes writebacks out
                          to the enclosing scope's name store.

    Post-condition (on success): no `hc.assign` / `hc.name_load` survives
    anywhere under the processed callables. Cross-kernel / cross-function
    name capture is out of scope; each callable is its own promotion
    root.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInferTypes : Pass<"hc-infer-types"> {
  let summary = "Infer concrete HC value types from pre-inference `!hc.undef` IR.";
  let description = [{
    Runs a sparse forward dataflow analysis over `hc` IR and rewrites values
    whose type can be refined without changing program structure. The first
    slice focuses on scalar/index facts: constants, symbols, launch geometry,
    slice construction, buffer dimensions, index arithmetic/comparisons, and
    structured loop carriers. Inference is monotonic: `!hc.undef` refines to a
    concrete type, and conflicting concrete facts fail the pass instead of
    being reconciled with a guess.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCVerifyStaticShapes : Pass<"hc-verify-static-shapes"> {
  let summary = "Verify SSA shape operands have static HC shape metadata.";
  let description = [{
    Checks `hc.load`, `hc.vload`, and shaped allocator ops after type
    inference. Shape operands must be tuple values whose element types are
    pinned `!hc.idx<...>` dimensions, and the resulting tensor/vector type must
    carry the same static shape. Load-like ops also require a statically-ranked
    source and index/slice operands whose arity stays within that source rank.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCCanonicalizeLayouts : Pass<"hc-canonicalize-layouts"> {
  let summary = "Fold explicit identity layouts and collapse layout chains.";
  let description = [{
    Cleanup pass for the `#hc.layout<...>` attribute introduced in
    `doc/layouts.md`. Three rewrites, in order:

      1. Strip explicit identity layouts from shaped Values. A layout
         is identity for an n-dim shape iff its shape_syms /
         index_syms have rank n, params is empty, storage_size is
         the product over shape_syms, and offset is the dot product
         of index_syms with tail-product strides over shape_syms
         (rightmost axis varies fastest), all under ixsimpl-canonical
         form. Matching values get `setType` to the layout-less type
         — the layout slot was made optional precisely so
         "absent equals identity" is the canonical form.

      2. ixsimpl-normalize layout payloads. This is structurally a
         no-op because `LayoutAttr` normalizes through the symbolic
         store at construction; the pass exercises the contract
         under `--debug` and serves as the documented place to
         reconstruct stale forms once a future API allows it.

      3. Collapse `hc.as_layout(hc.as_layout(%v, L1), L2)` to
         `hc.as_layout(%v, L2)`. The outer choice wins regardless of
         L1 / L2 — semantically `as_layout` overwrites the prior
         layout, not composes with it.

    Runs after `hc-infer-types`; no IR-shape changes elsewhere. The
    type-level fold also propagates to BlockArguments so function
    signatures and region bodies stay coherent. Function-type
    attributes (the cached signature on `hc.func` /
    `hc.kernel` / `hc.intrinsic`) are rebuilt to match.

    Implementation routes through MLIR's dialect-conversion driver so
    the type rewrite stays consistent with how SCF / func / call
    structural conversions are done elsewhere in the codebase. The
    pass picks up `func` and `scf` dialect dependencies for that
    reason — the populators they ship know how to clone-and-replace
    `func.func` signatures, `scf.for` iter_args, etc., while keeping
    the conversion driver's materialization tracking honest. Identity
    materializations are registered on the converter because the
    layout-stripped and layout-bearing forms are observationally
    interchangeable.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def HCShapedComputeToGeneric : Pass<"hc-shaped-compute-to-generic"> {
  let summary = "Rewrite hc.matmul / hc.reduce into hc.generic + identity fill.";
  let description = [{
    Folds the source-level `hc.matmul` and `hc.reduce` ops into a
    single body-driven `hc.generic` form. After this pass the
    post-flatten compute surface is one op (`hc.generic`); the source
    shapes (`hc.matmul`, `hc.reduce`, plus the per-element
    decomposition that lands later) are syntactic shorthands the
    frontend keeps for ergonomics.

    Each rewrite emits its iter bounds as an empty-binding
    `hc.idx_apply` carrying `!hc.idx<dim>` sourced directly from
    operand shapes — the rewriter has full structural knowledge of
    the iteration space and the bounds-inference pass would be a
    no-op if it ran after.

    `hc.matmul lhs, rhs : (T<M,K>, T<K,N>) -> T<M,N>` → an `hc.zeros`
    fill on the result type plus an `hc.generic` with two parallel
    iters (M, N) and one reduction iter (K). Inputs are addressed
    `[i,k]` and `[k,j]`, the output `[i,j]`. Body extends the inputs
    to the result element type via `hc.astype` when they differ, then
    runs `arith.mulf` + `arith.addf` (or `arith.muli` + `arith.addi`
    for integers). v0 only handles rank-2 operands and identical
    input element types; mixed-type and batched matmul are
    follow-ups.

    `hc.reduce val, kind = K, axis = a : T<...> -> T<...>` →
    an identity fill (`hc.zeros` for sum, `hc.full` with -inf / +inf
    for max / min on float operands) on the result shape, plus an
    `hc.generic` with one reduction iter on `axis a` and parallel
    iters on every other axis. Body picks the combinator: `arith.addf`
    / `arith.maximumf` / `arith.minimumf` for floats; integer support
    is sum-only in v0 (max / min identity for unsigned vs signed
    needs a separate slot decision). `keepdims = true` is not yet
    handled — the reshape it implies is a separate piece of work.

    Operands typed `!hc.undef` (pre-inference) and any case the v0
    matcher doesn't cover are left in place — downstream passes
    diagnose if they were the last remaining consumer.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::hc::HCDialect"
  ];
}

def HCLoadStoreToGeneric : Pass<"hc-load-store-to-generic"> {
  let summary = "Rewrite hc.load / hc.vload / hc.store into hc.generic.";
  let description = [{
    Funnels the remaining shaped memory ops into the body-driven
    `hc.generic` surface so post-flatten codegen has a single op
    family to lower. Counterpart to `hc-shaped-compute-to-generic` for
    the load/store side; both run pre-flatten on semantic shaped types
    so the per-axis offsets line up with the operand shapes before
    `hc-flatten-with-layouts` composes them through the layout.

    Rewrites:

    `hc.load %buf[%i_0, ..., %i_n], shape %shape -> T<...>` → an
    `hc.zeros` fill of the result shape plus an `hc.generic` with one
    parallel iter per result axis. The buffer/tensor input carries
    per-axis offsets `[i_0_expr + i_0, ..., i_n_expr + i_n]`,
    where each `i_k_expr` is the access op's k-th index symbol. The
    synthesised init is the `outs` operand with identity offsets. The
    body forwards the loaded element through `hc.yield`. `hc.vload`
    follows the same shape with a vector result.

    `hc.store %dst[%i_0, ..., %i_n], %src` → an `hc.generic` with
    `%src` as a value-typed `ins` (identity offsets) and `%dst` as a
    ptr/buffer `outs` (per-axis offsets `[i_0_expr + i_0, ...]`). The
    body forwards the source element through `hc.yield`; downstream
    `hc-lower-generic` materialises the implicit `hc.ptr_store` at the
    out's offset. No SSA result.

    Empty index list (`hc.load %buf[]` / `hc.store %dst[]`) reduces
    the per-axis offset on the memory operand to identity over the
    iter syms — same body, no addressing addend.

    Iter bounds are emitted as empty-binding `hc.idx_apply` ops
    carrying `!hc.idx<dim>` sourced directly from the op's tile
    shape, so the later `hc-infer-generic-bounds` pass would be a
    no-op if it ran after this one. Iter sym names use `i_<k>` to
    stay distinct from the symbolic names the access op's index
    operands carry.

    Out of scope (no rewrite, op survives for downstream diagnostics):

      * Pre-inference operands typed `!hc.undef` — no shape to source
        bounds from.
      * Index operands not typed `!hc.idx<expr>` — the rewriter has
        no symbolic name to bind to the iter sym, so the offset
        composition would lose information.
      * `hc.store` with a `mask` operand — masked stores need an
        scf.if-shaped body the v0 rewrite doesn't emit; deferred.
      * `hc.load_mask` — the bare-predicate companion stays a
        primitive; only data movement gets rewritten here.
      * `hc.buffer_view` and `hc.vec` — sub-view and broadcast
        semantics that fall under separate slices.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCElementwiseToGeneric : Pass<"hc-elementwise-to-generic"> {
  let summary = "Rewrite per-element shaped arith / cmp / astype ops into "
                "hc.generic.";
  let description = [{
    Funnels the per-element decomposition family into a single
    body-driven `hc.generic` form so the post-flatten pipeline only
    has to lower one shaped compute op. Pure source-level rewrite —
    no memory effects, no vectorization choices, no codegen.

    v0 covers the all-shaped (no scalar broadcast) cases for:

      * binary arith / boolean: `hc.add`, `hc.sub`, `hc.mul`,
        `hc.div`, `hc.mod`, `hc.and`, `hc.or`,
      * unary arith / boolean: `hc.neg`, `hc.not`,
      * comparisons: `hc.cmp.lt`, `hc.cmp.le`, `hc.cmp.gt`,
        `hc.cmp.ge`, `hc.cmp.eq`, `hc.cmp.ne`,
      * cast: `hc.astype`.

    Each rewrite emits:

      * one parallel iter sym per result axis (`i_0`, `i_1`, ...) —
        distinct from the source-level symbol names operands carry,
      * `iter_bounds` as `hc.undef_value` placeholders that
        `hc-infer-generic-bounds` resolves from the operand shapes,
      * shaped inputs as `ins` with identity per-axis offsets,
      * a synthesised `hc.zeros` / `hc.vzeros` init as the single
        `outs` with identity per-axis offsets,
      * a body whose scalar arith mirrors the source op, terminated
        by `hc.yield`.

    Out of scope (op stays in place; downstream passes still
    diagnose if the op is the only remaining consumer):

      * Mixed scalar / shaped operands (broadcast). The body would
        need a splat or a scalar capture pattern; deferred.
      * `hc.select`. The false-value is a scalar inactive fill that
        needs the same broadcast story.
      * Nullary fills (`hc.zeros`, `hc.ones`, `hc.full`, `hc.const`,
        `hc.empty`). The init scaffolding folds back on itself
        unless we grow a `hc.generic` form whose init is implicit.
      * Pre-inference `!hc.undef` operands.
      * Operands with mismatched ranks or per-axis dim expressions
        (no broadcast support yet).
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInferGenericBounds : Pass<"hc-infer-generic-bounds"> {
  let summary = "Fill placeholder iter_bounds on hc.generic from operand shapes.";
  let description = [{
    Walks every `hc.generic` and resolves any `iter_bounds` operand
    defined by `hc.undef_value` to a concrete `!hc.idx<...>` SSA value
    derived from operand shapes. Inference must run before any pass that
    composes per-axis offsets into a single linear expression — once the
    per-axis structure is gone, recovering iter ranges from a composed
    offset is intractable.

    Per iter sym `s` with a placeholder bound:

      1. For every (operand, axis) pair where the per-axis offset is
         exactly the bare symbol `s` under ixsimpl-canonical form,
         collect the operand's shape entry at `axis` as an implied
         bound. Non-identity offsets (`s + 1`, `2*s`, ...) are not yet
         understood and are skipped — the operand contributes no
         binding for `s` through this axis.

      2. Multiple identity occurrences must agree on the implied bound
         expression (hash-consed pointer equality through the dialect
         store). Conflicts diagnose and name the offending operand
         pair.

      3. The first agreed bound is materialized via an empty-binding
         `hc.idx_apply` carrying `!hc.idx<dim>` and used to replace
         the placeholder operand on the op. The original
         `hc.undef_value` op is left for canonicalization / DCE if no
         other use survives.

    A placeholder for which no identity occurrence is found diagnoses:
    the user wrote a body the pass cannot infer bounds for, and going
    further would let the placeholder leak past flatten where the lack
    of an answer becomes much harder to localize. The pass is a no-op
    when every bound on every `hc.generic` is already concrete, so
    rewrites that emit fully-resolved ops (`hc.reduce` / `hc.matmul` →
    `hc.generic`) pay nothing.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerGeneric : Pass<"hc-lower-generic"> {
  let summary = "Lower hc.generic to scf.parallel / scf.for + hc.ptr_offset / "
                "hc.ptr_load / hc.ptr_store.";
  let description = [{
    Unroll-and-merge codegen frame for `hc.generic`, documented in
    `doc/layouts.md`. Walks every `hc.generic` and rewrites it as an
    outer `scf.parallel` over the parallel iters plus an inner
    `scf.for` nest over the reduction iters with `iter_args` carrying
    the per-output accumulators. Pure-parallel ops collapse to a
    single `scf.parallel`; pure-reduction ops collapse to a single
    `scf.for`.

    The pass picks an `(axis-order, partition)` pair via a
    divisibility-pruned merge-score search: every per-axis factor
    `p_a` must provably divide its bound (probed via
    `ixs_check(cmp(ixs_mod(N_a, p_a), EQ, 0))`), and each surviving
    candidate is scored by the per-operand pairwise contig probe
    (`ixs_check(cmp(off[k+1] - off[k], EQ, 1))`). The body stays
    elementwise scalar at every total unroll: scalar contig groups
    lower to scalar `hc.ptr_load` / `hc.ptr_store`, contig groups of
    size > 1 lower to vector-typed `hc.ptr_load` / `hc.ptr_store` of
    width `G` plumbed through `vector.extract` /
    `vector.from_elements` at the body boundary. The trivial
    `(1, ..., 1)` partition collapses verbatim to the scalar
    baseline emission.

    Per-iteration body emission (per partition lane):

      * For each `ins` operand: evaluate its single composed offset
        expression to a `Value` of `index` type, emit
        `hc.ptr_offset` then `hc.ptr_load`. The loaded value
        replaces the corresponding body block argument.
      * For each `outs` operand: load the initial accumulator once
        per parallel-loop iteration via `hc.ptr_offset` +
        `hc.ptr_load` at the operand's parallel-only offset, thread
        it through the inner `scf.for`'s `iter_args`, and emit
        `hc.ptr_store` after the reduction nest with the final
        accumulator.
      * Body region is cloned in place with the block arguments
        remapped to the loaded values (ins) and the running carry
        (outs). The cloned `hc.yield` provides the per-iteration
        update for each output's accumulator.

    Per-operand offsets surface as `hc.idx_apply` carrying the
    composed `ExprAttr` (with the per-lane delta substituted into
    each iter sym) and binding the listed iter syms to the loop
    induction vars. The launch-body lowering downstream walks the
    expression and substitutes the listed operands plus any ambient
    kernel-bound symbols (shape dims, stride params).

    Pre-flight bail (the op stays in place; downstream passes still
    diagnose if the op is the only remaining consumer):

      * Any operand is not `!hc.ptr<...>` — value-typed bare
        carriers (`!hc.bare_tensor`, `!hc.bare_vector`),
        `!hc.buffer`, and `!hc.undef` are out of v0 scope.
      * Any operand has rank > 1 in its offset array — multi-axis
        offsets need linearization through the operand's shape,
        which in turn needs outer-scope shape-symbol bindings the
        v0 walker doesn't reach to in every case. Handled when
        `hc-flatten-with-layouts` lands and every op-time operand
        is rank-1.
      * Any iter bound is still a `hc.undef_value` placeholder.
      * Any offset references a free symbol the outer-scope walker
        can't resolve — the lookup walks parent ops looking for a
        block argument or op result whose type is `!hc.idx<sym>`
        with `sym` a bare-symbol expression matching the free name.
      * `scf.if` anywhere in the body — residual case from the
        predicated-bodies note in `doc/layouts.md`; rare in
        practice once masked load/store sites route through the
        predicated `hc.ptr_load_pred` / `hc.ptr_store_pred` ops.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::hc::HCDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def HCFoldPredicates : Pass<"hc-fold-predicates"> {
  let summary = "Fold `hc.predicate` by hoisting the mask to its producer "
                "or collapsing to `arith.select`.";
  let description = [{
    Walks every `hc.predicate %v mask %m passthrough %f` op and resolves
    it by inspecting the def of `$value`:

      * Trivial masks fold first — `m_One()` replaces the op with `$value`
        (drop the predicate), `m_Zero()` replaces with `$passthrough`
        (load elided).

      * `hc.ptr_load` / `hc.vload` / `hc.load`: clone the producer at
        its def site in predicated form (`hc.ptr_load_pred` etc.) with
        `$mask` / `$passthrough`. The predicate op's uses get
        redirected to the new predicated load. Each `hc.predicate` use
        of the same load creates its own clone — sharing is opt-in,
        not default. The original load is dropped once it has no
        remaining uses.

      * `vector.extract` of a vector value: emit `arith.select %mask,
        %value, %passthrough` at the predicate site. The underlying
        vector load stays unconditional; the predicate gates the lane
        only.

      * Anything else is a hard error — the allow-list is deliberate,
        extending it is a per-producer choice.

    Dominance: `$mask` and `$passthrough` must dominate the producer's
    def site when the lowering clones / rewrites there. Violations are
    emitted as diagnostics, the pass signals failure.

    Conceptually the pass is the lowering counterpart of
    `hc.yield_predicated`'s store-side path: both edges of an
    `hc.generic` body decompose into predicated mem ops or
    `arith.select` blends, depending on the producer / sink.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::hc::HCDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def HCFlattenWithLayouts : Pass<"hc-flatten-with-layouts"> {
  let summary = "Type-only flatten: every shaped value becomes 1D, "
                "no #hc.layout slot survives.";
  let description = [{
    Establishes the post-flatten *type* invariant: every
    `SymbolicallyShapedTypeInterface` value (except buffers) is 1D
    and carries no `#hc.layout` attribute. Op surfaces stay
    untouched — `hc.generic` keeps its per-axis offset arrays at
    their original logical rank, `hc.load` / `hc.store` /
    `hc.vload` / `hc.buffer_view` keep their multi-index lists.
    Whoever turns those nD logical accesses into concrete 1D
    addresses owns the layout->offset materialization in a
    follow-up slice.

    What runs:

      1. A `TypeConverter` whose dispatch on the shaped-type interface
         computes `storage_size_expr` (`layout.storage_size` after
         binding `layout.shape_syms` to the original shape entries,
         or the product of the original dimensions when no layout is
         attached) and rebuilds the type with a single-entry shape
         and a null layout slot. Tuple and function types recurse so
         nested carriers get the same fold without per-shell
         hand-rolling.

      2. Upstream populators thread the converter through `func.func` /
         `func.return` / `func.call` signatures and the SCF structural
         ops (`scf.for`, `scf.if`, `scf.while`, `scf.index_switch`).

      3. `hc.as_layout` collapses to its operand once both endpoints
         carry the converted form — the op was a layout-only relabel,
         and after the type fold both sides are observationally
         identical.

    Buffer types are deliberately exempt: their default strided
    layout's `storage_size` is the literal `0` placeholder the
    frontend emits (the host owns the allocation), so 1D-collapsing
    them would fold every buffer shape to `[0]`. They keep their
    original nD shape and lose only the layout slot for now; a
    separate slice owns the buffer-side ABI.

    Cooperates with `hc-canonicalize-layouts`: that pass folds
    explicit *identity* layouts back to absent; this one collapses
    every shape (with or without an explicit layout) to its 1D
    storage form. Not yet wired into the default `hc.compile`
    schedule — downstream passes still expect nD operand types and
    a layout-aware access path that the follow-up slice will
    materialize.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def HCDecomposeShapedValues
    : Pass<"hc-decompose-shaped-values"> {
  let summary = "Split semantic tensor/vector values into bare data and masks.";
  let description = [{
    Rewrites supported semantic `!hc.tensor` / `!hc.vector` producers and users
    into compiler-internal `!hc.bare_tensor` / `!hc.bare_vector` data plus
    matching bare `!hc.pred` masks. This is an HC-to-HC normalization boundary
    for upstream lowering; semantic shaped values that cross unsupported ops are
    diagnosed instead of being silently dropped.

    Ordering: scheduled after `hc-infer-types`, `hc-materialize-bound-exprs`,
    and `hc-verify-static-shapes`. The decomposition reads concrete shaped
    types (so the mask-type derivation can build matching `!hc.pred` carriers)
    and assumes bound symbolic expressions have already been severed from
    their producer chains, so it only sees the post-inference shape surface.

    Legality in `strict=false` mode is asymmetric: unknown ops are legal
    (passed through unchanged), but the explicit HC op list in
    `makePartialShapedDecompositionTarget` is still converter-legal-checked
    because every op in that list is a registered rewrite target — if one of
    them carries a semantic shaped operand or result, the partial conversion
    decomposes it even though the rest of the IR may keep its semantic shape
    via `builtin.unrealized_conversion_cast` boundaries. Strict mode collapses
    the asymmetry: every op must converter-legalize before the pass succeeds.
  }];
  let options = [
    Option<"strictMode", "strict", "bool", "true",
           "Fail unless every semantic shaped boundary is decomposed. Disable "
           "this to run a partial conversion that preserves unsupported "
           "boundaries with unrealized conversion casts.">
  ];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerBarePredicates
    : Pass<"hc-lower-bare-predicates"> {
  let summary = "Lower bare HC predicate containers to upstream i1 masks.";
  let description = [{
    Converts concrete `!hc.pred` values to `i1` and static `!hc.bare_vector`
    values to upstream `vector` values. Predicate vectors become
    `vector<...xi1>` masks, and supported HC mask operations are lowered to
    `arith`/`vector` operations. Bare tensors remain HC values for the later
    memory/tensor lowering. Unsupported HC boundaries are preserved with
    `builtin.unrealized_conversion_cast` so this pass can be staged before the
    full HC-to-upstream lowering.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::hc::HCDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def HCMaterializeBoundExprs
    : Pass<"hc-materialize-bound-exprs"> {
  let summary = "Sever bound symbolic HC expressions from their producer chains.";
  let description = [{
    Replaces uses of values typed as pinned `!hc.idx<expr>` or
    `!hc.pred<pred>` with an empty-binding `hc.idx_apply` /
    `hc.pred_apply` (no listed symbols, no operand SSA edges) when
    every symbolic leaf is a bound symbol or the expression is
    constant-only. The original producer is left in place for
    canonicalization/DCE, which lets this pass break dependence on
    launch-context query chains before scope normalization. The
    severed apply ops resolve their pinned expressions ambiently
    against the surrounding launch context at the launch-body
    lowering boundary.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInlineHelpers
    : Pass<"hc-inline-helpers"> {
  let summary = "Inline supported HC helper calls.";
  let description = [{
    Inlines `hc.call` sites targeting `hc.func` helpers. The pass owns HC
    helper call-graph concerns separately from scope-region normalization:
    recursion and unsupported callee body shapes are diagnosed here, and helpers
    that were actually inlined are erased once no `hc.call` references remain.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCNormalizeScopeRegions
    : Pass<"hc-normalize-scope-regions"> {
  let summary = "Flatten supported collective scope regions into SPMD HC IR.";
  let description = [{
    Normalizes HC toward per-workitem SPMD form before upstream GPU lowering.
    The first supported slice flattens resultless `hc.workitem_region` bodies
    after bound expression materialization has removed live uses of the
    workitem token. Result-producing workitem regions and subgroup regions are
    diagnosed until distributed value projection and subgroup-local workitem
    semantics are handled explicitly.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerKernelsToGPULaunch
    : Pass<"hc-lower-kernels-to-gpu-launch"> {
  let summary = "Wrap HC kernels in upstream GPU launch operations.";
  let description = [{
    Rewrites `hc.kernel` symbols into host `func.func` entry points that launch
    the former kernel body with `gpu.launch`. The host wrapper signature is a
    leading `!llvm.ptr` stream pointer followed by one `!llvm.ptr`
    (`PyObject *`) per non-`!hc.group` kernel argument; the wrapper body calls
    the `_mlir_ciface_hc_get_*` runtime helpers (declared as `func.func
    private` with `llvm.emit_c_interface`) to unpack each PyObject into a
    raw pointer / dim / stride / scalar before computing launch geometry.
    Each buffer argument materializes as an `(!hc.ptr<global, T>, dim*,
    stride*)` tuple sourced from `hc_get_ptr` + per-axis `hc_get_dim` /
    `hc_get_stride`; an N→1 `unrealized_conversion_cast` planted inside the
    `gpu.launch` region bridges the tuple back to `!hc.buffer<...>` so
    `gpu-kernel-outlining` captures the raw pointer / dim / stride values
    the kernel ultimately consumes. The stream pointer threads through to the
    runtime launch calls in `hc-lower-launch-func-to-runtime`; passing null
    keeps HIP's default-stream semantics.

    The original HC kernel body is kept inside the launch behind explicit cast
    boundaries for later memory and intrinsic lowering passes.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::hc::HCDialect"
  ];
}

def HCInterpretIntrinsicRecipes
    : Pass<"hc-interpret-intrinsic-recipes", "::mlir::ModuleOp"> {
  let summary = "Apply embedded intrinsic transform recipes to payload IR.";
  let description = [{
    Looks up a sibling top-level `builtin.module @__hc_intrinsic_lowerings__`
    (carrying `transform.with_named_sequence`) and applies every
    `transform.named_sequence` whose `hc.target` matches the requested target
    string against the surrounding module. The recipes were planted by the
    frontend emitter (`hc/_frontend_mlir.py`) and stick to MLIR transform IR
    end-to-end — no string round-trip.

    Recipes are expected to follow the standard hc transform-op shape: a
    `transform.hc.match_intrinsic_call` to find the call site, optional
    `transform.hc.get_intrinsic_*` handle ops to thread operands, result
    types, and attributes into a `transform.hc.create_op`, then a
    `transform.hc.replace_intrinsic_call` that swaps the original
    `hc.call_intrinsic` for the freshly constructed payload.

    After applying every selected sequence, the lowerings module is erased
    and the pass walks the payload one final time. Any surviving
    `hc.call_intrinsic` is a hard error: the user asked for target `T` but
    nothing under `__hc_intrinsic_lowerings__` knew how to lower the call,
    so we surface the gap here instead of letting it slip into the next
    pass as an unrecognized op.
  }];
  let options = [
    Option<"target", "target", "std::string", "\"\"",
           "Filter named sequences by their `hc.target` attribute. The "
           "empty string runs every sequence and is mostly useful for "
           "single-target tests.">
  ];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::transform::TransformDialect"
  ];
}

def HCLowerGPUToBinary
    : Pass<"hc-lower-gpu-to-binary", "::mlir::ModuleOp"> {
  let summary = "Compile each gpu.module to a HSACO blob via our pinned ld.lld.";
  let description = [{
    For every `gpu.module` carrying exactly one `#rocdl.target<...>`
    attribute the pass:

      1. translates the module to LLVM IR (`translateModuleToLLVMIR`),
      2. builds an `llvm::TargetMachine` from the rocdl target's triple/
         chip/features and pins the IR's data layout + triple,
      3. runs the standard optimize-then-codegen pipeline at the target's
         opt level (`makeOptimizingTransformer`),
      4. translates LLVM IR to ISA text
         (`LLVM::ModuleToObject::translateModuleToISA`),
      5. assembles ISA → ELF object via `ROCDL::assembleIsa`,
      6. links the ELF → HSACO via `ROCDL::linkObjectCode`, calling our
         pinned `ld.lld`.

    The result is attached as a `gpu::ObjectAttr` on a sibling
    `gpu.binary` op (CompilationTarget::Binary, with the original
    rocdl target preserved), and the source `gpu.module` is erased.

    `lld-path` resolves in this order:

      1. the option's value if non-empty,
      2. the `HC_LLD` environment variable (escape hatch for direct
         `hc-opt` invocations against a non-default toolchain),
      3. the first `ld.lld` on `$PATH`.

    The Python pipeline driver (`hc.compile`) always populates option
    1 from the bundled `hc/_native/bin/ld.lld`, so `HC_LLD` and `$PATH`
    only matter for callers reaching the pass directly via `hc-opt`.
    No ROCm install fallback — if none of the three succeed we error
    out rather than silently picking up a host's distro-installed lld.

    v0 deliberately does not link rocm-device-libs bitcode. The WMMA
    example doesn't reference `__ocml_*` / `__ockl_*`; kernels that do
    will need a follow-up pass that resolves the device-libs path and
    feeds bitcode files into the optimization pipeline before codegen.
  }];
  let options = [
    Option<"lldPath", "lld-path", "std::string", "\"\"",
           "Explicit path to ld.lld; falls back to HC_LLD then $PATH "
           "if empty.">,
    Option<"dumpIntermediates", "dump-intermediates", "std::string", "\"\"",
           "Directory to write per-stage codegen artifacts into; empty "
           "disables dumping. Per gpu.module the pass writes "
           "<module>.0-pre-opt.ll (LLVM IR before optimization), "
           "<module>.1-post-opt.ll (LLVM IR after optimization), "
           "<module>.2-isa.s (AMDGPU assembly), and "
           "<module>.3-binary.hsaco (linked HSACO blob). Pre-existing "
           "files are overwritten; the directory must already exist.">
  ];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::ROCDL::ROCDLDialect"
  ];
}

def HCLowerLaunchFuncToRuntime
    : Pass<"hc-lower-launch-func-to-runtime", "::mlir::ModuleOp"> {
  let summary = "Replace gpu.launch_func with hc_rt_* runtime calls and "
                "embed the HSACO blob as an LLVM global.";
  let description = [{
    Final lowering before LLJIT: each `gpu.launch_func` becomes a pair
    of calls into `libhc_hip_runtime.so` (`hc_rt_load_kernel` +
    `hc_rt_launch_kernel`) and the matching `gpu.binary`'s HSACO blob
    is materialised inline as an `llvm.mlir.global` so the JIT'd host
    wrapper carries everything it needs to launch without touching the
    filesystem.

    For each `gpu.launch_func` the pass:

      1. Looks up the sibling `gpu.binary` via `kernel_module_name`,
         pulls the first `#gpu.object` (we only emit one per binary
         today — single rocdl target), and reads its raw blob.
      2. Emits two per-callsite globals at module scope (internal
         linkage): a NUL-terminated string holding the HSACO blob
         (`<kernel>_data`) and a zero-init `!llvm.ptr` slot
         (`<kernel>_handle`) the runtime uses as a single-flight
         `hipFunction_t` cache. Names are uniqued through `SymbolTable`
         so multiple launches of the same kernel keep distinct slots
         (matches wave's recipe; the cache slot is per-callsite, not
         per-binary).
      3. Packs kernel operands as alloca-per-arg + an array-of-pointer
         (one `llvm.alloca` per arg storing its value, then an
         `LLVM::ArrayType` of `!llvm.ptr` built via `insertvalue` and
         spilled to a single alloca whose pointer is what the runtime
         reads). This is the HIP `kernelParams` convention; the
         `num_args` count is passed alongside the pointer for parity
         with wave's signature.
      4. Replaces the launch with `hc_rt_load_kernel(stream, &handle,
         data, size, name)` followed by `hc_rt_launch_kernel(stream,
         fn, smem, gx,gy,gz, bx,by,bz, cx,cy,cz, args, num_args)`.

    Stream: v0 always passes a null pointer (HIP's default stream).
    When the host wrapper grows an explicit stream argument (the
    upcoming Python invoke surface bead), this pass will switch to
    sourcing the stream from the enclosing function's first argument.

    All consumed `gpu.binary` ops are erased after the walk completes.
    Fails the pass if a launch's binary or selected object is missing,
    or if the binary carries something other than a single object.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

def HCEmitBenchWrapper
    : Pass<"hc-emit-bench-wrapper", "::mlir::ModuleOp"> {
  let summary = "Mint an i64-returning `wrapper_bench` sibling per host wrapper "
                "that calls hc_rt_launch_kernel_repeat for benchmarking.";
  let description = [{
    Runs after `-hc-lower-launch-func-to-runtime`, on the lowered LLVM
    IR. For each `llvm.func` that calls `hc_rt_launch_kernel` exactly
    once (the host wrapper shape `hc-lower-kernels-to-gpu-launch` /
    `hc-lower-launch-func-to-runtime` lays down), clones the function
    into a sibling `<name>_bench` with:

      * a trailing `i64 %n_inner` block argument appended to the
        original parameter list,
      * `i64` return type instead of `void`,
      * the `hc_rt_launch_kernel` callsite rewritten to
        `hc_rt_launch_kernel_repeat` with `%n_inner` threaded as the
        new trailing argument,
      * the function's `llvm.return` replaced with `llvm.return
        %elapsed : i64` where `%elapsed` is the repeat call's result.

    Per-callsite `<kernel>_data` / `<kernel>_handle` / `<kernel>` name
    globals are reused via symbol reference — the cloned `llvm.addressof`
    ops point at the same symbols as the original wrapper's, so the
    first `hc_rt_load_kernel` from either wrapper warms the shared
    `hipFunction_t` cache slot for both paths.

    Idempotent: functions whose symbol name already ends in `_bench`
    are skipped, so a second invocation of the pass is a no-op. Host
    wrappers that do not contain an `hc_rt_launch_kernel` call (the
    runtime decl itself, the helper unpackers, externs) are likewise
    skipped — only the launch-bearing wrappers get a bench sibling.

    Default off; `hc.compile(bench=True)` flips the `__HC_BENCH__`
    placeholder in `_GPU_LOWERING_PIPELINE` to splice this pass into
    the chain right before `symbol-dce`. With `bench=False` the
    placeholder is empty and the post-pipeline LLVM IR is
    byte-identical to today.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def HCLowerLaunchBody
    : Pass<"hc-lower-launch-body"> {
  let summary = "Lower HC launch-body operations to upstream dialects.";
  let description = [{
    Converts the HC subset that remains inside `gpu.launch` after kernel
    wrapping. Bound symbolic expressions become upstream SSA values derived
    from launch ids and buffer dimensions, simple arithmetic lowers to `arith`,
    `hc.for_range`/`hc.if` become `scf` control flow, static bare tensors
    materialize as `hc.alloc` + `hc.ptr_*` against `!hc.ptr<workgroup, T>`
    (the `hc-lower-to-llvm` slice owns the final memory lowering), and bare
    vectors lower to upstream `vector` values. Final masked stores lower to
    `hc.ptr_store_pred` (vector form) or guarded scalar stores. Kernel-arg
    buffers arrive as `(!hc.ptr<global, T>, dim*, stride*)` and lower to
    `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]` on the
    same `!hc.ptr` family. WMMA intrinsic boundaries stay on their existing
    `hc.call_intrinsic` path until the matching recipe runs.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::hc::HCDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def HCLowerToLLVM
    : Pass<"hc-lower-to-llvm"> {
  let summary = "Lower hc.ptr / hc.alloc / hc.ptr_* to LLVM dialect.";
  let description = [{
    Final lowering for the `!hc.ptr` family — see `doc/layouts.md`
    "hc.ptr and memory ops" for the contract. Operates on whatever scope
    it's invoked in (`builtin.module` for unit LITs, `gpu.module` inside
    the canonical AMDGPU pipeline) and rewrites the dialect surface
    launch-body and `hc-lower-generic` produce.

    Type conversion: `!hc.ptr<addr_space, T?>` → `!llvm.ptr<addr_space>`
    (LLVM has been opaque-pointers-only since LLVM 17; the carried element
    type is dropped here and recovered from the access op).

    Op rewrites:

      * `hc.alloc count = N : index -> !hc.ptr<workgroup, T>` —
        materializes a sibling `llvm.mlir.global private`
        `!llvm.array<N x T>` in addrspace 3 at the nearest enclosing
        symbol-table op, then plants `llvm.mlir.addressof` at the use
        site. `N` must be a constant integer (the doc invariant); a
        non-constant count diagnoses.
      * `hc.alloc count = N : index -> !hc.ptr<private, T>` —
        `llvm.alloca N x T` in addrspace 5 at the use site (runtime
        count is fine; the alloca's lifetime is per-call).
      * `hc.alloc count = N : index -> !hc.ptr<global, T>` — diagnoses;
        global-AS allocations come from the host (kernel arg) and have
        no in-kernel materialization.
      * `hc.ptr_offset %p, %i` — `llvm.getelementptr` with the carried
        element type as the GEP element. Index is `index`-cast to `i64`
        for the GEP index list.
      * `hc.ptr_load %p : ... -> T` — `llvm.load` of the access type
        (scalar or `vector<NxT>`); identical surface either way.
      * `hc.ptr_store %v, %p` — `llvm.store` of the access type.
      * `hc.ptr_load_pred` / `hc.ptr_store_pred` — vector form lowers
        to `llvm.intr.masked.load` / `llvm.intr.masked.store`. Scalar
        form lowers to `scf.if` + `llvm.load`/`llvm.store` + the
        passthrough as the else-yield (load) or no-op else (store).

    Element type comes from the pointer for typed `!hc.ptr<AS, T>` and
    from the access op's value type for opaque pointers — load/store
    always know the access type at their op.

    Out of scope (kept until follow-ups): kernel-arg `!hc.ptr<global>`
    boundary plumbing (gated on the host-wrapper migration), private-AS
    optimization, host-side stride binding fold-through.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::hc::HCDialect",
    "::mlir::scf::SCFDialect"
  ];
}

#endif // HC_TRANSFORMS_PASSES
