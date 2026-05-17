// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES
#define HC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCPromoteNames : Pass<"hc-promote-names"> {
  let summary = "Promote Python name-store ops to SSA values.";
  let description = [{
    Replaces `hc.assign` / `hc.name_load` with structured SSA, rebuilding
    region-carrying ops so each Python name becomes an SSA value, an
    iter_arg, a result, or a block arg. Pipeline contract is on
    `ConvertHCFrontToHC` in `include/hc/Conversion/HCFrontToHC/Passes.td`.

    Bottom-up over every function region tree
    (`hc.kernel` / `hc.func` / `hc.intrinsic`).
    `NameStoreRegionOpInterface` ops participate:

      - `hc.for_range`  : carried names → iter_args + yields. Leading
                          `hc.assign "<n>", %iv` (frontend's way of
                          publishing the IV under its Python name) is
                          recognised as a tautological self-bind, kept
                          loop-local, folded to direct block-arg uses.
      - `hc.if`         : carried names → op results + two-branch
                          yields; branches must agree on the write set.
      - `hc.{workitem,subgroup}_region` : nested Python scopes. Inner
                          reads lazily snapshot the outer binding;
                          inner writes shadow without leaking. Outbound
                          leak is opt-in via `hc.region_return <names>`
                          — the promoter adds one result per returned
                          name and pushes writebacks into the enclosing
                          name store.

    Post-condition: no `hc.assign` / `hc.name_load` under processed
    callables. Cross-kernel / cross-function capture is out of scope;
    each callable is its own promotion root.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInferTypes : Pass<"hc-infer-types"> {
  let summary = "Infer concrete HC value types from pre-inference `!hc.undef` IR.";
  let description = [{
    Sparse forward dataflow over `hc` IR; rewrites values whose type can
    refine without structural change. Covers scalar / index facts:
    constants, symbols, launch geometry, slice construction, buffer
    dimensions, index arith / comparisons, structured loop carriers.
    Monotonic: `!hc.undef` refines to concrete; conflicting concrete
    facts fail the pass instead of being reconciled by guess.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCSpecializeLiterals : Pass<"hc-specialize-literals"> {
  let summary = "Substitute literal-bound symbols with integer constants.";
  let description = [{
    Folds compile-time literal bindings into IR before shape-sensitive
    passes vote. Reads `literal_bindings` (`DictionaryAttr` of
    `name -> IntegerAttr<i64>`) on each `hc.kernel`, walks every
    `#hc.expr` / `#hc.pred` / `#hc.shape` / `#hc.layout` / `!hc.idx` /
    `!hc.pred` / shaped-type carrier inside the kernel, substitutes via
    ixsimpl, rebuilds the attr/type stack with `AttrTypeReplacer`.
    Consumed `literal_bindings` drops off so reruns are no-ops and
    `hc_ir_text` snapshots are self-contained.

    Bindings live on the kernel op (not on a pass option) so
    `compile().hc_ir_text` snapshots round-trip through
    `hc-opt -hc-specialize-literals`. Two slots:
    `literals = ["K"]` — which symbols *may* be bound;
    `literal_bindings = {K = 4 : i64}` — what they bind to here.

    Separate pass: once K is folded to a literal, every downstream pass
    sees concrete dims with no extra substitution plumbing. Single
    explicit specialization point in the schedule; dropping the pass
    keeps `front_ir` / `hc_ir` symbolic.

    Validation: every binding key must appear in `literals` when set;
    every value must be `IntegerAttr`. Both fail with op errors. Absent
    `literal_bindings` is the unspecialized case.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerPow : Pass<"hc-lower-pow"> {
  let summary = "Rewrite hc.pow with positive integer-literal rhs to a mul chain.";
  let description = [{
    `hc.pow` is the carrier for Python `**`. Pass:

      * Unfolds supported exponents into an `hc.mul` chain via binary
        squaring (`K = 2,3 → 1,2 mults`; `K = 4..7 → 2,3,3,4 mults`).
        Downstream passes understand mul chains.
      * Diagnoses unsupported shapes at one site. Today: anything
        non-positive-integer-literal — non-constant rhs (no runtime
        loop / `math.pow` yet), non-integer constant, zero (`x**0`
        needs a typed `1` we can't synthesize pre-inference), negative
        exponents (Python promotes to float, no reciprocal op).

    Runs after `hc-specialize-literals` (so literal-bound symbol
    exponents have folded to `hc.const<N>`) and before `hc-infer-types`
    (so the mul chain — not the carrier — gets refined).

    No-op when no `hc.pow` survives; safe to leave wired.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerMath : Pass<"hc-lower-math"> {
  let summary = "Rewrite hc.builtin_call to the matching upstream math op.";
  let description = [{
    Lowers `hc.builtin_call` by name:

      * `"numpy.sqrt"` -> `math.sqrt`
      * `"numpy.exp"`  -> `math.exp`

    v0: float scalar / vector / shaped operands compatible with `math`
    dialect. HC shaped types (`!hc.tensor`, `!hc.vector`) ride
    `hc-elementwise-to-generic` first so the carrier shows up scalar
    inside an `hc.generic` body before this pass fires.

    No-op when no `hc.builtin_call` survives. Unrecognised name
    diagnoses here rather than leaking to LLVM as an unknown op.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::math::MathDialect",
  ];
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
    Cleanup pass for `#hc.layout<...>`. Three rewrites, in order:

      1. Strip explicit identity layouts from shaped Values. Identity
         for n-dim shape iff shape_syms / index_syms have rank n,
         params empty, storage_size is product over shape_syms,
         offset is dot product of index_syms with tail-product
         strides (rightmost varies fastest), all ixsimpl-canonical.
         Matching values `setType` to the layout-less type — absent
         equals identity is the canonical form.

      2. ixsimpl-normalize layout payloads. Structurally a no-op
         (`LayoutAttr` normalizes at construction); kept as the
         documented place to reconstruct stale forms.

      3. Collapse `hc.as_layout(hc.as_layout(%v, L1), L2)` to
         `hc.as_layout(%v, L2)`. `as_layout` overwrites, never composes.

    Runs after `hc-infer-types`. Type-level fold propagates to
    BlockArguments; function-type attrs on `hc.func` / `hc.kernel` /
    `hc.intrinsic` rebuild to match.

    Routes through dialect-conversion driver to stay consistent with
    SCF / func / call structural conversions. Picks up `func` and
    `scf` dependencies for their populators. Identity materializations
    registered on the converter — layout-stripped and layout-bearing
    forms are observationally interchangeable.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def HCLowerStripLayout : Pass<"hc-lower-strip-layout"> {
  let summary = "Rewrite hc.strip_layout into hc.vzeros + hc.generic gather.";
  let description = [{
    Materialises ``hc.strip_layout`` (user-marked boundary that drops
    the operand's layout) into the body-driven ``hc.generic`` surface.
    Mirror of ``hc-load-store-to-generic`` and
    ``hc-shaped-compute-to-generic``: one shape-rewriting op funnels
    into ``hc.generic`` + identity init.

    ``%dst = hc.strip_layout %src : T_src -> T_dst`` becomes an
    ``hc.vzeros`` / ``hc.zeros`` sized to the result shape plus an
    ``hc.generic`` with parallel iters per result axis. Operand carries
    identity per-axis offsets (``[i_0, i_1, ...]``); flatten substitutes
    those into the operand's layout offset at ``index_syms[k]`` and
    yields the gather offset post-flatten. Result carries identity
    offsets composing to ``i_0 + i_1 * stride + ...`` against the
    dim-product layout. Body: trivial element copy.

    Layout-less operands collapse straight to the bare result without
    emitting the generic — flatten would compose identity over the
    operand's shape and the generic would be a no-op copy.

    Bails (op stays for downstream diagnostics):
      * Operand isn't an HC-shaped value (``!hc.undef``, raw pointer,
        non-shaped scalar) — verifier rejects at parse time.
      * Operand or result shape has non-``#hc.expr`` dim entries
        (``#hc.dyn`` sentinel, builder placeholder) — no symbolic
        bound for the iter range.
      * Operand has a layout but isn't a vector / tensor carrier
        (``!hc.buffer`` rejected by verifier).
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCShapedComputeToGeneric : Pass<"hc-shaped-compute-to-generic"> {
  let summary = "Rewrite hc.matmul / hc.reduce into hc.generic + identity fill.";
  let description = [{
    Folds source-level `hc.matmul` / `hc.reduce` into the body-driven
    `hc.generic` form. After this pass the post-flatten compute
    surface is one op; source shapes are frontend ergonomics shorthand.

    Iter bounds emit as empty-binding `hc.idx_apply` carrying
    `!hc.idx<dim>` sourced from operand shapes.

    `hc.matmul lhs, rhs : (T<M,K>, T<K,N>) -> T<M,N>` → `hc.zeros` fill
    on the result + `hc.generic` with two parallel iters (M, N) and one
    reduction iter (K). Inputs addressed `[i,k]` / `[k,j]`, output
    `[i,j]`. Body extends inputs to result elt type via `hc.astype`
    when needed, then `arith.mulf` + `arith.addf` (or `muli` + `addi`).
    v0: rank-2 only, identical input elt types.

    `hc.reduce val, kind = K, axis = a` → identity fill
    (`hc.zeros` for sum, `hc.full` with ±inf for min/max on floats)
    on the result + `hc.generic` with one reduction iter on `axis a`
    and parallel iters on the rest. Combinator picks: `arith.addf` /
    `arith.maximumf` / `arith.minimumf` for floats; integer is
    sum-only in v0 (signed vs unsigned min/max identity TBD).
    `keepdims = true` not yet handled.

    `!hc.undef` operands (pre-inference) and unmatched cases stay in
    place — downstream passes diagnose if needed.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::hc::HCDialect"
  ];
}

def HCLoadStoreToGeneric : Pass<"hc-load-store-to-generic"> {
  let summary = "Rewrite hc.load / hc.vload / hc.store into hc.generic.";
  let description = [{
    Funnels shaped memory ops into the body-driven `hc.generic` surface.
    Counterpart to `hc-shaped-compute-to-generic` for load/store. Both
    run pre-flatten on semantic shaped types so per-axis offsets line
    up with operand shapes before `hc-flatten-with-layouts` composes
    them through the layout.

    Rewrites:

    `hc.load %buf[%i_0, ..., %i_n], shape %shape -> T<...>` → `hc.zeros`
    fill of the result + `hc.generic` with one parallel iter per result
    axis. Input carries per-axis offsets `[i_0_expr + i_0, ...]`
    where `i_k_expr` is the access op's k-th index symbol. Init is the
    outs operand with identity offsets. Body forwards loaded element
    through `hc.yield`. `hc.vload` follows the same shape, vector result.

    `hc.store %dst[%i_0, ..., %i_n], %src` → `hc.generic` with `%src`
    as value-typed `ins` (identity offsets), `%dst` as ptr/buffer
    `outs` (per-axis offsets `[i_0_expr + i_0, ...]`). Body forwards
    source through `hc.yield`; `hc-lower-generic` materialises the
    implicit `hc.ptr_store`. No SSA result.

    Empty index list reduces the per-axis offset to identity over iter
    syms — same body, no addend.

    Iter bounds emit as empty-binding `hc.idx_apply` with `!hc.idx<dim>`
    sourced from the op's tile shape. Iter syms named `i_<k>` to stay
    distinct from the access op's index operand syms.

    Out of scope (op survives for downstream diagnostics):

      * Pre-inference `!hc.undef` operands — no shape to source bounds.
      * Index operands not `!hc.idx<expr>`-typed — no symbolic name to
        bind iter sym to; offset composition would lose information.
      * `hc.store` with `mask` operand — needs an `scf.if`-shaped body;
        deferred.
      * `hc.buffer_view`, `hc.vec` — sub-view / broadcast semantics in
        separate slices.

    `hc.load_mask` rewritten too: same parallel-iters shape, body is a
    single `hc.pred_apply` carrying per-axis bounds conjunction. Slice
    axes without pinned `!hc.idx` lower / step and source dims that
    aren't `#hc.expr` leaves fall through unchanged.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCDistributeWaveLayouts : Pass<"hc-distribute-wave-layouts"> {
  let summary = "Convert wave-cooperative layout-bearing carriers to per-lane.";
  let description = [{
    Bridges user-spelled wave-cooperative layouts (e.g. AMD gfx11 WMMA
    accumulator `vector<["32", "8"], <offset = 32*fi + lane,
    storage_size = 256, ...>>`) to the shape-suffix form the rest of
    the substrate understands.

    Recognition. For every shaped Value with `#hc.layout<...>` inside
    a workitem region the pass factors `offset` against the first
    index sym:

      `offset = index_syms[0] * <lane_stride> + <residue>`

    where the first shape dim equals `subgroup_size` and `<residue>`
    is independent of `index_syms[0]`. When the factorisation holds
    and `lane_stride * wave_size` divides `storage_size`, the type is
    *wave-distributable* along that first axis: each lane owns exactly
    `storage_size / wave_size` slots strided by `lane_stride`.

    Rewrite. Replaces every wave-distributable carrier with its
    per-lane peer (first shape dim dropped, lane sym substituted by
    `$WI0`, storage_size /= wave_size). Producers / consumers spanning
    the wave/lane boundary:

      * `hc.vzeros`/`hc.vones`/`hc.vfull`/`hc.zeros`/`hc.ones`/`hc.full`
        allocators carrying the wide type drop their first shape dim
        and layout attribute → per-lane bare (identity) or per-lane
        layout-bearing carrier.
      * `hc.generic` ops whose outs carry a wave-distributable type
        drop the lane-bound iter axis, substitute that iter sym with
        `$WI0` in every ins/outs offset, retype results in lockstep.
        Iter bounds for the dropped axis erased; ambient idx pinning
        left untouched (downstream resolves `$WI0` via workitem region).
      * `hc.buffer_view %carrier[%lane_id, ...]` collapses to a forward
        when the lane axis is the only one consumed by a scalar index.
      * `hc.strip_layout` whose operand is now layout-free becomes an
        identity forward.

    Boundary. Fires only inside `hc.workitem_region` bodies — outside
    the region a per-lane carrier has no scope. Escaping carriers get
    trailing-suffix treatment via `hc-normalize-scope-regions`; the
    two forms are not interchangeable. Lane sym from launch geo
    method's `LocalId` axis 0 (`$WI0`); wave size from kernel's
    `subgroup_size`.

    Bails (op stays for downstream diagnostics):
      * `offset` doesn't factor against `index_syms[0]` (mixed
        cross-axis, non-affine in lane sym, residue mentions lane).
      * `storage_size` not a const multiple of `wave_size`, or
        `lane_stride * wave_size > storage_size`.
      * First shape dim isn't a constant equal to `wave_size`.
      * Layout-bearing carrier lives outside any workitem region.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCElementwiseToGeneric : Pass<"hc-elementwise-to-generic"> {
  let summary = "Rewrite per-element shaped arith / cmp / astype ops into "
                "hc.generic.";
  let description = [{
    Funnels per-element shaped ops into the body-driven `hc.generic`
    form. Pure source-level rewrite — no memory effects, no vectorization.

    v0 covers all-shaped (no scalar broadcast):

      * binary arith / boolean: `hc.add`, `hc.sub`, `hc.mul`,
        `hc.div`, `hc.mod`, `hc.and`, `hc.or`,
      * unary arith / boolean: `hc.neg`, `hc.not`,
      * comparisons: `hc.cmp.{lt,le,gt,ge,eq,ne}`,
      * cast: `hc.astype`.

    Emits:

      * one parallel iter sym per result axis (`i_0`, `i_1`, ...) —
        distinct from operand source-level syms,
      * `iter_bounds` materialised directly from the result shape via
        empty-binding `hc.idx_apply` (no placeholder + inference step),
      * shaped inputs as `ins` with identity per-axis offsets,
      * synthesised `hc.zeros` / `hc.vzeros` init as single `outs` with
        identity offsets,
      * body mirroring the source op as scalar arith, terminated by
        `hc.yield`.

    Out of scope (op stays for downstream diagnostics):

      * Mixed scalar/shaped operands (broadcast) — needs splat / scalar
        capture pattern.
      * `hc.select` — false-value is a scalar inactive fill needing the
        same broadcast story.
      * Nullary fills (`hc.zeros`, `hc.ones`, `hc.full`, `hc.const`,
        `hc.empty`) — init scaffolding would fold back on itself.
      * Pre-inference `!hc.undef` operands.
      * Mismatched ranks or per-axis dim exprs.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInsertWorkgroupBarriers : Pass<"hc-insert-workgroup-barriers"> {
  let summary = "Insert gpu.barrier between hc.generic ops on shared "
                "workgroup-AS storage.";
  let description = [{
    Cross-generic sync on workgroup-AS storage (LDS). Runs after the
    first `hc-lower-launch-body` (workgroup tiles already rooted at
    `hc.alloc workgroup` + `hc.generic` on `!hc.ptr<workgroup, T>`)
    and before `hc-lower-generic` (per-axis offsets / iter syms still
    on the op for any future precise-elision pass).

    Conservative rule:

      1. For every `gpu.launch` body, walk regions block by block.
      2. Track per-block `pending` set of workgroup-AS storage roots
         written by an earlier `hc.generic` since the last barrier.
      3. At each `hc.generic`:
           - Collect storage roots `ins` / `outs` resolve to. Root is
             the SSA reached by walking back through single-input UCC
             and `hc.ptr_offset` chains.
           - If any read or write root intersects `pending`, insert
             `gpu.barrier` immediately before and clear `pending`.
             Read side catches writer→reader; write side catches
             writer→writer on the same slot family.
           - Add every workgroup-AS write root to `pending`.
      4. `gpu.barrier` in the scan clears `pending`; already-present
         barriers don't get a sibling.

    Stray non-generic loads / stores on workgroup-AS storage out of
    scope by design — contract is "workgroup-AS writes live inside
    `hc.generic`". A stray writer is an emitter bug.

    Nested regions (`scf.for`, `scf.if`) get a fresh per-block scan;
    `pending` doesn't thread across structural-control-flow. A write
    inside one `scf.if` branch doesn't seed a barrier in the sibling
    (can't, by construction); a write before `scf.for` barriers ahead
    of any read inside the loop. Precise affine elision and tighter
    cross-region tracking are follow-ups.

    No-op on payloads without `gpu.launch` or without `hc.generic`
    touching workgroup-AS; safe to leave wired.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::hc::HCDialect"
  ];
}

def HCLowerGeneric : Pass<"hc-lower-generic"> {
  let summary = "Lower hc.generic to scf.parallel / scf.for + hc.ptr_offset / "
                "hc.ptr_load / hc.ptr_store.";
  let description = [{
    Unroll-and-merge codegen frame for `hc.generic`. See
    `doc/layouts.md`. Rewrites every `hc.generic` as an outer
    `scf.parallel` over parallel iters + inner `scf.for` nest over
    reduction iters with `iter_args` carrying per-output accumulators.
    Pure-parallel collapses to one `scf.parallel`; pure-reduction
    collapses to one `scf.for`.

    Picks `(axis-order, partition)` via divisibility-pruned merge-score
    search: every factor `p_a` must provably divide its bound (probed
    via `ixs_check(cmp(ixs_mod(N_a, p_a), EQ, 0))`); survivors scored
    by per-operand pairwise contig probe
    (`ixs_check(cmp(off[k+1] - off[k], EQ, 1))`). Body stays elementwise
    scalar at every total unroll: scalar contig groups lower to scalar
    `hc.ptr_load` / `hc.ptr_store`; groups size > 1 lower to vector
    `hc.ptr_load` / `hc.ptr_store` of width `G` plumbed through
    `vector.extract` / `vector.from_elements` at the body boundary.
    `(1, ..., 1)` partition collapses to scalar baseline.

    Per-iteration body emission (per partition lane):

      * Each `ins`: evaluate composed offset to `index` Value, emit
        `hc.ptr_offset` + `hc.ptr_load`. Loaded value replaces the
        body block argument.
      * Each `outs`: load initial accumulator once per parallel-iter
        via `hc.ptr_offset` + `hc.ptr_load` at the parallel-only
        offset, thread through inner `scf.for`'s `iter_args`, emit
        `hc.ptr_store` after the reduction nest with the final value.
      * Body region cloned in place, block args remapped to loaded
        values (ins) and running carries (outs). Cloned `hc.yield`
        provides per-iter accumulator update.

    Per-operand offsets surface as `hc.idx_apply` carrying composed
    `ExprAttr` (per-lane delta substituted into each iter sym),
    binding listed iter syms to loop induction vars. Launch-body
    lowering walks the expression and substitutes operands plus
    ambient kernel-bound syms (shape dims, stride params).

    Pre-flight bail (op stays for downstream diagnostics):

      * Operand not `!hc.ptr<...>` — value-typed bare carriers
        (`!hc.bare_tensor`, `!hc.bare_vector`), `!hc.buffer`, and
        `!hc.undef` out of v0 scope.
      * Operand offset rank > 1 — multi-axis needs linearization
        through operand shape (needs outer-scope shape-symbol
        bindings). Handled when `hc-flatten-with-layouts` lands and
        every operand is rank-1.
      * Iter bound still a `hc.undef_value` placeholder.
      * Offset references a free symbol the outer-scope walker can't
        resolve.
      * `scf.if` in body — residual case (rare once masked load/store
        routes through `hc.ptr_load_pred` / `hc.ptr_store_pred`).
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
    Resolves `hc.predicate %v mask %m passthrough %f` by inspecting the
    def of `$value`:

      * Trivial masks fold first — `m_One()` → `$value` (drop the
        predicate), `m_Zero()` → `$passthrough` (load elided).

      * `hc.ptr_load` / `hc.vload` / `hc.load`: clone the producer at
        its def site in predicated form (`hc.ptr_load_pred` etc.) with
        `$mask` / `$passthrough`. Predicate op's uses redirect to the
        new predicated load. Each `hc.predicate` use of the same load
        gets its own clone — sharing is opt-in.

      * `vector.extract`: emit `arith.select %mask, %value, %passthrough`
        at the predicate site. Underlying vector load stays
        unconditional; predicate gates the lane only.

      * Anything else: hard error. Allow-list is deliberate.

    Dominance: `$mask` and `$passthrough` must dominate the producer's
    def site when cloning / rewriting. Violations diagnose and fail
    the pass.

    Lowering counterpart of `hc.yield_predicated`'s store-side: both
    edges of an `hc.generic` body decompose into predicated mem ops or
    `arith.select` blends.
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
    Post-flatten *type* invariant: every `SymbolicallyShapedTypeInterface`
    value (except buffers) is 1D and carries no `#hc.layout`. Op
    surfaces stay untouched — `hc.generic` keeps per-axis offset
    arrays at original logical rank; `hc.load`, `hc.store`, `hc.vload`,
    `hc.buffer_view` keep multi-index lists. Whoever turns nD logical
    accesses into 1D addresses owns the layout→offset materialization
    in a follow-up slice.

      1. `TypeConverter` dispatches on the shaped-type interface:
         computes `storage_size_expr` (`layout.storage_size` after
         binding `layout.shape_syms` to original shape entries, or the
         product of original dims when no layout is attached); rebuilds
         the type with single-entry shape and null layout. Tuple and
         function types recurse.

      2. Upstream populators thread the converter through `func.func` /
         `func.return` / `func.call` signatures and SCF structural ops
         (`scf.for`, `scf.if`, `scf.while`, `scf.index_switch`).

      3. `hc.as_layout` collapses to its operand once both endpoints
         carry the converted form — observationally identical.

    Buffer types deliberately exempt: default strided layout's
    `storage_size` is the literal `0` placeholder the frontend emits
    (host owns allocation), so 1D-collapsing would fold every buffer
    shape to `[0]`. Buffers keep their nD shape, lose only the layout
    slot; buffer-side ABI is a separate slice.

    Cooperates with `hc-canonicalize-layouts`: that folds explicit
    *identity* layouts back to absent; this collapses every shape
    (with or without explicit layout) to 1D storage form. Not yet in
    default `hc.compile` schedule — downstream still expects nD
    operand types and a layout-aware access path.
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
    Rewrites every semantic `!hc.tensor` / `!hc.vector` producer and
    user into compiler-internal `!hc.bare_tensor` / `!hc.bare_vector`
    data + matching bare `!hc.pred` masks. Contract boundary between
    the semantic shaped surface and downstream lowering: no
    `!hc.tensor` / `!hc.vector` may survive into
    `hc-shaped-compute-to-generic`, `hc-flatten-with-layouts`,
    `hc-lower-launch-body`, or anything below this pass. Survivors
    diagnose loudly — first full-conversion failure on whichever op
    carries the type, then a post-conversion walk fails the pass.

    Ordering: scheduled after `hc-infer-types`,
    `hc-materialize-bound-exprs`, and `hc-verify-static-shapes`.
    Reads concrete shaped types (mask-type derivation needs matching
    `!hc.pred` carriers); assumes bound symbolic exprs already severed
    from producers (sees post-inference shape surface only).
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCMaterializeBoundExprs
    : Pass<"hc-materialize-bound-exprs"> {
  let summary = "Sever bound symbolic HC expressions from their producer chains.";
  let description = [{
    Replaces uses of pinned `!hc.idx<expr>` / `!hc.pred<pred>` with an
    empty-binding `hc.idx_apply` / `hc.pred_apply` (no listed syms, no
    SSA edges) when every leaf is a bound symbol or the expr is
    constant-only. Original producer left for canon/DCE. Breaks
    dependence on launch-context query chains before scope
    normalization. Severed apply ops resolve pinned exprs ambiently
    against the launch context at the launch-body lowering boundary.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInlineHelpers
    : Pass<"hc-inline-helpers"> {
  let summary = "Inline supported HC helper calls.";
  let description = [{
    Inlines `hc.call` sites targeting `hc.func` helpers. Owns HC
    helper call-graph concerns separately from scope-region
    normalization: recursion and unsupported callee body shapes
    diagnose here. Inlined helpers are erased once no `hc.call`
    references remain.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCNormalizeScopeRegions
    : Pass<"hc-normalize-scope-regions"> {
  let summary = "Flatten supported collective scope regions into SPMD HC IR.";
  let description = [{
    Normalizes HC toward per-workitem SPMD before upstream GPU
    lowering. v0 flattens resultless `hc.workitem_region` bodies after
    bound expression materialization removes live uses of the workitem
    token. Result-producing workitem regions and subgroup regions
    diagnose until distributed value projection and subgroup-local
    workitem semantics land.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerKernelsToGPULaunch
    : Pass<"hc-lower-kernels-to-gpu-launch"> {
  let summary = "Wrap HC kernels in upstream GPU launch operations.";
  let description = [{
    Rewrites `hc.kernel` symbols into host `func.func` entry points
    that launch the former kernel body with `gpu.launch`. Host wrapper
    signature: leading `!llvm.ptr` stream + one `!llvm.ptr`
    (`PyObject *`) per non-`!hc.group` kernel arg. Body calls
    `_mlir_ciface_hc_get_*` helpers (declared as `func.func private`
    with `llvm.emit_c_interface`) to unpack each PyObject into raw
    pointer / dim / stride / scalar before computing geometry. Each
    buffer arg materialises as `(!hc.ptr<global, T>, dim*, stride*)`
    tuple from `hc_get_ptr` + per-axis `hc_get_dim` / `hc_get_stride`;
    an N→1 `unrealized_conversion_cast` inside `gpu.launch` bridges
    the tuple back to `!hc.buffer<...>` so `gpu-kernel-outlining`
    captures the raw pointer / dim / stride. Stream pointer threads
    to runtime launch calls in `hc-lower-launch-func-to-runtime`;
    null = HIP default stream.

    Original HC kernel body stays inside the launch behind explicit
    cast boundaries for later memory / intrinsic lowering.
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
    Looks up sibling top-level
    `builtin.module @__hc_intrinsic_lowerings__` (carrying
    `transform.with_named_sequence`) and applies every
    `transform.named_sequence` whose `hc.target` matches the requested
    target. Recipes are planted by the frontend emitter and stay in
    MLIR transform IR end-to-end — no string round-trip.

    Recipe shape: `transform.hc.match_intrinsic_call` to find the call
    site, optional `transform.hc.get_intrinsic_*` handle ops to thread
    operands / result types / attrs into `transform.hc.create_op`,
    then `transform.hc.replace_intrinsic_call` to swap the original
    `hc.call_intrinsic` for the constructed payload.

    After applying selected sequences the lowerings module is erased
    and the payload is walked once more. Any surviving
    `hc.call_intrinsic` is a hard error — user asked for target `T`
    but no recipe knew how to lower it.
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
    For every `gpu.module` carrying exactly one `#rocdl.target<...>`:

      1. translate to LLVM IR (`translateModuleToLLVMIR`),
      2. build `llvm::TargetMachine` from rocdl triple/chip/features;
         pin IR's data layout + triple,
      3. optimize-then-codegen at the target's opt level
         (`makeOptimizingTransformer`),
      4. LLVM IR → ISA text
         (`LLVM::ModuleToObject::translateModuleToISA`),
      5. ISA → ELF object via `ROCDL::assembleIsa`,
      6. ELF → HSACO via `ROCDL::linkObjectCode`, calling our pinned
         `ld.lld`.

    Result attached as `gpu::ObjectAttr` on a sibling `gpu.binary` op
    (CompilationTarget::Binary, original rocdl target preserved);
    source `gpu.module` erased.

    `lld-path` resolution order:

      1. option value if non-empty,
      2. `HC_LLD` env var (escape hatch for direct `hc-opt`),
      3. first `ld.lld` on `$PATH`.

    `hc.compile` always populates option 1 from bundled
    `hc/_native/bin/ld.lld`; `HC_LLD` and `$PATH` only matter for
    direct `hc-opt`. No ROCm install fallback — error rather than
    silently picking up a distro-installed lld.

    v0 doesn't link rocm-device-libs bitcode. WMMA example doesn't
    use `__ocml_*` / `__ockl_*`; kernels that do need a follow-up
    pass that resolves device-libs and feeds bitcode into the
    optimization pipeline.
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
    Final lowering before LLJIT. Each `gpu.launch_func` becomes a pair
    of calls into `libhc_hip_runtime.so` (`hc_rt_load_kernel` +
    `hc_rt_launch_kernel`); matching `gpu.binary`'s HSACO blob
    materialises inline as `llvm.mlir.global` so the JIT'd host wrapper
    carries everything to launch without touching the filesystem.

    Per `gpu.launch_func`:

      1. Look up sibling `gpu.binary` via `kernel_module_name`, pull
         the first `#gpu.object` (one per binary today — single rocdl
         target), read its raw blob.
      2. Emit two per-callsite globals at module scope (internal
         linkage): NUL-terminated string `<kernel>_data` holding the
         HSACO blob; zero-init `!llvm.ptr` slot `<kernel>_handle` used
         as single-flight `hipFunction_t` cache. Names uniqued via
         `SymbolTable` so multiple launches of the same kernel keep
         distinct slots (cache slot is per-callsite, not per-binary).
      3. Pack kernel operands as alloca-per-arg + array-of-pointer
         (one `llvm.alloca` per arg storing its value; an
         `LLVM::ArrayType` of `!llvm.ptr` built via `insertvalue` and
         spilled to a single alloca whose pointer the runtime reads).
         HIP `kernelParams` convention; `num_args` passed alongside.
      4. Replace launch with `hc_rt_load_kernel(stream, &handle, data,
         size, name)` + `hc_rt_launch_kernel(stream, fn, smem,
         gx,gy,gz, bx,by,bz, cx,cy,cz, args, num_args)`.

    Stream: v0 always passes null (HIP default). When the host wrapper
    grows an explicit stream arg, this pass switches to sourcing it
    from the enclosing function's first argument.

    Consumed `gpu.binary` ops erased after the walk. Fails if a
    launch's binary or selected object is missing, or if the binary
    carries more than one object.
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
    Runs after `-hc-lower-launch-func-to-runtime`, on lowered LLVM IR.
    For each `llvm.func` that calls `hc_rt_launch_kernel` exactly once
    (the host-wrapper shape), clones into a sibling `<name>_bench`:

      * trailing `i64 %n_inner` block arg appended,
      * `i64` return type (was `void`),
      * `hc_rt_launch_kernel` callsite rewritten to
        `hc_rt_launch_kernel_repeat` with `%n_inner` as new trailing arg,
      * `llvm.return` replaced with `llvm.return %elapsed : i64` where
        `%elapsed` is the repeat call's result.

    Per-callsite `<kernel>_data` / `<kernel>_handle` / `<kernel>` name
    globals reused via symbol reference — cloned `llvm.addressof` ops
    point at the same symbols, so first `hc_rt_load_kernel` from either
    wrapper warms the shared `hipFunction_t` cache slot for both.

    Idempotent: functions ending in `_bench` skipped. Wrappers without
    an `hc_rt_launch_kernel` call (runtime decl, unpackers, externs)
    skipped too — only launch-bearing wrappers get a bench sibling.

    Default off; `hc.compile(bench=True)` flips `__HC_BENCH__` in
    `_GPU_LOWERING_PIPELINE` to splice this pass before `symbol-dce`.
    `bench=False` leaves post-pipeline LLVM IR byte-identical.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def HCLowerLaunchBody
    : Pass<"hc-lower-launch-body"> {
  let summary = "Lower HC launch-body operations to upstream dialects.";
  let description = [{
    Converts the HC subset that survives inside `gpu.launch` after
    kernel wrapping. Bound symbolic exprs become upstream SSA from
    launch ids and buffer dims; simple arithmetic to `arith`;
    `hc.for_range` / `hc.if` to `scf`; static bare tensors to
    `hc.alloc` + `hc.ptr_*` on `!hc.ptr<workgroup, T>`
    (`hc-lower-to-llvm` owns final memory lowering); bare vectors to
    upstream `vector`. Masked stores → `hc.ptr_store_pred` (vector) or
    guarded scalar stores. Kernel-arg buffers arrive as
    `(!hc.ptr<global, T>, dim*, stride*)` and lower to
    `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`.
    WMMA intrinsic boundaries stay on `hc.call_intrinsic` until the
    matching recipe runs.
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
    "hc.ptr and memory ops". Operates on whatever scope it's invoked
    in (`builtin.module` for unit LITs, `gpu.module` in the AMDGPU
    pipeline); rewrites the surface that launch-body and
    `hc-lower-generic` produce.

    Type conversion: `!hc.ptr<addr_space, T?>` → `!llvm.ptr<addr_space>`
    (LLVM has been opaque-pointers-only since 17; carried elt type
    dropped here, recovered from the access op).

    Op rewrites:

      * `hc.alloc count = N : index -> !hc.ptr<workgroup, T>` — sibling
        `llvm.mlir.global private !llvm.array<N x T>` in addrspace 3
        at the nearest enclosing symbol-table op, plus
        `llvm.mlir.addressof` at the use site. `N` must be a constant;
        non-constant diagnoses.
      * `hc.alloc count = N : index -> !hc.ptr<private, T>` —
        `llvm.alloca N x T` in addrspace 5 at the use site (runtime
        count fine; per-call lifetime).
      * `hc.alloc count = N : index -> !hc.ptr<global, T>` — diagnoses
        (global-AS allocations come from host kernel arg, no in-kernel
        materialization).
      * `hc.ptr_offset %p, %i` — `llvm.getelementptr` with carried elt
        type. Index `index`-cast to `i64`.
      * `hc.ptr_load %p : ... -> T` — `llvm.load` of the access type
        (scalar or `vector<NxT>`).
      * `hc.ptr_store %v, %p` — `llvm.store` of the access type.
      * `hc.ptr_load_pred` / `hc.ptr_store_pred` — vector form lowers
        to `llvm.intr.masked.load` / `llvm.intr.masked.store`. Scalar
        form: `scf.if` + `llvm.load` / `llvm.store` + passthrough as
        else-yield (load) or no-op else (store).

    Elt type from the pointer for typed `!hc.ptr<AS, T>`; from the
    access op's value type for opaque pointers.

    Out of scope (follow-ups): kernel-arg `!hc.ptr<global>` boundary
    plumbing (gated on host-wrapper migration), private-AS
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
