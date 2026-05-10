# Layouts

This document is the design contract for layouts in the `hc` pipeline. It
covers how `index_map(...)` from `doc/langref.md` is represented in MLIR,
how layout-bearing types flow through the existing passes, and how the
post-decomposition lowering reaches LLVM without going through memref or
affine maps.

The two non-negotiables that fall out of the rest of the project:

* memref is not in the long-term picture — every memory carrier ends as
  a typed pointer plus a 1D extent, not a strided shaped buffer;
* affine maps are not in the long-term picture — every index expression
  is an `#hc.expr` and reasoning about it goes through ixsimpl, not the
  affine analysis.

## Skeleton

Four legs:

1. **Layout attr** — `#hc.layout<...>` built on `#hc.expr`. Attaches to
   `!hc.buffer` / `!hc.tensor` / `!hc.vector` / `!hc.bare_tensor` /
   `!hc.bare_vector` as an optional type parameter. Absent on tensors
   and vectors means identity (dense row-major); on buffers it is set
   to the host-supplied strided layout at frontend time.
2. **Flatten** — a normalization pass folds every nD+layout shaped value
   into a 1D bare value with no layout, materializing the offset
   expression at each access site.
3. **Memory carrier** — bare 1D tensor lowers to `(!hc.ptr, count)` and
   then to `!llvm.ptr` (AMDGPU) or a typed SPIR-V pointer. Bare 1D
   vector lowers to upstream `vector<NxT>`.
4. **Generic compute op** — `hc.generic` plays the `linalg.generic`
   role but indexes operands with `#hc.expr` instead of `affine_map`,
   and supports parallel + reduction iter kinds. Lowering uses the
   symbolic engine to decide vectorization.

## `#hc.layout<...>` attribute

A layout is three symbolic expressions sharing a small bound-name table:

```mlir
#hc.layout<
  shape_syms   = ["W", "H"],
  index_syms   = ["i", "j"],
  params       = ["row_stride" = #hc.expr<"H + 4">],
  storage_size = #hc.expr<"W * row_stride">,
  offset       = #hc.expr<"i * row_stride + j">
>
```

Bound-name conventions:

* `shape_syms` are bound positionally to the shape entries of the
  attached type. A layout attached to `!hc.tensor<f16, ["M", "N"], L>`
  substitutes `W := M`, `H := N` everywhere in `L`.
* `index_syms` are free until an access op binds them to its actual
  index operands. `hc.load %t[%i0, %i1]` pins `i := %i0`, `j := %i1`.
* `params` are derived expressions over `shape_syms`, given names so
  `storage_size` and `offset` can share intermediates without repetition.
  Semantically equivalent to inlining; whether the attribute keeps the
  shared form or pre-substitutes is an implementation detail of the
  parser/builder.

Equality: ixsimpl normalizes each `#hc.expr` payload at construction.
Two layouts that compute the same offset under different spellings hash
to the same canonical form, so MLIR type uniquing gives free equality.

Validity: `offset` injectivity over `[0, shape) ∩ ℤⁿ` and
`storage_size ≥ max(offset) + 1` are symbolic-engine queries. Whatever
ixsimpl can prove gets accepted; whatever it can't gets rejected with a
located `unverifiable_layout` diagnostic. **No restriction on the
expression grammar in v1**: we'll find the practical edge by feeding it
real layouts (block-cyclic, swizzle, padded, broadcast) and see what
ixsimpl chokes on. Documenting the supported subset comes after, not
before.

The legacy `HC_LayoutAttr` enum (`row_major | col_major`) is folded into
this representation. The Python-side `index_map` builder constructs the
canonical row-major / col-major instances when no `params` / `offset` is
supplied; the enum stops being a first-class IR concept.

## Type slot

The five shaped types grow an optional layout parameter:

```mlir
!hc.buffer<f16, ["M", "K"], #hc.layout<...>>      // host-supplied
!hc.tensor<f16, ["M", "N"]>                       // identity
!hc.tensor<f16, ["M", "N"], #hc.layout<...>>      // explicit
!hc.bare_tensor<f16, ["M", "N"], #hc.layout<...>>
!hc.vector<f32, ["8"], #hc.layout<...>>           // statically resolvable only
!hc.bare_vector<f16, ["16"], #hc.layout<...>>
```

`SymbolicallyShapedTypeInterface` extends with a `getLayout()` query.
For `!hc.tensor` / `!hc.vector` / their bare counterparts, `absent`
means identity row-major; the canonicalization pass folds an explicit
identity layout back to absent so type uniquing stays tight
(`hc-canonicalize-layouts`). For `!hc.buffer` the absent-as-identity
shortcut does **not** apply — see below.

Inference (`hc-infer-types`) propagates layouts:

* elementwise: result inherits the common operand layout if all match.
  Mixed default + explicit, or two distinct explicits, fails type
  inference with the `as_layout` requirement diagnostic from
  `doc/langref.md`.
* shape-changing (reshape, broadcast, reduction): drops to default.
* `hc.matmul`: default unless both operands carry the same layout *and*
  the result shape matches both of theirs (rare).
* `hc.as_layout`: replaces the operand layout with the operand of its
  attribute, no other change.

The collective return suffix from `@group.subgroups` / `@group.workitems`
is **not** part of `#hc.layout`. The langref rule "dense suffix appended
in canonical order" stays outside the user-visible descriptor, and
flatten treats the suffix as a trailing trivially-dense block.

## Buffer layouts and the host ABI

Buffer arguments carry a layout the same way tensors and vectors do.
v1 defaults every buffer argument to a fully strided np/torch-style
layout, with per-axis strides extracted from the runtime tensor at
launch:

```mlir
!hc.buffer<f16, ["M", "K"], #hc.layout<
  shape_syms = ["M", "K"],
  index_syms = ["i", "j"],
  params     = ["s0" = #hc.expr<"$STRIDE_0_a">,
                "s1" = #hc.expr<"$STRIDE_1_a">],
  offset     = #hc.expr<"i * s0 + j * s1">
>>
```

Per-axis stride symbols (`$STRIDE_0_a`, `$STRIDE_1_a`, ...) join `M`,
`K`, `N` in the kernel's `bound_symbols` list and the host wrapper's
ABI. The wrapper extracts them from `np.ndarray.strides` (in elements,
not bytes — divided by `itemsize` at the wrapper) or
`torch.Tensor.stride()` and binds them at launch alongside the existing
shape symbols. Contiguous inputs end up with the same expressions a
default row-major layout would produce; non-contiguous slices (a
column from a 2D tensor, a transposed view) work without a host-side
copy.

The buffer-attached layout differs from tensor/vector layouts in three
ways:

* The absent-as-identity shortcut doesn't apply — the absence of an
  explicit layout on a buffer means "the host hasn't been queried yet",
  not "dense row-major". The frontend always emits a strided layout for
  every buffer argument; the canonicalization pass leaves them alone.
* `storage_size` is informational at most. The host owns the
  allocation and the verifier doesn't require
  `storage_size ≥ max(offset) + 1` for buffer layouts.
* Buffer layouts can name launch-determined symbols (the stride
  symbols) that no other layout can. The langref already permits
  launch-determined values in tensor layouts; the buffer surface is
  the case where this matters in practice.

User-supplied buffer layouts (block-cyclic, swizzle, padded global)
are a follow-up. The IR machinery is the same; only the host wrapper
and the launch ABI need to learn how to consume a non-default layout
descriptor on a kernel argument. The default strided layout is the
natural starting point because every np/torch input already produces
one.

## Flatten pass — `hc-flatten-with-layouts`

Position: after `hc-decompose-shaped-values`, before any pointer
lowering.

The pass is **type-only**: every shaped value collapses to its 1D
form and loses its layout slot, but op surfaces stay untouched.
Per-axis offset arrays on `hc.generic` keep their original logical
rank, multi-index lists on `hc.load` / `hc.store` / `hc.vload` /
`hc.buffer_view` stay as-is. Composing the layout offset into those
access expressions is a separate slice — downstream fusion /
vectorization wants the per-axis structure available, and the
access expression materialization needs `hc.ptr` plumbing the
type-only slice doesn't own.

What runs:

1. The type changes from `<T, [d_0, ..., d_{n-1}], #hc.layout<L>>` to
   `<T, [storage_size_expr]>` (1D, no layout). `storage_size_expr` is
   the layout's `storage_size` after binding `shape_syms` to the
   original shape entries — or the dimension product when the type
   sits on the implicit identity-row-major contract (no explicit
   layout).
2. **Buffers** collapse to `<T, [?]>` — the `?` is the surface
   spelling for the `#hc.dyn` sentinel. Buffer storage extent is
   owned by the host descriptor (the verifier doesn't enforce
   `storage_size >= max(offset) + 1` on buffers, the default
   strided layout emits `storage_size = 0` as an informational
   placeholder); a `#hc.dyn` shape entry says that out loud.
   Consumers that need a concrete extent reach for the host
   descriptor instead of the in-IR symbol set.
3. The shape collapse and layout-slot strip propagate through every
   op via a `MatchAnyOpTypeTag` rebuild pattern scoped to the `hc`
   dialect. Body block args of `hc.generic` carry scalar element
   types and need no signature conversion. Function signatures /
   `func.return` / `func.call` / SCF structural ops route through
   the upstream populators.
4. `hc.as_layout` drops unconditionally: both endpoints route
   through the converter and the relabel becomes cosmetic.

Deferred slices (out of scope here):

* **Per-access offset materialization.** Every access op
  (`hc.load`, `hc.store`, `hc.vload`, `hc.buffer_view`, `hc.vec`)
  gets its multi-index rewritten into a single 1D offset. v0
  carries the multi-index forward as-is; lowering owns the
  composition.
* **`hc.as_layout` structural difference.** When source and
  destination layouts disagree under ixsimpl equality, the op
  becomes an `hc.generic` copy between the two offset expressions.
  v0 always drops the op (correctness is preserved only when the
  layouts agree on the underlying storage-size expression — strict
  layout mismatches need the materialization slice).
* **`hc.generic` operand offset composition.** Per-operand per-axis
  `#hc.expr` arrays stay nD post-flatten; composing the layout
  offset to produce a single 1D expression per operand is the
  "compose layout into generic" slice.
* **`i1` byte-per-element retirement.** The hand-coded `i1` mask
  handling in `HCLowerLaunchBodyPass.cpp` needs flatten to emit a
  byte-per-element layout for `i1` shaped types as a default. Until
  then the manual extract/insert sequences stay.

Post-flatten *type* invariant: no layout attribute survives on any
shaped type, every shaped value is 1D (buffers spell the unknown
extent as `#hc.dyn`). Op-level structural invariants (multi-index
access, per-axis offsets) carry through unchanged for the follow-up
slice to consume.

## `hc.ptr` and memory ops

Types and ops added to the `hc` dialect (no second dialect — keeping
the surface small):

```mlir
!hc.ptr<workgroup | global | private>           // opaque
!hc.ptr<workgroup, f16>                         // typed (SPIR-V)
```

Surface mirrors `!llvm.ptr<3>`: required address space first as a bare
keyword, optional element type after the comma. Element type is
optional — AMDGPU/LLVM lowering uses the opaque form, SPIR-V lowering
keeps the typed form. The element type is dropped at the lowering pass
that emits `llvm.ptr` (LLVM has been opaque-pointers since 17). Any
pass that needs the element type before that point gets it from the
`hc.ptr_load` / `hc.ptr_store` op, which always carries it.

```mlir
hc.alloc count = %n : index -> !hc.ptr<workgroup, f16>
hc.ptr_offset %p, %i : (!hc.ptr<workgroup, f16>, index) -> !hc.ptr<workgroup, f16>
%v = hc.ptr_load  %p          : !hc.ptr<workgroup, f16> -> f16
     hc.ptr_store %v, %p      : f16, !hc.ptr<workgroup, f16>
```

`hc.alloc` count is a single `index` SSA value. **In v1 the count is
required to be statically known after specialization** — the verifier
rejects allocations whose `count` is not a constant once the
literal-binding pass has run. Dynamic LDS sizes are a follow-up; the
shipping AMDGPU LDS path is static-LDS-only and we don't need a runtime
launch-time hook to make the example go.

Address space lowering:

* `workgroup` → `llvm.mlir.global private @<name>() {addr_space = 3}` +
  `llvm.mlir.addressof` at the use site;
* `private` → `llvm.alloca` in addrspace 5;
* `global` → externally-supplied pointer (kernel argument).

The 1D bare tensor → memory carrier transition replaces the entire
`memref<...xT, #gpu.address_space<workgroup>>` family currently used by
`HCLowerLaunchBodyPass.cpp`. The cooperative copy helper
(`emitCooperativeCopy`) is rewritten on top of the new ptr ops; the
end-of-pipeline lowers to `llvm.ptr` directly without a memref
intermediate.

The 1D bare vector → upstream `vector<NxT>` mapping is unchanged; `N`
must be statically resolved by this point (already an inference
postcondition).

## `hc.generic`

`hc.generic` is the **universal compute surface** in HC: every
shaped op (elementwise arith, fills, contractions, reductions,
loads, stores) eventually lowers into a single `hc.generic` body.
The op subsumes both `linalg.generic`'s tensor and memref regimes
on the same op, with one twist — `outs` may **freely mix** value
and ptr/buffer entries in the same op, so a fused
"compute-and-spill" lands as one loop nest instead of two.

Two forms share the surface:

```mlir
// value form: outs are shaped values, op produces SSA results.
%out = hc.generic
    iter_syms   = ["i", "j", "k"]
    iter_bounds = [%m, %n, %kn]
    iter_kinds  = ["parallel", "parallel", "reduction"]
    ins  (%a : !hc.bare_tensor<f16, [%m, %kn]>
              at [#hc.expr<"i">, #hc.expr<"k">],
          %b : !hc.bare_tensor<f16, [%kn, %n]>
              at [#hc.expr<"k">, #hc.expr<"j">])
    outs (%c : !hc.bare_tensor<f32, [%m, %n]>
              at [#hc.expr<"i">, #hc.expr<"j">])
    -> (!hc.bare_tensor<f32, [%m, %n]>) {
  ^bb0(%av: f16, %bv: f16, %cv: f32):
    %p  = arith.mulf %av, %bv : f16
    %pe = arith.extf %p : f16 to f32
    %s  = arith.addf %cv, %pe : f32
    hc.yield %s : f32
}

// in-place form: outs are ptr/buffer, op produces no SSA result and
// carries MemWrite. Body-arg for the destination sources via an
// implicit hc.ptr_load at the iteration's offset (same outs-as-init
// semantics as the value form), the yield routes through an implicit
// hc.ptr_store.
hc.generic
    iter_syms   = ["i"]
    iter_bounds = [%n]
    iter_kinds  = ["parallel"]
    ins  (%src : !hc.bare_tensor<f32, [%n]> at [#hc.expr<"i">])
    outs (%dst : !hc.ptr<global, f32>       at [#hc.expr<"i">]) {
  ^bb0(%sv: f32, %dv: f32):
    hc.yield %sv : f32
}

// mixed: produce a result tensor *and* spill a debug trace into LDS.
%out = hc.generic
    iter_syms   = ["i"]
    iter_bounds = [%n]
    iter_kinds  = ["parallel"]
    ins  (%a : !hc.bare_tensor<f32, [%n]> at [#hc.expr<"i">])
    outs (%c : !hc.bare_tensor<f32, [%n]> at [#hc.expr<"i">],
          %trace : !hc.ptr<workgroup, f32> at [#hc.expr<"i">])
    -> (!hc.bare_tensor<f32, [%n]>) {
  ^bb0(%av: f32, %cv: f32, %tv: f32):
    %r = arith.mulf %av, %av : f32
    hc.yield %r, %r : f32, f32
}
```

Same shape as `linalg.generic` with the affine-map slot replaced by a
per-axis `#hc.expr` array over `iter_syms ∪ shape_syms ∪ params`. The
op is logically nD on its operand axes regardless of where it appears
in the pipeline:

* **per-axis offsets.** Each operand carries an array of `#hc.expr`
  with length equal to the operand's rank. Pre-flatten that's the
  operand's own nD shape; post-flatten the operand is 1D and the array
  has a single entry — but the iter space stays nD and the per-axis
  structure is preserved on the inputs/outputs the rewriter chose to
  keep nD. Flatten composes each per-axis offset through the operand's
  layout offset and rewrites the operand to its 1D form, leaving a
  single composed expression in the offset slot.
* **SSA bounds with inference.** `iter_bounds` are SSA values
  (`HC_ValueType`), so dynamic, runtime-resolved bounds drop in
  naturally. Any subset may be a value of type `!hc.undef`; that's the
  "infer me" sentinel. A dedicated `hc-infer-generic-bounds` pass walks
  operand shapes and matches iter symbols in the per-axis offsets to
  fill the placeholders. Inference runs **before** flatten — once
  axes have been composed into a single linear expression, recovering
  per-axis ranges is intractable. Rewrites that have full structural
  knowledge (`hc.reduce` / `hc.matmul` → `hc.generic`) emit
  fully-resolved bounds directly; the frontend stays simple and emits
  `!hc.undef` placeholders for elementwise decomposition.
* **`iter_kinds`** lists `parallel | reduction` per iter sym. Mixing
  the two in one op is what makes it a contraction-shaped surface
  (matmul, dot, reductions over arbitrary axes) instead of a pure
  elementwise op.
* **outs as init.** Each output operand carries the iteration-start
  value through the body (block arg, last `outs.size()` positions).
  The body's `hc.yield` produces the next value, which becomes the
  output's value at the parallel index. For reduction iters this is
  the carried accumulator; for pure-parallel ops the `cv` arg is
  unused. Caller pre-fills value-typed outs with the reduction
  identity (`hc.zeros`, `hc.full`) before the op. Ptr/buffer-typed
  outs source the carry via an implicit `hc.ptr_load` at the
  iteration's offset, so the destination's existing memory is the
  reduction init — same shape as `linalg.generic` over a memref dest.
* **multiple outputs** are allowed and share the iteration space — fused
  reductions (sum + count for mean, max + argmax) become one op.
* **polymorphic outs (value + ptr/buffer, mixed allowed).** Any subset
  of `outs` may be ptr/buffer-typed (`!hc.ptr<...>` /
  `!hc.bare_*<...>`); the rest stay value-typed. This subsumes
  `linalg.generic`'s tensor/memref split into one op:
  - **result count** equals the number of *value-typed* outs, in
    declaration order. A pure-store generic (all-ptr outs) produces
    zero results; a mixed generic produces a result for each
    value-typed slot.
  - **memory effects** are per-operand: every ptr/buffer in `ins`
    contributes `MemoryEffects::Read` on its operand value;
    every ptr/buffer in `outs` contributes `Read+Write` (the read
    is for the carry). Value-typed outs contribute nothing — the op
    is then pure on those slots.
  - **body block args** still come `ins ++ outs` in declaration
    order, one element-typed arg per operand regardless of operand
    flavor (value or ptr/buffer).
  - **`hc.yield`** produces exactly `outs.size()` values, type-matched
    to each output's element type. The lowering routes value-typed
    yields into the SSA result and ptr-typed yields into an implicit
    `hc.ptr_store` at the operand's offset.

Verifier rules:

* per-operand offset array length equals the operand's rank.
* output offset expressions reference parallel iters only. A reduction
  iter in an output offset is the most likely surface mistake; the
  diagnostic names the offending iter and offset.
* body block arg arity is `ins.size() + outs.size()`; the trailing
  `outs.size()` args are the carried output values, in declaration
  order. Element type per arg = element type of the operand
  (value-typed: the operand's element type; ptr-typed: the pointee
  type).
* `hc.yield` produces exactly `outs.size()` values, type-matched to
  each output's element type.
* result types match the value-typed outs in declaration order; the
  ptr/buffer outs contribute no SSA result. A generic with no
  value-typed outs produces zero results.

This op is the single home for compute end-to-end:

* the existing per-element decomposition of `hc.add`, `hc.mul`, ... on
  bare values lowers into a single `hc.generic` with all-parallel
  iters and value-typed outs;
* `hc.reduce` rewrites into `hc.generic` with one reduction iter and
  the appropriate identity fill on the output (`hc-shaped-compute-
  to-generic`);
* `hc.matmul` rewrites into `hc.generic` with two parallel iters and
  one reduction iter on the K axis, plus an identity fill (`hc-
  shaped-compute-to-generic`); shape comes from the operand types
  directly, not from inference;
* `hc.load` / `hc.vload` rewrite into `hc.generic` with a ptr/buffer
  input and a value-typed out of the loaded shape (all-parallel
  iters, body forwards the loaded element);
* `hc.store` rewrites into `hc.generic` with a value-typed input and
  a ptr/buffer out (all-parallel iters, no SSA result, body
  forwards the input element through `hc.yield`);
* the cooperative copy helper becomes `hc.generic` with one
  ptr/buffer input and one ptr/buffer output, all parallel iters;
* `as_layout` reorderings — when they survive flatten — also become
  `hc.generic` between two offset expressions;
* fused compute-and-spill (e.g. matmul + LDS trace, GEMM + bias write)
  lands as a single mixed-outs `hc.generic` instead of two ops.

Mask handling stays out of v1: masks ride as ordinary bare-pred tensor
inputs that the body inspects with `scf.if`. A typed mask slot on the
op surface is a follow-up if vectorization needs it as a first-class
operand later.

Bound inference — `hc-infer-generic-bounds`:

Scans every `hc.generic` and fills any `iter_bounds` operand whose
defining op is `hc.undef`. The procedure: for each iter sym `s`,
collect every `(operand, axis)` pair where `s` appears in the offset
expression at position `axis`; match the operand's shape entry at that
axis against `[0, s)`; emit a `materialize_bound_expr` (or directly a
shape-symbol-reference) producing the resolved bound. Conflicts (two
operands implying different bounds for the same iter) are diagnostics,
not silent picks. The pass is no-op if every bound is already
concrete, so rewriters that emit fully-resolved ops pay nothing.

Lowering — `hc-lower-generic`:

Scalar baseline: nested `scf.for` over `iter_bounds`, reduction iters
as the inner loops with `iter_args` carrying the accumulators, parallel
iters as the outer loops with a load-once / store-once pattern.
Pre-fill of the outs is read into the outermost `iter_args`; the body
runs scalar; the result writes back at the parallel index.

Vector codegen: for each operand on the chosen vectorization axis, the
symbolic engine computes
`Δ = offset(iter[axis] + 1) - offset(iter[axis])`. Three outcomes:

* every `Δ == 1` on a *parallel* axis → contiguous vector load/store,
  body lifts to vector arithmetic. Same op covers scalar fallback and
  vectorized fast path.
* `Δ == 1` on a *reduction* axis → vector load + `vector.reduction`
  (horizontal reduce). The body's accumulator combinator picks the
  reduction kind.
* every `Δ` is a constant `s ≠ 1` → strided/shuffled load
  (target-dependent: AMDGPU DS strided loads, fallback to gather).
* unknown / mixed → scalar `scf.for` loop.

The vectorization decision is a symbolic-engine question, not a
heuristic. Layouts that ixsimpl can simplify produce vectorized code;
layouts that defeat it produce correct scalar code with a clear "missed
vectorization here" location for users to inspect.

## Pass pipeline (revised)

```
hc-front-to-hc                    capture layout= and as_layout(...)
hc-infer-types                    propagate #hc.layout through the lattice
hc-decompose-shaped-values        layout flows on data and mask halves
hc-canonicalize-layouts           identity → absent, ixsimpl normalize, fold double as_layout
hc-shaped-compute-to-generic      rewrite hc.matmul / hc.reduce into hc.generic + identity fill
hc-infer-generic-bounds           fill !hc.undef iter_bounds on hc.generic from operand shapes
hc-flatten-with-layouts           nD+layout → 1D no-layout, offsets at access sites
hc-lower-launch-body              operate on 1D bare values; emit hc.alloc / hc.ptr_*
hc-lower-generic                  symbolic stride analysis; scalar or vector codegen
hc-lower-to-llvm                  hc.ptr → llvm.ptr, hc.alloc → addrspace globals/allocas
```

`hc-decompose-shaped-values`, `hc-lower-launch-body`, and
`hc-lower-to-llvm` already exist and shrink in scope as the new passes
take over their indexing/memory work.

## Decisions taken

| | Decision | Rationale |
| --- | --- | --- |
| layout grammar | unrestricted in v1 | document the supported subset after seeing what ixsimpl handles; cheap to tighten later, expensive to loosen |
| dialect split | none, all in `hc` | small surface; new dialect is only worth it once the codegen ops outgrow the source ops |
| pointer typing | optional element type | LLVM target drops it, SPIR-V target keeps it; same op surface |
| `hc.alloc` count | static after specialization | matches the AMDGPU static-LDS path; dynamic LDS is a follow-up |
| layout slot | semantic and bare types | semantic types need it for `layout=` capture; bare types need it for the post-decompose passes; both is consistent |
| buffer layout default | fully strided (np/torch) | every np/torch input already produces one; non-contiguous slices work without a host-side copy; user-supplied buffer layouts extend the same machinery later |

## Out of scope (deferred)

* **Vector layouts that aren't carrier-permutation.** WMMA fragment
  lane-mapping as a first-class `#hc.layout` is interesting but not
  required to make the existing example go. Recipe-level encoding stays.
* **Dynamic LDS allocation.** A launch-time-sized `hc.alloc` needs a
  runtime hook the host wrapper doesn't have today.
* **User-supplied buffer layouts.** The IR slot is there from day one;
  the frontend / host wrapper only learn to consume the default strided
  layout in v1.
* **Layout-aware fusion.** Combining adjacent `hc.generic` ops with
  compatible offset expressions across the iteration space.
* **Auto-padding.** The user explicitly writes `H + 4` today. A
  heuristic that picks a padding stride to avoid bank conflicts is
  research, not infrastructure.
* **Block-cyclic / Z-order layouts.** Permitted by the unrestricted
  v1 grammar, but the symbolic engine almost certainly can't prove them
  injective today; they'll get rejected with `unverifiable_layout`
  until ixsimpl learns the moves.

## Delivery slices

Each slice is small enough to land + verify independently. None depend
on later slices.

1. **layout attr scaffold** — define `#hc.layout<...>` over `#hc.expr`,
   parser/printer/round-trip, ixsimpl normalization, no IR consumers.
2. **type slot** — add optional layout to the five shaped types
   (buffer + tensor + vector + bare_tensor + bare_vector),
   default-absent semantics for the four shaped types, inference
   no-ops; buffer slot still empty until slice 3.
3. **frontend capture** — `hc_front` collects `layout=` and
   `as_layout(...)` for tensors/vectors and emits the default strided
   layout for every buffer argument (referring to per-arg `$STRIDE_*`
   symbols). `hc.as_layout` grows the new attr kind alongside the
   enum (transitional).
4. **`hc-canonicalize-layouts`** — identity → absent for shaped types,
   double `as_layout` fold, ixsimpl normalization on attached attrs;
   buffer layouts left alone.
5. **host ABI: stride extraction** — host wrapper queries
   `np.ndarray.strides` / `torch.Tensor.stride()`, divides by element
   size, and binds the new `$STRIDE_*` symbols at launch alongside the
   existing shape symbols. Smoke-tested on a contiguous matmul (same
   numerics as today) and a transposed view (validates the strided
   path).
6. **`hc.ptr` family** — `!hc.ptr<...>`, `hc.alloc`, `hc.ptr_offset`,
   `hc.ptr_load`, `hc.ptr_store`. Round-trip + LIT. No flatten yet, no
   `hc.generic` yet.
7. **`hc-flatten-with-layouts`** — type-only collapse: every shaped
   value becomes 1D and loses its layout slot. Tensors / vectors get
   `<T, [storage_size_expr]>`; buffers get `<T, [?]>` (`#hc.dyn`
   sentinel) because their storage extent is host-owned. Op surfaces
   stay untouched: per-axis offset arrays on `hc.generic` and
   multi-index lists on `hc.load` / `hc.store` / `hc.vload` /
   `hc.buffer_view` carry through at their original logical rank for
   downstream fusion / vectorization to consume. Per-access offset
   materialization, `hc.as_layout` structural-difference handling,
   and `i1` byte-per-element retirement are separate slices
   documented under the pass section above.
8. **`hc.generic` op surface** — define the op (parallel + reduction
   iter kinds, outs-as-init, multiple outputs, per-operand `#hc.expr`
   offset slot, SSA `iter_bounds`). No lowering yet, no per-axis array
   yet — single offset per operand is the v0 surface.
9. **per-axis offset slot** — extend `hc.generic` so each operand
   carries an `ArrayAttr<#hc.expr>` of length equal to the operand's
   rank. Pre-flatten consumers emit nD per-axis arrays; post-flatten
   collapses to one entry per operand. Verifier enforces the rank
   match. Flatten composes per-axis offsets through the operand's
   layout.
10. **`hc-infer-generic-bounds`** — pre-flatten pass that walks operand
    shapes and per-axis offsets to fill any `!hc.undef`-typed
    `iter_bounds` on `hc.generic`. Conflicts diagnose; no-op when every
    bound is already concrete.
11. **`hc-shaped-compute-to-generic`** — rewrite `hc.matmul` and
    `hc.reduce` into `hc.generic` + an identity fill on the result.
    Runs pre-flatten on semantic shaped types so the per-axis offsets
    line up with the operand shapes directly; flatten then composes
    them through the layout the same way it composes any other
    `hc.generic`. v0 supports rank-2 matmul (uniform arith family,
    `hc.astype`-promoted body), reduce sum on float / integer, and
    reduce max / min on float; integer max / min, `keepdims = true`,
    and rank-0 result are deferred follow-ups.
12. **`hc.generic` polymorphic outs** — extend the op so `outs` may
    mix value and ptr/buffer entries. `MemoryEffectsOpInterface`
    derives Read/Write per ptr-typed operand; result count tracks
    value-typed outs only; body block-arg element types come from the
    operand's element type or pointee type. Verifier updates and
    round-trip LIT only — no rewriters or lowering changes here.
13. **`hc-elementwise-to-generic`** — rewrite the per-element
    decomposed family (`hc.add`, `hc.sub`, `hc.mul`, `hc.div`,
    `hc.mod`, `hc.and`, `hc.or`, `hc.neg`, `hc.not`, `hc.cmp.*`,
    `hc.astype`) into a single `hc.generic` with all-parallel iters
    and value-typed outs. Pure source-level rewrite; runs pre-flatten
    so per-axis offsets are identity over the operand shape. Iter
    bounds are emitted as `!hc.undef` placeholders that
    `hc-infer-generic-bounds` later resolves from the operand shapes.
    `hc.select`, scalar / shaped broadcast, and the nullary fills
    (`hc.zeros`, `hc.full`, ...) are deferred — they need either an
    init-scaffolding rework or a broadcast story this slice doesn't
    pin down.
14. **`hc-load-store-to-generic`** — rewrite `hc.load` / `hc.vload`
    into `hc.generic` with a ptr/buffer in and a value-typed out, and
    `hc.store` into `hc.generic` with a value-typed in and a
    ptr/buffer out. Picks up the multi-index offset arrays the
    per-access materialization slice (`hc-flatten-with-layouts`
    follow-up) feeds it. Cooperative copy helpers fold into a single
    ptr-in / ptr-out generic. Masked stores, tensor-dst stores, and
    `hc.load_mask` are deferred to follow-ups.
15. **scalar `hc-lower-generic`** — lower `hc.generic` (all three
    forms: value-out, ptr-out, mixed) to a scalar `scf.for` nest
    only. No vectorization yet. Implicit `hc.ptr_load` for the
    ptr-out carry and implicit `hc.ptr_store` on the yield.
16. **retire per-op lowering paths** — once every shaped op funnels
    into `hc.generic`, drop the dedicated lowering paths in
    `hc-lower-launch-body` and the bare-value decomposition machinery
    that fed them. Single codegen surface from this slice forward.
17. **switch `hc-lower-launch-body` from memref to `hc.ptr`** —
    wholesale replacement of the cooperative-load/store machinery;
    memref drops out. WMMA still on its existing intrinsic path.
18. **`hc-lower-to-llvm` for `hc.ptr` and `hc.alloc`** — the example
    runs end-to-end on the new stack.
19. **symbolic stride vectorization** — same `hc-lower-generic`,
    smarter codegen. The actual win.

Slices 1–5 are pure additive (no observable behavior change beyond the
strided buffer ABI, which preserves contiguous numerics). Slices 6–18
are the risky middle. Slice 19 is what the design exists for.
