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
   and vectors means the identity layout (dense, rightmost axis varies
   fastest); on buffers it is set to the host-supplied strided layout
   at frontend time.
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

`hc.as_layout` and the type-attached `#hc.layout` slot take only the
structured form above. The Python-side `index_map` builder constructs
canonical row-first / column-first instances when no `params` /
`offset` is supplied.

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
means the identity layout; the canonicalization pass folds an explicit
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
* `hc.as_layout`: replaces the operand layout with the op attribute.
  Operand and result may also differ in rank / shape; the verifier
  guards safety by comparing effective `storage_size` expressions
  under ixsimpl. See *Late-bound symbols and reinterpret /
  Shape-changing `hc.as_layout`* below.
* `hc.vload` / `hc.load` / `hc.vec` (and allocators `hc.vzeros`,
  `hc.vones`, `hc.zeros`, `hc.ones`, `hc.empty`, `hc.vfull`,
  `hc.full`) carry an optional `layout` attribute. Frontend `layout=`
  kwargs are stamped directly on the producer op rather than emitting
  a separate `hc.as_layout` overlay; inference reads the attribute
  and bakes it into the result type. This is what lets non-injective
  layouts (broadcasts, per-lane fragments where
  `storage_size < product(shape)`) flow through the pipeline: the
  producer never reports the bare `product(shape)` storage that the
  old `hc.as_layout` overlay path would have compared against
  `layout.storage_size`. `hc.as_layout` itself stays useful for the
  explicit `.as_layout(...)` surface — e.g. reinterpreting a value
  returned from elsewhere, or shape-changing reinterprets — and its
  storage_size verifier still runs in that path for value-semantic
  operands (tensor / vector / bare) where the dims *are* the
  allocation. For pointer-rooted operands (`!hc.buffer`) the storage
  check is skipped: a buffer's named dims are a proxy for the
  addressable extent, not a declaration of its byte size, so layouts
  whose `storage_size` legitimately diverges from `product(dims)`
  (broadcasts, per-lane WMMA fragments, multibuffered LDS reinterprets)
  are accepted on the ptr side. The runtime gather/scatter path bounds-
  clips against the real flat allocation rather than the layout's
  declared `storage_size`. Element-type and shape-rank checks stay
  strict on both sides.
* `hc.buffer_view`: a layout-bearing source composes scalar subscripts
  and non-trivial slice rebinds into the residual layout —
  `index_syms[k]` substitutes the scalar's expression, `shape_syms[k]`
  substitutes the operand's dim entry, and both slots are dropped
  from the residual layout. For non-trivial slices (`[lo:hi:st]` with
  `lower != 0` or `step != 1` or sliced extent `!=` operand dim),
  `index_syms[k]` rebinds to `lower + step * index_syms[k]` and
  `shape_syms[k]` substitutes the operand's dim (slots stay to keep
  the rank invariant). Trivial slices (`[:]`, `[0:dim:1]`) and
  implicit pass-through axes keep their slots unchanged. See
  *Late-bound symbols and reinterpret / `hc.buffer_view`* below.

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
shape symbols. Contiguous inputs end up with the same expressions the
identity layout would produce; non-contiguous slices (a column from a
2D tensor, a transposed view) work without a host-side copy.

The buffer-attached layout differs from tensor/vector layouts in three
ways:

* The absent-as-identity shortcut doesn't apply — the absence of an
  explicit layout on a buffer means "the host hasn't been queried yet",
  not "the dense identity layout". The frontend always emits a strided
  layout for every buffer argument; the canonicalization pass leaves
  them alone.
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

The pass collapses every shaped value to its 1D form and drops its
layout slot. On `hc.generic` and per-access ops it also composes the
per-operand offset addressing into the post-flatten 1D form: the
per-axis `#hc.expr` array on each `hc.generic` operand becomes a
single composed `#hc.expr` (substituted through the operand's
`#hc.layout` offset, or the identity-layout fallback when no layout
is attached); multi-index lists on `hc.load` / `hc.store` / `hc.vload`
collapse to a single `hc.idx_apply`-materialized 1D base offset by
the same path. `hc.load_mask` is rewritten to an `hc.generic` upstream
by `hc-load-store-to-generic`, so it reaches flatten via the
generic-offset compose path with the rest. `hc.buffer_view` on a
layout-bearing source folds away once `hc-infer-types` has composed
scalar subscripts into the residual layout — the flat carriers on
both endpoints describe the same physical storage, so flatten
forwards the source's expansion through unchanged (see *Late-bound
symbols and reinterpret / `hc.buffer_view`*); the layout-less /
strided-slice path still does its own offset composition below.

What runs:

1. The type changes from `<T, [d_0, ..., d_{n-1}], #hc.layout<L>>` to
   `<T, [storage_size_expr]>` (1D, no layout). `storage_size_expr` is
   the layout's `storage_size` after binding `shape_syms` to the
   original shape entries — or the dimension product when the type
   sits on the implicit identity-layout contract (no explicit
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
   through the converter, their `storage_size` expressions agree
   (the verifier guarantees it), and the relabel becomes cosmetic.
   Shape-changing reinterpret (e.g. 1-D bare carrier presented as
   N-D layout view) collapses to the same flat carrier on both
   sides, so the operand expansion passes through verbatim.

Deferred slices (out of scope here):

* **`hc.as_layout` with disagreeing storage.** Operand and result
  whose effective `storage_size` expressions don't unify under
  ixsimpl are rejected by the verifier — there's no `hc.generic`
  copy fallback, layouts that need to materially repack data go
  through `hc.generic` directly.
* **`hc.buffer_view` layout-less strided-slice offset composition.**
  Buffer views whose source has *no* layout still rely on the
  layout-less strided-slice branch below for rank-reduction. Buffer
  views with a layout-bearing source compose the slice's `lower` /
  `step` into the residual layout's `index_syms[k]` at type
  inference time (see *`hc.buffer_view` composes scalar indices and
  slice rebinds into the layout*) so the flatten identity branch
  picks them up. Today the layout-less chain walk lives in
  `hc-lower-launch-body`; eventually flatten will fuse the chain
  into the sliced operand's layout directly.
* **`i1` byte-per-element retirement.** The hand-coded `i1` mask
  handling in `HCLowerLaunchBodyPass.cpp` needs flatten to emit a
  byte-per-element layout for `i1` shaped types as a default. Until
  then the manual extract/insert sequences stay.

Post-flatten invariants:

* **Types.** No layout attribute survives on any shaped type, every
  shaped value is 1D (buffers spell the unknown extent as `#hc.dyn`).
* **Offsets.** `hc.generic`'s per-operand offset arrays carry one
  composed `#hc.expr` per operand axis (one entry post-flatten on
  the 1D operand); per-access ops carry a single 1D `!hc.idx<expr>`
  base operand. Free symbols (iter syms, dim / stride params,
  workgroup IDs) survive in the composed expressions for the
  downstream lowering to bind.

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

// vector forms — same op, vector-typed result/operand:
%vv = hc.ptr_load  %p         : !hc.ptr<workgroup, f16> -> vector<8xf16>
      hc.ptr_store %vv, %p    : vector<8xf16>, !hc.ptr<workgroup, f16>
```

`hc.ptr_load` / `hc.ptr_store` are **polymorphic on the result /
operand type**: a scalar element type denotes a single-element
load/store, a `vector<NxT>` denotes a contiguous N-element load/store
starting at the pointer. Width is unconstrained at this op — target
lowering at `hc-lower-to-llvm` splits oversized vectors into
hardware-sized chunks (AMDGPU dwordx4, SPIR-V's natural vector width)
without touching the IR before that. The vec lowering path
(`hc-lower-generic` at `U > 1`) emits the wide form directly at
merged contig groups.

The pair has predicated counterparts for masked memory access:

```mlir
// scalar predicated load: load if pred true, else passthrough.
%v = hc.ptr_load_pred %p, %pred passthrough %fill
    : !hc.ptr<workgroup, f16>, i1, f16 -> f16

// vector predicated load: per-lane mask, per-lane passthrough.
%vv = hc.ptr_load_pred %p, %mask passthrough %fillv
    : !hc.ptr<workgroup, f16>, vector<8xi1>, vector<8xf16>
    -> vector<8xf16>

// scalar / vector predicated store: no-op on inactive lanes.
hc.ptr_store_pred %v, %p, %pred : f16, !hc.ptr<workgroup, f16>, i1
hc.ptr_store_pred %vv, %p, %mask
    : vector<8xf16>, !hc.ptr<workgroup, f16>, vector<8xi1>
```

`hc.ptr_load_pred` / `hc.ptr_store_pred` mirror the polymorphic
surface of the unconditional pair plus a predicate operand and (for
the load) a mandatory `passthrough` fill. Shape parity is enforced
by the verifier: scalar value ↔ `i1` predicate; `vector<NxT>` value
↔ `vector<Nxi1>` predicate of the same N. The passthrough is
required so producers commit to the inactive-lane value at the IR
boundary (no implicit poison/undef); the common "zero on miss"
spelling is one `arith.constant 0` away.

Lowering at `hc-lower-generic` decomposes by partition width: at
`U = 1` (scalar slice) the predicated forms lower to `scf.if` +
`hc.ptr_load` / `hc.ptr_store`; at `U > 1` (vec slice, contig
group) they lower directly to upstream `vector.maskedload` /
`vector.maskedstore`. This is the v1 surface for masked memory —
see the lowering section's "Predicated bodies" notes for how the
search interacts with `scf.if` residuals.

The value-side counterpart for predicated reads is `hc.predicate`:

```mlir
// scalar `mask ? value : passthrough` — semantically equivalent to
// arith.select, but with a producer-hoist lowering contract.
%r = hc.predicate %v mask %m passthrough %fill : f32, i1

// vector form — per-lane mask, per-lane passthrough.
%rv = hc.predicate %vv mask %mv passthrough %fillv : vector<4xf32>, vector<4xi1>
```

`hc.predicate` is a `Pure` value op with no parent restriction —
producer-side rewriters and the `hc.generic` body lowering share the
same surface. Mask shape parity follows the same rule the predicated
mem ops carry (scalar ↔ `i1`, `vector<NxT>` ↔ `vector<Nxi1>` of the
same N); passthrough is mandatory, no implicit poison fallback.

The producer-hoist lowering is split across two passes:

* `hc-lower-generic` does *not* touch `hc.predicate` — it clones the op
  verbatim through the body so the predicate lands in the lowered
  `scf` nest right next to its now-explicit `hc.ptr_load` producer.
* `hc-fold-predicates` runs after, walks each `hc.predicate`, and
  rewrites the *producer* rather than emitting a speculative load
  followed by a blend — OOB lanes never materialise the underlying
  read. Dispatch is on the def site (strict allow-list, anything else
  is a hard diagnostic):

  * `hc.ptr_load` → clone in place as `hc.ptr_load_pred` with the
    mask / passthrough. Each `hc.predicate` use clones one predicated
    op at the producer's def site, so multiple decorators on the same
    load each get their own predicated form. The unpredicated load is
    erased only when it has no remaining uses.
  * `vector.extract` of a vector value → `arith.select` at the
    predicate site. The underlying vector load stays unconditional;
    the predicate only gates the lane.
  * Block-arg, arbitrary other ops, or `hc.ptr_load_pred` itself
    (double-predicating) → diagnostic, pass fails.

Splitting fold-from-emit keeps `hc-lower-generic` unaware of mask
shapes / producer kinds and lets the fold run on every emitter that
plants a predicate, not just on `hc-lower-generic`'s output.

Dominance: the mask and passthrough must dominate the producer's def
site when the fold pass clones / rewrites there. The user has to
schedule the mask before the load; otherwise the fold errors
out. Always-true masks fold the predicate away
(`m_One()` → `result := value`); always-false elide the load
(`m_Zero()` → `result := passthrough`, dead producer cleaned up).

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

The 1D bare tensor → memory carrier transition has retired the
workgroup-AS `memref<...xT, #gpu.address_space<workgroup>>` family
*and* the kernel-arg `memref<?x?xT>` envelope from
`HCLowerLaunchBodyPass.cpp`. Workgroup-staged tiles flow through the
`hc.generic` pipeline (`hc-load-store-to-generic` →
`hc-flatten-with-layouts` → `hc-lower-generic`); the per-op
load/store/select patterns in launch-body emit `hc.alloc` +
`hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]` directly. The end-of-pipeline `hc-lower-to-llvm`
pass rewrites that family into `!llvm.ptr` (workgroup → addrspace-3
`llvm.mlir.global private` + `llvm.mlir.addressof`, private →
`llvm.alloca` addrspace-5, predicated forms → `scf.if` / masked
intrinsics) without a memref intermediate. Kernel arguments now flow
on the same `!hc.ptr<global, T>` carrier: `hc-lower-kernels-to-gpu-launch`
materializes a `(ptr, dim*, stride*)` tuple per buffer at the host —
`hc_get_ptr` for the data pointer, `hc_get_dim` / `hc_get_stride` for
extents — and bridges back to `!hc.buffer<...>` via an N→1
`unrealized_conversion_cast` planted inside the launch body so
`gpu-kernel-outlining` captures the raw values. `hc-lower-launch-body`
walks that UCC to recover the source ptr + per-axis strides and lowers
every kernel-arg load/store via `hc.ptr_offset` + `hc.ptr_load[_pred]`
/ `hc.ptr_store[_pred]`. The final `hc-lower-to-llvm` step rewrites
`!hc.ptr<global, T>` to `!llvm.ptr<1>` on `gpu.func` and `func.func`
signatures and emits an `llvm.addrspacecast` at the host boundary for
the generic→global pointer transition. No memref reaches the LLVM
backend.

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
  has a single composed entry. Flatten substitutes the per-axis exprs
  positionally into the operand's `#hc.layout` offset formula
  (binding `index_syms` to the per-axis exprs and `shape_syms` to
  the operand dim entries) — or falls back to the identity layout when
  no layout is attached — to produce that single composed offset. The
  iter space stays nD on `iter_syms`; only the per-operand addressing
  collapses to 1D.
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
  - **`hc.yield_predicated`** is the per-value masked counterpart of
    `hc.yield`: same arity and same per-slot element-type parity as
    `hc.yield`, plus one `i1` (or `vector<Nxi1>` for vector slots)
    mask per yielded value. Slot `i` publishes only on lanes where
    `$masks[i]` is true; masked-out lanes leave the outs-as-init
    carry in place. The lowering routes value-typed predicated yields
    into an `arith.select` blend against the carry and ptr-typed
    yields into `hc.ptr_store_pred` at the operand's offset.

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
* the body terminates with either `hc.yield` or `hc.yield_predicated`;
  both produce exactly `outs.size()` values, type-matched to each
  output's element type. `hc.yield_predicated` additionally carries
  one mask per value (strict parity) — scalar value with `i1`,
  `vector<NxT>` value with `vector<Nxi1>` of the same N.
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
* workgroup-staged copy becomes `hc.generic` with one ptr/buffer
  input and one ptr/buffer output, all parallel iters;
* `as_layout` reorderings — when they survive flatten — also become
  `hc.generic` between two offset expressions;
* fused compute-and-spill (e.g. matmul + LDS trace, GEMM + bias write)
  lands as a single mixed-outs `hc.generic` instead of two ops.

No typed mask slot on `hc.generic` itself in v1 — masks ride as
ordinary bare-pred tensor inputs when precomputed, or on body-side
ops when computed in the iteration scope. Both edges of the body
carry their own mask channel:

* **Input side**: `hc.predicate %sv mask %m passthrough %f` decorates
  the block-arg load. `hc-lower-generic` clones the predicate into the
  lowered `scf` body next to the now-explicit `hc.ptr_load`;
  `hc-fold-predicates` then hoists the mask into the load
  (`hc.ptr_load_pred`), so masked-out lanes never speculate.
* **Output side**: the body terminates with `hc.yield_predicated`
  instead of `hc.yield`; per-value masks gate the publish. Lowering
  picks `hc.ptr_store_pred` for ptr-typed outs and an `arith.select`
  blend against the outs-as-init carry for value-typed outs.

That keeps masked code vectorizable end-to-end without falling back
to `scf.if` shapes in the body (see the `hc.ptr` section). A typed
mask slot on the op surface remains a follow-up only if a workload
shows the precomputed-tensor input story needs first-class operand
treatment beyond what the body-side `hc.predicate` already covers.

Tile sizes that don't divide the work bound generate trailing
iterations with some lanes OOB. Both edges express the validity
condition at the source-rewrite level:

```mlir
hc.generic
    iter (parallel i = %n : index)
    ins (%src at [#hc.expr<"i">] : !hc.ptr<global, f32>)
    outs (%dst at [#hc.expr<"i">] : !hc.ptr<global, f32>)
    -> () {
^bb0(%sv: f32, %dv: f32):
  %valid = hc.pred_apply (%i as "i") : (!hc.idx<"i">) -> !hc.pred<"i < M">
  %m = hc.cast %valid : !hc.pred<"i < M"> -> i1
  %fill = arith.constant 0.0 : f32
  // Predicated load: lowering hoists %m to the implicit hc.ptr_load,
  // OOB lanes never read past the buffer.
  %sv_pred = hc.predicate %sv mask %m passthrough %fill : f32, i1
  // Predicated store: lowering routes through hc.ptr_store_pred at
  // %dst's ins-slot offset.
  hc.yield_predicated %sv_pred mask %m : (f32), (i1)
}
```

The launch-body / `hc.generic` lowering picks up the predicated load
and store off the mask directly — no hand-rolled OOB guard inside the
rewriter, the predicate lives at the source-rewrite level on both
sides.

Bound inference — `hc-infer-generic-bounds`:

Scans every `hc.generic` and fills any `iter_bounds` operand whose
defining op is `hc.undef`. The procedure: for each iter sym `s`,
collect every `(operand, axis)` pair where `s` appears in the offset
expression at position `axis`; match the operand's shape entry at that
axis against `[0, s)`; emit an empty-binding `hc.idx_apply` (or
directly a shape-symbol-reference) producing the resolved bound. Conflicts (two
operands implying different bounds for the same iter) are diagnostics,
not silent picks. The pass is no-op if every bound is already
concrete, so rewriters that emit fully-resolved ops pay nothing.

Lowering — `hc-lower-generic`:

Loop nest shape: parallel iters become an outer `scf.parallel` with
the parallel iters as multi-dim induction variables; reduction iters
become an inner `scf.for` nest with `iter_args` carrying the
accumulators. Pure-parallel ops collapse to a single `scf.parallel`
with no inner loop. The body runs once per innermost iteration,
sources value-typed `outs` carries via the `iter_args` chain (or via
implicit `hc.ptr_load` at the operand offset for ptr/buffer-typed
outs), and writes back at the parallel index after the reduction
nest finishes.

Codegen frame: **unroll-and-merge**. The pass picks an iter-axis
order and a partition `p` of an unroll budget across the iters,
constrained by divisibility — every per-axis factor `p_i` must
provably divide its axis bound `N_i`, queried via
`ixs_check(cmp(ixs_mod(N_i, p_i), EQ, 0))`. The scalar baseline pins
`p = (1, ..., 1)` (trivial divisibility, total unroll 1); the vector
slice enumerates partitions with total unroll `prod(p_i) ≤ 32`
subject to the divisibility filter. **No tail loop is emitted** —
when divisibility doesn't hold for the larger factors the pass
simply discards them and picks a smaller valid partition (e.g.
`(8, 2)` instead of `(32, 1)`); when no axis divides above 1 the
pass falls back to scalar `p = (1, ..., 1)`.

For each surviving candidate `(order, partition)` the pass
symbolically generates `prod(p_i)` per-operand offset expressions
over the unrolled iter positions, then runs a pairwise contiguity
probe — `ixs_check(cmp(offset[k+1] − offset[k], EQ, 1))` — to
identify maximal contig groups in each operand's offset list. Score
= total merged elements summed across all input/output operands.
Pick the `(order, partition)` with the maximum score, ties broken
lexicographically on the order.

Search bounds: axis-order search is capped at `n ≤ 4`; for `n ≥ 5`
the pass uses declaration order without permuting. Partition
enumeration is bounded by `prod(p_i) ≤ 32` and pruned by the
divisibility filter before the merge probe even runs.

Emission: the body stays **elementwise scalar** at every total
unroll. The pass clones the body `prod(p_i)` times in declaration
order; reduction-axis unrolls thread the carry through all clones
in sequence (single accumulator, sequential adds — LLVM and the
loop vectorizer fuse to vector reductions if profitable). Loads and
stores at the body boundary route through the merge result:

* contig group of size `G > 1` → one vector-typed `hc.ptr_load` /
  `hc.ptr_store` of width `G` for the whole group; individual body
  copies read their lane via `vector.extract` / write via
  `vector.insert` (or pack at the store boundary). Width is
  **maximal at the merge** — target lowering splits oversized
  vectors into hardware-sized chunks at the `hc-lower-to-llvm`
  boundary, no target-aware width selection in this pass.
* size-1 group → scalar-typed `hc.ptr_load` / `hc.ptr_store`.

Because every chosen partition factor provably divides its axis
bound, the unrolled main loop covers the entire iteration space
exactly — no tail loop is ever emitted.

Predicated bodies — the dominant case is masked memory access,
expressed two ways depending on which side of the body owns the
mask channel:

* **Input loads** ride on `hc.predicate %v mask %m passthrough %f`
  in the body. `hc-lower-generic` clones the predicate through verbatim;
  `hc-fold-predicates` then rewrites the producer of `%v`: a body-
  emitted `hc.ptr_load` is cloned in place as `hc.ptr_load_pred`, a
  `vector.extract` collapses to `arith.select` at the predicate site.
  Each `hc.predicate` use generates its own producer-site clone —
  sharing is opt-in, not default. The merge analyzer probes the
  resulting predicated loads the same way it probes the unconditional
  pair (the predicate doesn't change offset arithmetic). Emission
  decomposes the chosen partition: scalar groups lower to `scf.if` +
  `hc.ptr_load`, vector groups lower
  directly to `vector.maskedload`.
* **Output stores and value-out blends** ride on
  `hc.yield_predicated` at the body terminator. The lowering peeks
  at the terminator kind: an `hc.yield` produces unpredicated stores
  / SSA result writes as before; an `hc.yield_predicated` per-value
  mask drives `hc.ptr_store_pred` for ptr-typed outs and
  `arith.select` against the outs-as-init carry for value-typed
  outs. Always-true masks fold back to the unpredicated path
  (canonical `m_One()` match), constant-false masks elide the store
  / select entirely; the producer materialises `arith.constant true`
  to keep one verifier rule — every mask is an `i1` SSA — without
  inventing a sentinel "no mask" op.

Masked code stays vectorizable end-to-end without intermediate
`scf.if` shapes defeating the search.

`scf.if` *in the body* — i.e. control flow not lifted into a
predicated memory op — remains as the residual case. A single
`scf.if` anywhere in the body short-circuits the partition search
to `(1, ..., 1)` and emits the scalar baseline with a "missed
vectorization: scf.if in body" remark. This is conservative — a
pure-arith `scf.if` (predicated select, predicated SSA carry
update) would compose with merged loads/stores at the boundary —
but the residual is rare in practice once masked load/store sites
are routed through the predicated ops, and a smarter policy that
bails only on `scf.if` whose regions have memory effects is a
follow-up.

Body vectorization (lifting scalar arith back to vector ops) is a
follow-up — SLP / loop-vectorizer / target codegen handles the
common cases. The pass commits to elementwise scalar arith
internally; the only vectorization is at the memory boundary.

The vectorization decision is a symbolic-engine question, not a
heuristic. Layouts that ixsimpl can simplify produce contig-merged
loads/stores; layouts that defeat it produce correct scalar code
with a clear "missed vectorization here" location for users to
inspect.

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

## Non-injective layouts

`LayoutAttr` enforces `index_syms.size() == shape_syms.size()`: every
logical axis has both an extent (`shape_syms[k]`) and a coordinate
(`index_syms[k]`), and access ops bind one operand per axis. Per-lane
fragments, broadcast layouts, and similar "many logical indices share
one storage slot" patterns express that **through the offset formula**,
not by adding extra index_syms.

Concretely: a layout is *non-injective* when distinct logical-index
tuples map to the same storage offset (so `storage_size` can be — and
usually is — strictly smaller than the logical-shape product). The
verifier doesn't try to prove injectivity in either direction; it's
the layout author's contract. Example for a per-lane row fragment of
a WMMA tile expressed as a 3-D logical shape `(M, K, LANE)` collapsed
to a 1-D `K`-sized storage:

```mlir
#hc.layout<shape_syms = ["M", "K", "LANE"],
           index_syms = ["i", "j", "lane"],
           params = {},
           storage_size = #hc.expr<"K">,
           offset = #hc.expr<"j">>
```

`shape_syms` names the three logical extents. `index_syms` names the
three access coordinates (one per axis, same rank as `shape_syms`).
`storage_size = K` says the physical storage is a single K-column;
`offset = j` says element `(i, j, lane)` lives at flat slot `j`, so
every `(i, *, lane)` slice reads the *same* K row. That's a deliberate
broadcast — multiple lanes seeing the same row data.

Consumer rules:

* `hc.load`, `hc.vload`, `hc.store`, `hc.load_mask` bind one operand
  per `index_syms` entry; `hc-verify-static-shapes` enforces
  `indices.size() <= rank` (fewer is legal — trailing axes pick up
  iter syms from the result tile). Each operand can be a scalar
  (pinned coord) or `!hc.slice` (iter sweep).
* `hc-canonicalize-layouts` only folds layouts where the offset
  formula matches the canonical "rightmost-fastest * dims" identity.
  Non-injective offsets and offsets referencing fewer than `rank`
  index_syms aren't the identity, so the pass leaves them alone.
* `hc-decompose-shaped-values` mirrors the layout from the semantic
  `!hc.tensor` / `!hc.vector` onto both bare halves of the
  `(bare data, bare mask)` pair. The mask carrier inherits the same
  layout so `hc.store`'s mask/source verifier still finds the pair
  structurally consistent post-decompose; `hc-flatten-with-layouts`
  then reads the preserved layout off the bare carrier when composing
  per-access offsets, the same way it does for the semantic input.
  Dropping the layout here would collapse every non-injective access
  to identity indexing — materialising the broadcast before flatten
  ever sees it.
* `hc-load-store-to-generic` consults the result type's layout on a
  non-injective `hc.vload` whose source rank is smaller than the tile
  rank (canonical case: rank-1 flat storage fanning out into a
  multi-axis logical tile via a broadcast `offset`). Per-source-axis
  offsets come from substituting iter syms into the result layout's
  `offset` expression so the verifier's "one offset per operand axis"
  contract holds across rank changes. Same-rank accesses keep the
  existing per-axis `base + step * iter` form.
* `hc-flatten-with-layouts` composes the indices against the layout's
  `offset` expression by zip-substituting `index_syms[k] -> indices[k]`
  and `shape_syms[k] -> source_dims[k]`, then handing the resulting
  single-element offset to the post-flatten access op.
* The frontend's `index_map(..., offset=lambda i, j, lane, M, K, L: ...)`
  serialises both `index_syms` and `shape_syms` from the lambda's
  positional parameter names. `Buffer[..., layout]` accepts any uniform
  layout; per-call access goes through standard subscripting:
  `group.vload(buf[i, j, lane], shape=...)`.

What does **not** yet do anything special with non-injective layouts:

* Access op lowering. `hc.vload` reads contiguously from the composed
  base offset post-flatten. If the offset doesn't reference every
  iter axis, the corresponding axes broadcast — but that broadcast
  is a side-effect of `hc-lower-generic`'s per-iter scalar emit (one
  `hc.idx_apply` + `hc.ptr_load` per iteration, lifted into the
  result vector via `vector.insert` / `vector.from_elements` when
  outs is a `!hc.bare_vector`, or replayed against the same flat
  offset when outs is `!hc.ptr`). The two layers — flatten dropping
  the unreferenced iter sym from the composed offset, and
  lower-generic emitting a `hc.idx_apply` that only lists the iter
  syms the offset mentions — are pinned by
  `test/HC/flatten-with-layouts.mlir` (`@generic_noninjective_layout`)
  and `test/HC/lower-generic.mlir` (`@noninjective_broadcast_2d`).
  There is no explicit gather primitive; "layout-driven per-thread
  gather" with a dedicated op stays on the deferred list.

What does:

* Simulator `group.load` / `group.vload`. A layout-bearing read
  ravels the source and reads `flat_source[layout.offset(*logical,
  *shape[, params])]` per logical position. The same O(prod(shape))
  walk the bounds probe already runs; OOB offsets clip to a zero /
  false slot, same masking contract as the no-layout overlap copy.
  Pinned by `tests/test_simulator.py`
  (`test_group_vload_gathers_through_noninjective_broadcast_layout`,
  `test_group_vload_gather_clips_out_of_bounds_offsets`,
  `test_group_load_accepts_layout_and_preserves_logical_contents`).
  Free-sym layouts still bail at `resolve_layout` — the simulator
  has no surrounding kernel scope to bind from.

Non-injectivity is orthogonal to where the symbols *bind*: a layout
can be injective on its declared `(shape_syms, index_syms)` set yet
still reference names from the surrounding kernel scope (a
`!hc.idx<"buf_idx">` kernel arg, an ancestor `scf.for`'s induction
var). Those late-bound names — and the IR moves that introduce them
(authoring a free sym directly, slicing a layout-bearing source with
`hc.buffer_view`, reinterpreting a 1-D carrier as a multi-D view with
`hc.as_layout`) — are covered next.

## Late-bound symbols and reinterpret

Three IR moves let a layout reach beyond its own slot lists or change
the logical structure of the value it's attached to:

* **Free symbols in `offset` / `storage_size`** — names not declared
  on the layout that the surrounding kernel scope supplies at
  lowering time. Authored directly on the layout.
* **`hc.buffer_view` composes scalar subscripts and non-trivial slice
  rebinds into the residual layout** — a scalar subscript substitutes
  the matching `index_syms` entry into the layout's offset /
  storage_size and drops both slots from the residual rank. A
  non-trivial slice (`[lo:hi:st]` whose `lower != 0`, `step != 1`, or
  whose sliced extent differs from the operand's dim) rebinds
  `index_syms[k] -> lower + step * index_syms[k]` and substitutes
  `shape_syms[k]` with the operand's dim, both slots staying to keep
  the rank invariant. The substituted scalar / slice bounds typically
  pin kernel-scope syms, which surface as free symbols in the view's
  residual layout.
* **Shape-changing `hc.as_layout`** — reinterprets a value's logical
  rank / extents without touching storage, guarded by an effective-
  `storage_size` equality check under ixsimpl. The 1-D-allocation-as-
  N-D-layout-bearing-view pattern (multibuffered LDS) routes through
  here.

The common thread: layouts describe addressing, not allocation, and
the lowering pipeline binds the symbolic surface against the actual
SSA values that pass through the access site. Validation is lazy on
purpose — the same composed offset may lower under different ambient
scopes, so the binding contract is a property of the *lowering site*,
not the layout.

### Free symbols in `offset` / `storage_size`

A layout's `offset` and `storage_size` formulas may reference symbol
names that aren't in `shape_syms`, `index_syms`, or `params`. They're
called *free symbols* and represent kernel-scope bindings the
lowering pipeline resolves at access time:

* Kernel-arg aux idx values surfaced by `hc-flatten-with-layouts`'
  type expansion (the leading positional carrier + one `!hc.idx<sym>`
  aux per name in `collectImplicitSyms`).
* Ancestor block arguments that pin a single bare sym in their type
  (an `scf.for` induction variable typed `!hc.idx<sym>`, an
  `hc_front` kernel scope sym, …).
* Ambient launch geometry (`$WG0`, `$WI1`, `$WGS2`, …) that
  `hc-lower-launch-body` injects into the apply's bound-values map.

`LayoutAttr::verify` accepts any combination of free symbol names;
nothing is rejected up front. Lowering is **lazy**: the apply emitted
at the access site lists the bindings it can supply (operand
expansion + scope walk), and `hc-lower-launch-body` walks the
composed offset expression at the apply node. If every free symbol
resolves, the apply lowers to plain `arith` ops. If any doesn't, the
pass fails with a diagnostic that names the unresolved sym, the
apply op, and the candidate scopes it searched — see
`test/HC/lower-launch-body-invalid.mlir` (`@unknown_symbol`) for the
exact wording. There is no flatten-time "you forgot to bind X"
diagnostic by design.

Authoring shape:

```mlir
#hc.layout<shape_syms = ["M", "N"],
           index_syms = ["i0", "i1"],
           params = {},
           storage_size = #hc.expr<"M*N">,
           offset = #hc.expr<"(row0 + i0)*N + i1">>
```

`row0` is a free sym. Attaching this layout to a `!hc.tensor<f32,
["M", "N"], #hc.layout<...>>` gives the 1-to-N type expansion the
implicit syms `{M, N, row0}` — `row0` appears as an extra aux
`!hc.idx<"row0">` trailing the flat carrier on every function arg /
call result the type passes through. Access ops bind `row0` from
that aux in the post-flatten `hc.idx_apply`; see
`test/HC/flatten-with-layouts.mlir` (`@free_sym_in_offset`) for the
exact CHECK lines.

Frontend (`hc.core.index_map`):

```python
my_layout = index_map(
    storage_size=lambda M, N: M * N,
    offset=lambda i, j, M, N, *, row0, col0:
        (row0 + i) * N + col0 + j,
    free_syms=("row0", "col0"),
)
```

The `free_syms=` kwarg names the kernel-scope symbols; each lambda
binds the subset it references as keyword-only parameters after `*`.
`_classify_index_map` runs the lambda with `hc.symbols.Symbol`
instances for the declared free names, so the resulting `#hc.expr`
keeps `row0` / `col0` as bare leaves. Frontend collision checks
guarantee `free_syms` is disjoint from `shape_syms` / `index_syms` /
`params` keys before MLIR sees the dict attribute. Pytest coverage:
`tests/test_resolve.py::test_index_map_classifier_accepts_free_syms`,
`..._rejects_undeclared_kwonly`, `..._rejects_free_sym_collision`.

What the simulator does today: free-sym layouts route through
`resolve_layout` and bail with a `SimulatorError` that names the
declared free syms. The simulator path has no surrounding kernel
scope to query and can't make up runtime values. The non-injective
gather path documented above runs against in-layout symbols only;
free-sym binding for the simulator is its own follow-up. Pinned by
`tests/test_simulator.py::test_resolve_layout_rejects_free_syms`.

### `hc.buffer_view` composes scalar indices and slice rebinds into the layout

`hc.buffer_view` on a layout-bearing source is the second free-sym
introduction site. `inferBufferViewResult`
(`lib/IR/HCInferTypeOpInterface.cpp::composeBufferViewLayout`)
composes the operand layout against the per-axis subscript stream:

* **Scalar axes** (`v[buf_idx]`): substitute `index_syms[k]` with the
  scalar's `IdxType` expression, substitute `shape_syms[k]` with the
  operand's actual dim entry, and drop both names from the residual
  layout's slot lists.
* **Trivial slice axes** (`v[:]`, `v[0:dim:1]` — `lower` defaults to
  0, `step` defaults to 1, and the sliced extent structurally equals
  the operand's dim): keep both slots unchanged. The original sym
  names continue to refer to the operand's axis `k`.
* **Non-trivial slice axes** (`v[lo:hi:st]` — any of the three departs
  from the trivial defaults): rebind `index_syms[k]` to `lower + step
  * index_syms[k]` and (when the sliced extent differs from the
  operand's dim) substitute `shape_syms[k]` with the operand's dim
  entry. Both slots remain in the residual lists so the rank invariant
  `shape_syms.size() == index_syms.size() == result_rank` survives;
  the substituted names are reachable through their handles in the
  rebound expression but no longer appear as bare references in the
  offset / storage_size / params formulas.
* **Implicit pass-through axes** (subscripts shorter than the source
  rank): same as trivial slice — keep slots, no substitution.

Substituted scalars and slice bounds typically pin kernel-scope syms
(a `!hc.idx<"buf_idx">` kernel arg, a block-arg loop IV, a kernel-arg
`row0`); those syms then surface in the residual layout as free
symbols and bind through the same mechanism described above.

Two motivating shapes:

* **Multi-buffered LDS**: a 4-D layout `(BUF, M, N, LANE)` sliced with
  `lds_4d[buf_idx]` collapses to a 3-D residual whose `offset` already
  bakes in the `buf_idx*M*N*L` shift. The caller never declares
  `buf_idx` on the layout; it surfaces as a free sym after composition
  and binds from the buffer_view's index operand at lowering time.
* **WMMA per-lane fragments**: a 3-D layout `(M, N, LANE)` over a 2-D
  buffer accessed with `c.as_layout(...)[row0 : row0 + WMMA_M : 2,
  col0, lane]` — the leading slice's `lower = row0` and `step = 2`
  rebind `index_syms[0]` to `row0 + 2 * index_syms[0]`, the trailing
  scalars substitute `col0` / `lane`, and the residual is a 1-D
  per-lane fragment whose `storage_size = M*N` matches the operand's
  flat span so the flatten identity branch forwards through.

The flatten identity branch keys off the residual `storage_size`
structurally matching the operand's. `composeBufferViewLayout` keeps
this contract for non-trivial slices by substituting `shape_syms[k]`
with the operand's dim entry whenever the sliced extent departs from
it — without that substitution the residual `storage_size` would bind
the layout's local name to the sliced extent and the identity branch
would mis-fire. The downstream access patterns
(`ComposeLoadOffsets` / `ComposeVLoadOffsets` / `ComposeStoreOffsets`
in `lib/Transforms/HCFlattenWithLayoutsPass.cpp`) accept 1-D
layout-bearing sources too, so a 1-D residual feeds the post-flatten
offset compose just like a multi-D one.

LIT coverage:
`@buffer_view_layout_multibuf`,
`@buffer_view_layout_mixed_scalar_slice`,
`@buffer_view_layout_all_slice`,
`@buffer_view_layout_strided_slice`,
`@buffer_view_layout_lower_slice` in `test/HC/infer-types.mlir`;
`@buffer_view_layout_multibuf_forwards_source`,
`@buffer_view_strided_slice_then_vload` in
`test/HC/flatten-with-layouts.mlir`.

### Shape-changing `hc.as_layout`

`hc.as_layout` is a pure relabel — same storage, new logical
descriptor. The relaxed verifier accepts operand and result types
that differ in rank / shape, provided both sides address the same
physical span. "Same span" is the effective `storage_size`:

* with a layout, `layout.storage_size` after substituting
  `layout.shape_syms` with the type's dim entries;
* without a layout, the product of the type's dims (the default
  identity layout's storage size).

ixsimpl hash-consing makes the comparison structural — two textually
different expressions that reduce to the same canonical form
compare equal. Element type must still match; payload reinterpretation
is `hc.astype`'s job, not this op's.

Operand and result accept either flavor of the shaped split
(`!hc.tensor`, `!hc.vector`, `!hc.bare_tensor`, `!hc.bare_vector`);
the op is a pure relabel and composes with the decomposition that
introduces the bare carriers.

The motivating shape: a 1-D bare allocation reinterpreted as a 4-D
layout-bearing view, so `hc.buffer_view` can then slice across the
leading axis (e.g. a multi-buffered LDS arena allocated as a flat
`!hc.bare_tensor<f32, ["2*M*N*L"]>` and presented as
`!hc.tensor<f32, ["2","M","N","L"], #hc.layout<...>>`). Flatten
collapses both endpoints back to the same 1-D carrier, and
`DropAsLayout` forwards the operand expansion through.

Same `computeStorageSizeExpr` (`lib/IR/HCAttrs.cpp`) feeds both the
verifier and the flatten 1-D-collapse, so the canonical handle the
two compare against is the one and only canonical handle for that
storage expression.

LIT coverage:
`@as_layout_shape_change_storage_match` in
`test/HC/ops-buffer-data.mlir` for the positive form;
`test/HC/verify-hc.mlir` for the storage-mismatch and
element-type-mismatch diagnostics;
`@as_layout_shape_change_collapses_to_1d` in
`test/HC/flatten-with-layouts.mlir` for the flatten round-trip.

## Out of scope (deferred)

* **Layout-driven per-thread gather op.** A standalone `hc.gather`
  whose semantics evaluate `offset` per logical element and pull
  scalars from the source. The WMMA example's B-fragment column
  reads and accumulator strided-row reads are the motivating shape,
  but they're expressible today through the standard subscript +
  vload pattern plus the implicit broadcast described above. A
  dedicated op would give backend lowerings a cleaner contract.
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
   symbols). `hc.as_layout` carries the structured `#hc.layout<...>`
   attribute.
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
7. **`hc-flatten-with-layouts`** — every shaped value becomes 1D
   and loses its layout slot. Tensors / vectors get
   `<T, [storage_size_expr]>`; buffers get `<T, [?]>` (`#hc.dyn`
   sentinel) because their storage extent is host-owned. Operand
   addressing also collapses post-flatten: `hc.generic`'s per-axis
   `#hc.expr` arrays compose through the operand's `#hc.layout`
   offset (or the identity layout when no layout is attached) into
   a single 1D `#hc.expr` per operand; per-access multi-index
   lists on `hc.load` / `hc.store` / `hc.vload` collapse to a single
   `hc.idx_apply`-materialized base offset. `hc.load_mask` is rewritten
   to an `hc.generic` upstream and arrives here via the generic-offset
   compose path.
   `hc.buffer_view`, `hc.as_layout` structural-difference handling,
   and `i1` byte-per-element retirement are separate slices
   documented under the pass section above.
8. **`hc.generic` op surface** — define the op (parallel + reduction
   iter kinds, outs-as-init, multiple outputs, per-operand `#hc.expr`
   offset slot, SSA `iter_bounds`). No lowering yet, no per-axis array
   yet — single offset per operand is the v0 surface.
9. **per-axis offset slot** — `hc.generic` carries an
   `ArrayAttr<#hc.expr>` per operand of length equal to the
   operand's rank. Pre-flatten the entries are the per-axis
   addressing on the operand's logical nD shape; flatten composes
   them through the operand's `#hc.layout` offset (or the identity
   layout) into a single entry on the post-flatten 1D operand.
   The verifier enforces rank parity in both regimes.
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
    into `hc.generic` with a ptr/buffer in and a value-typed out,
    `hc.store` into `hc.generic` with a value-typed in and a
    ptr/buffer out, and `hc.load_mask` into a value-typed-outs
    `hc.generic` whose body emits a single `hc.pred_apply` carrying
    the per-axis bounds conjunction. Picks up the multi-index offset
    arrays the per-access materialization slice
    (`hc-flatten-with-layouts` follow-up) feeds it. Cooperative copy
    helpers fold into a single ptr-in / ptr-out generic. Masked
    `hc.store` rides on the same rewrite — the mask operand becomes
    a second ins slot with identity offsets and the body terminates
    with `hc.yield_predicated` instead of `hc.yield`, so the
    lowering routes through `hc.ptr_store_pred` at the dst's ins-slot
    offset. Tensor-dst stores remain a deferred follow-up.
15. **scalar `hc-lower-generic`** — lower `hc.generic` (all three
    forms: value-out, ptr-out, mixed) to an outer `scf.parallel`
    over the parallel iters with an inner `scf.for` nest over the
    reduction iters carrying the accumulator via `iter_args`. The
    unroll-and-merge framework lands here at `U = 1` (degenerate
    partition, no merges, all loads/stores scalar) so the vector
    slice is a delta and not a rewrite. Implicit `hc.ptr_load` for
    the ptr-out carry and implicit `hc.ptr_store` on the yield.
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
    enables axis-order search (capped at 4) and partition
    enumeration with total unroll `prod(p_i) ≤ 32`, gated by an
    `ixs_check`-driven divisibility filter on each axis bound; runs
    the pairwise contiguity probe to find merged groups; emits
    maximal-width vector-typed `hc.ptr_load` / `hc.ptr_store` at
    each merged group while keeping the body elementwise scalar.
    No tail loop — partitions that don't divide cleanly are pruned
    in favor of smaller ones, degenerating to `p = (1, ..., 1)`
    when no axis divides above 1. The actual win.

Slices 1–5 are pure additive (no observable behavior change beyond the
strided buffer ABI, which preserves contiguous numerics). Slices 6–18
are the risky middle. Slice 19 is what the design exists for.
