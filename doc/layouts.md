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
4. **Elementwise op** — `hc.elementwise` plays the `linalg.generic` role
   but indexes operands with `#hc.expr` instead of `affine_map`. Its
   lowering uses the symbolic engine to decide vectorization.

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

For every shaped value with a non-default layout:

1. The type changes from `<T, [d_0, ..., d_{n-1}], #hc.layout<L>>` to
   `<T, [storage_size_expr]>` (1D, no layout). `storage_size_expr` is
   the layout's `storage_size` after binding `shape_syms` to the
   original shape entries.
2. Every access op (`hc.load`, `hc.store`, `hc.vload`, `hc.buffer_view`,
   `hc.vec`) is rewritten so its multi-index becomes a single 1D offset
   computed by binding `index_syms` and inlining `offset(...)`. The
   inlined expression goes through ixsimpl, so common cases collapse to
   short SSA chains, not opaque expression trees.
3. `hc.as_layout`: if the source and destination layouts agree on the
   carrier (the offset expressions are equal under ixsimpl), it becomes
   a no-op cast. Otherwise it expands into an `hc.elementwise` copy
   whose input and output use the two different offset expressions over
   the same iteration space.
4. The `i1` mask byte-per-element behavior currently hand-coded in
   `HCLowerLaunchBodyPass.cpp` falls out as a default: `i1` shaped types
   pick a `byte-per-element` layout out of the box, and the manual
   extract/insert sequences in the launch-body lowering retire.

Post-flatten invariant: no layout attribute survives on any shaped
type. Every shaped value is 1D, every access carries the offset
explicitly. This is the boundary the pointer/elementwise lowering
operates against.

## `hc.ptr` and memory ops

Types and ops added to the `hc` dialect (no second dialect — keeping
the surface small):

```mlir
!hc.ptr<addrspace = workgroup | global | private>           // opaque
!hc.ptr<f16, addrspace = workgroup>                         // typed (SPIR-V)
```

Element type is optional. AMDGPU/LLVM lowering uses the opaque form;
SPIR-V lowering keeps the typed form. The element type is dropped at
the lowering pass that emits `llvm.ptr` (LLVM has been opaque-pointers
since 17). Any pass that needs the element type before that point gets
it from the `hc.ptr_load` / `hc.ptr_store` op, which always carries it.

```mlir
hc.alloc count = %n : !hc.ptr<f16, addrspace = workgroup>
hc.ptr_offset %p, %i : (!hc.ptr<addrspace=?>, index) -> !hc.ptr<addrspace=?>
%v = hc.ptr_load  %p          : !hc.ptr<addrspace=?> -> f16
     hc.ptr_store %v, %p      : (f16, !hc.ptr<addrspace=?>) -> ()
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

## `hc.elementwise`

```mlir
%out = hc.elementwise
    iter_syms   = ["i", "j"]
    iter_bounds = [%w, %h]
    ins  (%a : !hc.bare_tensor<f16, [%n_a]> at #hc.expr<"i * Sa + j">,
          %b : !hc.bare_tensor<f16, [%n_b]> at #hc.expr<"j * Sb + i">)
    outs (%c : !hc.bare_tensor<f16, [%n_c]> at #hc.expr<"i * Sc + j">) {
  ^bb0(%av: f16, %bv: f16):
    %s = arith.addf %av, %bv : f16
    hc.yield %s : f16
}
```

Same idea as `linalg.generic` but with the affine-map slot replaced by
a per-operand `#hc.expr` over `iter_syms ∪ shape_syms ∪ params`. The
body operates on scalar SSA values; `hc.yield` returns one value per
output operand.

This op is the single home for elementwise math after flatten:

* the existing per-element decomposition of `hc.add`, `hc.mul`, ... on
  bare values lowers into a single `hc.elementwise`;
* mask-aware stores (the guarded scalar-store path in the current
  launch-body lowering) become `hc.elementwise` with a predicate
  operand and an `scf.if` in the body;
* the cooperative copy helper becomes `hc.elementwise` with two
  different offset expressions on input and output;
* `as_layout` reorderings — when they survive flatten — also become
  `hc.elementwise` between two offset expressions.

Lowering — `hc-lower-elementwise`:

For each operand, the symbolic engine computes
`Δ_axis = offset(iter[axis] + 1) - offset(iter[axis])` over the chosen
vectorization axis. Three outcomes:

* every `Δ == 1` → emit a contiguous vector load/store, lift the body
  to vector arithmetic. This is the payoff: the same op covers scalar
  fallback and vectorized fast path.
* every `Δ` is a constant `s ≠ 1` → emit a strided/shuffled load
  (target-dependent: AMDGPU has DS strided loads, fallback to gather).
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
hc-flatten-with-layouts           nD+layout → 1D no-layout, offsets at access sites
hc-lower-launch-body              operate on 1D bare values; emit hc.alloc / hc.ptr_*
hc-lower-elementwise              symbolic stride analysis; scalar or vector codegen
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
* **Layout-aware fusion.** Combining adjacent `hc.elementwise` ops with
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
   `elementwise` yet.
7. **`hc-flatten-with-layouts`** — nD+layout → 1D-flat. Output uses
   bare types + per-access offset expressions; not yet on `hc.ptr`.
   Covers buffer, tensor, vector, and their bare counterparts uniformly.
8. **`hc.elementwise` op + scalar `hc-lower-elementwise`** — define
   the op, lower it to a scalar `scf.for` loop only. No vectorization
   yet.
9. **switch `hc-lower-launch-body` from memref to `hc.ptr`** —
   wholesale replacement of the cooperative-load/store machinery;
   memref drops out. WMMA still on its existing intrinsic path.
10. **`hc-lower-to-llvm` for `hc.ptr` and `hc.alloc`** — the example
    runs end-to-end on the new stack.
11. **symbolic stride vectorization** — same `hc-lower-elementwise`,
    smarter codegen. The actual win.

Slices 1–5 are pure additive (no observable behavior change beyond the
strided buffer ABI, which preserves contiguous numerics). Slices 6–10
are the risky middle. Slice 11 is what the design exists for.
