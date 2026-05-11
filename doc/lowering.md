# RFC: Lowering pipeline for high-level GPU kernel API

## Summary

This document describes an MLIR-first lowering path for the high-level kernel
API defined in `doc/langref.md`.

For the concrete pass schedule and live pipeline shape, see
[`doc/schedules.md`](schedules.md) and the
"hc.ptr and memory ops" section of [`doc/layouts.md`](layouts.md).

The main design goal is to get a useful end-to-end compiler running quickly
while keeping the Python frontend intentionally thin. Type inference, semantic
checks, specialization, and non-trivial rewrites should live in MLIR, not in a
Python-side semantic IR.

The recommended initial path is:

1. parse a restricted Python subset with the standard Python AST,
2. emit textual MLIR in a source-faithful frontend dialect,
3. run MLIR passes to legalize that frontend dialect into a typed kernel
   dialect,
4. perform type inference, scope/layout/mask verification, and specialization
   in MLIR,
5. use a launcher-driven launch-validation step and lower to standard MLIR
   dialects and backend-specific IR.

## Goals

The lowering path should:

* preserve source structure well enough to support useful diagnostics,
* handle the structured control-flow model from `doc/langref.md`,
* support masks, layouts, specialization, helper functions, and intrinsics,
* make MLIR the home of non-trivial compiler logic,
* reuse standard MLIR dialects where practical,
* keep the initial implementation small enough to get something running early.

## Non-goals

The initial lowering path does not try to:

* implement arbitrary Python semantics,
* support arbitrary Python control flow on dynamic symbolic values,
* build a Python semantic IR that duplicates MLIR compiler logic,
* encode every language invariant directly in MLIR types on day one,
* build a perfect long-term dialect stack before first execution,
* replace later optimizations or more specialized lowerings.

## Frontend model

This document assumes an AST-first frontend, but only for a restricted Python
subset. The intention is not to compile full Python, only the kernel DSL.

Frontend capture may initially be implemented with:

* `inspect.getsource(fn)` to recover the source of kernel/helper/intrinsic
  functions,
* `ast.parse(...)` to build a Python AST,
* a structured AST visitor that accepts only the supported subset and emits
  frontend MLIR.

The frontend implementation may still be split internally into:

* source recovery and AST parsing,
* a restricted AST visitor,
* a tiny emitter interface whose real implementation constructs `hc_front`.

This split is about code structure and testability, not about introducing a
third compiler IR. The emitter boundary should be designed around the intended
`hc_front` operation set and should stay close to it.

Milestone 0 should assume kernels, helpers, and intrinsic fallback bodies are
defined in importable `.py` files where source recovery succeeds.
REPL/notebook/lambda/generated-function cases may be rejected in Milestone 0
with a stable frontend error. Alternative registration paths may be added
later.

Unsupported constructs are compile errors.

The Python frontend should stay as dumb as practical. Its job is to:

* recover source and source locations,
* parse the supported AST subset,
* preserve names, literals, annotations, decorators, and region structure,
* serialize that structure into MLIR.

It should not do type inference, shape inference, mask/layout reasoning,
specialization, capture analysis, resource checks, or any other non-trivial
compiler transformation.

In particular, the frontend should not grow a durable Python-side semantic tree
or pseudo-IR. A small fake emitter is acceptable for unit testing the visitor,
but it must remain an ephemeral test harness rather than a new compiler layer.

## Supported AST subset

The initial subset should include only what the current language requires:

* `FunctionDef`
* `Assign`
* `AugAssign` where needed
* `Return`
* `Expr`
* `If`
* `For`
* `Call`
* `Name`
* `Constant`
* `Tuple` / `List` literals where needed
* assignment targets built from `Name`, `Subscript`, and tuple
  destructuring where needed
* `BinOp`, `UnaryOp`, `Compare`
* `Subscript`
* `Attribute`
* nested `FunctionDef` bodies used by `@group.subgroups` and
  `@group.workitems`

Everything else should be rejected explicitly.

This keeps the frontend small while still covering the current language design.
A bootstrap fake emitter may still start narrower, for example by only
accepting `for` over `range(...)` in its first milestone. However, the real
`hc_front` boundary should preserve source forms such as generic `For`,
tuple-shaped or subscript assignment targets, and `AugAssign`, leaving any
desugaring to later legalization out of `hc_front`.

## Frontend dialect

The first IR form should be MLIR, not a Python semantic IR. This document uses
`hc_front` as the textual name of the frontend IR family.

In the current native bootstrap implementation, that frontend family lives in
its own registered MLIR dialect namespace `hc_front`. Textual IR therefore
spells frontend operations and types as `hc_front.*` and `!hc_front.*`.

`hc_front` should be source-faithful and intentionally low intelligence. It is
not the semantic source of truth for the language; it is a serialized AST-like
form that feeds the real MLIR compiler pipeline.

### Design principles for `hc_front`

`hc_front` should:

* preserve source-level structure closely,
* keep source locations on every op,
* preserve unresolved names and calls where possible,
* avoid doing semantic interpretation in Python,
* be easy to emit as textual MLIR.

The recommended frontend boundary is therefore:

* the AST visitor targets a tiny emitter protocol,
* the real emitter builds textual or builder-based `hc_front`,
* test emitters may record the same boundary for unit tests.

That protocol should expose only operations that closely mirror `hc_front`
construction, such as beginning a kernel/function, emitting constants/calls,
opening structured regions, and emitting returns. It should not become a rich
Python IR with independent semantics.

### Suggested `hc_front` operation set

The exact set can evolve, but an initial frontend dialect may include:

* `hc_front.kernel`
* `hc_front.func`
* `hc_front.intrinsic`
* `hc_front.constant`
* `hc_front.name`
* `hc_front.assign`
* `hc_front.aug_assign`
* `hc_front.target_name`
* `hc_front.target_tuple`
* `hc_front.target_subscript`
* `hc_front.attr`
* `hc_front.subscript`
* `hc_front.slice`
* `hc_front.call`
* `hc_front.keyword`
* `hc_front.tuple`
* `hc_front.list`
* `hc_front.binop`
* `hc_front.unaryop`
* `hc_front.compare`
* `hc_front.if`
* `hc_front.for`
* `hc_front.return`
* `hc_front.subgroup_region`
* `hc_front.workitem_region`

Control ops such as `hc_front.if` and `hc_front.for` should own regions for
their structural parts rather than introducing separate top-level ops just for
`condition`, `then`, `else`, `target`, `iter`, or `body`.

This is intentionally more source-faithful than the narrowest bootstrap fake
emitter. For example, if a helper or intrinsic fallback spells a loop as
`for index, row in enumerate(rows):` or uses `accum += x`, that syntax should
survive into `hc_front` rather than being rewritten in Python. If the semantic
dialect later wants only `for_range` or plain assignment, that rewrite should
happen in `hc_front` to `hc` legalization.

### Frontend type strategy

To keep the AST translation layer dumb, `hc_front` may use a very small set of
opaque frontend types such as:

* `!hc_front.value` for expression results,
* `!hc_front.typeexpr` for annotation/type syntax when needed.

Names, decorators, annotations, and literal syntax can remain as attributes or
frontend ops until MLIR legalization resolves them.

Region ops such as `hc_front.subgroup_region` and `hc_front.workitem_region`
should also carry explicit syntactic capture lists naming outer bindings
referenced in the nested body. Recording lexical captures is still frontend
serialization, not semantic capture analysis.

## Semantic dialect

The real compiler IR should be a separate semantic layer. This document uses
`hc` as the semantic dialect for that layer.

The current bootstrap integration now uses a second registered MLIR dialect
namespace for that layer: `hc_front` stays source-faithful, while `hc` carries
the semantic IR.

`hc` is where the language becomes semantic rather than syntactic. It should
own:

* typed values and operations,
* scope and region semantics,
* tensor/vector/buffer semantics,
* mask and layout semantics,
* intrinsic contracts,
* specialization and launch/resource validation hooks.

### Progressive typing in `hc`

`hc_front → hc` is a mechanical structural rewrite. It does not perform type,
shape, dtype, or symbol inference. Every SSA value it produces is typed
`!hc.undef` unless a concrete type is trivially evident (e.g. from a buffer
parameter annotation).

A later inference pipeline — type propagation through control flow, symbol
synthesis for launch-geometry outputs and loop induction variables, constraint
binding — refines `!hc.undef` into concrete semantic types. Types form a
refinement lattice:

```
!hc.undef
  → !hc.idx                                                   (is an index)
  → !hc.idx<#hc.expr<"...">>                                  (index + expression)
  → !hc.pred                                                  (is a boolean)
  → !hc.pred<#hc.pred<"...">>                                 (boolean + expression)
  → !hc.tensor<elem, #hc.shape<...>>                          (tensor, possibly partial)
  → !hc.vector<elem, #hc.shape<...>>                          (vector, possibly partial)
  → !hc.buffer<elem, #hc.shape<...>>                          (buffer)
  → builtin index / i* / f* / tensor<…> / vector<…>           (post-lowering)
```

Boundaries between refinement levels are crossed with casts:

* **`unrealized_conversion_cast`** (MLIR builtin) — used by compiler-internal
  progressive-conversion passes to bridge type changes that later passes are
  expected to resolve. These casts are transient scaffolding.
* **`hc.cast`** — user-visible type conversion matching source-level constructs
  like `astype(np.float32)`, explicit layout changes, and the occasional
  `!hc.idx → index` materialization at the boundary where the symbolic domain
  ends. These casts persist until a lowering pattern consumes them.

### `hc` type strategy

Semantic dialect types:

* `!hc.undef` — inference has not yet pinned this value
* `!hc.idx` — an index-kind value; symbolic expression parameter optional.
  Bare `!hc.idx` is the unpinned form produced by inference before the
  expression is known; the pinned form accepts both `!hc.idx<"M + 1">`
  (inline string) and `!hc.idx<#hc.expr<"M + 1">>` (full attribute). Type
  uniquing across ixsimpl-canonicalized `#hc.expr` gives free expression
  equality.
* `!hc.pred` — a boolean-kind value; symbolic predicate parameter optional
  (`!hc.pred<"M < N">` inline or `!hc.pred<#hc.pred<"M < N">>` full form)
* `!hc.buffer<elem, #hc.shape<...>>` — externally visible buffer
* `!hc.tensor<elem, #hc.shape<...>>` — workgroup-local tensor
* `!hc.vector<elem, #hc.shape<...>>` — immutable fixed-size vector, carrying
  activity mask semantics and an optional collective-return suffix
* `!hc.group<...>` — kernel launch-context token carrying invariant
  `work_shape`, `group_shape`, and `subgroup_size` metadata when available
* `!hc.slice` — first-class slice value with optional low/high/step, built by
  `hc.slice_expr` and consumed by load/store/subscript
* builtin `index`, `i*`, `f*` are first-class operand/result types of the
  generic arithmetic ops and coexist with `hc` types without forced casting

Generic ops use an `HC_ValueType` constraint defined as an `AnyOf<[…]>` over
every type listed above. Ops with clear semantic categories tighten further:

* `HC_NumericValueType` — arithmetic operands (`hc.add`, `hc.sub`, `hc.mul`,
  `hc.div`, `hc.mod`, `hc.neg`) and `hc.cmp.*` inputs; excludes `!hc.pred`,
  `!hc.slice`, `!hc.buffer`
* `HC_ShapedValueType` — ops that only make sense on semantic tensors/vectors
  (`hc.matmul`, `hc.reduce`, `hc.vec`, `hc.with_inactive`, `hc.as_layout`)
* `HC_DecomposableShapedValueType` — semantic or bare shaped values at
  decomposition-aware boundaries such as `hc.store`'s `$source`
* `HC_BufferValueType` — buffer handles for `hc.buffer_dim`, `hc.load`;
  excludes everything non-buffer
* `HC_BufferOrTensorValueType` — destinations/sources that accept either
  (`hc.vload.$source`, `hc.store.$dest`)
* `HC_ViewRootValueType` — generic subscript roots
  (`hc.buffer_view.$buffer`), admitting buffers, tensors, and vectors
* `HC_GroupValueType` — launch-geometry query root (`!hc.group` or
  pre-inference `!hc.undef`)

Every narrow constraint still admits `!hc.undef` so the `hc_front -> hc`
pass stays mechanical. Further narrowings (booleans for `hc.and/or/not`)
wait on structural support for "integer width = 1" / "tensor-of-i1".

Symbolic surface attributes:

* `#hc.expr<...>` — scalar symbolic expression
* `#hc.pred<...>` — symbolic predicate
* `#hc.shape<[...]>` — list of `#hc.expr` dims, printed inline as a list of
  symbolic strings
* `#hc.constraints<[...]>` — set of `#hc.pred`
* `#hc.scope<"WorkGroup" | "SubGroup" | "WorkItem">` — scope for
  `hc.func`/`hc.intrinsic`/attribute use; the verifier rejects any other
  string
* `#hc.effects<"Pure" | "Read" | "Write" | "ReadWrite">` — intrinsic effect
  class; verified similarly
* `#hc.layout<...>` — symbolic layout descriptor (shape_syms / index_syms /
  params / storage_size / offset over `#hc.expr`); see `doc/layouts.md`
  for the design and consumers

The bound name on `hc.symbol` lives in the result type (`!hc.idx<"name">`);
type uniquing gives symbol equality for free, so a dedicated
`#hc.symbol<"name">` attribute is not needed.

### `hc` operation set

The initial implementation of this surface is in place as of the current
bootstrap; what follows describes the implemented shape, not an aspirational
one. Additional folders, type-inference passes, and target lowerings remain
TODO and land behind this IR.

Every op below accepts `!hc.undef` operands so the mechanical `hc_front → hc`
pass can emit them without doing inference work. Verifiers tighten as types
refine; folders dispatch on the concrete-type combinations they recognize.

#### Declarations and regions

* `hc.kernel @name` — compiled kernel; carries a module-scoped symbol name
  plus symbolic launch geometry, declared bound-symbol metadata for launch
  queries and kernel input ABI symbols, parameter annotations, and
  literal-symbol set attributes. `Symbol` trait so references go through the
  symbol table.
* `hc.func @name` — helper callable referenced by `hc.call`. Also a
  `Symbol`. An optional inline `function_type` signature makes the op
  self-describing; when present, `hc.call` sites get arity and type
  parity checked at verify time (`!hc.undef` on either side is a
  progressive-typing wildcard).
* `hc.intrinsic @name` — intrinsic declaration carrying `scope = #hc.scope<...>`
  and an optional `effects = #hc.effects<...>`; its body region holds the
  fallback implementation (may be empty, in which case lowering patterns
  handle the op directly). Also a `Symbol`, with the same optional
  `function_type` signature story as `hc.func`; the Python decorator may
  publish this from explicit `operand_types` / `result_types` metadata.
  `function_type` inputs follow the intrinsic's declared parameter order after
  filtering out Python `const_attrs` names; those filtered names are stored as
  the MLIR-side `const_kwargs = ["wave_size", "arch", ...]` list. Each
  `hc.call_intrinsic` must carry the declared constant kwargs as attributes;
  missing entries fail at verify time. Extra attributes on the call site stay
  allowed so targets can attach their own decorations without modifying the
  declaration.
* `hc.subgroup_region`, `hc.workitem_region` — collective regions with
  syntactic capture lists

#### Terminators

* `hc.return` — kernel/func/intrinsic terminator
* `hc.yield` — block terminator for `hc.for_range` / `hc.if` and other
  structured-value ops

#### Structured control flow

* `hc.for_range %lo to %hi step %step iter_args (...)` — semantic loop matching
  `for i in range(...)`. Induction variable is typed `!hc.undef` out of the
  frontend pass; inference pins it to `!hc.idx<_symN>` with a constraint that
  `_symN` lies in `[lo, hi)` at step `step`. The verifier checks that the
  body block has `1 + iter_args.size()` arguments, that iter-arg types line
  up with block argument types and with result types one-to-one, and that
  the terminating `hc.yield` produces the same number and types of values as
  the op's results. `!hc.undef` is accepted on either side so pre-inference
  IR round-trips.
* `hc.if %cond -> (...) : type($cond)` — structured conditional with an
  optional else region. The verifier matches each non-empty region's yield
  count and types against the op's result signature, and requires an `else`
  region when the op produces results. Conditions use any `HC_ValueType`
  (`i1`, `!hc.pred`, `!hc.undef`, bool tensor, ...).

#### Scalars, captures, and launch geometry

* `hc.const <value>` — binds a literal-bound Python capture (e.g.
  `WMMA_M = 16`) to a scalar SSA value; result is `!hc.undef` from the
  frontend pass and refines to `!hc.idx<"16">` / `i64` / `f32` / … during
  inference
* `hc.symbol : !hc.idx<"name">` — binds a symbolic capture (e.g.
  `M = sym.M`). The bound name lives *in the result type* so it is visible
  at one place and type uniquing gives free equality. The verifier rejects
  any result type other than `!hc.idx` with a pinned expression.
* `hc.cast` — generic value-level conversion: refinement-lattice
  transitions, symbolic-to-builtin exits, vector layout changes. Distinct
  from `unrealized_conversion_cast` (compiler-internal, transient) and
  from `hc.astype` (Python-surface element-type cast carrying a target
  attribute on the op).
* `hc.group_id`, `hc.local_id`, `hc.subgroup_id`, `hc.group_shape`,
  `hc.group_size`, `hc.work_offset`, `hc.work_shape`, `hc.wave_size` —
  multi-result where appropriate (one `!hc.undef` per dimension; refined to
  axis-specific `!hc.idx<...>` types from the `!hc.group` metadata).
  Marked `Pure` under a **pipeline invariant** (see below); duplicate reads
  of the same axis are therefore interchangeable and CSE/DCE may freely
  collapse or drop them.

Source-level geometry queries such as `group.group_id` are preserved as one
`hc.tuple` value wrapping those independent SSA components; Python indexing
then lowers to `hc.getitem`.

##### Launch-geometry invariant

Launch-geometry ops are valid only in contexts where the operand `%group`
denotes a single launch instance. Inside one such region every axis is
invariant, which is what makes `Pure` sound: two reads of `hc.wave_size %g`
are observationally equal. Any pass that fuses regions across launches, or
that introduces a second launch context into an existing region, must
either insert fresh SSA values for the new context or weaken the effects
on the relevant ops before running — otherwise CSE will silently merge
values that came from different launches.

`Pure` does *not* imply cross-invocation folding: different launches
produce different launch state.

#### Generic arithmetic, comparison, boolean, and reduction ops

One op per semantic notion; operand and result types are open. The same op
covers the symbolic domain (`!hc.idx`/`!hc.pred`), builtin scalars
(`i*`/`f*`/`index`), and tensor/vector values.

* `hc.add`, `hc.sub`, `hc.mul`, `hc.div`, `hc.mod`, `hc.neg`
* `hc.cmp.lt`, `hc.cmp.le`, `hc.cmp.gt`, `hc.cmp.ge`, `hc.cmp.eq`, `hc.cmp.ne`
* `hc.and`, `hc.or`, `hc.not`
* `hc.matmul`
* `hc.reduce %v, kind = sum | max | min, axis = N, keepdims = bool` —
  kind is a typed `#hc<reduce_kind ...>` enum, so wrong spellings fail at
  parse rather than verify. Axis must be non-negative and, once the operand
  type carries a concrete shape, less than the rank.
* `hc.astype %v, target = T` — explicit numeric conversion; target must be
  a builtin `int`, `index`, or `float` type and must agree with the op's
  declared result (element-wise for tensor/vector results, directly for
  scalars; `!hc.undef` escapes both checks). Lowers via `hc.cast`.

Folders dispatch on concrete-type combinations:

* both operands `!hc.idx<E1>` / `!hc.idx<E2>` → result `!hc.idx<simplify(op(E1, E2))>`
  computed via ixsimpl at op construction time
* both operands constant builtin scalars → constant fold
* otherwise no fold

Comparison result typing:

* `!hc.idx<A> cmp.lt !hc.idx<B>` → `!hc.pred<#hc.pred<"A < B">>` when ixsimpl
  can form the predicate
* builtin scalar operands → `i1`
* tensor/vector operands → `!hc.tensor<i1, S>` / `!hc.vector<i1, S>` with
  ordinary broadcast rules
* any `!hc.undef` operand → `!hc.undef`

#### Casts

* `hc.cast %x : SrcT -> DstT` — user-visible conversion. Covers `astype`,
  layout changes on vectors, and the symbolic-to-builtin exit
  (`!hc.idx<E> -> index`). Canonicalization drops identity and
  refinement-erasing casts; lowering patterns consume the rest.
* `unrealized_conversion_cast` (MLIR builtin) — compiler-internal boundary
  between progressive-conversion passes; always transient.

#### Buffer views and subscript construction

* `hc.buffer_dim %buf, axis = N` — symbolic dimension (matches `a.shape[1]`);
  refines to `!hc.idx<"dim_N_expr">`. Axis is a non-negative I64 and, once
  the buffer type carries a concrete shape, less than the rank. Negative
  axis indexing is a frontend-time convenience and canonicalized before
  landing in `hc`.
* `hc.slice_expr(lower = %lo upper = %hi step = %st)` — builds an
  `!hc.slice`. Each keyword is optional and space-separated (no commas);
  the printed form lists only the parts the frontend supplied, mirroring
  Python slice syntax (`x[1:]`, `x[1:10:2]`, `x[:]`, `x[::2]`, ...).
  Implementation: MLIR's `Optional<>` + `AttrSizedOperandSegments`, so
  there is no flag/operand drift.
* `hc.buffer_view %buf[%idx...]` — sub-view of a buffer, tensor, or vector with
  the slice-reduced shape, for cases that do not require data movement.
  Inference preserves the storage class (`!hc.buffer` or `!hc.tensor`) and
  removes axes consumed by scalar indices. Vector roots refine to
  `!hc.vector` fragments when slices/trailing dimensions remain, or to the
  element type when scalar indices consume every axis.
* `hc.tuple(%value...)` — first-class aggregate using MLIR's builtin
  `tuple<...>` type, preserving Python tuple structure as a single SSA value
* `hc.getitem %base[%idx...]` — generic Python square-bracket indexing kept
  before the base kind is known. Inference can refine tuple item access when
  the index converges to a concrete integer. Once the base is known to be a
  tuple, failure to produce exactly one static integer index is diagnosed in
  `hc-infer-types`; concrete non-indexable bases are rejected there too. Vector
  bases use the same element/fragment inference as `hc.buffer_view`, while
  buffer and tensor bases remain deferred to `hc.buffer_view`-style lowering.

Multi-axis subscripts on `hc.getitem`/`hc.load`/`hc.store`/`hc.buffer_view`
take variadic operands directly; `hc.tuple` is reserved for source-level tuple
values and explicit aggregates. During `hc_front` conversion, tuple literals
expand only at syntax sites whose target `hc` op is variadic: multi-axis
subscript indices and `shape=(...)` keyword folding. Binding, returning, and
passing a tuple preserves one tuple SSA value.

#### Data movement

Rule of thumb for memory effects in this dialect: n-ary tensor/vector
**math** (`hc.matmul`, `hc.reduce`, generic arithmetic, ...) is modeled as
value-semantic and carries no effect (`Pure`); only ops that materialize or
observe **addressable workgroup state** — `hc.load`, `hc.vload`, `hc.store`,
`hc.vec` — carry memory effects. Allocators carry `MemAlloc`; opaque calls
carry the conservative `MemRead + MemWrite` pair until `hc.func` /
`hc.intrinsic` grow per-target effect annotations.

* `hc.load %buf[%idx...], shape %shape` — buffer → tensor. Carries a
  `MemRead` effect. The shape is an SSA value, usually a tuple of idx-typed
  dimensions; missing source shape is represented as `!hc.undef`.
* `hc.vload %src[%idx...], shape %shape` — buffer/tensor → vector. Same
  effect and SSA shape convention as `hc.load`.
* `hc.store %dst[%idx...], %src` — tensor or vector source; carries a
  `MemWrite` effect. After shaped-value decomposition, bare data stores may
  carry `mask %preds` with matching bare `!hc.pred` validity.
* `hc.vec %t` — tensor → vector materialization. Carries a `MemRead` effect
  because the source tensor is addressable memory; an interleaved store can
  change what two sibling `hc.vec` ops observe, so CSE must not collapse
  them blindly.
* `hc.with_inactive %v, %inactive` — replace inactive elements. The inactive
  fill is a scalar SSA value; once inference pins both operands its numeric
  domain must match the shaped value's element type.
* `hc.as_layout %v, layout = row_major | col_major` — change layout.
  The `layout` payload is a typed `#hc<layout ...>` enum, so garbage
  spellings fail at parse. v0 admits `row_major` and `col_major`; later
  schemes can extend the enum without changing call sites.
* `hc.vzeros shape %shape, dtype = T`, `hc.vones shape %shape, dtype = T`,
  `hc.vfull %fill, shape %shape, dtype = T` — vector allocators (any scope).
  The dtype is optional when the result or fill operand already fixes the
  element type.
* `hc.zeros shape %shape, dtype = T`, `hc.ones shape %shape, dtype = T`,
  `hc.full %fill, shape %shape, dtype = T` — tensor allocators (`WorkGroup`
  scope only), with the same optional dtype convention.
* `hc.empty shape %shape, dtype = T` — uninitialized tensor (`WorkGroup`
  scope); dtype is required for inference unless the result is already typed.

All allocators carry a `MemAlloc` effect: each op hands back a fresh,
distinct storage slab, so CSE cannot collapse two sibling allocations into
one even when they agree on shape (and fill operand, for the fill
variants). This matches `memref.alloc`'s convention and gives `hc.empty`
the right semantics uniformly with the rest — no special-casing required.

Scope verification runs in each tensor allocator's verifier: it walks the
parent chain, stops at the nearest `hc.kernel` / `hc.func` / `hc.intrinsic`
(or module-like op), and rejects the op if it finds an enclosing
`hc.subgroup_region` or `hc.workitem_region` along the way. Keeping the
check local to the op keeps the diagnostic next to the offender; a pass
walking the IR top-down would surface the same error much later in the
pipeline.

#### Calls

* `hc.call @name(...)` — call into an `hc.func`. Verified as a
  `SymbolUserOp`: `verifySymbolUses` resolves `@name` against the nearest
  symbol table, checks that the target op is actually an `hc.func`, and —
  if the callee declares a `function_type` — checks arity and per-arg/result
  type parity. `!hc.undef` on either side of a parity check is a wildcard,
  so partial inference at the call site or the declaration is not a verify
  error.
* `hc.call_intrinsic @name(...) {const_kwarg = ...}` — call into an
  `hc.intrinsic`, verified the same way against `hc.intrinsic` (including
  the optional signature check). Specialization-required keyword arguments
  live as op attributes rather than SSA operands so verify/specialize hooks
  see them without operand bookkeeping.

Both ops implement `MemoryEffectOpInterface` and inherit their effect set
from the callee's declared `effects` class — `pure | read | write |
read_write`. Absence on the callee falls back to `MemRead + MemWrite`,
the conservative default that keeps effect-aware passes from reordering
opaque calls past loads and stores. Concretely:

* `effects = pure` → no effects, so CSE can merge identical calls and
  LICM can hoist them.
* `effects = read` / `write` → exactly one of the sides is reported.
* `effects = read_write` (explicit or default) → both sides are
  reported; behaves like the old hard-coded trait did.

`test/HC/cse.mlir` keeps this honest: a `pure`-annotated helper collapses
under `-cse`, an unannotated one does not.

### Example: gfx11 WMMA kernel at two pipeline stages

The example `examples/amdgpu_gfx11_wmma_matmul.py` is the shape-first
integration target. Two snapshots are shown: immediately after `hc_front → hc`
(structural ops with metadata-seeded group, buffer, and intrinsic contract
types, but no dataflow inference), and after type / symbol inference.

#### After `hc_front → hc` (mechanical, with annotation-seeded contracts)

```mlir
hc.kernel @tiled_gfx11_wmma_matmul(
  %group: !hc.group<work_shape = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>,
                    group_shape = #hc.shape<["32", "1"]>,
                    subgroup_size = #hc.expr<"32">>,
  %a: !hc.buffer<f16, ["M", "K"]>,
  %b: !hc.buffer<f16, ["K", "N"]>,
  %c: !hc.buffer<f32, ["M", "N"]>) attributes {
  work_shape    = #hc.shape<["32*ceiling(1/16*M)", "ceiling(1/16*N)"]>,
  group_shape   = #hc.shape<["32", "1"]>,
  subgroup_size = 32 : i32,
  bound_symbols = ["$WG0", "$WG1", "$WI0", "$WI1", "$SG0", "$SG1",
                   "$WGS0", "$WGS1", "$WO0", "$WO1", "$WS0", "$WS1",
                   "$GSZ0", "$WV0", "M", "K", "N"]
} {
  %c0 = hc.const <0 : i64>  : !hc.undef
  %c1 = hc.const <1 : i64>  : !hc.undef
  %WM = hc.const <16 : i64> : !hc.undef
  %WK = hc.const <16 : i64> : !hc.undef

  %gid:2     = hc.group_id %group      : (!hc.group<...>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  %gid_tuple = hc.tuple(%gid#0, %gid#1)
                 : (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
                   -> tuple<!hc.idx<"$WG0">, !hc.idx<"$WG1">>
  %gr       = hc.getitem %gid_tuple[%c0] : (tuple<!hc.idx<"$WG0">, !hc.idx<"$WG1">>, !hc.undef) -> !hc.undef
  %gc       = hc.getitem %gid_tuple[%c1] : (tuple<!hc.idx<"$WG0">, !hc.idx<"$WG1">>, !hc.undef) -> !hc.undef
  %row0     = hc.mul %gr, %WM          : !hc.undef
  %col0     = hc.mul %gc, %WM          : !hc.undef

  %a_k     = hc.buffer_dim %a, axis = 1 : !hc.buffer<f16, ["M", "K"]> -> !hc.undef
  %acc0    = hc.call @init_wmma_acc(%group) : (!hc.group<...>) -> !hc.undef

  %acc_final = hc.for_range %k0 = %c0 to %a_k step %WK
                            iter_args (%acc = %acc0) : !hc.undef {
    %row_hi = hc.add %row0, %WM : !hc.undef
    %col_hi = hc.add %k0,   %WK : !hc.undef
    %row_sl = hc.slice_expr(lower = %row0 upper = %row_hi)
                  : (!hc.undef, !hc.undef) -> !hc.undef
    %col_sl = hc.slice_expr(lower = %k0 upper = %col_hi)
                  : (!hc.undef, !hc.undef) -> !hc.undef
    %tile_shape = hc.tuple(%WM, %WK)
        : (!hc.undef, !hc.undef) -> tuple<!hc.undef, !hc.undef>
    %a_tile = hc.load %a[%row_sl, %col_sl], shape %tile_shape
              : (!hc.buffer<f16, ["M", "K"]>, !hc.undef, !hc.undef,
                 tuple<!hc.undef, !hc.undef>) -> !hc.undef
    // …analogous slice + load for b_tile…
    %acc1   = hc.call @issue_wmma_tile(%group, %a_tile, %b_tile, %acc)
              : (!hc.group<...>, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
    hc.yield %acc1 : !hc.undef
  }

  hc.call @store_wmma_tile(%group, %c, %row0, %col0, %acc_final)
      : (!hc.group<...>, !hc.buffer<f32, ["M", "N"]>,
         !hc.undef, !hc.undef, !hc.undef) -> ()
}

hc.intrinsic @wmma_gfx11(
  %group: !hc.undef,
  %a_tile: !hc.tensor<f16, ["16", "16"]>,
  %b_tile: !hc.tensor<f16, ["16", "16"]>,
  %a_frag: !hc.vector<f16, ["16"]>,
  %b_frag: !hc.vector<f16, ["16"]>,
  %acc_frag: !hc.vector<f32, ["8"]>,
  %lane: !hc.idx) -> !hc.vector<f32, ["8"]>
  scope = <"WorkItem"> effects = pure
  const_kwargs = ["arch", "wave_size"]
  parameters = ["group", "a_tile", "b_tile", "a_frag", "b_frag",
                "acc_frag", "lane", "wave_size", "arch"]
  keyword_only = ["lane", "wave_size", "arch"] { }
```

Every op is structural, and most expression results still start as `!hc.undef`.
The exceptions are metadata-backed surfaces: kernel buffer annotations seed
element dtypes, launch-context parameters become group/workitem types, and
intrinsic `operand_types` / `result_types` publish the callee `function_type`
before dataflow inference runs.

#### After type / symbol inference

```mlir
hc.intrinsic @wmma_gfx11(
  !hc.undef,
  !hc.tensor<f16, ["16", "16"]>, !hc.tensor<f16, ["16", "16"]>,
  !hc.vector<f16, ["16"]>, !hc.vector<f16, ["16"]>,
  !hc.vector<f32, ["8"]>, !hc.idx) -> !hc.vector<f32, ["8"]>
  scope = <"WorkItem"> effects = pure
  const_kwargs = ["arch", "wave_size"] { }

hc.func @init_wmma_acc(%group: !hc.group<...>)
    -> !hc.vector<f32, ["8"]> attributes {scope = #hc.scope<"WorkGroup">} (...)
hc.func @issue_wmma_tile(%group: !hc.group<...>,
                         %a_tile: !hc.tensor<f16, ["16", "16"]>,
                         %b_tile: !hc.tensor<f16, ["16", "16"]>,
                         %acc: !hc.vector<f32, ["8"]>)
    -> !hc.vector<f32, ["8"]> attributes {scope = #hc.scope<"WorkGroup">} (...)
hc.func @store_wmma_tile(%group: !hc.group<...>,
                         %c: !hc.buffer<f32, ["M", "N"]>,
                         %row0: !hc.idx<"16*$WG0">,
                         %col0: !hc.idx<"16*$WG1">,
                         %acc: !hc.vector<f32, ["8"]>)
    attributes {scope = #hc.scope<"WorkGroup">} (...)

hc.kernel @tiled_gfx11_wmma_matmul attributes { /* ...same attrs... */ }
          (%group, %a, %b, %c) {
  %c0  = hc.const <0>  : !hc.idx<0>
  %WM  = hc.const <16> : !hc.idx<16>
  %WK  = hc.const <16> : !hc.idx<16>

  %gr, %gc = hc.group_id %group            // : (!hc.idx<_gid0>, !hc.idx<_gid1>)
  %row0    = hc.mul %gr, %WM               // : !hc.idx<_gid0 * 16>
  %col0    = hc.mul %gc, %WM               // : !hc.idx<_gid1 * 16>

  %a_k     = hc.buffer_dim %a, axis = 1    // : !hc.idx<K>
  %acc0    = hc.call @init_wmma_acc(%group)
                 : (!hc.group<...>) -> !hc.vector<f32, ["8"]>

  %acc_final = hc.for_range %k0 = %c0 to %a_k step %WK    // %k0 : !hc.idx<"_k0">
                            iter_args (%acc = %acc0) {
    %row_hi = hc.add %row0, %WM            // : !hc.idx<"_gid0 * 16 + 16">
    %col_hi = hc.add %k0,   %WK            // : !hc.idx<"_k0 + 16">
    %row_sl = hc.slice_expr(lower = %row0 upper = %row_hi) : !hc.slice
    %col_sl = hc.slice_expr(lower = %k0   upper = %col_hi) : !hc.slice
    %tile_shape = hc.tuple(%WM, %WK) : tuple<!hc.idx<"16">, !hc.idx<"16">>
    %a_tile = hc.load %a[%row_sl, %col_sl], shape %tile_shape
              : (!hc.buffer<f16, ["M", "K"]>,
                 !hc.slice, !hc.slice,
                 tuple<!hc.idx<"16">, !hc.idx<"16">>)
                -> !hc.tensor<f16, ["16", "16"]>
    // ...
    %acc1 = hc.call @issue_wmma_tile(%group, %a_tile, %b_tile, %acc)
        : (!hc.group<...>, !hc.tensor<f16, ["16", "16"]>,
           !hc.tensor<f16, ["16", "16"]>, !hc.vector<f32, ["8"]>)
          -> !hc.vector<f32, ["8"]>
    hc.yield %acc1
  }

  hc.call @store_wmma_tile(%group, %c, %row0, %col0, %acc_final)
      : (!hc.group<...>, !hc.buffer<f32, ["M", "N"]>,
         !hc.idx<"16*$WG0">, !hc.idx<"16*$WG1">,
         !hc.vector<f32, ["8"]>) -> ()
}
```

Same op shape, refined types. The `%row_hi - %row0 = 16` kind of consistency
check between the loaded view, `%tile_shape`, and the result type is a plain
ixsimpl query on the typed expressions.

Notes on the legalization contract these sketches assume:

* Python helpers that are not decorated with `@kernel.func`/`@kernel.intrinsic`
  (e.g. `_tile_origin`) must be resolved during legalization — either inlined
  against their captured constants or rejected. They never become `hc.call`s.
* `range(lo, hi, step)` is matched into the operands of `hc.for_range` rather
  than becoming a first-class `hc.range` op.
* `ceil_div(M, WMMA_M) * WAVE_LANES` in `work_shape=` lives on the decorator;
  the resolved symbolic expression becomes a `#hc.shape` attribute on
  `hc.kernel`, not an SSA chain.
* All names from `hc_front.name` are eliminated during SSA construction.
  Literal-bound captures become `hc.const`, symbolic captures become
  `hc.symbol`, loop-carried state becomes `hc.for_range` iter args, and
  region-captured values become explicit captures on `hc.subgroup_region` /
  `hc.workitem_region`.

## Compile-time and launch-time ownership

The lowering stack has three authority layers:

* the Python frontend, which recovers source, preserves syntax, and emits
  `hc_front`,
* compile-time MLIR, which builds and verifies symbolic `hc`,
* a launcher/specialization driver, which binds literal symbols and concrete
  launch values, chooses default `group_shape`, and drives launch-time
  validation before execution.

The public entry point is `hc.compile(kernel_fn, symbols=..., schedule=...)`,
which runs whichever of those layers is wired up today and returns a
`CompiledKernel` handle. The handle carries the pre-pipeline `hc_front`
module under `front_ir` / `front_ir_text`, the post-pipeline `hc` module
under `hc_ir` / `hc_ir_text` (or `None` on pipeline failure), and any
MLIR diagnostics emitted during the pipeline run under
`pipeline_diagnostics`. `compiled.invoke(*args, stream=None)` runs the
kernel: it lazy-builds an `hc.execution_engine.ExecutionEngine`, JITs
the post-pipeline LLVM-dialect module, and calls the host wrapper with
each user argument as a `PyObject *` (the wrapper unpacks tensors via
the `_mlir_ciface_hc_get_*` helpers from `libhc_rt_helpers.so` and
dispatches the GPU launch through the `hc_rt_load_kernel` /
`hc_rt_launch_kernel` shim from `libhc_hip_runtime.so`); see the
"End-to-end execution" section below. Partial `symbols` maps are legal;
unbound literals stay symbolic and later stages refine them. Bindings
are recorded on the handle for later stages to consume.

The `hc_front -> hc` stage runs under a transform-dialect schedule, not
a fixed pass list — see [`doc/schedules.md`](schedules.md). The default
schedule lives at `hc/schedules/front_to_hc.mlir`; callers can pass
`schedule=Path(...)` to point at an alternative schedule file, or
`schedule="..."` to inline a transform-module string. Pipeline failures
are non-fatal: the handle stays introspectable at the `hc_front` stage
and the diagnostics tuple on the handle reports what went wrong.

`hc.compile` does not lower a single function in isolation. Starting from
`kernel_fn`, the driver transitively walks every `@kernel.func` /
`@kernel.intrinsic` reachable through globals and closures and lowers the
closed set into one combined `hc_front` module — kernel first, helpers and
intrinsics in discovery order. The handle exposes the resulting symbol
names via `front_ir_symbols` so downstream tooling can enumerate the dep
set without re-parsing.

Every load-context `hc_front.name` in that module carries a `ref` DictAttr
classifying the identifier so the `hc_front -> hc` pass can dispatch in
MLIR without reaching back into Python state. The kinds recognized on
names are `param`, `iv`, `local` (stamped by the frontend from scope
state), `constant`, `symbol` (captured `hc.symbols.Symbol`), `callee`
(`@kernel.func` helper), `intrinsic` (`@kernel.intrinsic`), `inline`
(undecorated Python helper), `builtin` (`range`, `len`, ...), and `module`
(whole-module alias, currently `numpy` only). Attribute accesses
(`hc_front.attr`) carry `dsl_method` when rooted in a param/iv/local,
`numpy_dtype_type` when rooted in the numpy module and naming a scalar
dtype, or `numpy_attr` for any other numpy access. Unresolvable name loads
surface as a frontend error pointing at the file and line.

Compile-time MLIR should operate primarily on symbolic launch parameters and
kernel structure. Concrete buffer shapes, launch shapes, and device limits
enter only when the launcher supplies them to specialization or launch
validation.

MLIR may still own most of the validation logic, but launch-time checks are not
ordinary ahead-of-time `mlir-opt` passes. They are driven by the
launcher/specialization driver after it supplies concrete bindings and target
limits to the validation pipeline.

## MLIR pass pipeline

The current language has multiple classes of validation and transformation.
Those should happen as MLIR passes and verifiers, not as Python IR logic.

### Frontend emission

The Python side should only:

* reject unsupported syntax,
* preserve source structure,
* emit `hc_front` IR.

Postcondition: the module contains only `hc_front` syntax plus builtin/module
attributes needed to preserve source structure.

### `hc_front` to `hc` legalization

The first real compiler stage should:

* fold the "`@group.workitems` / `@group.subgroups` def + immediate
  call" pattern Python uses to open a collective region. The
  `-hc-front-fold-region-defs` pass erases the ghost
  `hc_front.name {ref.kind = "local"} + hc_front.call` trail the
  emitter leaves next to the region op — the region itself is
  already the lowering, so the trail is dead code. Must precede
  `-convert-hc-front-to-hc`; a surviving `ref.kind = "local"` call
  otherwise fires the converter's `unsupported callee ref.kind
  'local'` diagnostic.
* expand undecorated Python helpers that the resolver emitted as
  `hc_front.func` with `ref = {kind = "inline", ...}` into the caller
  (the `-hc-front-inline` pass; must precede `-convert-hc-front-to-hc`
  because the latter refuses to lower a surviving
  `ref.kind = "inline"` call with a located diagnostic),
* resolve decorators and annotations into semantic form,
* recognize DSL constructs such as `group.load`, `group.vload`,
  `with_inactive`, `as_layout`, and region declarations,
* materialize subgroup/workitem regions together with their explicit syntactic
  capture lists,
* build semantic `hc` operations while preserving symbolic launch parameters.

The canonical pipeline for a module the resolver may have stamped
folding / inline markers on is:

    hc-front-fold-region-defs → hc-front-inline
                              → convert-hc-front-to-hc
                              → hc-promote-names
                              → hc-infer-types
                              → hc-materialize-bound-exprs
                              → hc-verify-static-shapes
                              → hc-decompose-shaped-values(strict=false)
                              → hc-inline-helpers
                              → hc-materialize-bound-exprs
                              → hc-canonicalize-layouts
                              → hc-shaped-compute-to-generic
                              → hc-elementwise-to-generic
                              → hc-load-store-to-generic
                              → hc-infer-generic-bounds
                              → hc-normalize-scope-regions
                              → hc-lower-kernels-to-gpu-launch
                              → hc-lower-launch-body

Both `-hc-front-fold-region-defs` and `-hc-front-inline` are no-ops
when nothing is marked, so both are safe to keep in the pipeline
unconditionally.

`hc.compile` drives this pipeline through a transform-dialect schedule
shipped as `hc/schedules/front_to_hc.mlir`; `hc-opt` is the CLI handle
on the same pass list, so

    hc-opt --hc-front-fold-region-defs --hc-front-inline \
           --convert-hc-front-to-hc --hc-promote-names \
           --hc-infer-types --hc-materialize-bound-exprs \
           --hc-verify-static-shapes \
           --hc-decompose-shaped-values=strict=false \
           --hc-inline-helpers --hc-materialize-bound-exprs \
           --canonicalize \
           --hc-canonicalize-layouts \
           --hc-shaped-compute-to-generic \
           --hc-elementwise-to-generic \
           --hc-load-store-to-generic \
           --hc-infer-generic-bounds \
           --hc-normalize-scope-regions --canonicalize --cse \
           --hc-lower-kernels-to-gpu-launch \
           --hc-lower-launch-body --canonicalize --cse

and `hc.compile(...)` with the default schedule produce identical
output. See [`doc/schedules.md`](schedules.md) for the schedule format
and override API.

Postcondition: semantic `hc` operations and explicit region structure exist,
name-based bindings have been promoted into SSA, inferable HC types have been
refined, bound symbolic expressions declared by kernel `bound_symbols` have
been materialized as SSA, static tensor/vector shape operands have been verified,
and supported semantic shaped values may have been split into bare data/masks.
The scheduled decomposition is non-strict: helper-call signatures, call sites,
stores, and structured/collective region boundaries are decomposed, while
intrinsic boundaries that are not decomposed yet are preserved with
`builtin.unrealized_conversion_cast`. Symbolic launch parameters may still
remain. Supported helper calls and workitem scope regions are then normalized
away so the executable HC body is closer to per-workitem SPMD form before
upstream lowering. Bound-expression materialization runs a second time after
helper inlining because inlined helper bodies can expose fresh launch-geometry
producer chains; DCE/canonicalization removes the dead scope-token producers
before region normalization checks for remaining live scope-token uses. The
generic-pipeline rewriters (`hc-canonicalize-layouts`,
`hc-shaped-compute-to-generic`, `hc-elementwise-to-generic`,
`hc-load-store-to-generic`, `hc-infer-generic-bounds`) run after the
DCE pair: each one is conservative and only fires on inputs that match
its v0 surface (rank-2 matmul / reduce, all-shaped per-element arith,
pinned `!hc.idx<expr>` indices on load and store) — anything outside
that surface flows through untouched and reaches the per-op handlers in
`hc-lower-launch-body`. `hc-lower-generic` is wired immediately after
`hc-lower-launch-body` and lowers any `hc.generic` whose operands are
already `!hc.ptr` (post-launch-body) to an outer `scf.parallel` over
the parallel iters and an inner `scf.for` nest over reduction iters;
v0 bails on inputs whose per-axis offset arrays haven't been collapsed
to single-entry, so the pass is a no-op for current real workloads.
`hc-flatten-with-layouts` is intentionally NOT in the schedule yet:
its per-access offset composer (`ComposeLoadOffsets` and friends) folds
load/store/vload index lists from rank-N to a single 1D offset, while
the launch-body per-op patterns still expect a rank-N index list to
match the rank-N kernel-arg buffer carrier. That contract gap blocks
wiring flatten until launch-body learns the 1D-index path.
`hc-lower-launch-body` is now memref-free end to end: workgroup-AS
storage lands on `!hc.ptr<workgroup, T>` and kernel-arg loads/stores
go through `hc.ptr_offset` + `hc.ptr_load[_pred]` /
`hc.ptr_store[_pred]` against the `(ptr, dim*, stride*)` tuple
`hc-lower-kernels-to-gpu-launch` plants at the host boundary
([`doc/layouts.md`](layouts.md) covers both contracts). Kernel
lowering then converts each semantic `hc.kernel` into a host
`func.func` with a `gpu.launch`, exposing buffer ABI arguments as
`!hc.ptr<global, T>` rather than memrefs and leaving remaining HC
body operations behind explicit conversion boundaries for the
subsequent upstream-lowering slices. `hc-lower-to-llvm` finishes the
job: `!hc.ptr<global, T>` → `!llvm.ptr<1>` on `gpu.func` and
`func.func` signatures, with an `llvm.addrspacecast` planted at the
host boundary to bridge the generic pointer (returned by
`hc_get_ptr`) to the global pointer the kernel consumes.

### SSA construction

These MLIR passes should:

* eliminate name-based frontend bindings in favor of SSA values,
* lower reassignment, conditional results, and loop-carried state into explicit
  SSA/block-argument/yield structure or equivalent state-carrying ops,
* make captured and region-carried values explicit enough for later scope and
  barrier verification.

Postcondition: no unresolved `hc_front.name`-style bindings remain in semantic
IR, and loop/region-carried state is explicit.

### Semantic inference and verification

These compile-time MLIR passes should:

* infer tensor/vector/scalar result types,
* resolve symbol and shape relationships that depend only on symbolic
  compile-time facts,
* classify captures according to the language rules,
* verify scope legality, capture rules, and barrier placement,
* diagnose non-static vector requirements where required.

Postcondition: the compiler has a semantically well-formed symbolic `hc`
module, but concrete launch values and device caps may still be unknown.

### Specialization

Specialization is driven by the launcher/specialization driver when a concrete
kernel variant is requested. It supplies bound literal symbols and other
specialization-time constants to MLIR.

These MLIR passes should:

* bind literal symbols into specialized variants,
* enforce static vector and layout requirements that become concrete only after
  literal binding,
* infer and verify mask behavior,
* attach and validate layout descriptors,
* run intrinsic verify/infer hooks.

Postcondition: the compiler has a variant-specific `hc` module in which all
literal-dependent requirements are concrete.

### Launch binding and validation

Launch binding and validation are driven by the launcher/specialization driver,
not by a standalone ahead-of-time pass pipeline.

The launcher should:

* bind concrete symbol values from runtime arguments,
* choose the default `group_shape` when omitted,
* provide device limits, capabilities, and any target-specific launch metadata,
* invoke launch validators over the specialized `hc` module plus those
  concrete bindings.

These validators should check:

* symbol consistency checks,
* launch-shape legality,
* subgroup divisibility,
* tensor materialization footprint legality,
* device workgroup and LDS limit checks,
* any target-specific constraints tied to the chosen launch configuration.

Postcondition: the launch is either rejected before execution or a fully bound,
validated kernel instance is ready to lower/execute.

### Lowering to standard and target dialects

After semantic verification, and after any specialization or launch facts
required by a given backend have been supplied, `hc` should lower to a mix of:

* `func.func`
* `arith.*`
* `scf.*`
* other standard dialects as needed,
* target-specific dialects or backend IR.

### End-to-end execution

For the gfx11 WMMA path the lowering chain runs end-to-end today:
`hc.compile(kernel_fn, target="amdgpu-gfx11").invoke(*args)` JITs a host
wrapper that drives the AMDGPU device binary through the bundled HIP
shim, with no external ROCm install on the host. The pieces are:

1. **Schedule + GPU lowering pipeline.** The transform schedule
   (`hc/schedules/front_to_hc.mlir`) lowers `hc_front` to `hc`,
   replaces every `hc.kernel` with a `gpu.launch` (`hc-lower-kernels-to-gpu-launch`)
   wrapping a `func.func` host wrapper, lowers `hc` ops in the launch
   body to upstream dialects (`hc-lower-launch-body`, including the
   cooperative LDS-staged copy for `group.load`), and stamps each
   outlined `gpu.module` with a `#rocdl.target` carrying the resolved
   chip and `target-features` (the wave32 feature is mandatory on
   gfx10+ — without it WMMA silently miscompiles to a wave64 fragment
   layout). The Python driver in `hc/_pipeline.py` then appends a
   fixed device-side chain (`hc-lower-to-llvm` to take the
   `!hc.ptr`/`hc.alloc` family launch-body emits down to LLVM IR
   first, then `convert-amdgpu-to-rocdl`,
   `convert-gpu-to-rocdl`, `gpu-to-llvm`, ...), terminated by:
   * `hc-lower-gpu-to-binary` — runs the LLVM AMDGPU backend on each
     `gpu.module`, links the resulting object with the bundled
     `ld.lld` (resolved Python-side from `_native_paths.lld_path`,
     no external `rocm-llvm` lookup), and replaces the module with a
     `gpu.binary` op holding the HSACO blob.
   * `hc-lower-launch-func-to-runtime` — embeds each HSACO as an
     LLVM private global and rewrites every `gpu.launch_func` into
     paired `hc_rt_load_kernel` (cached, returns an opaque
     `hipFunction_t`) + `hc_rt_launch_kernel` calls. After this pass
     the post-pipeline module is pure LLVM dialect with externs for
     the runtime helpers and HIP shim — no `gpu.binary`, no
     `gpu.launch_func`, no MLIR-side device modules.
2. **Runtime helpers (`libhc_rt_helpers.so`).** The host wrapper
   takes one `PyObject *` per user argument and unpacks each one
   through the `_mlir_ciface_hc_get_ptr` / `_mlir_ciface_hc_get_int64`
   / `_mlir_ciface_hc_get_float64` / `_mlir_ciface_hc_get_dim` /
   `_mlir_ciface_hc_get_stride` helpers. The buffer ABI is the raw
   pointer + per-axis dim/stride trio — no memref descriptor on the
   wire. These helpers borrow the buffer protocol /
   `__cuda_array_interface__` view of the object — they do not
   allocate, copy, or take ownership; the caller keeps the tensor
   alive across the launch.
3. **HIP shim (`libhc_hip_runtime.so`).** Statically dlopens
   `libamdhip64.so` on first use (`hc_rt_init`, mutex-serialized and
   double-checked, idempotent) and exposes `hc_rt_load_kernel` /
   `hc_rt_launch_kernel` against the function-pointer cache it
   populates. There is no link-time dependency on ROCm: the shim
   resolves HIP symbols at runtime and reports a clean error if HIP
   is unavailable.
4. **JIT (`hc.execution_engine.ExecutionEngine`).** `compiled.invoke`
   resolves the runtime helper + HIP shim symbol addresses with
   `ctypes` (no `LoadLibraryPermanently` side effects) and hands them
   to `ExecutionEngineOptions.set_symbol_map` so the JIT'd host
   wrapper binds against the in-process implementations. It then
   `lookup`s `@<kernel_name>` and calls it through a
   `ctypes.CFUNCTYPE(None, py_object * N)` thunk — bypassing the
   upstream MLIR engine's packed-args wrapper entirely. Engine and
   cfunc are cached per `CompiledKernel`, so the second `invoke` on
   the same handle skips JIT.

`tests/test_examples.py::test_gfx11_wmma_example_invokes_on_real_hardware`
is the executable specification of this chain: it compiles the WMMA
matmul example for `amdgpu-gfx11`, runs it through the full stack on a
gfx11 GPU (gated on `HC_RT_RUN_HIP_INVOKE_TEST=1`), and checks the
result against a numpy reference.

### Per-pass IR dumps (`HC_DUMP_PASSES=1`)

Setting `HC_DUMP_PASSES=1` in the environment turns on payload IR
printing across the whole compile, in two surfaces:

* The device-side `PassManager` (`_GPU_LOWERING_PIPELINE` in
  `hc/_pipeline.py`) gets `enable_ir_printing(...)`. This is upstream
  `--mlir-print-ir-after-all`; it dumps to **stderr** with the
  conventional `// ----- // IR Dump After <PassName> ...` headers.
  Catches every device-side pass (`fold-memref-alias-ops`, the
  rocdl-conversion chain, `gpu-to-llvm`, `hc-lower-gpu-to-binary`,
  `hc-lower-launch-func-to-runtime`, ...) and the
  `transform-interpreter` pass itself.
* The transform schedule (`hc/schedules/front_to_hc.mlir`) is
  rewritten in memory before being handed to the interpreter: a
  `transform.print` op is spliced in after every payload-mutating
  transform op (`apply_registered_pass`, `apply_patterns`,
  `apply_cse`, `apply_dce`). The interpreter then prints to
  **stdout** with the `[[[ IR printer: after-<pass> ]]]` headers.
  This works around the fact that the interpreter spawns throwaway
  per-op `PassManager`s inside `apply_registered_pass` that don't
  inherit instrumentation from the parent PM.

Multi-threading is automatically disabled on the compile context
when this knob is on (MLIR's IR printer requires it). Stdout is
where the schedule's probe prints land, so combining
`HC_DUMP_PASSES=1` with `--dump-hc-ir` interleaves the two streams
on stdout — fine for grep, awkward for an editor; redirect to a
file and search.

### Codegen artifacts (`HC_DUMP_DIR=PATH`)

`HC_DUMP_PASSES` covers MLIR-level IR but stops at
`hc-lower-gpu-to-binary` — past that the device chain is C++ inside
the pass: LLVM IR optimization, ISA emission, AMDGPU MC assembly,
ld.lld linking. Setting `HC_DUMP_DIR=/some/path` (the directory
must exist) makes the pass write four artifacts per `gpu.module`:

```
<module>.0-pre-opt.ll      (LLVM IR before the optimizer runs)
<module>.1-post-opt.ll     (LLVM IR after the standard pipeline)
<module>.2-isa.s           (AMDGPU assembly out of translateModuleToISA)
<module>.3-binary.hsaco    (linked HSACO blob — same bytes the runtime loads)
```

The numeric prefix sorts ls/glob output in pipeline order. Pre-opt
gets dumped *before* the optimizer runs so it survives an optimizer
crash — historically the most informative artifact when codegen
miscompiles. The four artifacts are independent; they're useful for
diffing against `roc-obj` / `llvm-objdump` output, comparing
generated assembly to a reference `rocm`-installed compiler, and
re-feeding hand-edited `.ll` / `.s` files into a downstream
toolchain.

Plumbing matches the `ld.lld` path: Python-side substitution into
the pass's `dump-intermediates=` option via the `__HC_DUMP_DIR__`
placeholder, so calling `hc.compile()` from Python is the only
supported way to set it (no env-var lookup happens inside the C++
pass — the pass option is the source of truth).

## MLIR strategy

The fastest implementation path is still to emit textual MLIR.

Reasons:

* easy to implement,
* easy to inspect and diff in tests,
* easy to feed into the normal MLIR parser/verifier,
* avoids large Python builder boilerplate for the initial compiler.

The recommended implementation pattern is to keep the AST visitor independent of
the final emission format by targeting a very small emitter interface. The
initial real emitter may still produce textual `hc_front` directly, while test
emitters record visitor actions without needing the real MLIR dialect
implementation.

However, the generated MLIR should now be treated as the primary compiler IR,
not as a serialization target for a Python semantic IR. The Python frontend may
emit textual `hc_front`, and all substantial compiler work starts once that IR
is parsed by MLIR.

Textual emission is the bootstrap path, not a forever constraint. Later
implementations may switch to builder-based or bytecode-backed construction
without changing the phase boundaries described here.

## Minimum test strategy

Milestone 0 should be backed by:

* unit tests for the AST visitor using a tiny fake recording emitter that
  mirrors the `hc_front` boundary closely enough to validate syntax handling,
  scoping, decorator capture, unsupported-construct diagnostics, and region
  structure,
* golden `hc_front` and `hc` textual MLIR tests that parse and verify in CI,
* a small shared corpus of kernels cross-checked against the simulator in
  `doc/simulator.md` for defined inputs and launch failures,
* explicit tests for source-capture failures and unsupported environments.

The fake emitter should record only frontend-structural events, for example:

* begin/end kernel or helper,
* emit assign/aug-assign, constant/name/call/subscript,
* begin/end `if` or loop structure,
* begin/end subgroup/workitem region,
* emit return / tuple construction / target construction where needed.

A bootstrap fake emitter may still use a narrower event vocabulary such as
`for_range` while the visitor subset is intentionally restricted. The real
`hc_front` dialect should remain the source of truth for the broader
source-faithful operation set above.

It must not become a separately designed frontend IR with its own invariants.
Its purpose is only to test that the AST visitor walks and classifies the
supported Python subset correctly before the real `hc_front` dialect exists or
while its builder/textual APIs are still in flux.

## Dialect strategy

The initial lowering should assume at least two custom dialects:

* `hc_front` for source-faithful AST serialization,
* `hc` for semantic kernel IR.

Standard dialects should be introduced once `hc` is semantically well-formed,
not used as the primary representation of unresolved frontend syntax.

This is a deliberate tradeoff: a slightly larger MLIR stack up front keeps the
compiler logic in one place instead of splitting it between Python and MLIR.

## Lowering of current language features

### Tensors

Tensor syntax should first be emitted as `hc_front` operations without Python
side type interpretation. MLIR legalization should then recognize tensor
constructs and build typed `hc` tensor operations carrying shape, dtype, and
layout semantics.

### Vectors

Vectors lower the same way: frontend emission preserves syntax, while MLIR
passes infer vector types and verify the stronger static requirements:

* logical vector shape must be static,
* layout parameters must be static after specialization,
* vector layout affects physical storage order, not logical semantics.

### Layouts

`index_map(...)` and `as_layout(...)` should be serialized into frontend MLIR
first and interpreted by MLIR legalization/passes. The initial implementation
does not need a large family of dedicated layout ops; layout can live in
attributes and a small number of semantic ops until more structure is needed.

### Masks

Mask syntax should first be preserved in `hc_front`, then inferred and verified
in `hc`. The initial implementation may represent mask behavior through a mix
of:

* explicit `hc.mask` / `hc.with_inactive`-style operations,
* attributes on values/results where appropriate,
* structured lowering of masked loads/stores/reductions.

### Scopes and regions

The current structured execution model maps naturally to region-bearing ops:

* the enclosing `WorkGroup` body remains the top-level kernel region,
* `@group.subgroups` should first lower to `hc_front.subgroup_region` and then
  to `hc.subgroup_region`,
* `@group.workitems` should first lower to `hc_front.workitem_region` and then
  to `hc.workitem_region`.

This keeps AST translation dumb while still preserving the structure needed for
later scope verification and lowering.

The `hc_front` region ops should preserve lexical capture lists so later passes
do not need to rediscover closure structure from arbitrary name use.

### Helper functions

`@kernel.func` definitions should first lower to `hc_front.func`. MLIR passes
may then turn them into `hc.func` symbols, inline them, or eventually lower
them to internal `func.func` symbols. This choice must preserve language
semantics, though compile time, debug information, and diagnostics may differ.

### Intrinsics

`@kernel.intrinsic` definitions cross two tracks that share a name but never
share a body:

* **Simulator fallback** — the function body of `@kernel.intrinsic` is a
  pure-Python implementation the host-side simulator runs unchanged. It
  uses NumPy and the simulator's mask/tensor types and is *never*
  compiled. The fallback body lowers to `hc_front.intrinsic` for IR
  fidelity but its contents are deliberately discarded by
  `ConvertHCFrontToHC`: the resulting `hc.intrinsic` is a declaration-only
  shell carrying signature, scope, effects, and `const_kwargs`.
* **Compiler lowering recipes** — separate Python callables registered via
  `@<intrinsic>.lower(target="...")`. Each recipe constructs a transform
  IR program that rewrites a matching `hc.call_intrinsic` into target-
  specific payload (e.g. `amdgpu.wmma` for gfx11). Recipes ride through the
  pipeline as real MLIR ops, not as serialized strings.

Keeping these tracks separate is the contract that lets the simulator stay
expressive (Python+NumPy, full Python types) while the compiler stays
structural (MLIR transform IR, verifier-checked).

#### Authoring recipes

Each recipe callback receives a builder `t` and a typed call view `call`:

```python
@wmma_gfx11.lower(target="amdgpu-gfx11")
def _lower_wmma(t, call):
    op = t.create(
        "amdgpu.wmma",
        result_types=["vector<8xf32>"],
        operands=[
            call.operand("a_frag.data", expected_type="vector<16xf16>"),
            call.operand("b_frag.data", expected_type="vector<16xf16>"),
            call.operand("acc_frag.data", expected_type="vector<8xf32>"),
        ],
        attrs={"m": t.i32(16), "n": t.i32(16), "k": t.i32(16)},
    )
    return (
        t.cast(op.result(0), to=call.result_type(0)),
        call.operand("acc_frag.mask"),
    )
```

* `call.operand(name_or_index, expected_type=...)` — a runtime SSA
  operand handle. The optional `expected_type=` argument is sugar for
  `t.cast(call.operand(name), to=expected_type)`: it wraps the operand
  in `builtin.unrealized_conversion_cast` to the requested upstream
  type, bridging the bare HC types `hc-lower-launch-body` keeps at
  `hc.call_intrinsic` boundaries to the upstream types target ops like
  `amdgpu.wmma` expect. Naming follows `hc-decompose-shaped-values`:
  every shaped operand splits into `<name>.data` + `<name>.mask`
  channels at the call site once decomposition runs.
* `call.result_type(index)` — the call's result type as a transform type
  parameter (typically a `!hc.bare_*` type post-launch-body).
* `call.attr(name)` — a value of one of the declared `const_kwargs`.
* `t.i32(value)` / `t.i64(value)` — width-annotated integer literal
  helpers (plain Python `int`s default to `i64`; widths matter when the
  target op declares confined integer attributes).
* `t.literal_type("vector<16xf16>")` — materialize a literal MLIR type
  as a recipe-side type handle (deduped by literal text). Strings
  passed to `t.create(result_types=[...])` or `t.cast(..., to=...)`
  are routed through this helper automatically.
* `t.cast(value, to=type)` — bridge a value through
  `builtin.unrealized_conversion_cast` to the target type. The C++
  side skips the cast when source and target types already match, so
  recipes can opportunistically request a type without paying for an
  extra UCC. The `to=` argument accepts either a literal type string or
  a type handle (e.g. `call.result_type(0)` to bridge an upstream
  result back to the call's bare type for replacement).
* `t.create(name, operands=..., result_types=..., attrs=...)` — emit a
  payload op; returns a handle whose `.result(i)` is the replacement for
  the matched call's `i`-th result. `result_types` accepts both
  literal type strings (sugar for `t.literal_type(...)`) and type
  handles from `call.result_type(...)`.
* `t.require_attr(call, name, expected)` — pre-rewrite assertion that a
  named call attribute equals the literal value. Lowers to
  `transform.hc.require_intrinsic_attr` and fails the apply with a
  definite diagnostic on mismatch (so the user sees "wrong arch"
  instead of the generic "no recipe matched").
* The callback returns the SSA values that replace the matched call —
  one value per post-decomposition result. Tuple/list returns become
  multi-result replacements; a single value or `CreatedOpHandle`
  unwraps as appropriate.

Operand-shape and attribute-value validation falls out of the target op's
own verifier: when the recipe constructs `amdgpu.wmma`, the upstream
verifier rejects mismatched vector lengths or out-of-range `m`/`n`/`k`.

##### Bare ↔ upstream bridging

`hc-lower-launch-body` keeps `!hc.bare_vector` / `!hc.bare_tensor`
operands across every `hc.call_intrinsic` boundary, planting paired
`builtin.unrealized_conversion_cast` ops on either side that bridge the
surrounding upstream-typed values into and out of the call. Target
payload ops like `amdgpu.wmma` reject bare types — they want plain
`vector<NxF>` — so recipes use `expected_type=` / `t.cast` to insert a
matching pair around the freshly created op:

```text
%upstream_a = ...                                           // surrounding upstream values
%bare_a = unrealized_conversion_cast %upstream_a            // launch-body planted
%bare_a_up = unrealized_conversion_cast %bare_a             // recipe-inserted by expected_type
%out = amdgpu.wmma %bare_a_up, %bare_b_up, %bare_acc_up
%bare_out = unrealized_conversion_cast %out                 // recipe-inserted by t.cast
%upstream_out = unrealized_conversion_cast %bare_out        // launch-body planted
```

The recipe-inserted casts pair with the launch-body casts, so the
round-trip `upstream → bare → upstream` chain on each side collapses to
identity. A `--canonicalize` pass after `-hc-interpret-intrinsic-recipes`
folds every UCC away, leaving `amdgpu.wmma` between plain upstream
vectors with no leftover bridging machinery. Recipes that target
already-upstream-typed operands (no surrounding bare types) pay nothing:
`cast_value` short-circuits when source and target types already match.

#### Generic HC transform ops

The recipe layer compiles the Python callback into a stable set of HC
transform ops (defined in `hc/include/hc/TransformOps/HCTransformOps.td`):

* `transform.hc.match_intrinsic_call %root @callee target = "..."` — walk
  payload and surface every `hc.call_intrinsic` calling `@callee`.
* `transform.hc.get_intrinsic_operand`,
  `transform.hc.get_intrinsic_result_type`,
  `transform.hc.get_intrinsic_attr` — thread call operands, result types,
  and constant kwargs into transform handles.
* `transform.hc.require_intrinsic_attr %call {expected = ..., name =
  "..."}` — fail the apply (with a pinpointed diagnostic) when the call's
  named attribute doesn't equal the literal. Used for pre-rewrite
  invariant checks the target op itself can't express.
* `transform.hc.constant_type <type-attr> : !transform.type` —
  materialize a literal MLIR type as a transform parameter handle.
  Recipes pair this with `cast_value` and `create_op` to assert
  upstream types without going through a payload-derived handle.
* `transform.hc.cast_value %value to %type` — insert a
  `builtin.unrealized_conversion_cast` from the source value to the
  target type at the value's definition site, returning a value handle
  to the cast result. No-op when source/target types already match.
* `transform.hc.create_op "<op_name>" at %call (operands) result_types(types)
  dynamic_attrs [names](values) static_attrs = {...}` — emit a payload op
  at the call site; consumes operand/type/attr handles.
* `transform.hc.replace_intrinsic_call %call with %values` — RAUW the
  matched call with the freshly created payload values.

The transform op surface is intentionally narrow. Anything more
target-specific lives in the recipe body, not in the op set, so adding a
new target requires a new recipe rather than new ops.

#### Embedding and lifecycle

The frontend emitter packs every registered recipe into a sibling
top-level module:

```mlir
module {
  hc_front.kernel "..." { ... }
  hc_front.intrinsic "wmma_gfx11" attributes {...} { ... }

  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    transform.named_sequence @__hc_lower_wmma_gfx11_amdgpu_gfx11(
        %arg0: !transform.any_op) attributes {hc.target = "amdgpu-gfx11"} {
      // match_intrinsic_call → get_intrinsic_* → create_op → replace
    }
  }
}
```

The conversion pass (`ConvertHCFrontToHC`) only collects `hc_front.*`
ops, so the lowerings module passes through untouched and the named
sequences travel alongside the kernel into the `hc` dialect.

#### Pipeline placement

`-hc-interpret-intrinsic-recipes` is the consumer. It runs *after*
`-hc-lower-launch-body`, when scalar, vector, ptr, and mask types have
been materialized — the recipe authors see the same operand types the
target op expects:

1. find sibling `module @__hc_intrinsic_lowerings__`;
2. select named sequences by `hc.target` attribute (the pass's `target`
   option); empty target runs every sequence and is mostly useful for
   tests;
3. invoke `transform::applyTransformNamedSequence` on each;
4. erase the spent lowerings module so downstream passes don't trip over
   stray transform IR;
5. walk the payload one final time and emit a clear `no intrinsic
   lowering recipe matched @<callee> for target '<t>'` diagnostic for
   every surviving `hc.call_intrinsic`. Silent passthrough would just
   relocate the gap to the next pass;
6. sweep top-level `hc.intrinsic` declarations whose last call site was
   rewritten so the post-interpretation IR is free of stray HC ops
   without forcing every caller to run a separate symbol-DCE pass.

Earlier `hc` passes (verify/infer hooks, type inference, decomposition,
launch wrapping) operate on the structural `hc.intrinsic` *declaration*
without touching the recipes — the recipes describe the target rewriting
contract, not how to type-check the intrinsic itself.

## Recommended first implementation order

### Milestone 0: fixed-shape straight-line WorkGroup kernels

Implement:

* source capture for supported `.py`-defined kernels,
* AST parsing for straight-line kernels,
* textual `hc_front` emission,
* parsing/verifying that frontend MLIR,
* `hc_front` to `hc` legalization for assignments, calls, and returns,
* SSA construction for straight-line blocks,
* minimal `hc` typing plus explicit loads/stores and simple arithmetic on a
  narrow kernel family.

Milestone 0 is intentionally smaller than the pairwise-distance example below.
It does not yet require reductions, rich broadcasting, or the full NumPy
surface from `doc/langref.md`.

### Milestone 1: structured control flow

Add:

* `if`
* `for range(...)`
* block arguments / yields or equivalent explicit state-carrying structure,
* `hc.subgroup_region`
* `hc.workitem_region`

### Milestone 2: masks and layouts

Add:

* reductions and richer NumPy surface needed by the motivating workgroup
  examples,
* explicit mask propagation,
* `with_inactive`,
* `tensor.mask` / `vector.mask`,
* `index_map(...)`,
* `as_layout(...)`

### Milestone 3: intrinsics and helpers

Add:

* `@kernel.func`
* `@kernel.intrinsic`
* target-specific lowering hooks
* verify/infer hooks

## Example lowering shape

For a source like the workgroup pairwise-distance kernel from `doc/langref.md`:

```python
@kernel(work_shape=(W1, W2))
def pairwise_distance_kernel(group: CurrentGroup,
                             X1: Buffer[W1, H],
                             X2: Buffer[W2, H],
                             D: Buffer[W1, W2]):
    gid = group.work_offset
    x1 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))
    x2 = group.load(X2[gid[1]:], shape=(group.shape[1], X2.shape[1]))
    diff = ((x1[None, :, :] - x2[:, None, :])**2).sum(axis=2)
    group.store(D[gid[0]:, gid[1]:], np.sqrt(diff))
```

This example is intentionally beyond Milestone 0. It becomes a target once
reductions and a richer NumPy surface are available.

The initial compiler should aim to produce:

* an `hc_front` form that preserves source structure, names, and unresolved
  calls closely,
* a semantic `hc` form that still preserves:

  * explicit workgroup-local loads,
  * explicit logical tensor operations,
  * reduction structure,
  * explicit store,
  * enough shape/layout metadata for later lowering.

Milestone 0 does not require immediate lowering to the final target dialect,
but it does require a working `hc_front` to `hc` path and a correct and
inspectable semantic MLIR representation.

## Rationale

The main reason to choose this path is implementation speed.

Compared to a Python-side semantic IR, this approach:

* avoids duplicating type inference and semantic checks in Python and MLIR,
* keeps non-trivial transformations in the IR that will remain long-term,
* lets verification, canonicalization, and lowering share the same
  representation.

Compared to a full Python compiler, this approach:

* supports source-structured diagnostics,
* keeps control-flow semantics explicit,
* avoids dependence on tracing tricks for correctness,
* but still keeps the compiler small by limiting the accepted syntax and using
  textual MLIR emission.

Compared to a pure tracer-first design, this approach:

* gives earlier and clearer syntax/semantic errors,
* makes region/scoping rules explicit in the frontend,
* avoids having the runtime execution model define the compiler capture model.

## Open questions

This document intentionally leaves a few issues open for later refinement:

* how much mask/layout information should eventually live in `hc` types versus
  attributes or dedicated ops once the initial phase boundaries stabilize,
* whether helper functions should lower as internal functions or always inline
  once debug and ABI requirements are clearer,
* whether some structured region ops should eventually lower directly to
  standard dialect regions,
* whether production implementations should eventually replace textual emission
  with builder-based or bytecode-backed construction while preserving the same
  phase structure.
