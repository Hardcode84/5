# RFC: Lowering pipeline for high-level GPU kernel API

## Summary

This document describes an MLIR-first lowering path for the high-level kernel
API defined in `doc/langref.md`.

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
* `HC_ShapedValueType` — ops that only make sense on tensors/vectors
  (`hc.matmul`, `hc.reduce`, `hc.vec`, `hc.with_inactive`, `hc.as_layout`,
  `hc.store`'s `$source`)
* `HC_BufferValueType` — buffer handles for `hc.buffer_dim`, `hc.load`;
  excludes everything non-buffer
* `HC_BufferOrTensorValueType` — destinations/sources that accept either
  (`hc.buffer_view.$buffer`, `hc.vload.$source`, `hc.store.$dest`)
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
* `#hc.layout<...>` — reserved for `index_map(...)` descriptors; unused in v0

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
  plus symbolic launch geometry, parameter annotations, and literal-symbol
  set attributes. `Symbol` trait so references go through the symbol table.
* `hc.func @name` — helper callable referenced by `hc.call`. Also a
  `Symbol`. An optional inline `function_type` signature makes the op
  self-describing; when present, `hc.call` sites get arity and type
  parity checked at verify time (`!hc.undef` on either side is a
  progressive-typing wildcard).
* `hc.intrinsic @name` — intrinsic declaration carrying `scope = #hc.scope<...>`
  and an optional `effects = #hc.effects<...>`; its body region holds the
  fallback implementation (may be empty, in which case lowering patterns
  handle the op directly). Also a `Symbol`, with the same optional
  `function_type` signature story as `hc.func`. An optional
  `const_kwargs = ["wave_size", "arch", ...]` list declares specialization
  attributes every `hc.call_intrinsic` must carry; missing entries fail at
  verify time. Extra attributes on the call site stay allowed so targets
  can attach their own decorations without modifying the declaration.
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
  `!hc.idx<"_symN">` with fresh per-launch symbols at inference time).
  Marked `Pure` under a **pipeline invariant** (see below); duplicate reads
  of the same axis are therefore interchangeable and CSE/DCE may freely
  collapse or drop them.

There is no `!hc.index_tuple` and no `hc.getindex`; geometry ops return
independent SSA components.

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
* `hc.buffer_view %buf[%idx...]` — sub-view of a buffer or tensor with the
  slice-reduced shape, for cases that do not require data movement
* `hc.getitem %base[%idx...]` — generic Python square-bracket indexing kept
  before the base kind is known. Inference can refine tuple item access when
  the index converges to a concrete integer; later passes may specialize
  buffer/tensor cases into view or data-movement ops.

Multi-axis subscripts on `hc.getitem`/`hc.load`/`hc.store`/`hc.buffer_view`
take variadic operands directly; there is no separate tuple-construction op.

#### Data movement

Rule of thumb for memory effects in this dialect: n-ary tensor/vector
**math** (`hc.matmul`, `hc.reduce`, generic arithmetic, ...) is modeled as
value-semantic and carries no effect (`Pure`); only ops that materialize or
observe **addressable workgroup state** — `hc.load`, `hc.vload`, `hc.store`,
`hc.vec` — carry memory effects. Allocators carry `MemAlloc`; opaque calls
carry the conservative `MemRead + MemWrite` pair until `hc.func` /
`hc.intrinsic` grow per-target effect annotations.

* `hc.load %buf[%idx...] {shape = #hc.shape<[...]>}` — buffer → tensor.
  Carries a `MemRead` effect. The optional shape, when present, must have
  the same rank as the index list.
* `hc.vload %src[%idx...] {shape = #hc.shape<[...]>}` — buffer/tensor →
  vector. Same effect/verifier shape as `hc.load`.
* `hc.store %dst[%idx...], %src` — tensor or vector source; carries a
  `MemWrite` effect.
* `hc.vec %t` — tensor → vector materialization. Carries a `MemRead` effect
  because the source tensor is addressable memory; an interleaved store can
  change what two sibling `hc.vec` ops observe, so CSE must not collapse
  them blindly.
* `hc.with_inactive %v {inactive = T}` — replace inactive elements. The
  `inactive` payload is a typed scalar literal (int/float/bool); once
  inference pins the operand to a shaped type its numeric domain must
  match the element type.
* `hc.as_layout %v, layout = row_major | col_major` — change layout.
  The `layout` payload is a typed `#hc<layout ...>` enum, so garbage
  spellings fail at parse. v0 admits `row_major` and `col_major`; later
  schemes can extend the enum without changing call sites.
* `hc.vzeros`, `hc.vones`, `hc.vfull` — vector allocators (any scope).
* `hc.zeros`, `hc.ones`, `hc.full` — tensor allocators (`WorkGroup` scope
  only).
* `hc.empty` — uninitialized tensor (`WorkGroup` scope).

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
(mostly `!hc.undef`, with trivially seeded group/buffer parameters, generic
ops, no inference), and after type / symbol inference.

#### After `hc_front → hc` (mechanical, mostly `!hc.undef`)

```mlir
hc.kernel @tiled_gfx11_wmma_matmul(
  %group: !hc.group<work_shape = #hc.shape<["ceil_div(M, 16) * 32", "ceil_div(N, 16)"]>,
                    group_shape = #hc.shape<["32", "1"]>,
                    subgroup_size = 32 : i32>,
  %a: !hc.buffer<!hc.undef, ["M", "K"]>,
  %b: !hc.buffer<!hc.undef, ["K", "N"]>,
  %c: !hc.buffer<!hc.undef, ["M", "N"]>) attributes {
  work_shape    = #hc.shape<["ceil_div(M, 16) * 32", "ceil_div(N, 16)"]>,
  group_shape   = #hc.shape<["32", "1"]>,
  subgroup_size = 32 : i32,
  literals      = ["WMMA_M", "WMMA_N", "WMMA_K", "WAVE_LANES"]
} {
  %c0 = hc.const <0 : i64>  : !hc.undef
  %WM = hc.const <16 : i64> : !hc.undef
  %WK = hc.const <16 : i64> : !hc.undef

  %gr, %gc = hc.group_id %group       : (!hc.group<...>) -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
  %row0    = hc.mul %gr, %WM           : !hc.undef
  %col0    = hc.mul %gc, %WM           : !hc.undef

  %a_k     = hc.buffer_dim %a, axis = 1 : !hc.buffer<!hc.undef, ["M", "K"]> -> !hc.undef
  %acc0    = hc.call @init_wmma_acc(%group) : (!hc.group<...>) -> !hc.undef

  %acc_final = hc.for_range %k0 = %c0 to %a_k step %WK
                            iter_args (%acc = %acc0) : !hc.undef {
    %row_hi = hc.add %row0, %WM : !hc.undef
    %col_hi = hc.add %k0,   %WK : !hc.undef
    %row_sl = hc.slice_expr(lower = %row0 upper = %row_hi)
                  : (!hc.undef, !hc.undef) -> !hc.undef
    %col_sl = hc.slice_expr(lower = %k0 upper = %col_hi)
                  : (!hc.undef, !hc.undef) -> !hc.undef
    %a_tile = hc.load %a[%row_sl, %col_sl]
                     {shape = #hc.shape<["WMMA_M", "WMMA_K"]>}
              : (!hc.undef, !hc.undef, !hc.undef) -> !hc.undef
    // …analogous slice + load for b_tile…
    %acc1   = hc.call @issue_wmma_tile(%group, %a_tile, %b_tile, %acc)
              : (!hc.undef, !hc.undef, !hc.undef, !hc.undef) -> !hc.undef
    hc.yield %acc1 : !hc.undef
  }

  hc.call @store_wmma_tile(%group, %c, %row0, %col0, %acc_final)
      : (!hc.undef, !hc.undef, !hc.undef, !hc.undef, !hc.undef) -> ()
}
```

Every op is structural, no semantic checks yet. The pass is a tree rewrite
over `hc_front` with no per-op inference work.

#### After type / symbol inference

```mlir
hc.intrinsic @wmma_gfx11 scope = #hc.scope<"WorkItem">
    effects = #hc.effects<"Pure"> {}
// Specialization kwargs (`wave_size`, `arch`, ...) travel on the matching
// `hc.call_intrinsic` sites as op attributes; a typed kwarg-descriptor
// attribute on `hc.intrinsic` itself is a follow-up.

hc.func @init_wmma_acc   attributes {scope = #hc.scope<"WorkGroup">} (...)
hc.func @issue_wmma_tile attributes {scope = #hc.scope<"WorkGroup">} (...)
hc.func @store_wmma_tile attributes {scope = #hc.scope<"WorkGroup">} (...)

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
                 : (!hc.buffer<f16, #hc.shape<["M","K"]>>)
                -> !hc.vector<f32, #hc.shape<["WMMA_ACC_FRAGMENT","32","1"]>>

  %acc_final = hc.for_range %k0 = %c0 to %a_k step %WK    // %k0 : !hc.idx<"_k0">
                            iter_args (%acc = %acc0) {
    %row_hi = hc.add %row0, %WM            // : !hc.idx<"_gid0 * 16 + 16">
    %col_hi = hc.add %k0,   %WK            // : !hc.idx<"_k0 + 16">
    %row_sl = hc.slice_expr(lower = %row0 upper = %row_hi) : !hc.slice
    %col_sl = hc.slice_expr(lower = %k0   upper = %col_hi) : !hc.slice
    %a_tile = hc.load %a[%row_sl, %col_sl]
                {shape = #hc.shape<["WMMA_M", "WMMA_K"]>}
              : !hc.tensor<f16, #hc.shape<["WMMA_M","WMMA_K"]>>
    // ...
    %acc1 = hc.call @issue_wmma_tile(%group, %a_tile, %b_tile, %acc)
    hc.yield %acc1
  }

  hc.call @store_wmma_tile(%group, %c, %row0, %col0, %acc_final)
}
```

Same op shape, refined types. The `%row_hi - %row0 = 16` kind of derivation
used by `hc.load` shape verification is now a plain ixsimpl query on the
typed expressions.

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
`pipeline_diagnostics`. Invoking the compiled handle launches — until
specialization and launch stages land, invocation raises
`NotImplementedError`. Partial `symbols` maps are legal; unbound literals
stay symbolic and later stages refine them. Bindings are recorded on the
handle for later stages to consume.

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

Both `-hc-front-fold-region-defs` and `-hc-front-inline` are no-ops
when nothing is marked, so both are safe to keep in the pipeline
unconditionally.

`hc.compile` drives this pipeline through a transform-dialect schedule
shipped as `hc/schedules/front_to_hc.mlir`; `hc-opt` is the CLI handle
on the same pass list, so

    hc-opt --hc-front-fold-region-defs --hc-front-inline \
           --convert-hc-front-to-hc --hc-promote-names

and `hc.compile(...)` with the default schedule produce identical
output. See [`doc/schedules.md`](schedules.md) for the schedule format
and override API.

Postcondition: semantic `hc` operations and explicit region structure exist, but
name-based bindings, partially unknown types, and symbolic launch parameters may
still remain.

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

`@kernel.intrinsic` definitions should first lower to `hc_front.intrinsic`.
Later MLIR passes should:

* apply verify/infer hooks,
* lower intrinsic calls to `hc.intrinsic.call` or directly to target-specific
  IR when appropriate,
* preserve the language-visible semantics of the intrinsic contract and
  fallback body when one is present,
* otherwise lower the fallback body as ordinary kernel code.

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
