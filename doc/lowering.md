<!--
SPDX-FileCopyrightText: 2024 The HC Authors
SPDX-FileCopyrightText: 2025 The HC Authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# RFC: Lowering pipeline for high level GPU kernel API

Alexander Kalistratov, Ivan Butygin

## Summary

This document describes an MLIR-first lowering path for the high level kernel
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
4. perform type inference, scope/layout/mask verification, specialization, and
   launch/resource validation in MLIR,
5. lower to standard MLIR dialects and backend-specific IR.

## Goals

The lowering path should:

* preserve source structure well enough to support useful diagnostics,
* handle the structured control-flow model from `langref.md`,
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

Frontend capture may be implemented with:

* `inspect.getsource(fn)` to recover the source of kernel/helper/intrinsic
  functions,
* `ast.parse(...)` to build a Python AST,
* a structured AST visitor that accepts only the supported subset and emits
  frontend MLIR.

Unsupported constructs are compile errors.

The Python frontend should stay as dumb as practical. Its job is to:

* recover source and source locations,
* parse the supported AST subset,
* preserve names, literals, annotations, decorators, and region structure,
* serialize that structure into MLIR.

It should not do type inference, shape inference, mask/layout reasoning,
specialization, capture analysis, resource checks, or any other non-trivial
compiler transformation.

## Supported AST subset

The initial subset should include only what the current language requires:

* `FunctionDef`
* `Assign`
* `Return`
* `Expr`
* `If`
* `For` over `range(...)`
* `Call`
* `Name`
* `Constant`
* `Tuple` / `List` literals where needed
* `BinOp`, `UnaryOp`, `Compare`
* `Subscript`
* `Attribute`
* nested `FunctionDef` bodies used by `@group.subgroups` and
  `@group.workitems`

Everything else should be rejected explicitly.

This keeps the frontend small while still covering the current language design.

## Frontend dialect

The first IR form should be MLIR, not a Python semantic IR. This document uses
`hc.front` as the placeholder name for the frontend dialect.

`hc.front` should be source-faithful and intentionally low intelligence. It is
not the semantic source of truth for the language; it is a serialized AST-like
form that feeds the real MLIR compiler pipeline.

### Design principles for `hc.front`

`hc.front` should:

* preserve source-level structure closely,
* keep source locations on every op,
* preserve unresolved names and calls where possible,
* avoid doing semantic interpretation in Python,
* be easy to emit as textual MLIR.

### Suggested `hc.front` operation set

The exact set can evolve, but an initial frontend dialect may include:

* `hc.front.kernel`
* `hc.front.func`
* `hc.front.intrinsic`
* `hc.front.constant`
* `hc.front.name`
* `hc.front.assign`
* `hc.front.attr`
* `hc.front.subscript`
* `hc.front.call`
* `hc.front.tuple`
* `hc.front.if`
* `hc.front.for_range`
* `hc.front.return`
* `hc.front.subgroup_region`
* `hc.front.workitem_region`

### Frontend type strategy

To keep the AST translation layer dumb, `hc.front` may use a very small set of
opaque carrier types such as:

* `!hc.front.value` for expression results,
* `!hc.front.typeexpr` for annotation/type syntax when needed.

Names, decorators, annotations, and literal syntax can remain as attributes or
frontend ops until MLIR legalization resolves them.

## Semantic dialect

The real compiler IR should be a separate MLIR dialect. This document uses `hc`
as the placeholder name for that dialect.

`hc` is where the language becomes semantic rather than syntactic. It should
own:

* typed values and operations,
* scope and region semantics,
* tensor/vector/buffer semantics,
* mask and layout semantics,
* intrinsic contracts,
* specialization and launch/resource validation hooks.

### Suggested `hc` type strategy

The initial semantic dialect may use:

* `!hc.buffer<...>`
* `!hc.tensor<...>`
* `!hc.vector<...>`
* builtin scalar/index types where appropriate

Mask and layout information may initially live partly in dedicated ops and
attributes rather than being forced fully into types.

### Suggested `hc` operation set

An initial set may include:

* `hc.kernel`
* `hc.func`
* `hc.tensor.alloc`
* `hc.vector.alloc`
* `hc.load`
* `hc.vload`
* `hc.store`
* `hc.mask`
* `hc.with_inactive`
* `hc.as_layout`
* `hc.subgroup_region`
* `hc.workitem_region`
* `hc.intrinsic.call`

## MLIR pass pipeline

The current language has multiple classes of validation and transformation.
Those should happen as MLIR passes and verifiers, not as Python IR logic.

### Frontend emission

The Python side should only:

* reject unsupported syntax,
* preserve source structure,
* emit `hc.front` IR.

### `hc.front` to `hc` legalization

The first real compiler stage should:

* resolve decorators and annotations into semantic form,
* translate name-based frontend ops into SSA-based semantic ops,
* recognize DSL constructs such as `group.load`, `group.vload`,
  `with_inactive`, `as_layout`, and region declarations,
* build semantic `hc` operations.

### Type, shape, and scope inference

These MLIR passes should:

* infer tensor/vector/scalar result types,
* resolve symbol and shape relationships,
* verify scope legality, capture rules, and barrier placement,
* diagnose non-static vector requirements where required.

### Mask, layout, and specialization passes

These MLIR passes should:

* infer and verify mask behavior,
* attach and validate layout descriptors,
* bind literal symbols into specialized variants,
* run intrinsic verify/infer hooks.

### Launch and resource validation

These MLIR passes should happen after symbol binding and default `group_shape`
selection but before execution:

* symbol consistency checks,
* launch-shape legality,
* subgroup divisibility,
* tensor materialization footprint legality,
* device workgroup and LDS limit checks,
* any target-specific constraints tied to the chosen launch configuration.

### Lowering to standard and target dialects

After semantic verification, `hc` should lower to a mix of:

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

However, the generated MLIR should now be treated as the primary compiler IR,
not as a serialization target for a Python semantic IR. The Python frontend may
emit textual `hc.front`, and all substantial compiler work starts once that IR
is parsed by MLIR.

## Dialect strategy

The initial lowering should assume at least two custom dialects:

* `hc.front` for source-faithful AST serialization,
* `hc` for semantic kernel IR.

Standard dialects should be introduced once `hc` is semantically well-formed,
not used as the primary representation of unresolved frontend syntax.

This is a deliberate tradeoff: a slightly larger MLIR stack up front keeps the
compiler logic in one place instead of splitting it between Python and MLIR.

## Lowering of current language features

### Tensors

Tensor syntax should first be emitted as `hc.front` operations without Python
side type interpretation. MLIR legalization should then recognize tensor
constructs and build typed `hc` tensor operations carrying shape, dtype, and
layout semantics.

### Vectors

Vectors lower the same way: frontend emission preserves syntax, while MLIR
passes infer vector types and verify the stronger static requirements:

* logical vector shape must be static,
* layout parameters must be static after specialization,
* vector layout affects carrier order, not logical semantics.

### Layouts

`index_map(...)` and `as_layout(...)` should be serialized into frontend MLIR
first and interpreted by MLIR legalization/passes. The initial implementation
does not need a large family of dedicated layout ops; layout can live in
attributes and a small number of semantic ops until more structure is needed.

### Masks

Mask syntax should first be preserved in `hc.front`, then inferred and verified
in `hc`. The initial implementation may represent mask behavior through a mix
of:

* explicit `hc.mask` / `hc.with_inactive`-style operations,
* attributes on values/results where appropriate,
* structured lowering of masked loads/stores/reductions.

### Scopes and regions

The current structured execution model maps naturally to region-bearing ops:

* WorkGroup remains the enclosing kernel body,
* `@group.subgroups` should first lower to `hc.front.subgroup_region` and then
  to `hc.subgroup_region`,
* `@group.workitems` should first lower to `hc.front.workitem_region` and then
  to `hc.workitem_region`.

This keeps AST translation dumb while still preserving the structure needed for
later scope verification and lowering.

### Helper functions

`@kernel.func` definitions should first lower to `hc.front.func`. MLIR passes
may then turn them into `hc.func` symbols, inline them, or eventually lower
them to internal `func.func` symbols. This choice should not be observable.

### Intrinsics

`@kernel.intrinsic` definitions should first lower to `hc.front.intrinsic`.
Later MLIR passes should:

* apply verify/infer hooks,
* lower intrinsic calls to `hc.intrinsic.call` or directly to target-specific
  IR when appropriate,
* otherwise lower the fallback body as ordinary kernel code.

## Recommended first implementation order

### Milestone 0: straight-line workgroup kernels

Implement:

* AST parsing for straight-line kernels,
* textual `hc.front` emission,
* parsing/verifying that frontend MLIR,
* `hc.front` to `hc` legalization for assignments, calls, and returns,
* minimal `hc` typing plus tensor/vector loads and stores.

### Milestone 1: structured control flow

Add:

* `if`
* `for range(...)`
* `hc.subgroup_region`
* `hc.workitem_region`

### Milestone 2: masks and layouts

Add:

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

For a source like:

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

the initial compiler should aim to produce:

* an `hc.front` form that preserves source structure, names, and unresolved
  calls closely,
* a semantic `hc` form that still preserves:

  * explicit workgroup-local loads,
  * explicit logical tensor operations,
  * reduction structure,
  * explicit store,
  * enough shape/layout metadata for later lowering.

The first milestone does not require immediate lowering to the final target
dialect, but it does require a working `hc.front` to `hc` path and a correct
and inspectable semantic MLIR representation.

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

* exact split of responsibility between `hc.front` and `hc`,
* how much mask/layout information should eventually live in `hc` types versus
  attributes or dedicated ops,
* which validations should be verifier logic versus dedicated analysis passes,
* whether helper functions should lower as internal functions or always inline,
* whether some structured region ops should eventually lower directly to
  standard dialect regions.
