<!--
SPDX-FileCopyrightText: 2024 The HC Authors
SPDX-FileCopyrightText: 2025 The HC Authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# RFC: Host-side Python simulator for high level GPU kernel API

Alexander Kalistratov, Ivan Butygin

## Summary

This document describes a pure Python simulator for the high level kernel API
defined in `doc/langref.md`.

The simulator runs kernels on the host in the Python interpreter without MLIR
lowering, code generation, or target compilation. Its purpose is to provide a
reference executor for language semantics, debugging, and functional testing.

The recommended initial path is:

1. bind symbols and choose launch geometry on the host,
2. construct host runtime objects for buffers, tensors, vectors, masks,
   layouts, and scope objects,
3. execute kernel, helper, and intrinsic fallback bodies directly as ordinary
   Python code,
4. map `@group.subgroups` and `@group.workitems` to host-side region runners,
5. emulate `group.barrier()` with coroutine scheduling, for example via
   `greenlet`.

This simulator is intentionally not a second compiler. It should model the
language semantics directly and stay entirely in Python.

## Goals

The simulator should:

* execute the same kernel definitions described by `doc/langref.md`,
* run entirely on the host in Python,
* avoid MLIR emission, code generation, and device compilation,
* minimize simulator-specific frontend machinery,
* provide deterministic, debuggable behavior,
* make semantic errors visible early and clearly,
* serve as a reference implementation for tests and examples.

## Non-goals

The simulator does not try to:

* model hardware performance, occupancy, or scheduling fidelity,
* reproduce target-specific memory banking or instruction timing,
* statically validate every source-level portability rule from the compiled
  path,
* replace the MLIR lowering path,
* optimize kernels.

## Overall Model

The simulator is a reference interpreter for the kernel DSL, not a tracing
runtime and not a compiler pipeline.

It should accept the same decorated kernel definitions as the compiled path,
but instead of lowering them to MLIR it executes them directly in a host-side
runtime.

The runtime consists of:

* a launch/runtime layer that binds symbols and manages execution scopes,
* Python objects that model buffers, tensors, vectors, layouts, masks, and
  scope objects,
* a scheduler that executes workgroups, subgroups, and workitems in a
  deterministic order,
* a coroutine mechanism for workitem barriers.

The simulator should call the original Python kernel bodies directly. Tensor,
vector, mask, and scope semantics come from the runtime objects passed into
those bodies, while workitem barriers are handled by host-side coroutine
scheduling.

## Relationship to Lowering

The simulator and the MLIR lowering path are parallel implementations of the
same language contract.

The lowering path in `doc/lowering.md` exists to compile kernels. The simulator
exists to execute them directly on the host. The simulator should not depend on
MLIR or reuse compiled IR as an execution format.

Both paths should agree on:

* supported source constructs,
* symbol binding rules,
* scope legality,
* tensor/vector semantics,
* mask and layout semantics,
* intrinsic fallback behavior.

The two paths need not use the same frontend mechanism. The compiled path may
parse and lower source, while the simulator may execute Python bodies directly.

## Portability Boundary

The simulator does not need AST parsing as part of its core design.

Kernel, helper, and intrinsic fallback bodies should execute as ordinary Python
functions against simulator runtime objects. This keeps the simulator small and
avoids building a second frontend.

As a consequence, the simulator may successfully run some Python constructs that
the compiled path does not support or that `langref.md` treats as ill-formed.
Such cases are non-portable. The portability contract is:

* programs that are legal under `langref.md` should behave consistently in the
  simulator and in the compiled path,
* programs that only happen to run in the simulator are outside the portable
  language contract.

If a project wants stricter portability checking, that should be provided by a
separate validator or compiler-side diagnostic pass, not by the simulator core.

## Launch and Target Model

The simulator should follow the same launch contract as `langref.md`.

### Symbol Binding

At launch, the simulator binds symbols from:

* buffer shapes,
* tuple-typed kernel arguments,
* literal symbols,
* explicit `work_shape`, `group_shape`, and `subgroup_size` declarations.

Launch fails before execution if:

* a required symbol is missing,
* bindings for the same symbol disagree,
* an explicit launch parameter conflicts with argument shapes,
* a constant-required context uses a non-literal symbol.

### Literal Symbols

Literal symbols change compilation policy in the compiled path, but they do not
change language semantics.

Because the simulator does no kernel compilation, it should not create compiled
variants for literal symbols. Instead, it should treat their bound values as
ordinary launch-time constants and use them to satisfy constant-required
contexts such as vector shapes and constant attributes.

The simulator may cache launch metadata or other host runtime metadata keyed by
literal values when useful, but this is an implementation detail and must not
be observable as specialization.

### Launch Geometry

The simulator must enforce the same launch rules:

* launch-shape values must be non-negative integers,
* negative launch-shape values are launch-time errors,
* zero-sized `work_shape` is legal and results in a no-op launch,
* non-divisible `work_shape / group_shape` is legal and rounds up the number of
  workgroups,
* `subgroup_size` must be constant or literal,
* `L0 % subgroup_size == 0` is required when subgroup execution is used.

### Simulator Target

Even though the simulator runs on the host, it still needs a target profile for
launch validation. The simulator should therefore execute against a configurable
`SimulatorTarget` object that provides:

* a deterministic default `group_shape` policy,
* subgroup support and default subgroup size behavior if needed,
* optional workgroup-size and LDS limits for validation,
* any other implementation-defined limits that the language exposes at launch.

If `group_shape` is omitted, the simulator chooses exactly one deterministic
default from its `SimulatorTarget` and validates it like an explicit
`group_shape`. It must not search multiple alternatives.

The default simulator target should prioritize simplicity and determinism over
hardware fidelity.

## Runtime Values

The simulator should represent language-visible values as explicit host-side
runtime objects.

### Buffers

Kernel buffer arguments should initially be backed by `numpy.ndarray`.

Buffers are persistent storage owned by the caller. They are not workgroup-local
and are not copied on kernel entry. Buffer slicing and indexing should follow
the same logical rules as in `langref.md`.

### Tensors

Workgroup-local tensors should be represented by a host object such as
`SimTensor` containing at least:

* payload storage,
* activity mask,
* logical shape,
* dtype,
* layout descriptor,
* view metadata when needed.

`group.load` must copy from the source buffer or tensor view into workgroup-local
tensor storage. It must never produce a direct alias to the source buffer.

Tensor-producing operations have value semantics in the simulator just as they
do in the language. Semantically, each tensor-producing operation creates a new
workgroup-local value unless `out=` is used.

The simulator does not need physical LDS, but it should still model tensors as
separate workgroup-local storage and keep them distinct from persistent buffers.

### Vectors

Vectors should be represented by an immutable host object such as `SimVector`
containing at least:

* payload storage,
* activity mask,
* logical shape,
* dtype,
* layout descriptor.

The simulator must enforce the same vector rules as the language:

* vector shapes are statically known after literal binding,
* mixed tensor/vector and buffer/vector operations are illegal,
* `vec()` is the only tensor-to-vector value conversion in v1,
* there is no vector-to-tensor value conversion in v1,
* vector results stay vectors,
* vector operations are valid only when the result shape is statically known.

### Masks and Poison

Tensors and vectors carry an explicit activity mask.

Inactive payload is poison semantically. The simulator should model this
explicitly instead of silently substituting arbitrary concrete values.

A practical host representation is:

* data payload stored separately from mask bits,
* a `Poison` sentinel for scalar reads or scalar reduction results that are
  inactive,
* eager failure when Python code tries to consume poison in a way that would
  observe it.

This means:

* shaped results may remain masked values,
* reading an inactive scalar should produce poison,
* using poison in Python arithmetic, comparisons, indexing, or control flow
  should raise a simulator error with a clear diagnostic.

`with_inactive(value=...)` should resolve inactive elements exactly as specified
in `langref.md` and produce a fully active result.

`.mask` should return a fully active bool tensor/vector view. The simulator may
materialize it lazily and does not need to allocate dedicated backing storage
when that is unnecessary.

### Layouts

Layouts should be represented directly by host layout descriptors using the
same `index_map(...)` contract as `langref.md`.

The simulator should preserve layout semantics rather than silently normalizing
everything to dense contiguous storage. In particular:

* tensor load/store and indexing semantics must honor the layout descriptor,
* vector layout must be preserved across operations when the language says it is
  preserved,
* `as_layout(...)` must explicitly change vector layout while preserving logical
  contents.

An initial implementation may represent tensor/vector storage as flat host
arrays plus a layout descriptor and logical view metadata.

## Direct Execution Model

The simulator should execute kernels by calling the original Python functions
directly.

### Kernel Invocation

At launch, the simulator wrapper should:

* bind symbols,
* choose and validate launch geometry,
* iterate over workgroups in deterministic order,
* construct the `group` object for each workgroup,
* call the original Python kernel body with that `group` object and the user
  arguments.

### Execution Environment

Normal Python name lookup, closures, globals, imported libraries, and helper
function calls should work as ordinary Python execution. The simulator does not
need to reimplement Python evaluation rules.

Its job is only to provide runtime objects whose methods and operator overloads
implement the language semantics for tensors, vectors, masks, layouts, and
scopes.

### Caching

The simulator may cache resolved signatures, symbol-binding metadata, layout
descriptors, and launch-derived runtime metadata. Such caching is a performance
detail only and must not change semantics.

### Region Runners

`@group.subgroups` and `@group.workitems` should be ordinary Python decorators
implemented by the simulator runtime.

When a nested function is decorated with one of these decorators, the result is
a callable region object. Invoking that region object executes the original
Python body multiple times under the appropriate subgroup or workitem contexts.

This allows collective regions to be expressed in normal Python without AST
rewriting.

## Execution Scopes

### WorkGroup Scope

The kernel body executes once per workgroup.

For each workgroup, the simulator constructs a `CurrentGroup`-like host object
containing at least:

* `group_id`,
* `group_shape`,
* `work_shape`,
* `work_offset`,
* size/shape helpers required by the API,
* workgroup-local tensor storage owned by that workgroup.

Workgroups should be executed in a deterministic lexicographic order.

### SubGroup Scope

`@group.subgroups` executes its body once per subgroup in the current
workgroup.

The simulator should partition the workgroup exactly as described in
`langref.md`: dimension `0` is fastest-varying, and subgroups split that
dimension into contiguous blocks of size `subgroup_size`.

Each subgroup execution receives a host `SubGroup` object providing subgroup id
and size queries. Subgroups should execute in deterministic subgroup-id order.

Returning from the subgroup region implies a join back to the enclosing
WorkGroup scope.

### WorkItem Scope

`@group.workitems` executes its body once per workitem in the current
workgroup.

`@group.workitems` should be implemented by wrapping the Python region body in a
set of host-side coroutines, one per workitem. A package such as `greenlet`
fits this model well, though equivalent coroutine mechanisms are also fine.

Each workitem receives:

* its own local environment for names defined inside the region,
* a host `WorkItem` object providing local/global id queries,
* access to captured outer values according to the capture rules in
  `langref.md`.

### Barrier Semantics

The workitem scheduler should coordinate the workitem coroutines in
barrier-delimited phases.

One workable model is:

1. create one coroutine per workitem,
2. start each coroutine by calling the original Python region body with its
   `WorkItem` object,
3. when `group.barrier()` is called, suspend the current coroutine and yield to
   the scheduler,
4. once all live workitems have reached the same barrier generation, resume
   them,
5. when all workitems return, join back to WorkGroup scope.

If some workitems reach a barrier and others skip it or reach a different
barrier generation, the simulator must raise an execution error for divergent
barrier use.

The simulator should model the language guarantee that `group.barrier()`
orders accesses to workgroup-local tensor storage, including masks.

### Captures and Mutability

The simulator must implement the same capture and mutation rules as the
language:

* immutable scalars, tuples, and symbol values are captured by value,
* buffers and `group` are captured as handles to the same underlying objects,
* workgroup-local tensors are captured as references to the same tensor
  storage,
* vectors are immutable values and may be captured freely,
* rebinding captured outer names is invalid.

Rules that naturally fall out of direct Python execution and runtime object
semantics should be enforced directly. Rules that require full source-level
portability analysis are outside the simulator core and may be diagnosed only
when practical.

### Conflicting Accesses

Conflicting unordered accesses from different subgroups or workitems are
invalid according to `langref.md`.

The simulator should aim to diagnose obvious conflicts in strict mode,
especially within workitem regions separated by barriers. Full dynamic race
detection is optional, but the design should leave room for it.

## Functions, Intrinsics, and Libraries

### Helper Functions

`@kernel.func` helpers should execute through the same simulator path as
kernels.

Helpers should run as ordinary Python functions against the same simulator
runtime objects. Scope restrictions are checked exactly as they are for
kernels.

There is no need for a separate device-function lowering path in the simulator.

### Intrinsics

The simulator does not use target-specific lowerings.

Therefore:

* `@name.lower(target=...)` hooks are ignored by the simulator,
* the intrinsic fallback body is the executable definition,
* if an intrinsic has an empty body and no host fallback semantics, simulator
  execution fails before the intrinsic is used.

Optional `verify` hooks may still run as host-side validation if present and if
doing so is useful. `infer` hooks are generally unnecessary for execution,
because the simulator obtains result values directly from the fallback body.

### Device Libraries

Device libraries are ordinary Python modules and should work in the simulator
exactly as they do for the compiled path, subject to the same supported kernel
subset and scope rules.

Imported helpers, intrinsics, layout descriptors, and constants are resolved
through the defining module environment.

## Validation and Diagnostics

The simulator should be strict by default.

It should reject or report:

* illegal scope use,
* illegal tensor allocation outside WorkGroup scope,
* vector operations with non-static result shape,
* illegal tensor/vector mixing,
* non-uniform or divergent barrier use,
* missing intrinsic fallback semantics,
* launch-time symbol and geometry errors.

Diagnostics should prefer determinism and clarity over fidelity to hardware
failure modes. In particular, where the language says behavior is invalid or
poison, the simulator should prefer a precise host-side error over silently
continuing with arbitrary behavior.

The simulator does not need to guarantee that every non-portable Python source
construct is rejected up front.

## Recommended First Implementation Order

### Milestone 0: WorkGroup-only reference execution

Implement:

* direct execution for straight-line WorkGroup kernels,
* symbol binding and launch validation,
* host buffer/tensor/vector objects,
* tensor and vector v1 operations,
* mask and poison behavior.

### Milestone 1: Helper functions and layouts

Add:

* `@kernel.func`,
* device library imports,
* `index_map(...)`,
* `as_layout(...)`.

### Milestone 2: SubGroup and WorkItem scopes

Add:

* subgroup partitioning,
* workitem scheduling,
* collective region joins,
* `group.barrier()`.

### Milestone 3: Intrinsics and stricter diagnostics

Add:

* intrinsic fallback execution,
* optional verify-hook support,
* clearer poison diagnostics,
* optional conflict/race detection in strict mode.

## Rationale

The simulator is valuable because it gives the project a semantic reference
executor that does not depend on compiler completeness.

Compared to the compiled MLIR path, the simulator:

* is simpler to bootstrap,
* can expose language errors early with host-side diagnostics,
* provides a correctness oracle for tests,
* makes debugging easier because values remain ordinary host-side Python
  objects.

Compared to calling the original Python functions directly without
runtime support, this design:

* preserves a single deterministic execution model,
* can implement collective regions and barriers correctly,
* keeps legal language programs aligned with the same semantics as the compiled
  path.

## Open Questions

This document intentionally leaves a few issues open for later refinement:

* how aggressive strict-mode conflict detection should be,
* whether the default simulator target should emulate a specific GPU family or
  remain purely abstract,
* whether the simulator should offer a simpler fast path for kernels that never
  enter `@group.workitems`,
* whether coroutine support should depend on `greenlet` specifically or allow
  multiple interchangeable host implementations.
