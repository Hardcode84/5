<!--
SPDX-FileCopyrightText: 2024 The HC Authors
SPDX-FileCopyrightText: 2025 The HC Authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# RFC: High level GPU kernel API

Alexander Kalistratov, Ivan Butygin

## Summary

We propose new high-level kernel API (TBD)

## Motivation

Current low-level Kernel API is too verbose and not very convenient for fast
prototyping.
Current high-level APIs (array API and prange), on the other hand, provide too
little low level control over GPU execution.

## Proposal

We propose a new Workgroup-level API, with direct access to Numpy array
operations and ability to acess workitem level API directly.


### Kernel definition
Simple example of pairwise distance kernel:
```python
# Current OpenCL/SYCL style kernel
@kernel
def pairwise_distance_kernel(X1, X2, D):
    i, j = nb.get_global_id()

    if i < X1.shape[0] and j < X2.shape[0]:
        d = 0.0
        # calculating distance with loop by dimensions
        for k in range(X1.shape[1]):
            tmp = X1[i, k] - X2[j, k]
            d += tmp * tmp
        D[i, j] = np.sqrt(d)

# New api, immediately switching to workitem level.
W1 = sym.W1
W2 = sym.W2
H = sym.H
@kernel(work_shape=(W1, W2))
def pairwise_distance_kernel(group: CurrentGroup,
                             X1: Buffer[W1, H],
                             X2: Buffer[W2, H],
                             D: Buffer[W1, W2]):
    # switch to workitem level
    # parallel loop over work items
    @group.workitems
    def inner(ind):
        i, j = ind.global_id()
        if i < X1.shape[0] and j < X2.shape[0]:
            # using high-level array api to calculate distance
            d = ((X1[i] - X2[j])**2).sum()
            D[i, j] = np.sqrt(d)

    inner()

# Using WG level api
@kernel(work_shape=(W1, W2))
def pairwise_distance_kernel(group: CurrentGroup,
                             X1: Buffer[W1, H],
                             X2: Buffer[W2, H],
                             D: Buffer[W1, W2]):
    gid = group.work_offset # global offset to current WG (i.e. group_size * group_id)

    # Create tensor of specified shape, but with boundary checks of X1 and X2
    x1 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))
    x2 = group.load(X2[gid[1]:], shape=(group.shape[1], X2.shape[1]))

    # calculating pairwise distance with numpy-style broadcasting
    diff = ((x1[None, :, :] - x2[:, None, :])**2).sum(axis=2)

    # store result to D, but with boundary checks
    group.store(D[gid[0]:, gid[1]:], np.sqrt(diff))
```


### Launching the kernel:
In the new API, launch geometry is determined by solving the symbols that
appear in the kernel signature and in kernel decorator arguments. Buffer
annotations, tuple-typed kernel arguments, literal symbols, and explicit
`work_shape`, `group_shape`, and `subgroup_size` declarations all participate
in the same binding step.

In `pairwise_distance_kernel`, `W1` is bound from `X1.shape[0]`, `W2` is bound
from `X2.shape[0]`, `H` is bound from the shared second dimension of `X1` and
`X2`, and `D` must have shape `(W1, W2)`. Since the decorator declares
`work_shape=(W1, W2)`, the global work shape is inferred as
`(X1.shape[0], X2.shape[0])`.

Kernel launch succeeds only if every required symbol has a unique value and all
uses of the same symbol are consistent. The runtime must report an error if a
symbol is missing, if two bindings for the same symbol disagree, or if an
explicit launch parameter conflicts with the argument shapes.

```python
# Current kernel API
pairwise_distance_kernel[global_size, local_size](X1, X2, D)

# New API: launch geometry is inferred from symbol bindings.
pairwise_distance_kernel(X1, X2, D)
```
While kernel function takes `CurrentGroup` as argument, it's not passed to the
kernel invocation directly. The runtime provides it after symbol binding
determines the launch geometry.

If user wants to specify work/group shapes explicitly they may do so by
binding the corresponding symbols through ordinary kernel arguments:
```python
@kernel(work_shape=(G1,G2,G3), group_shape=(L1,L2,L3))
def test(gr: CurrentGroup,
         gsize: tuple[G1, G2, G3],
         lsize: tuple[L1, L2, L3]):
    # `gsize` and `lsize` are ordinary kernel arguments whose values are used
    # to bind launch symbols.
    ...

test((1024, 1, 1), (64, 1, 1))
```

### Symbols
Symbols express equality relationships between kernel arguments and launch
geometry. Each occurrence of the same symbol refers to the same runtime value.

During kernel launch, the runtime collects symbol values from the actual
arguments and checks that all occurrences are consistent. A symbol may appear
in buffer dimensions, tuple-typed kernel arguments, or launch attributes such
as `work_shape`, `group_shape`, and `subgroup_size`:
```python
W, H = sym.W, sym.H
@kernel(work_shape=(W, H))
def pairwise_distance_kernel(group: CurrentGroup,
                             X1: Buffer[W, H],
                             X2: Buffer[W, H]):
    # Kernel expects 2 buffers with same shape and global work shape will be
    # equal to that shape.
    ...
```

By default symbols are treated as dynamic values, i.e. the same compiled kernel
can be used with any runtime values that satisfy the symbol constraints.

Symbols can also be declared as literals, and in this case, runtime will compile
separate versions of the kernel for each distinct symbol value:
```python
# H is usually small and won't change between kernel invocations, declaring it as
# literal so compiler can unroll it instead of doing dynamic loop.
@kernel(work_shape=(W1, W2), literals={H})
def pairwise_distance_kernel(group: CurrentGroup,
                             X1: Buffer[W1, H],
                             X2: Buffer[W2, H],
                             D: Buffer[W1, W2]):
```

Literal symbols define specialization points. Only literal symbols create
specialized kernel variants; launch-determined values that are not literal do
not.

A specialized kernel variant is identified by:
* the kernel definition
* the target device or backend-specific compilation target
* the bound values of all literal symbols for that launch

Declaring a symbol literal changes only compilation policy, not semantics:
different bound values of that symbol require distinct specialized variants.

The runtime may compile specialized variants lazily on the first launch that
requires a new specialization key, or eagerly before launch. In either case,
all required specialization, validation, and compilation must complete before
device execution begins.

If a launch requires a new specialized variant and the implementation cannot
create it, the launch fails before execution.

Implementations may cache specialized variants and may evict cached variants.
Cache persistence, residency, and eviction policy are implementation-defined.
Eviction must not change program semantics; it may only cause recompilation on
a later launch.

Implementations may impose implementation-defined limits on the number of
cached specialized variants and may provide configurable diagnostics when many
distinct literal specializations are created. However, an implementation must
not silently stop specializing a symbol that was declared literal.

Literal symbols can be used in context where constant is expected (e.g. vector
dimensions). Using a non-literal symbol in such a context is ill-formed, even
if that symbol is otherwise launch-determined.

Subgroup size must always be a constant or literal symbol:
```python
@kernel(work_shape=(G1,G2,G3), group_shape=(L1,L2,L3), subgroup_size=SG, literals={SG})
def test(gr: CurrentGroup,
         gsize: tuple[G1, G2, G3],
         lsize: tuple[L1, L2, L3],
         sgsize: SG):
    ...
```

For buffer it's also possible to declare specific dimension as constant if it's
known beforehead:
```python
@kernel(work_shape=(W1, W2))
def pairwise_distance_kernel(group: CurrentGroup,
                             X1: Buffer[W1, 3],
                             X2: Buffer[W2, 3],
                             D: Buffer[W1, W2]):
```

### Tensors and arrays
Numpy arrays passed as arguments to the kernel can be accessed directly inside
but we also provide `tensor` object as a convenient way to access data inside
the kernel.

Tensors can be of arbitrary, possibly dynamic, shape and support masking access.

Creating tensor from array
```python
x1 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))
```

Resulting tensor is always of requested shape, but if source slice was of
smaller shape, some elements will be masked.

Copying data back into array:
```python
group.store(D[gid[0]:, gid[1]:], tensor)
```

If tensor is masked, only active elements will be written.

Tensors are workgroup-local objects stored in shared local memory (LDS).
`group.load` copies data into tensor storage; it never returns a direct view
into the source array. Mutating a tensor never modifies the source array
directly. If user wants to make changes visible, it must call
`group.store` explicitly.

Allocating new tensor:
```python
arr = group.empty(shape=(...), dtype=dtyp)
arr = group.zeros(shape=(...), dtype=dtyp)
arr = group.ones(shape=(...), dtype=dtyp)
arr = group.full(shape=(...), dtype=dtyp, fill_value=...)
```
The implementation must not silently place tensors in global memory.

Tensor-producing operations have value semantics. Semantically, each such
operation creates a fresh tensor in workgroup-local storage unless an `out=`
argument is provided.

The shape of any tensor materialization must be determined entirely from
launch-determined values: constants, literal symbols, kernel argument shapes,
symbol-bound kernel arguments, and launch geometry values such as
`work_shape`, `group_shape`, `subgroup_size`, `group.shape`, and `group.size`,
including pure integer expressions over these values.

Tensor materialization must not depend on values computed during kernel
execution, including values loaded from tensors, vectors, or buffers, or values
derived from workgroup, subgroup, or workitem indices. If the compiler cannot
prove that a tensor materialization is launch-determined, the program is
ill-formed.

For legal tensor materializations, the runtime computes the required LDS usage
before kernel launch and rejects launches that exceed the device's shared local
memory limits.

Tensors support usual numpy operations, including fancy indexing and
broadcasting:
```python
diff = ((x1[None, :, :] - x2[:, None, :])**2).sum(axis=2)
```
Numpy ops follows usual Numpy semantics by returning newly allocated tensor as
result unless an `out=` argument is provided.

An implementation may eliminate, fuse, or reuse tensor storage when this does
not change observable behavior. In particular, it may remove storage for unused
results, merge allocations with non-overlapping lifetimes, or eliminate
temporary tensors created by fused computations. However, these optimizations
must preserve the semantics of fresh non-aliasing tensor results.

An implementation may provide a configurable diagnostic when required LDS usage
depends on launch-determined values rather than being compile-time constant.
This diagnostic may be suppressed if optimization proves that no such dynamic
materialization remains.

Supported Numpy ops on tensors: (TBD)

User also can pass output tensor explicitly:
```python
arr = group.zeros(shape=(...), dtype=dtyp)
res = np.subtract(x1, x2, out=arr)
```
Passing `out=` suppresses the logical fresh allocation for that operation,
though the compiler may still remove or reuse storage when this is
unobservable.

Tensor allocation is only allowed on workgroup level, they are not allowed on
subgroup or workitem level.


### Vectors

In addition to `tensor` objects compiler supports operations over `vector` types.
Vectors are immutable, statically-sized values with no observable aliasing.


New vector allocation:
```python
arr = group.vzeros(shape=(...), dtype=dtyp)
arr = group.vones(shape=(...), dtype=dtyp)
arr = group.vfull(shape=(...), dtype=dtyp, fill_value=...)
```

Creating vector from array:
```python
x1 = group.vload(X1[gid[0]:], shape=(W, H))
```
Vector allocation shape must be deteminable at compile time.

Creating vector from tensor:
```python
x1 = group.load(X1[gid[0]:], shape=(W, H)).vec()
x2 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))[x:x+W,y:y+W].vec()
x3 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1])).vec(shape=(W,H))
```
`vec()` shape must be deteminable at compile time. If shape is omitted, source
tensor shape will be used and must be static.
`vec()` is the only value conversion from tensor to vector in v1.

There is no vector-to-tensor value conversion in v1. To materialize a vector
into tensor or buffer storage, user must call `group.store` explicitly.


Storing vector back into array/tensor:
```python
group.store(D[gid[0]:, gid[1]:], vector)
```
Note: tensor and vector stores have the same data movement and masking
semantics. Scope restrictions on where `group.store` may be called are defined
by the execution model.

Vectors can be masked. If source tensor was masked, resulting vector will be
masked as well.

Vector operations are permitted on workgroup, subgroup and workitem level.

Mixed operations between vectors and tensors, or between vectors and source
arrays/buffers, are not allowed. Users must convert explicitly with `vec()`
before entering vector computations.

Vector operations never implicitly create tensors. For operations whose result
is vector-shaped, the result is always a vector. Such operations are valid only
when the compiler can determine the result shape at compile time; otherwise the
program is ill-formed. Numpy-style broadcasting is supported only within this
statically-shaped vector domain. `out=` argument is not supported.

Supported Numpy ops on vectors: (TBD)


### Masking

Tensors and vectors carry an activity mask. An element is active iff its
corresponding mask element is `True`; inactive iff it is `False`.

The payload of an inactive element is poison. Mask-aware tensor/vector
operations do not directly observe the payload of inactive elements, but
directly reading an inactive element yields poison.

`group.load/vload` accept either `shape=` or `mask=` argument, but not both.

When `shape=` is used, the result has the requested shape and elements whose
source access falls outside source bounds become inactive.

When `mask=` is used, the mask argument must be a `bool` tensor/vector. The
result shape is `mask.shape`. Result element `i` is active iff `mask[i]` is
active, `mask[i] == True`, and the corresponding source access is in bounds.
Otherwise result element `i` is inactive.

Comparison operations on tensors/vectors (e.g. `a < b`) return `bool`
tensors/vectors and follow the same mask propagation rules as other
tensor/vector operations.

`group.store` and tensor element assignment via `[]` only update destination
elements whose source elements are active. Updated destination elements become
active. Source elements that are inactive leave destination payload and
destination activity unchanged. Scalar source values are treated as fully
active.

There is intentionally no API to mutate activity bits in place. New inactive
elements arise only by creating new tensors/vectors, e.g. through masked or
out-of-bounds loads or through ordinary tensor/vector operations with mask
propagation.

For tensor/vector operations, an output element is active only if all source
elements it semantically depends on are active, unless operation-specific rules
say otherwise.

Reduction functions only consider active elements. For reductions that produce
a shaped tensor/vector, each output element is active iff its reduction domain
has at least one active contributor; otherwise that output element is inactive.
If a reduction returns a scalar and there are no active contributors, the
scalar result is poison.

Allocation functions `group.(v)zeros`,`group.(v)ones`,`group.(v)full` mark all
elements as active. `group.empty` marks all elements as inactive and leaves
their payload poison.

`with_inactive(value=...)` returns a tensor/vector of the same shape and dtype
as the source. Active elements preserve their original value. Inactive elements
are replaced with `value` converted to the source dtype. The result is always
fully active.

Mask storage is allocated in the same storage type as tensor/vector data and
contributes to storage requirements unless elided. The compiler may elide mask
storage when it can prove that the value is fully active.

`tensor.mask` and `vector.mask` return read-only `bool` tensors/vectors of the
same shape. A mask element is `True` iff the corresponding source element is
active.

The result of `.mask` is always fully active, regardless of the activity of the
source value, so reading `.mask` never yields poison. `.mask` follows the same
tensor/vector rules and has its own `.mask` property, but implementations must
always elide dedicated storage for `.mask` values. `x.mask.mask` is therefore
well-defined and yields a fully-active all-`True` bool tensor/vector.


### Switching to SubGroup or WorkItem scope

The default execution scope is WorkGroup: the kernel body executes once per
workgroup.

Let `group_shape = (L0, L1, ..., Ln-1)`, where dimension `0` is the
fastest-varying workgroup dimension. Workitems in the workgroup have local ids
`(i0, i1, ..., in-1)` with `0 <= ik < Lk`.

If `subgroup_size = SG`, then the workgroup is partitioned into subgroups by
splitting dimension `0` into contiguous blocks of size `SG`. For each fixed
tuple `(i1, ..., in-1)`, the workitems
`(k*SG : (k+1)*SG, i1, ..., in-1)` belong to one subgroup.

This requires `L0 % SG == 0`. If not, kernel launch is invalid and the runtime
must report an error before execution.

`@group.subgroups` executes its body once per subgroup. `sg.subgroup_id()`
returns the subgroup index within the current workgroup, in the ordering
induced by the partition above, and `sg.size()` returns `subgroup_size`.

`@group.workitems` executes its body once per workitem.

Entering `@group.subgroups` or `@group.workitems` is a collective operation on
the enclosing scope, and returning from the inner function implies a join back
to the enclosing scope. After `inner()` returns, all subgroups or workitems in
that region have completed.

The only explicit barrier in v1 is `group.barrier()`. It is permitted only
inside `@group.workitems` and synchronizes all workitems in the current
workgroup. `group.barrier()` also orders accesses to workgroup-local tensor
storage, including activity masks, making writes performed before the barrier
visible to reads performed after the barrier.

Every workitem in the workgroup must execute the same `group.barrier()` calls
in the same order. A barrier in non-uniform control flow is invalid. The
implementation should reject such kernels when it can prove the violation;
otherwise execution is invalid and may fail at execution time.

No explicit barrier is provided at WorkGroup or SubGroup scope, because those
scopes are collective by construction.

Inner `@group.subgroups` and `@group.workitems` functions may capture kernel
arguments, `group`, workgroup-local tensors, vectors, scalar locals, tuples,
and symbol-bound values from the enclosing scope.

Scalars, tuples, symbol values, and other immutable values are captured by
value and are read-only inside the inner scope. Rebinding a captured outer name
is invalid.

Buffer arguments and `group` are captured as handles to the same underlying
objects. Buffer arguments may be accessed directly from subgroup or workitem
scope.

Workgroup-local tensors are captured as references to the same tensor storage.
Subgroup or workitem scope may read from a captured tensor and may update its
elements via indexing. However, subgroup or workitem scope must not create new
tensor values; tensor allocation and tensor-producing operations remain
WorkGroup-only.

Vectors are immutable values with no observable aliasing. They may be captured
and used in any scope, but mutating a vector or rebinding a captured vector
name is invalid.

`group.load` and tensor allocation APIs create tensors and are therefore
WorkGroup-only. `group.vload` and vector creation APIs may be used in any
scope. `group.store` with a tensor source is WorkGroup-only; `group.store`
with a vector source may be used in any scope.

Writes performed by subgroup or workitem regions to captured tensors or buffer
arguments become visible to the enclosing WorkGroup scope when the region
returns. Within `@group.workitems`, `group.barrier()` only adds ordering
guarantees for workgroup-local tensor storage, including activity masks.

Conflicting unordered accesses to the same tensor or buffer location from
different subgroups or workitems are invalid. In particular, overlapping writes
or a read racing with a write are invalid unless the accesses occur in
different ordered phases of execution.

SG Level:
```python
@kernel
def foo(group, X1, X2, D):
    @group.subgroups
    def inner(sg):
        id = sg.subgroup_id()
        size = sg.size()

    inner()
```

Workitem scope:
```python
@kernel
def foo(group, X1, X2, D):
    @group.workitems
    def inner(wi):
        i, j, k = wi.global_id()

    inner()
```

Programming on workitem scope is close to usual OpenCL programming.
