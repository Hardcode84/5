# Upstream MLIR lowering plan for WMMA

This document sketches the first executable lowering target for
`examples/amdgpu_gfx11_wmma_matmul.py`: a host `func.func` that embeds the
kernel body in a `gpu.launch` op.

The intent is not to legalize all of HC at once. The first pass should accept a
fully inferred WMMA-shaped subset, lower it to upstream MLIR dialects, and fail
early on anything outside that subset.

## Target shape

The initial executable form is a host function with memref arguments and an
embedded GPU launch:

```mlir
func.func @tiled_gfx11_wmma_matmul(%a: memref<?x?xf16>,
                                   %b: memref<?x?xf16>,
                                   %c: memref<?x?xf32>) {
  %c1 = arith.constant 1 : index
  %tx = arith.constant 32 : index
  %ty = arith.constant 1 : index
  %gx = ... : index  // ceiling(M / 16)
  %gy = ... : index  // ceiling(N / 16)

  gpu.launch blocks(%bx, %by, %bz) in (%gx, %gy, %c1)
             threads(%lane, %thread_y, %thread_z) in (%tx, %ty, %c1) {
    // Former hc.kernel body.
    gpu.terminator
  }

  return
}
```

For the current WMMA example, `work_shape` is the total logical workitem grid:

```text
work_shape  = (32 * ceiling(M / 16), ceiling(N / 16))
group_shape = (32, 1)
```

The launch grid is therefore:

```text
blocks = ceildiv(work_shape, group_shape) = (ceiling(M / 16), ceiling(N / 16))
threads = group_shape = (32, 1)
```

The lowering must encode that division explicitly rather than treating
`work_shape` as a block grid.

## Accepted HC input

Lower from post-inference `hc`, not from `hc_front`. For the WMMA path, require:

* `hc.kernel @tiled_gfx11_wmma_matmul` with typed buffer parameters:
  `!hc.buffer<f16, ["M", "K"]>`, `!hc.buffer<f16, ["K", "N"]>`,
  `!hc.buffer<f32, ["M", "N"]>`.
* Static `group_shape = #hc.shape<["32", "1"]>` and
  `subgroup_size = #hc.expr<"32">`.
* `hc.func` helpers with inferred signatures for the tile and fragment flow.
* `hc.intrinsic @wmma_gfx11` with a typed runtime SSA contract:
  tensors `16x16xf16`, fragments `vector<16xf16>` / `vector<8xf32>`, and lane
  index.
* No remaining `!hc.undef` on operations that must choose upstream types or
  launch geometry.

Anything outside this subset should produce a located diagnostic before building
partial upstream IR.

## Mask and shaped-value decomposition

Before lowering to upstream dialects, HC should make masked shaped semantics
explicit. The semantic types:

```text
!hc.tensor<T, shape>
!hc.vector<T, shape>
```

represent data plus validity. Upstream dialect conversion should not have to
remember that this validity is implicit. Introduce compiler-internal bare shaped
types:

```text
!hc.bare_tensor<T, shape>
!hc.bare_vector<T, shape>
```

`T` may be a numeric element type or `!hc.pred`. A mask is therefore not a
separate type family; it is ordinary bare shaped data whose element type is
`!hc.pred`:

```text
!hc.bare_tensor<!hc.pred, ["16", "16"]>
!hc.bare_vector<!hc.pred, ["8"]>
```

The decomposition invariant is:

```text
!hc.tensor<T, shape>
  -> (!hc.bare_tensor<T, shape>, !hc.bare_tensor<!hc.pred, shape>)

!hc.vector<T, shape>
  -> (!hc.bare_vector<T, shape>, !hc.bare_vector<!hc.pred, shape>)
```

After this boundary, no operation prepared for upstream conversion should carry
semantic `!hc.tensor` / `!hc.vector` values. It should carry bare data values and
matching bare predicate values.

For example, a potentially partial tile load becomes:

```mlir
%tile, %tile_mask = hc.load %a[%row_sl, %k_sl], shape %shape
    : (...) -> (!hc.bare_tensor<f16, ["16", "16"]>,
               !hc.bare_tensor<!hc.pred, ["16", "16"]>)
```

Views decompose both data and validity with the same indexing rules:

```mlir
%frag, %frag_mask = hc.buffer_view %tile[%lane, :]
    : (...) -> (!hc.bare_tensor<f16, ["16"]>,
               !hc.bare_tensor<!hc.pred, ["16"]>)
```

Vectorization preserves the split:

```mlir
%frag_v = hc.vec %frag
    : !hc.bare_tensor<f16, ["16"]> -> !hc.bare_vector<f16, ["16"]>
%frag_m = hc.vec %frag_mask
    : !hc.bare_tensor<!hc.pred, ["16"]>
      -> !hc.bare_vector<!hc.pred, ["16"]>
```

Inactive-lane filling becomes an explicit select over the predicate vector:

```mlir
%filled = hc.select %frag_m, %frag_v, %zero
    : (!hc.bare_vector<!hc.pred, ["16"]>,
       !hc.bare_vector<f16, ["16"]>, f16)
      -> !hc.bare_vector<f16, ["16"]>
```

The `wmma_gfx11` intrinsic should consume filled bare fragments, not masked
semantic vectors. Its result is bare data:

```mlir
%acc_next = hc.call_intrinsic @wmma_gfx11(..., %a_filled, %b_filled,
                                         %acc_data, %lane)
    : (...) -> !hc.bare_vector<f32, ["8"]>
```

If that accumulator re-enters a semantic HC context before final lowering, pair
it with a full `!hc.bare_vector<!hc.pred, ["8"]>` validity value. For the
upstream path, keep it bare and thread the predicate value separately to stores.

## Dialect mapping

The first conversion target is:

```text
hc -> func + gpu + scf + arith + memref + vector + amdgpu/rocdl
```

Suggested operation mapping:

| HC concept | Upstream MLIR target |
| --- | --- |
| `hc.kernel` | host `func.func` containing `gpu.launch` |
| `!hc.buffer<T, shape>` | ranked or dynamic `memref<...xT>` host arguments |
| `hc.group_id` | `gpu.launch` block ids |
| `hc.workitem_region` | region inlined into `gpu.launch` using thread ids |
| `hc.local_id` | `gpu.launch` thread ids |
| `hc.for_range` | `scf.for` |
| `hc.const`, arithmetic ops | `arith` over `index`, integer, or float types |
| `!hc.bare_tensor<T, shape>` | workgroup-memory `memref<...xT, #gpu.address_space<workgroup>>` |
| `!hc.bare_tensor<!hc.pred, shape>` | workgroup-memory `memref<...xi1, #gpu.address_space<workgroup>>` |
| `!hc.bare_vector<T, shape>` | `vector<...xT>` after converting HC element types |
| `!hc.bare_vector<!hc.pred, shape>` | `vector<...xi1>` |
| `hc.slice_expr` | offset / size / stride SSA tuples for memref/vector ops |
| decomposed `hc.load` tile | `vector.transfer_read` with predicate mask and padding |
| decomposed `hc.store` tile | `vector.transfer_write` with predicate mask, or guarded stores |
| `hc.call_intrinsic @wmma_gfx11` | `amdgpu` WMMA/MFMA op, or a ROCDL/LLVM intrinsic wrapper |

## Lowering stages

### 1. Host wrapper and launch geometry

Create a host `func.func` for each executable `hc.kernel`.

* Convert kernel buffer parameters to memref function arguments.
* Compute symbolic dimensions from `memref.dim` and any launch bindings.
* Lower `work_shape / group_shape` to launch block counts.
* Emit `gpu.launch` with block ids and thread ids bound to the names expected by
  the inlined kernel body.

The HC kernel body then lowers inside the `gpu.launch` region. Kernel-scope
`hc.return` becomes `gpu.terminator`; the host function ends with
`func.return`.

### 2. Inline helper bodies for the initial path

For the first executable WMMA path, inline `hc.func` helpers into the launch
rather than materializing device functions.

This avoids early decisions about GPU function ABI, address spaces, and
cross-function lowering of vector fragments. Later, helpers can become private
`gpu.func` or LLVM functions once the ABI is intentional.

### 3. Lower index and slice expressions

Use `index`-typed SSA for launch geometry and buffer coordinates.

* `!hc.idx<expr>` becomes `index` for SSA operands and intrinsic contracts.
* `hc.idx_apply` / `hc.pred_apply` lower their pinned expression
  through any explicitly bound operand SSA, then resolve any unlisted
  free symbols ambiently — launch symbols such as `$WG0` / `$WI0`
  from `gpu.launch` block/thread ids, ABI symbols such as `M` / `N`
  from `memref.dim`.
* `hc.const`, simple index arithmetic/comparisons, `hc.buffer_dim`, and
  `hc.for_range` lower to `arith`, `memref.dim`, and `scf.for`.
* `hc.slice_expr` and tile shape tuples remain HC carriers for the memory
  lowering slice, but their scalar operands are upstream `index` values.

The static shape verifier should already have rejected dynamic tile dimensions
for the WMMA subset.

### 4. Decompose semantic tensors and vectors

Run a dedicated HC normalization pass before converting memory and vector ops to
upstream dialects.

Responsibilities:

* Rewrite semantic `!hc.tensor` and `!hc.vector` producers to bare data plus
  bare `!hc.pred` validity values.
* Rewrite views, tuple/getitem users, `hc.vec`, `hc.with_inactive`, and stores
  to consume the split representation.
* Fold full predicate values aggressively so interior-tile cases do not carry
  unnecessary mask operations.
* Reject any remaining semantic shaped value at the upstream conversion
  boundary.

This pass should remain HC-to-HC. It preserves symbolic predicate meaning as
`!hc.pred` while HC-level simplification can still reason about bounds.

### 5. Lower tile loads

The main loop loads two logical tiles:

```text
A[row0 : row0 + 16, k0   : k0 + 16] -> tensor<16x16xf16>
B[k0   : k0 + 16, col0 : col0 + 16] -> tensor<16x16xf16>
```

After decomposition, each load produces bare data plus a bare predicate tile.
Initial implementation options for the data path:

* Direct global `vector.transfer_read` from memrefs into `vector<16x16xf16>`.
* Or introduce LDS/shared-memory staging later with `memref.alloca` in a GPU
  address space and cooperative loads.

The current `hc-lower-launch-body` slice stages the direct global read into
workgroup memory: data tiles become workgroup `memref`s, `hc.load_mask` builds a
`vector.create_mask` and writes it into a matching predicate memref, and
`hc.buffer_view` forms subviews over tensor fragments in workgroup storage. Only
`hc.vec` materializes those tensor fragments as upstream `vector` values;
`hc.select` then becomes `arith.select`. Final masked stores scalarize to
guarded `memref.store` operations; the WMMA intrinsic remains an HC boundary
with explicit casts until its lowering slice lands.

### 6. Lower fragment extraction

The helper bodies compute lane-local fragments:

```text
a_frag_data : !hc.bare_vector<f16, ["16"]>
a_frag_mask : !hc.bare_vector<!hc.pred, ["16"]>
b_frag_data : !hc.bare_vector<f16, ["16"]>
b_frag_mask : !hc.bare_vector<!hc.pred, ["16"]>
acc_data    : !hc.bare_vector<f32, ["8"]>
```

Lower fragment extraction from the `16x16xf16` workgroup tile memrefs using the
launch thread id `lane = gpu.thread_id x`. Apply the same extraction to the
predicate tile memrefs so data and validity stay aligned.

The current HC code already models collective suffix axes on accumulator vector
views, so the post-inference shape for:

```python
acc[:, lane, 0]
```

is the local `!hc.vector<f32, ["8"]>` fragment. Preserve that interpretation in
the decomposition and upstream lowering: `lane` chooses the collective lane, not
an additional element of the local vector.

Before the WMMA intrinsic, fill inactive A/B fragment lanes:

```mlir
%a_filled = hc.select %a_frag_mask, %a_frag_data, %zero
%b_filled = hc.select %b_frag_mask, %b_frag_data, %zero
```

### 7. Lower WMMA intrinsic

Lower:

```mlir
hc.call_intrinsic @wmma_gfx11(..., %a_filled, %b_filled, %acc_data, %lane)
  {arch = "gfx11", wave_size = 32 : i64}
```

to the target-specific operation selected for gfx11 by running the
intrinsic's lowering recipe. Recipes are authored in Python via
`@<intrinsic>.lower(target=...)` and emitted as
`transform.named_sequence` ops inside a sibling
`module @__hc_intrinsic_lowerings__` (see the **Intrinsics** section in
`doc/lowering.md` for the recipe authoring + transform-op surface). The
`-hc-interpret-intrinsic-recipes` pass walks that module, applies the
named sequences whose `hc.target` matches the requested target (e.g.
`"amdgpu-gfx11"`), erases the spent lowerings module, and surfaces an
uncovered-call diagnostic if any `hc.call_intrinsic` remains.

For gfx11 the canonical recipe targets upstream `amdgpu.wmma`. If a
future target needs an intrinsic the upstream `amdgpu` dialect doesn't
model, the recipe can emit a ROCDL/LLVM wrapper instead — the
interpreter pass is target-agnostic.

The recipe layer relies on three independent verifiers, by design:

* `hc.intrinsic` const_kwargs gating ensures every call site carries
  `arch` and `wave_size` attributes; missing entries fail at IR verify
  time.
* The simulator-side `@<intrinsic>.verify` hook validates `arch == "gfx11"`,
  `wave_size == 32`, and the expected f16/f32 vector shapes when running
  Python.
* The target op's own verifier (`amdgpu.wmma` here) rejects mismatched
  vector lengths or out-of-range `m`/`n`/`k` when the recipe constructs
  the payload op.

The intrinsic boundary should be unmasked unless a future target intrinsic
explicitly models masked execution.

### 8. Lower stores

The final store scatters each lane-owned accumulator fragment into `C`.

The HC source uses slice-shaped destination views so right-edge tiles become
empty overlap rather than out-of-bounds scalar columns. Preserve that behavior
with one of:

* masked `vector.transfer_write` using the bare `!hc.pred` store predicate,
* guarded scalar stores,
* or a target-specific store pattern if vector transfer legality is awkward.

The first correctness target should favor explicit masks/guards over clever
stores.

The current launch-body lowering implements the guarded scalar-store path. It
extracts each accumulator element and its predicate bit, applies destination
slice offsets and strides, and emits the `memref.store` inside `scf.if` so
inactive edge lanes never perform an out-of-bounds write.

## Pass structure

A practical first implementation could be:

1. `HCPrepareGpuLaunch`
   * Validate the accepted WMMA subset.
   * Compute launch block/thread counts from `work_shape` and `group_shape`.
2. `HCDecomposeShapedValues`
   * Rewrite semantic tensors/vectors to bare data plus bare `!hc.pred`
     validity values.
   * Canonicalize full predicates and explicit inactive-lane fills.
3. `ConvertHCToGpuLaunch`
   * Convert `hc.kernel` to host `func.func` + `gpu.launch`.
   * Inline the kernel body into the launch.
   * Map group/workitem ids to launch block/thread ids.
4. `ConvertHCStructuredOps`
   * Lower `hc.for_range`, arithmetic, constants, tuples, getitem, slices, and
     buffer dimensions to upstream dialects.
5. `ConvertHCMemoryOps`
   * Lower tile loads, buffer views, vectorization, inactive-lane handling, and
     stores using the explicit predicate values.
6. `HCInterpretIntrinsicRecipes`
   * Walks the sibling `module @__hc_intrinsic_lowerings__` planted by the
     frontend emitter, applies each `transform.named_sequence` whose
     `hc.target` matches the requested target (e.g. `wmma_gfx11` →
     `amdgpu.wmma`), and erases the spent lowerings module. Any
     `hc.call_intrinsic` left at this point is a hard error: the user
     either asked for a target with no recipe or registered a recipe that
     skipped the call.
7. `ConvertHCPredicates`
   * Lower `!hc.pred` scalar and bare predicate containers to upstream `i1`
     scalar/vector values once HC-level predicate simplification is complete.
8. Existing upstream pipelines
   * Canonicalization, CSE, vector lowering, GPU-to-ROCDL/LLVM lowering.

These can start as one pass while the subset is tiny, but keeping the phases
separate in the code will make diagnostics and later expansion easier.

## Test plan

Add a WMMA-focused lowering suite before broadening the accepted subset.

Useful checks:

* `hc.kernel` becomes `func.func` containing `gpu.launch`.
* Launch blocks are derived from `work_shape / group_shape`, not copied from
  `work_shape`.
* `gpu.launch` thread ids replace `hc.local_id`.
* Buffer arguments become f16/f32 memrefs.
* `scf.for` replaces the K loop.
* Semantic `!hc.tensor` / `!hc.vector` values are gone before upstream
  conversion.
* Masks are represented as `!hc.bare_tensor<!hc.pred, ...>` or
  `!hc.bare_vector<!hc.pred, ...>` before becoming upstream `i1` vectors.
* No `hc.*` ops remain after the complete upstream conversion pipeline.
* The WMMA intrinsic lowers to the selected `amdgpu` / `rocdl` operation.
* Edge-tile stores include masks or guards.

Start with FileCheck tests over the WMMA example, then add small hand-written
negative tests for unsupported residual `!hc.undef`, unsupported group shapes,
and missing intrinsic contracts.

## Open decisions

* Whether the first tile representation is `vector<16x16xf16>`, a memref
  subview, or a target-specific fragment abstraction.
* Whether LDS staging is part of the first executable path or a later
  optimization.
* Which upstream AMDGPU/ROCDL operation exactly represents gfx11 WMMA.
* How much symbolic expression metadata should survive after converting
  `!hc.idx<expr>` to plain `index`.
* How much symbolic predicate metadata should survive after converting
  `!hc.pred` masks to upstream `i1`.
* Whether non-inlined helpers become private `gpu.func` or remain an HC-level
  optimization boundary for now.
