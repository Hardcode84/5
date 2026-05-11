# Upstream MLIR lowering plan for WMMA

> **Historical.** This document was the design plan that produced the first
> executable WMMA lowering. Most of it predates the post-2026-05 retirement
> of memref from the HC pipeline (kernel-arg ABI as
> `(!hc.ptr<global, T>, dim*, stride*)`, workgroup tiles via `hc.alloc` /
> `!hc.ptr<workgroup, T>`, no `vector.transfer_*` survivors past
> `hc-lower-launch-body`).
>
> For the live contract, read:
>
> * [`doc/layouts.md`](layouts.md) — `hc.ptr` and memory ops, layout flow.
> * [`doc/lowering.md`](lowering.md) — pass schedule and pipeline stages.
> * [`doc/schedules.md`](schedules.md) — the transform-dialect schedule and
>   the `hc-opt` pass list it composes against.
>
> The body below is preserved as a snapshot of the original design, not as a
> description of the running pipeline.

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
