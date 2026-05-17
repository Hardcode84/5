# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pairwise Euclidean distance kernel from the langref RFC.

This is the workgroup-level form from `doc/langref.md`'s "Kernel definition"
section: each workgroup computes a `(group.shape[0], group.shape[1])` tile
of the output distance matrix `D[W1, W2]` by loading the corresponding
`(group.shape[0], H)` slice of `X1` and `(group.shape[1], H)` slice of `X2`,
broadcasting them into a rank-3 difference, squaring, summing over `H`,
and writing `sqrt(...)` back to `D` with the boundary mask carried by the
`group.load` / `group.store` boundary checks.

The example exists as a small end-to-end pass through the workgroup-level
API that exercises broadcasting + reduction + sqrt — useful both against
the simulator and as the smallest non-WMMA native-compile target.

`H` (the inner reduction dim) is in the decorator's `literals=` set: the
workgroup LDS tile of shape `(g0, H)` / `(g1, H)` has to materialize with
all dims resolved to integer literals at `hc-lower-launch-body` time
(runtime-sized LDS for symbolic carriers is the open feature in the
tracker). The frontend-resolve step substitutes the literal binding into
the kernel IR, so a different `H` per call means a recompile — the
`compile_pairwise_distance` helper wires that up via `hc.compile(symbols=
{H: x1.shape[1]})`. The simulator path doesn't care about literal
binding and runs against arbitrary `H` directly.

The decorator also pins `group_shape=(8, 8)` to a concrete integer-literal
pair. The langref RFC text leaves `group_shape` implicit (the dispatcher
picks it), but the native amdgpu lowering currently needs `group_shape`
integer-literal at frontend-resolve time so the `$WGS0`/`$WGS1` system
symbols seed `literal_bindings` for downstream LDS-tile static-shape
checks. Symbolic `group_shape` with a user-supplied binding is the
eventual surface — same tracker, separate axis.

`(8, 8)` is a 64-thread square tile - one full wave on wave64 (gfx9-),
two waves on wave32 (gfx10+). The 2D shape matches the `(g0, g1, H)`
broadcast naturally; the parallel-iter chunk loop in the workgroup-
collective lowering hands each thread one or two output slots depending
on wave size. Smaller tiles (e.g. `(2, 2)` = 4 threads) leave the wave
mostly idle; larger tiles (e.g. `(16, 16)` = 256 threads) lift LDS
pressure and per-workgroup register usage without payoff for an
illustrative example.

Run from the repository root with:

    python -m examples.pairwise_distance
"""

from __future__ import annotations

import numpy as np

import hc.simulator as sim
from hc import Buffer, kernel, sym

W1 = sym.W1
W2 = sym.W2
H = sym.H


@kernel(work_shape=(W1, W2), group_shape=(8, 8), literals=[H])
def pairwise_distance_wg_kernel(
    group,
    X1: Buffer[W1, H, np.float32],
    X2: Buffer[W2, H, np.float32],
    D: Buffer[W1, W2, np.float32],
) -> None:
    # `group.work_offset` is `(group_id[0] * group.shape[0],
    # group_id[1] * group.shape[1])` — the upper-left corner of the
    # tile this workgroup owns. The trailing `gid[0]:` / `gid[1]:`
    # open-ended slices let `group.load` clip against `X1.shape[0]` /
    # `X2.shape[0]` and stamp the boundary mask onto the loaded tile.
    gid = group.work_offset
    g0, g1 = group.shape

    x1 = group.load(X1[gid[0] :], shape=(g0, X1.shape[1]))
    x2 = group.load(X2[gid[1] :], shape=(g1, X2.shape[1]))

    # Broadcasting subtraction lifts both operands to rank-3.
    # `x1[:, None, :]` shape `(g0, 1, H)`, `x2[None, :, :]` shape
    # `(1, g1, H)`, broadcast difference shape `(g0, g1, H)`. `**2`
    # then `sum(axis=2)` collapses over `H`, leaving `(g0, g1)`
    # aligned with the `D[gid[0]:gid[0]+g0, gid[1]:gid[1]+g1]`
    # destination tile so `D[i, j] = sqrt(sum_k (X1[i, k] - X2[j,
    # k])**2)`.
    diff = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(axis=2)

    # Explicit-stop slicing pins the destination tile to `(g0, g1)` so
    # `group.store` can validate `source.shape == dest_slice.shape`.
    # NumPy clips the slice at the buffer edge for boundary
    # workgroups; the source's mask (from `group.load`'s shape= bounds
    # check) carries the boundary OOB cells as False so the
    # per-element overlap drops them rather than producing garbage.
    group.store(D[gid[0] : gid[0] + g0, gid[1] : gid[1] + g1], np.sqrt(diff))


def reference_pairwise_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """NumPy reference for `pairwise_distance_wg_kernel`.

    Computes `D[i, j] = sqrt(sum_k (X1[i, k] - X2[j, k])**2)` directly
    in float32; the langref kernel's element dtype is whatever the
    caller supplies via the `Buffer[...]` annotations (float32 in
    `make_demo_inputs` below).
    """

    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("X1 and X2 must be rank-2 matrices")
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("X1.shape[1] must match X2.shape[1]")
    diff = x1[:, None, :] - x2[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def simulate_pairwise_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("X1 and X2 must be rank-2 matrices")
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("X1.shape[1] must match X2.shape[1]")

    d = np.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype)
    sim.launch(pairwise_distance_wg_kernel, x1, x2, d)
    return d


def compile_pairwise_distance(
    x1: np.ndarray, x2: np.ndarray, *, target: str = "amdgpu-gfx11"
):
    """Native compile the kernel with `H` bound to `x1.shape[1]`.

    The LDS tile dim has to be a compile-time literal (see module
    docstring for why), so the binding has to be supplied per-input
    and a fresh shape means a fresh `hc.compile` call. Returns the
    compiled handle; invoke it with device buffers (`handle.invoke
    (x1_dev, x2_dev, d_dev)`) on a HIP-visible build.
    """

    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("X1 and X2 must be rank-2 matrices")
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("X1.shape[1] must match X2.shape[1]")

    import hc

    return hc.compile(
        pairwise_distance_wg_kernel,
        symbols={H: x1.shape[1]},
        target=target,
    )


def make_demo_inputs(
    *,
    w1: int = 6,
    w2: int = 5,
    h: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, size=(w1, h)).astype(np.float32)
    x2 = rng.uniform(-1.0, 1.0, size=(w2, h)).astype(np.float32)
    return x1, x2


def main() -> None:
    x1, x2 = make_demo_inputs()
    out = simulate_pairwise_distance(x1, x2)
    ref = reference_pairwise_distance(x1, x2)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    print("pairwise_distance simulator output matches numpy reference")


if __name__ == "__main__":
    main()
