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

The point of the example here is twofold:

  * pin a small end-to-end pass through the workgroup-level API that
    exercises broadcasting + reduction + sqrt against the simulator,
    independently of the gfx11-specific WMMA pipeline; and

  * surface, as a single failing artifact, whichever frontend / lowering
    gaps still stand between the literal langref text and the existing
    substrate.

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


@kernel(work_shape=(W1, W2))
def pairwise_distance_wg_kernel(
    group,
    X1: Buffer[W1, H],
    X2: Buffer[W2, H],
    D: Buffer[W1, W2],
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
    #
    # `doc/langref.md`'s WG-level form writes the broadcast as
    # `(x1[None, :, :] - x2[:, None, :])` instead — that pattern
    # produces `(g1, g0)` and stores the transpose of the intended
    # distance matrix. The kernel here keeps the workitem-level
    # contract from the same section (`D[i, j] = sqrt(sum_k (X1[i,
    # k] - X2[j, k])**2)`) and corrects the broadcasting axes
    # accordingly. The langref text is being tracked separately.
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
