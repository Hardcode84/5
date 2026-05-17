# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pairwise Euclidean distance kernel from the langref RFC.

Each workgroup computes a `(g0, g1)` tile of `D[W1, W2]` from `(g0, H)`
and `(g1, H)` slices of `X1` / `X2`: broadcast into rank-3 diff, square,
reduce over `H`, `sqrt`. `group.load` / `group.store` carry the boundary
mask. Smallest non-WMMA native-compile target; covers broadcast +
reduction + sqrt end-to-end.

`H` is in `literals=` because LDS tiles need compile-time int shapes
(symbolic LDS unimplemented); a fresh `H` means a fresh `hc.compile`.
The simulator runs symbolic `H` directly.

`group_shape=(8, 8)` is a literal pair. Native amdgpu lowering needs
`group_shape` int-literal so `$WGS0` / `$WGS1` seed `literal_bindings`
for LDS static-shape checks. `(8, 8) = 64` threads — one wave64 / two
wave32. Smaller leaves wave idle; larger raises LDS without payoff.

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
    # Tile's upper-left corner. Open-ended `gid[0]:` / `gid[1]:` let
    # `group.load` clip against buffer extents and stamp boundary mask.
    gid = group.work_offset
    g0, g1 = group.shape

    x1 = group.load(X1[gid[0] :], shape=(g0, X1.shape[1]))
    x2 = group.load(X2[gid[1] :], shape=(g1, X2.shape[1]))

    # `(g0, 1, H) - (1, g1, H) -> (g0, g1, H)`, reduce axis 2 for
    # `D[i, j] = sqrt(sum_k (X1[i, k] - X2[j, k])**2)`.
    diff = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(axis=2)

    # Explicit stops pin dest to `(g0, g1)` so `group.store` validates
    # `source.shape == dest_slice.shape`. NumPy clips at the buffer
    # edge; the source mask drops OOB cells per-element.
    group.store(D[gid[0] : gid[0] + g0, gid[1] : gid[1] + g1], np.sqrt(diff))


def reference_pairwise_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """NumPy reference: `D[i, j] = sqrt(sum_k (X1[i, k] - X2[j, k])**2)`."""

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
    """Native compile with `H` bound to `x1.shape[1]`.

    LDS needs compile-time literal — fresh shape, fresh compile.
    Invoke the returned handle with device buffers on a HIP-visible build.
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


def _require_torch_cuda(surface: str):
    """Import torch and confirm a HIP/ROCm device is visible.

    Raises `RuntimeError` (not `ImportError`) so the message names what's
    missing. `surface=` names the caller for diagnostics.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"{surface} needs the `torch` package to allocate device "
            "buffers; `pip install torch` (with a ROCm-enabled wheel) and "
            "retry."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{surface} needs a HIP/ROCm device visible to torch.cuda; "
            "torch.cuda.is_available() returned False."
        )
    return torch


def run_on_hardware(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> np.ndarray:
    """Compile for gfx11 and invoke through the bundled HIP shim.

    `torch.cuda` tensors back device buffers — `.data_ptr()` returns
    a HIP pointer for `_mlir_ciface_hc_get_ptr`. Binds `H` to
    `x1.shape[1]` and checks against the NumPy reference.
    """

    torch = _require_torch_cuda("run_on_hardware")

    x1_dev = torch.from_numpy(x1).cuda()
    x2_dev = torch.from_numpy(x2).cuda()
    d_dev = torch.zeros(x1.shape[0], x2.shape[0], dtype=torch.float32, device="cuda")

    compiled = compile_pairwise_distance(x1, x2)
    compiled.invoke(x1_dev, x2_dev, d_dev)

    out = d_dev.cpu().numpy()
    reference = reference_pairwise_distance(x1, x2)
    np.testing.assert_allclose(out, reference, rtol=rtol, atol=atol)
    return out


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
