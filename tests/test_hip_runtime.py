# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sanity checks for `libhc_hip_runtime.so`. Default surface is ROCm-free:
# ctypes-load the shim, confirm C ABI presence, confirm no accidental
# `libamdhip64` link. `HC_RT_RUN_HIP_INIT_TEST=1` opt-in actually calls
# `hc_rt_init` (dlopens `libamdhip64.so`) — needs ROCm on the host.

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from pathlib import Path

import pytest

from build_tools import hc_native_tools, llvm_toolchain
from hc._native_paths import hip_runtime_lib_path

_SYMBOLS = (
    "hc_rt_init",
    "hc_rt_load_kernel",
    "hc_rt_launch_kernel",
    "hc_rt_launch_kernel_repeat",
)


def _resolve_hip_runtime_path() -> Path:
    override = os.environ.get("HC_RT_HIP_RUNTIME_PATH")
    if override:
        return Path(override).resolve()
    candidates: list[Path] = [hip_runtime_lib_path()]
    install_root = hc_native_tools.hc_native_tools_layout(
        llvm_toolchain.llvm_toolchain_layout(
            llvm_toolchain.load_llvm_lock()
        ).install_root
    ).install_root
    candidates.append(install_root / "lib" / "libhc_hip_runtime.so")
    project_root = Path(__file__).resolve().parents[1]
    cached = sorted(
        (project_root / ".hc" / "native" / "install").glob(
            "*/lib/libhc_hip_runtime.so"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(cached)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@pytest.fixture(scope="module")
def hip_runtime_path() -> Path:
    path = _resolve_hip_runtime_path()
    if not path.exists():
        pytest.skip(f"libhc_hip_runtime.so not built: {path}")
    return path


@pytest.fixture(scope="module")
def hip_runtime(hip_runtime_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(hip_runtime_path))
    lib.hc_rt_init.argtypes = []
    lib.hc_rt_init.restype = None
    # Wide launch signatures declared so ctypes resolves them. Real
    # calls come from JIT'd code.
    lib.hc_rt_load_kernel.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.POINTER(ctypes.c_void_p),  # cached_kernel_handle
        ctypes.c_void_p,  # binary_pointer
        ctypes.c_size_t,  # binary_size
        ctypes.c_char_p,  # kernel_name
    ]
    lib.hc_rt_load_kernel.restype = ctypes.c_void_p
    lib.hc_rt_launch_kernel.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # function
        ctypes.c_int,  # shared_memory_bytes
        ctypes.c_int,  # grid_x
        ctypes.c_int,  # grid_y
        ctypes.c_int,  # grid_z
        ctypes.c_int,  # block_x
        ctypes.c_int,  # block_y
        ctypes.c_int,  # block_z
        ctypes.c_int,  # cluster_x
        ctypes.c_int,  # cluster_y
        ctypes.c_int,  # cluster_z
        ctypes.POINTER(ctypes.c_void_p),  # args
        ctypes.c_int,  # num_args
    ]
    lib.hc_rt_launch_kernel.restype = None
    # Bench variant: same launch signature + `size_t n_inner` arg +
    # `uint64_t` elapsed-ns return.
    lib.hc_rt_launch_kernel_repeat.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # function
        ctypes.c_int,  # shared_memory_bytes
        ctypes.c_int,  # grid_x
        ctypes.c_int,  # grid_y
        ctypes.c_int,  # grid_z
        ctypes.c_int,  # block_x
        ctypes.c_int,  # block_y
        ctypes.c_int,  # block_z
        ctypes.c_int,  # cluster_x
        ctypes.c_int,  # cluster_y
        ctypes.c_int,  # cluster_z
        ctypes.POINTER(ctypes.c_void_p),  # args
        ctypes.c_int,  # num_args
        ctypes.c_size_t,  # n_inner
    ]
    lib.hc_rt_launch_kernel_repeat.restype = ctypes.c_uint64
    return lib


def test_hip_runtime_exports_expected_symbols(hip_runtime: ctypes.CDLL) -> None:
    for name in _SYMBOLS:
        assert getattr(hip_runtime, name) is not None, name


def test_hip_runtime_has_no_rocm_link(hip_runtime_path: Path) -> None:
    """`.so` must load on hosts without ROCm — dlopen-at-init is the
    whole point. Catch accidental `-lamdhip64` (or transitive ROCm
    CMake deps) at build-verification time.
    """
    result = subprocess.run(
        ["ldd", str(hip_runtime_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = re.compile(r"libamdhip64|libhsa|librocm|libhip", re.IGNORECASE)
    matches = [line for line in result.stdout.splitlines() if forbidden.search(line)]
    assert not matches, f"libhc_hip_runtime.so leaked a ROCm dep: {matches}"


def test_hip_runtime_loads_in_subprocess_without_rocm(
    hip_runtime_path: Path,
) -> None:
    """Import + ctypes-load must work without `libamdhip64.so`. Subprocess
    with cleared `LD_LIBRARY_PATH` so a locally installed ROCm can't
    hide a misconfiguration.
    """
    # Mirror `_SYMBOLS` via the module constant — new entries land in
    # one place.
    script = (
        "import ctypes\n"
        f"lib = ctypes.CDLL({str(hip_runtime_path)!r})\n"
        f"for name in {_SYMBOLS!r}:\n"
        "    getattr(lib, name)\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    result = subprocess.run(
        ["python", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.skipif(
    os.environ.get("HC_RT_RUN_HIP_INIT_TEST") != "1",
    reason="set HC_RT_RUN_HIP_INIT_TEST=1 on a host with libamdhip64.so to run",
)
def test_hc_rt_init_dlopens_libamdhip64(hip_runtime: ctypes.CDLL) -> None:
    """Opt-in: exercise the dlopen path. Two calls verify idempotency.
    Without `libamdhip64`, first call throws `std::runtime_error` →
    C ABI translates to abort. Hence the explicit gate.
    """
    hip_runtime.hc_rt_init()
    hip_runtime.hc_rt_init()
