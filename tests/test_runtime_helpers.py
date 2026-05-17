# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Smoke tests for `libhc_rt_helpers.so`: ctypes-load, hand the lib
# tensor-like Python objects, assert returned descriptors/scalars match
# what `hc.compile`'s host wrapper sees from JIT'd code.

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from build_tools import hc_native_tools, llvm_toolchain
from hc._native_paths import runtime_helpers_lib_path

_HELPER_NAMES = (
    "_mlir_ciface_hc_get_ptr",
    "_mlir_ciface_hc_get_int64",
    "_mlir_ciface_hc_get_float64",
    "_mlir_ciface_hc_get_dim",
    "_mlir_ciface_hc_get_stride",
)


class _NumpyTensor:
    """Duck-typed torch.Tensor stand-in — drives the helpers without
    pulling torch into the test suite."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def data_ptr(self) -> int:
        return int(self._array.ctypes.data)

    def size(self, dim: int) -> int:
        return int(self._array.shape[dim])

    def stride(self, dim: int) -> int:
        return int(self._array.strides[dim] // self._array.itemsize)


def _resolve_helpers_path() -> Path:
    override = os.environ.get("HC_RT_HELPERS_PATH")
    if override:
        return Path(override).resolve()
    candidates: list[Path] = [runtime_helpers_lib_path()]
    install_root = hc_native_tools.hc_native_tools_layout(
        llvm_toolchain.llvm_toolchain_layout(
            llvm_toolchain.load_llvm_lock()
        ).install_root
    ).install_root
    candidates.append(install_root / "lib" / "libhc_rt_helpers.so")
    # Tolerant of a stale build: if the lock changed but LLVM rebuild
    # hasn't run, the freshest cached install is still useful.
    project_root = Path(__file__).resolve().parents[1]
    cached = sorted(
        (project_root / ".hc" / "native" / "install").glob("*/lib/libhc_rt_helpers.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(cached)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@pytest.fixture(scope="module")
def helpers() -> ctypes.CDLL:
    path = _resolve_helpers_path()
    if not path.exists():
        pytest.skip(f"libhc_rt_helpers.so not built: {path}")
    lib = ctypes.CDLL(str(path))
    lib._mlir_ciface_hc_get_ptr.argtypes = [ctypes.py_object]
    lib._mlir_ciface_hc_get_ptr.restype = ctypes.c_void_p
    lib._mlir_ciface_hc_get_int64.argtypes = [ctypes.py_object]
    lib._mlir_ciface_hc_get_int64.restype = ctypes.c_int64
    lib._mlir_ciface_hc_get_float64.argtypes = [ctypes.py_object]
    lib._mlir_ciface_hc_get_float64.restype = ctypes.c_double
    lib._mlir_ciface_hc_get_dim.argtypes = [ctypes.py_object, ctypes.c_int32]
    lib._mlir_ciface_hc_get_dim.restype = ctypes.c_int64
    lib._mlir_ciface_hc_get_stride.argtypes = [ctypes.py_object, ctypes.c_int32]
    lib._mlir_ciface_hc_get_stride.restype = ctypes.c_int64
    return lib


def test_helpers_export_expected_symbols(helpers: ctypes.CDLL) -> None:
    for name in _HELPER_NAMES:
        assert getattr(helpers, name) is not None, name


def test_get_ptr_returns_data_pointer(helpers: ctypes.CDLL) -> None:
    # `hc_get_ptr` is the `!hc.ptr<global, T?>` kernel-arg entry —
    # returns the same address as `data_ptr()`.
    array = np.arange(64, dtype=np.float16)
    tensor = _NumpyTensor(array)
    expected_ptr = int(array.ctypes.data)
    assert helpers._mlir_ciface_hc_get_ptr(tensor) == expected_ptr


def test_get_int64_round_trips_python_int(helpers: ctypes.CDLL) -> None:
    assert helpers._mlir_ciface_hc_get_int64(0) == 0
    assert helpers._mlir_ciface_hc_get_int64(1) == 1
    assert helpers._mlir_ciface_hc_get_int64(-1) == -1
    assert helpers._mlir_ciface_hc_get_int64(2**31 - 1) == 2**31 - 1
    assert helpers._mlir_ciface_hc_get_int64(-(2**31)) == -(2**31)
    assert helpers._mlir_ciface_hc_get_int64(2**62) == 2**62


def test_get_float64_round_trips_python_float(helpers: ctypes.CDLL) -> None:
    assert helpers._mlir_ciface_hc_get_float64(0.0) == 0.0
    assert helpers._mlir_ciface_hc_get_float64(1.5) == 1.5
    assert helpers._mlir_ciface_hc_get_float64(-3.25) == -3.25


def test_get_dim_reads_size(helpers: ctypes.CDLL) -> None:
    array = np.zeros((7, 11, 13), dtype=np.int32)
    tensor = _NumpyTensor(array)
    assert helpers._mlir_ciface_hc_get_dim(tensor, 0) == 7
    assert helpers._mlir_ciface_hc_get_dim(tensor, 1) == 11
    assert helpers._mlir_ciface_hc_get_dim(tensor, 2) == 13


def test_get_stride_reads_element_strides(helpers: ctypes.CDLL) -> None:
    array = np.zeros((4, 8), dtype=np.int64)
    tensor = _NumpyTensor(array)
    # Contiguous: stride(0)=8, stride(1)=1 (elements, not bytes).
    assert helpers._mlir_ciface_hc_get_stride(tensor, 0) == 8
    assert helpers._mlir_ciface_hc_get_stride(tensor, 1) == 1


# Error paths not exercised here: helpers throw `std::runtime_error` on
# bad input; unwinding through the C ABI into ctypes is
# implementation-defined (in practice: process aborts because the JIT'd
# caller has no unwind tables). Host wrapper is responsible for never
# feeding invalid objects; runtime tracebacks would need a subprocess
# harness.
