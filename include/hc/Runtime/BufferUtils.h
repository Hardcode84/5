// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Public C surface of `libhc_rt_helpers.so`. JIT-emitted host wrappers call
// these `_mlir_ciface_*` functions to unpack incoming PyObject arguments
// into raw pointers and integers without forcing the Python interpreter to
// do per-arg marshalling on the Python side.
//
// The helpers do not own the PyObjects: they only read attributes via the
// Python C API under the GIL and return descriptors that point back into
// the caller's memory. The host wrapper is responsible for keeping the
// PyObjects alive across the kernel launch (in practice the ctypes call
// retains them for its duration).
//
// This header is callable from C and includable from compiled MLIR via
// `llvm.func` declarations.

#ifndef HC_RUNTIME_BUFFERUTILS_H
#define HC_RUNTIME_BUFFERUTILS_H

#include <cstdint>

// Forward declaration to keep this header free of <Python.h> so that the
// compiler-side build (which has no Python dep beyond the helpers TU) does
// not need to thread CPython headers through the include path.
extern "C" struct _object;
typedef struct _object PyObject;

extern "C" {

// Read `obj.data_ptr()` and return the raw pointer. Used by the
// `!hc.ptr<global, T?>` kernel-arg ABI: the host wrapper passes the
// pointer to `gpu.launch_func` directly, with dim and stride values
// arriving as separate scalar operands (`hc_get_dim` / `hc_get_stride`).
void *_mlir_ciface_hc_get_ptr(PyObject *obj);

// Coerce a Python int to int64. Raises `std::runtime_error` (which the
// JIT'd wrapper does not catch — surfaces back through the ctypes call as
// a Python exception) on overflow or non-int input.
int64_t _mlir_ciface_hc_get_int64(PyObject *obj);

// Coerce a Python float to f64. Raises on non-float input.
double _mlir_ciface_hc_get_float64(PyObject *obj);

// Read `obj.size(dim_idx)`. Host wrapper feeds the result into the
// per-axis dim slot of the kernel-arg `(ptr, dim*, stride*)` UCC.
int64_t _mlir_ciface_hc_get_dim(PyObject *obj, int32_t dim_idx);

// Read `obj.stride(dim_idx)` (in elements, matching torch's convention,
// not bytes). Host wrapper feeds the result into the per-axis stride slot
// of the kernel-arg `(ptr, dim*, stride*)` UCC; the launch-body lowering
// uses it to linearize `hc.ptr_offset` indices.
int64_t _mlir_ciface_hc_get_stride(PyObject *obj, int32_t dim_idx);
}

#endif // HC_RUNTIME_BUFFERUTILS_H
