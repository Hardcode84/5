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

// Layout-compatible with `mlir::StridedMemRefType<T, N>` from
// `mlir/ExecutionEngine/CRunnerUtils.h`. We redeclare the struct locally to
// avoid pulling in the heavier MLIR runtime header on the consumer side.
template <typename T, int N> struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

using HcMemRef1Di8 = StridedMemRefType<uint8_t, 1>;

extern "C" {

// Materialize a tensor-like PyObject (anything quacking like torch.Tensor:
// `data_ptr()`, `size(i)`, `stride(i)`) into an opaque `memref<?xi8>`
// descriptor. `sizes[0]` is set to `-1` as a sentinel — the host wrapper
// immediately reinterprets via `memref.view` + `memref.reinterpret_cast`
// to a typed memref of the kernel's expected shape, so the byte length is
// never consumed and computing it would just be an extra C-API roundtrip
// per call.
void _mlir_ciface_hc_get_buffer(HcMemRef1Di8 *ret, PyObject *obj);

// Coerce a Python int to int64. Raises `std::runtime_error` (which the
// JIT'd wrapper does not catch — surfaces back through the ctypes call as
// a Python exception) on overflow or non-int input.
int64_t _mlir_ciface_hc_get_int64(PyObject *obj);

// Coerce a Python float to f64. Raises on non-float input.
double _mlir_ciface_hc_get_float64(PyObject *obj);

// Read `obj.size(dim_idx)`. Used by the host wrapper to derive dynamic
// memref shape arguments at launch time.
int64_t _mlir_ciface_hc_get_dim(PyObject *obj, int32_t dim_idx);

// Read `obj.stride(dim_idx)` (in elements, matching torch's convention,
// not bytes). Used by the host wrapper for non-trivially-strided memref
// args.
int64_t _mlir_ciface_hc_get_stride(PyObject *obj, int32_t dim_idx);
}

#endif // HC_RUNTIME_BUFFERUTILS_H
