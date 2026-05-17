// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Public C ABI of `libhc_rt_helpers.so`. JIT'd host wrappers call these
// to unpack PyObject args into raw pointers / ints under the GIL. Helpers
// borrow the PyObject; caller keeps it alive for the launch.

#ifndef HC_RUNTIME_BUFFERUTILS_H
#define HC_RUNTIME_BUFFERUTILS_H

#include <cstdint>

// Forward-declared to keep <Python.h> off the include path of the
// compiler-side build.
extern "C" struct _object;
typedef struct _object PyObject;

extern "C" {

// `obj.data_ptr()` -> raw pointer. Fed straight to `gpu.launch_func`.
void *_mlir_ciface_hc_get_ptr(PyObject *obj);

// Python int -> int64. Throws `std::runtime_error` on overflow / wrong type;
// unwinds through the ctypes thunk as a Python exception.
int64_t _mlir_ciface_hc_get_int64(PyObject *obj);

// Python float -> f64. Throws on non-float.
double _mlir_ciface_hc_get_float64(PyObject *obj);

// `obj.size(dim_idx)`.
int64_t _mlir_ciface_hc_get_dim(PyObject *obj, int32_t dim_idx);

// `obj.stride(dim_idx)` in elements, matching torch.
int64_t _mlir_ciface_hc_get_stride(PyObject *obj, int32_t dim_idx);
}

#endif // HC_RUNTIME_BUFFERUTILS_H
