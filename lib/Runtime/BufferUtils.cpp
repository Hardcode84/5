// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implementation of the Python-unpacking runtime helpers. JIT'd code calls
// in here once per kernel argument; everything happens under a borrowed
// GIL because the host wrapper itself runs from a thread that already
// holds the interpreter (it was entered through ctypes from Python). We
// still call `PyGILState_Ensure` defensively in case the wrapper is ever
// invoked from a non-Python-managed thread — `Ensure` is a no-op in that
// common case and a real acquire otherwise.

#include "hc/Runtime/BufferUtils.h"

#include <Python.h>

#include <stdexcept>
#include <string>

namespace {

// RAII wrapper around the GIL state, so every helper has a single
// well-defined exit point even when an exception unwinds the C++ stack.
class GilGuard {
public:
  GilGuard() : state_(PyGILState_Ensure()) {}
  ~GilGuard() { PyGILState_Release(state_); }
  GilGuard(const GilGuard &) = delete;
  GilGuard &operator=(const GilGuard &) = delete;

private:
  PyGILState_STATE state_;
};

// Stealing-style smart pointer for owned PyObject references. Treats
// nullptr as "an exception was raised" — caller must check before use.
class PyRef {
public:
  PyRef() = default;
  explicit PyRef(PyObject *obj) : obj_(obj) {}
  ~PyRef() {
    if (obj_)
      Py_DECREF(obj_);
  }
  PyRef(const PyRef &) = delete;
  PyRef &operator=(const PyRef &) = delete;
  PyRef(PyRef &&other) noexcept : obj_(other.obj_) { other.obj_ = nullptr; }

  PyObject *get() const { return obj_; }
  explicit operator bool() const { return obj_ != nullptr; }

private:
  PyObject *obj_ = nullptr;
};

// Fetch the latest Python error (if any), wrap it in a C++ exception, and
// clear it from the interpreter. The host wrapper does not catch this; it
// unwinds back through ctypes which surfaces a Python `RuntimeError`.
[[noreturn]] static void raisePythonError(const char *context) {
  std::string message(context);
  if (PyErr_Occurred()) {
    PyObject *type = nullptr;
    PyObject *value = nullptr;
    PyObject *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    if (value) {
      PyRef str(PyObject_Str(value));
      if (str) {
        const char *utf8 = PyUnicode_AsUTF8(str.get());
        if (utf8) {
          message += ": ";
          message += utf8;
        }
      }
    }
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
  }
  throw std::runtime_error(message);
}

// Call `obj.<method>(int_arg)` and convert the result via PyLong_AsLongLong.
// Used by both `get_dim` and `get_stride` — they share the entire flow
// modulo the method name.
static int64_t callIntegerAccessor(PyObject *obj, const char *method,
                                   int32_t index) {
  PyRef bound(PyObject_GetAttrString(obj, method));
  if (!bound)
    raisePythonError(
        std::string("hc_rt: missing attribute ").append(method).c_str());
  PyRef arg(PyLong_FromLong(static_cast<long>(index)));
  if (!arg)
    raisePythonError("hc_rt: failed to box dim index");
  PyRef result(PyObject_CallOneArg(bound.get(), arg.get()));
  if (!result)
    raisePythonError(
        std::string("hc_rt: ").append(method).append("() raised").c_str());
  long long value = PyLong_AsLongLong(result.get());
  if (value == -1 && PyErr_Occurred())
    raisePythonError(std::string("hc_rt: ")
                         .append(method)
                         .append("() returned non-int")
                         .c_str());
  return static_cast<int64_t>(value);
}

} // namespace

// Shared `data_ptr()` extraction. Both the legacy `hc_get_buffer` and the
// `hc.ptr`-native `hc_get_ptr` reach back to the same Python attribute;
// keeping the call shape in one place avoids the two surfaces silently
// drifting on attribute name / error-message wording.
static void *fetchDataPtr(PyObject *obj, const char *who) {
  PyRef accessor(PyObject_GetAttrString(obj, "data_ptr"));
  if (!accessor)
    raisePythonError(std::string("hc_rt: ")
                         .append(who)
                         .append(": tensor argument missing data_ptr()")
                         .c_str());
  PyRef raw(PyObject_CallNoArgs(accessor.get()));
  if (!raw)
    raisePythonError(std::string("hc_rt: ")
                         .append(who)
                         .append(": data_ptr() raised")
                         .c_str());
  void *ptr = PyLong_AsVoidPtr(raw.get());
  if (ptr == nullptr && PyErr_Occurred())
    raisePythonError(std::string("hc_rt: ")
                         .append(who)
                         .append(": data_ptr() returned non-int")
                         .c_str());
  return ptr;
}

extern "C" void _mlir_ciface_hc_get_buffer(HcMemRef1Di8 *ret, PyObject *obj) {
  GilGuard gil;
  void *ptr = fetchDataPtr(obj, "hc_get_buffer");
  ret->basePtr = static_cast<uint8_t *>(ptr);
  ret->data = static_cast<uint8_t *>(ptr);
  ret->offset = 0;
  // Sentinel: see header — the byte length is never consumed because the
  // host wrapper immediately reinterprets to a typed memref.
  ret->sizes[0] = -1;
  ret->strides[0] = 1;
}

extern "C" void *_mlir_ciface_hc_get_ptr(PyObject *obj) {
  GilGuard gil;
  return fetchDataPtr(obj, "hc_get_ptr");
}

extern "C" int64_t _mlir_ciface_hc_get_int64(PyObject *obj) {
  GilGuard gil;
  long long value = PyLong_AsLongLong(obj);
  if (value == -1 && PyErr_Occurred())
    raisePythonError("hc_rt: int argument is not a Python int");
  return static_cast<int64_t>(value);
}

extern "C" double _mlir_ciface_hc_get_float64(PyObject *obj) {
  GilGuard gil;
  double value = PyFloat_AsDouble(obj);
  if (value == -1.0 && PyErr_Occurred())
    raisePythonError("hc_rt: float argument is not a Python float");
  return value;
}

extern "C" int64_t _mlir_ciface_hc_get_dim(PyObject *obj, int32_t dim_idx) {
  GilGuard gil;
  return callIntegerAccessor(obj, "size", dim_idx);
}

extern "C" int64_t _mlir_ciface_hc_get_stride(PyObject *obj, int32_t dim_idx) {
  GilGuard gil;
  return callIntegerAccessor(obj, "stride", dim_idx);
}
