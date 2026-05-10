// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implementation of the HIP launcher shim. Linux-only v0 — the wave
// reference also ports to Windows but we have no consumer there yet, so
// we error out early to keep the surface area honest.

#include "hc/Runtime/HipRuntime.h"

#include "hc/Runtime/HipTypes.h"

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
using ModuleHandle = void *;
#else
#error "hc_rt_init only supports Linux today; port libamdhip64 dlopen first"
#endif

namespace {

// Function pointers resolved by `hc_rt_init`. They are written once
// during the (mutex-serialized) init; the launch helpers then only read
// them, so no atomic load is needed beyond the init's release fence.
hipModuleLaunchKernel_t g_hipModuleLaunchKernel = nullptr;
hipDrvLaunchKernelEx_t g_hipDrvLaunchKernelEx = nullptr;
hipGetErrorName_t g_hipGetErrorName = nullptr;
hipGetErrorString_t g_hipGetErrorString = nullptr;
hipModuleUnload_t g_hipModuleUnload = nullptr;
hipModuleLoadData_t g_hipModuleLoadData = nullptr;
hipModuleGetFunction_t g_hipModuleGetFunction = nullptr;

static void *symbolOrNull(ModuleHandle module, const char *name) {
#if defined(__linux__)
  return dlsym(module, name);
#endif
}

template <typename Fn>
static Fn requireSymbol(ModuleHandle module, const char *name) {
  void *raw = symbolOrNull(module, name);
  if (!raw)
    throw std::runtime_error(
        "hc_rt_init: missing mandatory libamdhip64 symbol: " +
        std::string(name));
  return reinterpret_cast<Fn>(raw);
}

// Translates a hipError_t into a runtime_error with both the symbolic
// name and the human-readable message, including the location where the
// call originated (so a launch failure points at the launch site rather
// than the helper).
[[noreturn]] static void throwHipError(hipError_t code, const char *expression,
                                       const char *file, int line) {
  std::ostringstream msg;
  msg << "hc_rt: HIP error " << code;
  if (g_hipGetErrorName) {
    if (const char *name = g_hipGetErrorName(code))
      msg << " (" << name << ")";
  }
  msg << " at " << file << ":" << line << ": " << expression;
  if (g_hipGetErrorString) {
    if (const char *text = g_hipGetErrorString(code))
      msg << ": " << text;
  }
  throw std::runtime_error(msg.str());
}

} // namespace

#define HC_HIP_CHECK(expr)                                                     \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e)                                                                    \
      throwHipError(_e, #expr, __FILE__, __LINE__);                            \
  } while (0)

extern "C" void hc_rt_init() {
  // Mutex-serialized double-checked initialization. The check is `&&`'d
  // across all mandatory symbols so a partial init (e.g. previous run
  // threw mid-bind) re-runs cleanly.
  static std::mutex init_mutex;
  if (g_hipModuleLaunchKernel && g_hipGetErrorName && g_hipGetErrorString &&
      g_hipModuleUnload && g_hipModuleLoadData && g_hipModuleGetFunction)
    return;

  std::lock_guard<std::mutex> guard(init_mutex);
  if (g_hipModuleLaunchKernel && g_hipGetErrorName && g_hipGetErrorString &&
      g_hipModuleUnload && g_hipModuleLoadData && g_hipModuleGetFunction)
    return;

#if defined(__linux__)
  ModuleHandle module = dlopen("libamdhip64.so", RTLD_NOW);
  if (!module) {
    const char *err = dlerror();
    throw std::runtime_error("hc_rt_init: failed to dlopen libamdhip64.so: " +
                             std::string(err ? err : "(no error message)"));
  }
#endif

  g_hipModuleLaunchKernel =
      requireSymbol<hipModuleLaunchKernel_t>(module, "hipModuleLaunchKernel");
  g_hipGetErrorName =
      requireSymbol<hipGetErrorName_t>(module, "hipGetErrorName");
  g_hipGetErrorString =
      requireSymbol<hipGetErrorString_t>(module, "hipGetErrorString");
  g_hipModuleUnload =
      requireSymbol<hipModuleUnload_t>(module, "hipModuleUnload");
  g_hipModuleLoadData =
      requireSymbol<hipModuleLoadData_t>(module, "hipModuleLoadData");
  g_hipModuleGetFunction =
      requireSymbol<hipModuleGetFunction_t>(module, "hipModuleGetFunction");

  // Optional — older HIPs predate `hipDrvLaunchKernelEx`. We only need
  // it on the cluster-launch path; missing here is reported lazily.
  g_hipDrvLaunchKernelEx = reinterpret_cast<hipDrvLaunchKernelEx_t>(
      symbolOrNull(module, "hipDrvLaunchKernelEx"));
}

extern "C" void *hc_rt_load_kernel(void * /*stream*/,
                                   void **cached_kernel_handle,
                                   const void *binary_pointer,
                                   size_t /*binary_size*/,
                                   const char *kernel_name) {
  // Acquire-load the cache slot so we synchronize with any prior
  // release-store from a concurrent first caller. wave's reference does
  // a plain read here and races: two threads can both observe nullptr,
  // both call hipModuleLoadData, and one module silently leaks while the
  // loser's function pointer wins the store-back. We close that with the
  // atomic + the slow-path mutex below.
  void *cached = __atomic_load_n(cached_kernel_handle, __ATOMIC_ACQUIRE);
  if (cached)
    return cached;

  static std::mutex loader_mutex;
  std::lock_guard<std::mutex> guard(loader_mutex);

  cached = __atomic_load_n(cached_kernel_handle, __ATOMIC_ACQUIRE);
  if (cached)
    return cached;

  hipModule_t mod = nullptr;
  HC_HIP_CHECK(g_hipModuleLoadData(&mod, binary_pointer));
  hipFunction_t function = nullptr;
  HC_HIP_CHECK(g_hipModuleGetFunction(&function, mod, kernel_name));

  __atomic_store_n(cached_kernel_handle, function, __ATOMIC_RELEASE);
  return function;
}

extern "C" void hc_rt_launch_kernel(void *stream, void *function,
                                    int shared_memory_bytes, int grid_x,
                                    int grid_y, int grid_z, int block_x,
                                    int block_y, int block_z, int cluster_x,
                                    int cluster_y, int cluster_z, void **args,
                                    int /*num_args*/) {
  if (cluster_x * cluster_y * cluster_z > 1) {
    if (!g_hipDrvLaunchKernelEx)
      throw std::runtime_error(
          "hc_rt_launch_kernel: cluster launch requested but the loaded "
          "libamdhip64.so does not export hipDrvLaunchKernelEx");

    hipLaunchAttribute attrs[1] = {};
    attrs[0].id = hipLaunchAttributeClusterDimension;
    int *cluster_dims = reinterpret_cast<int *>(attrs[0].val.pad);
    cluster_dims[0] = cluster_x;
    cluster_dims[1] = cluster_y;
    cluster_dims[2] = cluster_z;

    HIP_LAUNCH_CONFIG cfg = {
        static_cast<unsigned>(grid_x),
        static_cast<unsigned>(grid_y),
        static_cast<unsigned>(grid_z),
        static_cast<unsigned>(block_x),
        static_cast<unsigned>(block_y),
        static_cast<unsigned>(block_z),
        static_cast<unsigned>(shared_memory_bytes),
        stream,
        attrs,
        1,
    };
    HC_HIP_CHECK(g_hipDrvLaunchKernelEx(&cfg, function, args, nullptr));
    return;
  }

  HC_HIP_CHECK(g_hipModuleLaunchKernel(
      function, static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
      static_cast<unsigned>(grid_z), static_cast<unsigned>(block_x),
      static_cast<unsigned>(block_y), static_cast<unsigned>(block_z),
      static_cast<unsigned>(shared_memory_bytes), stream, args, nullptr));
}
