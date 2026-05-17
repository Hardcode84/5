// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Public C ABI of `libhc_hip_runtime.so` -- launcher shim called by JIT'd
// host wrappers. `libamdhip64.so` is dlopen'd lazily so the wheel has zero
// ROCm build/load dependency; ROCm-free hosts can import `hc` and only
// `hc_rt_init` actually touches the runtime. Launch path is allocation-
// free so JIT'd LLVM IR doesn't trip hidden malloc.

#ifndef HC_RUNTIME_HIPRUNTIME_H
#define HC_RUNTIME_HIPRUNTIME_H

#include <cstddef>
#include <cstdint>

extern "C" {

// dlopen libamdhip64.so and bind the launch-path entry points. Idempotent.
// Throws `std::runtime_error` on missing library / missing mandatory
// symbol; from JIT'd code without unwind tables that aborts. Callers that
// want graceful failure must catch on the C++ side.
void hc_rt_init();

// Resolve `kernel_name` inside `binary_pointer` (HSACO blob) and return
// the `hipFunction_t` as `void*`.
//
// `cached_kernel_handle` is caller-owned single-flight memoization
// storage: first caller loads + binds + stores, concurrent callers spin
// on a global mutex, observe the cached value, skip the load. Acquire /
// release atomics around the slow-path mutex.
//
// `stream` and `binary_size` are ABI-compat slots, unused today (load is
// stream-agnostic, `hipModuleLoadData` reads a self-describing blob).
//
// The underlying `hipModule_t` leaks for process lifetime by design -- a
// handful of kernels per run, an unload path is a separate problem.
void *hc_rt_load_kernel(void *stream, void **cached_kernel_handle,
                        const void *binary_pointer, size_t binary_size,
                        const char *kernel_name);

// Launch with the given grid / block / shared-mem. Cluster dim > 1 routes
// through `hipDrvLaunchKernelEx` (its symbol is optional in `hc_rt_init`
// so missing it is only fatal on actual cluster launches); otherwise
// `hipModuleLaunchKernel`. `args` follows the HIP `kernelParams`
// convention; `num_args` is unused (HIP reads until trailing `nullptr`).
void hc_rt_launch_kernel(void *stream, void *function, int shared_memory_bytes,
                         int grid_x, int grid_y, int grid_z, int block_x,
                         int block_y, int block_z, int cluster_x, int cluster_y,
                         int cluster_z, void **args, int num_args);

// Bench variant: launch `n_inner` times back-to-back on `stream`,
// `hipStreamSynchronize`, return wall-clock ns for the (N launches + sync)
// window. Timing sampled in C via `hc_clock_now_ns` so the sample window
// never crosses the language boundary. Routing rule, args, and HIP-error
// path match `hc_rt_launch_kernel`; partial samples are lost on throw.
// `n_inner == 0` is well-defined: no launches, sync still drains prior
// work, returns clock-pair overhead. `uint64_t` return type lets Python
// receive the value through ctypes without signed-overflow concerns.
uint64_t hc_rt_launch_kernel_repeat(void *stream, void *function,
                                    int shared_memory_bytes, int grid_x,
                                    int grid_y, int grid_z, int block_x,
                                    int block_y, int block_z, int cluster_x,
                                    int cluster_y, int cluster_z, void **args,
                                    int num_args, size_t n_inner);
}

#endif // HC_RUNTIME_HIPRUNTIME_H
