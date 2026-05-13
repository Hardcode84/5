// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Public C ABI of `libhc_hip_runtime.so` — the small launcher shim that
// JIT'd host wrappers call instead of linking HIP directly. The shim
// dlopen's `libamdhip64.so` lazily so the wheel itself has zero ROCm
// build-time or load-time dependency: a host without ROCm can still
// import `hc` and resolve these symbols; only `hc_rt_init` will actually
// touch the runtime.
//
// All entry points are `extern "C"` and the launch path is intentionally
// allocation-free so it can be reached from JIT'd LLVM IR without
// surprising hidden malloc traffic.

#ifndef HC_RUNTIME_HIPRUNTIME_H
#define HC_RUNTIME_HIPRUNTIME_H

#include <cstddef>
#include <cstdint>

extern "C" {

// dlopen libamdhip64.so and bind the function pointers used by the
// launch path. Idempotent: subsequent calls are no-ops once the
// mandatory entry points are bound. Throws `std::runtime_error` on
// failure (missing library, missing mandatory symbol). The exception
// unwinds to abort when called from JIT'd code that lacks unwind tables;
// callers that care about graceful failure should wrap the first
// invocation in a try/catch on the C++ side.
void hc_rt_init();

// Resolve `kernel_name` inside `binary_pointer` (an HSACO blob in
// memory) and return the resulting `hipFunction_t` as a `void*`.
//
// `cached_kernel_handle` points to caller-owned storage that we use for
// single-flight memoization: the first caller loads the module + binds
// the function and stores the result; concurrent callers spin on a
// global mutex, see the cached value, and skip the load. Wave's
// equivalent uses a plain non-atomic read which races and can leak
// modules under contention; we close that gap with acquire/release
// atomics around the slow-path mutex.
//
// `stream` and `binary_size` are accepted for ABI compatibility with
// wave's signature but unused today (load is stream-agnostic in HIP and
// `hipModuleLoadData` consumes a NUL-terminated/self-describing blob).
//
// Note: the underlying `hipModule_t` is intentionally leaked for the
// process lifetime. We never expect more than a handful of distinct
// kernels per run, and an explicit cache + unload path is a separate
// design problem we'll only owe once we ship long-running services.
void *hc_rt_load_kernel(void *stream, void **cached_kernel_handle,
                        const void *binary_pointer, size_t binary_size,
                        const char *kernel_name);

// Launch `function` (returned by `hc_rt_load_kernel`) with the given
// grid/block dims and dynamic shared-memory bytes. If any cluster dim
// is > 1 we route through `hipDrvLaunchKernelEx` with the cluster
// attribute; otherwise we use the simpler `hipModuleLaunchKernel`. The
// cluster path requires a HIP that exposes `hipDrvLaunchKernelEx` —
// `hc_rt_init` makes that symbol optional, so missing it is only fatal
// when a cluster launch is actually requested.
//
// `args` is an array of `void*` pointing at each kernel argument's
// storage (matching the HIP `kernelParams` convention). `num_args` is
// passed for parity with wave's signature but currently unused — the
// underlying HIP entry points read until the trailing `nullptr`.
void hc_rt_launch_kernel(void *stream, void *function, int shared_memory_bytes,
                         int grid_x, int grid_y, int grid_z, int block_x,
                         int block_y, int block_z, int cluster_x, int cluster_y,
                         int cluster_z, void **args, int num_args);

// Bench variant: launch `function` `n_inner` times back-to-back on `stream`,
// `hipStreamSynchronize` at the end, and return the wall-clock nanoseconds
// elapsed for the (N launches + sync) window. Timing is sampled in C via
// `hc_clock_now_ns` (CLOCK_MONOTONIC on Linux today) so the measurement
// never crosses the language boundary inside the sample window — the
// caller's outer benchmark loop only needs to collect the returned value
// per outer iteration, not bracket the call with `perf_counter_ns`.
//
// Same arg-conventions as `hc_rt_launch_kernel`: cluster dim > 1 routes
// through `hipDrvLaunchKernelEx`, otherwise `hipModuleLaunchKernel`. The
// args array is reused verbatim for every iteration — caller is
// responsible for any per-launch state rotation (e.g. cache-cold input
// reshuffling), this entry intentionally measures the hot path with
// the args held constant.
//
// HIP errors propagate via the same `throw std::runtime_error` path the
// single-shot launch uses; the partial sample is lost. `n_inner == 0` is
// well-defined: no launches, still calls `hipStreamSynchronize` (drains
// any prior work on `stream`), returns the clock-pair overhead.
//
// Return type is `uint64_t` (5+ centuries of headroom) so wrappers can
// pass the value straight through to a Python caller via ctypes without
// signed-overflow concerns at extreme sample sizes.
uint64_t hc_rt_launch_kernel_repeat(void *stream, void *function,
                                    int shared_memory_bytes, int grid_x,
                                    int grid_y, int grid_z, int block_x,
                                    int block_y, int block_z, int cluster_x,
                                    int cluster_y, int cluster_z, void **args,
                                    int num_args, size_t n_inner);
}

#endif // HC_RUNTIME_HIPRUNTIME_H
