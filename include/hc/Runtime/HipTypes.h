// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Hand-coded subset of the HIP runtime ABI we touch from the launcher
// shim. Mirroring wave's `wave_lang/kernel/wave/runtime/hip_types.h` —
// the entire point is to drop the build-time dependency on ROCm so this
// translation unit (and downstream consumers) compile on any host with
// just a C++17 toolchain. The actual HIP entry points are dlsym'd from
// `libamdhip64.so` at runtime by `hc_rt_init`.

#ifndef HC_RUNTIME_HIPTYPES_H
#define HC_RUNTIME_HIPTYPES_H

#include <cstddef>

// Sentinel pointer values consumed by the legacy `extra` array passed to
// `hipModuleLaunchKernel`. We never use that path (we always go through
// `hipDrvLaunchKernelEx`'s `kernelParams` instead), but the constants are
// part of the public ABI surface and may surface in future codepaths.
#define HC_HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define HC_HIP_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define HC_HIP_LAUNCH_PARAM_END ((void *)0x03)

using hipError_t = int;
using hipStream_t = void *;
using hipFunction_t = void *;
using hipModule_t = void *;

enum hipLaunchAttributeID {
  hipLaunchAttributeAccessPolicyWindow = 1,
  hipLaunchAttributeCooperative = 2,
  hipLaunchAttributeSynchronizationPolicy = 3,
  hipLaunchAttributeClusterDimension = 4,
  hipLaunchAttributePriority = 8,
  hipLaunchAttributeMemSyncDomainMap = 9,
  hipLaunchAttributeMemSyncDomain = 10,
  hipLaunchAttributeMax,
};

// 64-byte payload — large enough to hold any of the upstream attribute
// value structs (cluster dim is three ints, the rest are smaller). We
// don't unpack the variants here because the only attribute we set today
// is the cluster dimension, which we write through a `int*` reinterpret.
union hipLaunchAttributeValue {
  char pad[64];
};

struct hipLaunchAttribute {
  hipLaunchAttributeID id;
  // Padding so the union starts at offset 8, matching the upstream layout
  // (where the enum is followed by an explicit 4-byte pad to 8-byte align
  // the union).
  char pad[8 - sizeof(hipLaunchAttributeID)];
  union {
    hipLaunchAttributeValue val;
    hipLaunchAttributeValue value;
  };
};

struct HIP_LAUNCH_CONFIG {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  hipStream_t hStream;
  hipLaunchAttribute *attrs;
  unsigned int numAttrs;
};

using hipModuleLaunchKernel_t = hipError_t (*)(hipFunction_t, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               hipStream_t, void **, void **);

using hipDrvLaunchKernelEx_t = hipError_t (*)(const HIP_LAUNCH_CONFIG *,
                                              hipFunction_t, void **, void **);

using hipGetErrorName_t = const char *(*)(hipError_t);
using hipGetErrorString_t = const char *(*)(hipError_t);
using hipModuleUnload_t = hipError_t (*)(hipModule_t);
using hipModuleLoadData_t = hipError_t (*)(hipModule_t *, const void *);
using hipModuleGetFunction_t = hipError_t (*)(hipFunction_t *, hipModule_t,
                                              const char *);

#endif // HC_RUNTIME_HIPTYPES_H
