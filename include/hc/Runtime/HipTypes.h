// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Hand-coded subset of the HIP ABI touched by the launcher shim. Drops
// the build-time ROCm dependency -- host needs only a C++17 toolchain.
// Entry points are dlsym'd from `libamdhip64.so` by `hc_rt_init`.

#ifndef HC_RUNTIME_HIPTYPES_H
#define HC_RUNTIME_HIPTYPES_H

#include <cstddef>

// Sentinels for `hipModuleLaunchKernel`'s legacy `extra` array. Unused
// today (we route through `hipDrvLaunchKernelEx`'s `kernelParams`), kept
// for public-ABI completeness.
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

// 64 bytes -- fits any upstream attribute value struct. Cluster dim is the
// only attribute we set today; written via `int*` reinterpret.
union hipLaunchAttributeValue {
  char pad[64];
};

struct hipLaunchAttribute {
  hipLaunchAttributeID id;
  // Union at offset 8, matching upstream's explicit 4-byte pad after the
  // enum.
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
using hipStreamSynchronize_t = hipError_t (*)(hipStream_t);

#endif // HC_RUNTIME_HIPTYPES_H
