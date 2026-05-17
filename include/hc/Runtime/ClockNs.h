// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal nanosecond monotonic clock for `libhc_hip_runtime.so`. Backed by
// `clock_gettime(CLOCK_MONOTONIC)` on Linux to match `time.perf_counter_ns`.
// Two-helper split leaves room for QPC / RDTSC ports without touching call
// sites; non-Linux fails to compile rather than picking a worse clock.

#ifndef HC_RUNTIME_CLOCKNS_H
#define HC_RUNTIME_CLOCKNS_H

#include <cstdint>

#if defined(__linux__)
#include <time.h>

static inline uint64_t hc_clock_now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
}

static inline uint64_t hc_clock_diff_ns(uint64_t start_ns, uint64_t end_ns) {
  // Saturating: CLOCK_MONOTONIC is non-decreasing, but swapped args
  // shouldn't wrap to ~18e18.
  return end_ns >= start_ns ? end_ns - start_ns : 0;
}

#else
#error                                                                         \
    "hc clock helpers only implement Linux (CLOCK_MONOTONIC) today; port for Windows QPC or RDTSC before targeting this platform"
#endif

#endif // HC_RUNTIME_CLOCKNS_H
