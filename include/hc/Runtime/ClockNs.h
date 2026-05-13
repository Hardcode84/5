// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal nanosecond-resolution monotonic clock for `libhc_hip_runtime.so`.
// Two `static inline` helpers wrap the actual time source so callers stay
// platform-agnostic:
//
//   uint64_t hc_clock_now_ns();
//   uint64_t hc_clock_diff_ns(uint64_t start, uint64_t end);
//
// v0 backs both with `clock_gettime(CLOCK_MONOTONIC)` on Linux. That
// matches what Python's `time.perf_counter_ns` does under the hood, so a
// host-side timer driven by `hc_clock_now_ns` can be cross-checked
// against `perf_counter_ns` brackets at sub-µs resolution without
// surprising clock-source skew.
//
// CLOCK_MONOTONIC is preferred over CLOCK_MONOTONIC_RAW: NTP slewing is
// sub-µs over the second-scale samples our bench harness collects, well
// below the noise floor, and matching perf_counter_ns is more valuable
// than skipping the slew.
//
// Future-portability shape: the two-helper split is here so a Windows
// port can implement `hc_clock_now_ns` via `QueryPerformanceCounter` +
// the QPC frequency, or a low-overhead variant via `__rdtsc` + the TSC
// frequency, without touching any call site. Until either lands, the
// non-Linux build refuses to compile rather than silently degrading to
// some less-accurate clock.

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
  // Saturating subtract. CLOCK_MONOTONIC is non-decreasing under POSIX,
  // so this branch is defensive: a caller bug that swaps start/end
  // shouldn't wrap into ~18 quintillion nanoseconds.
  return end_ns >= start_ns ? end_ns - start_ns : 0;
}

#else
#error                                                                         \
    "hc clock helpers only implement Linux (CLOCK_MONOTONIC) today; port for Windows QPC or RDTSC before targeting this platform"
#endif

#endif // HC_RUNTIME_CLOCKNS_H
