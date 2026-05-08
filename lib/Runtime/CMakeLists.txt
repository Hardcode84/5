## SPDX-FileCopyrightText: 2026 hc contributors
##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Self-contained Python discovery. We use `Python3::Python` (full libpython
# link) rather than `Python3::Module` because the MLIR/LLVM build inherits
# `-Wl,--no-undefined` from `HandleLLVMOptions`; with `Module` the GIL/refcount
# entry points would be left unresolved at link time even though dlopen would
# resolve them at runtime against the loading interpreter. Linking libpython
# costs us no flexibility here — the helpers are only ever loaded into a
# Python process.
find_package(Python3 COMPONENTS Development REQUIRED)

add_library(hc_rt_helpers SHARED BufferUtils.cpp)
target_include_directories(
  hc_rt_helpers
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(hc_rt_helpers PRIVATE Python3::Python)

install(TARGETS hc_rt_helpers LIBRARY DESTINATION lib)

# HIP launcher shim — Linux-only v0. We deliberately do NOT depend on a
# system ROCm install (no `find_package(HIP)`, no `-lamdhip64`); the
# shim dlopen's libamdhip64.so at runtime via `hc_rt_init`. That keeps
# the wheel buildable on hosts without ROCm and importable on hosts
# that have neither ROCm nor a GPU.
add_library(hc_hip_runtime SHARED HipRuntime.cpp)
target_include_directories(
  hc_hip_runtime
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(hc_hip_runtime PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS hc_hip_runtime LIBRARY DESTINATION lib)
