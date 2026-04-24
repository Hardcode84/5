# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(HCIXSimpl STATIC "${PROJECT_SOURCE_DIR}/third_party/ixsimpl/ixsimpl_amalg.c")
target_include_directories(
  HCIXSimpl
  PUBLIC
  "${PROJECT_SOURCE_DIR}/third_party/ixsimpl/include"
  PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/ixsimpl/src"
)

add_subdirectory(IR)
add_subdirectory(Front/IR)
add_subdirectory(Front/Transforms)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(CAPI)
