# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(hc/IR)
add_subdirectory(hc/Front/IR)
