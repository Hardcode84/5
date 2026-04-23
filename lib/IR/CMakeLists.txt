# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect_library(MLIRHCDialect
  HCAttrs.cpp
  HCDialect.cpp
  HCOps.cpp
  HCTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hc/IR

  DEPENDS
  MLIRHCAttrsIncGen
  MLIRHCOpsIncGen
  MLIRHCTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(
  MLIRHCDialect
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
