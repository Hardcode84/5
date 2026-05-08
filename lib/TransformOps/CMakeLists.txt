# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect_library(MLIRHCTransformOps
  HCTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hc/TransformOps

  DEPENDS
  MLIRHCTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRHCDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTransformDialect
)

target_include_directories(
  MLIRHCTransformOps
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
