# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_library(MLIRHCFrontTransforms
  HCFrontFoldRegionDefsPass.cpp
  HCFrontInlinePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hc/Front/Transforms

  DEPENDS
  MLIRHCFrontTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRHCFrontDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)

target_include_directories(
  MLIRHCFrontTransforms
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
