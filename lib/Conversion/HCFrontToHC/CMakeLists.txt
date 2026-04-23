# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_library(MLIRHCFrontToHC
  HCFrontToHCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hc/Conversion/HCFrontToHC

  DEPENDS
  MLIRHCFrontToHCConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRHCDialect
  MLIRHCFrontDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)

target_include_directories(
  MLIRHCFrontToHC
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
