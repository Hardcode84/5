# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_library(MLIRHCTransforms
  HCDecomposeShapedValuesPass.cpp
  HCInferTypesPass.cpp
  HCLowerBarePredicatesPass.cpp
  HCMaterializeBoundExprsPass.cpp
  HCPromoteNamesPass.cpp
  HCVerifyStaticShapesPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hc/Transforms

  DEPENDS
  MLIRHCTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRHCDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRVectorDialect
)

target_include_directories(
  MLIRHCTransforms
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
