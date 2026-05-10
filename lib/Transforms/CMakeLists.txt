# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_library(MLIRHCTransforms
  HCCanonicalizeLayoutsPass.cpp
  HCDecomposeShapedValuesPass.cpp
  HCFlattenWithLayoutsPass.cpp
  HCInferGenericBoundsPass.cpp
  HCInferTypesPass.cpp
  HCInlineHelpersPass.cpp
  HCInterpretIntrinsicRecipesPass.cpp
  HCLoadStoreToGenericPass.cpp
  HCLowerBarePredicatesPass.cpp
  HCLowerGPUToBinaryPass.cpp
  HCLowerKernelsToGPULaunchPass.cpp
  HCLowerLaunchBodyPass.cpp
  HCLowerLaunchFuncToRuntimePass.cpp
  HCMaterializeBoundExprsPass.cpp
  HCNormalizeScopeRegionsPass.cpp
  HCPromoteNamesPass.cpp
  HCShapedComputeToGenericPass.cpp
  HCVerifyStaticShapesPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hc/Transforms

  DEPENDS
  MLIRHCTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRExecutionEngineUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRGPUDialect
  MLIRHCDialect
  MLIRHCTransformOps
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRROCDLDialect
  MLIRROCDLTarget
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTargetLLVM
  MLIRTargetLLVMIRExport
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRVectorDialect
)

target_include_directories(
  MLIRHCTransforms
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
