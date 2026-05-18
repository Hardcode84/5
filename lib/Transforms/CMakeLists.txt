# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_library(MLIRHCTransforms
  HCBridgeIntrinsicsPass.cpp
  HCCanonicalizeLayoutsPass.cpp
  HCDecomposeShapedValuesPass.cpp
  HCDistributeWaveLayoutsPass.cpp
  HCElementwiseToGenericPass.cpp
  HCEmitBenchWrapperPass.cpp
  HCFlattenWithLayoutsPass.cpp
  HCFoldPredicatesPass.cpp
  HCInferTypesPass.cpp
  HCInlineHelpersPass.cpp
  HCInsertWorkgroupBarriersPass.cpp
  HCInterpretIntrinsicRecipesPass.cpp
  HCLoadStoreToGenericPass.cpp
  HCLowerApplyPass.cpp
  HCLowerGenericPass.cpp
  HCLowerGPUToBinaryPass.cpp
  HCLowerKernelsToGPULaunchPass.cpp
  HCLowerLaunchBodyPass.cpp
  HCLowerLaunchFuncToRuntimePass.cpp
  HCLowerLaunchScalarOpsPass.cpp
  HCLowerLaunchShapedConstantsPass.cpp
  HCLowerMathPass.cpp
  HCLowerPowPass.cpp
  HCLowerStripLayoutPass.cpp
  HCLowerToLLVMPass.cpp
  HCMaterializeBoundExprsPass.cpp
  HCNormalizeScopeRegionsPass.cpp
  HCPromoteNamesPass.cpp
  HCReconcileGenericOperandsPass.cpp
  HCShapedComputeToGenericPass.cpp
  HCSpecializeLiteralsPass.cpp
  HCVerifyBareCarriersPass.cpp
  HCVerifyStaticShapesPass.cpp
  LaunchUtils.cpp

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
  MLIRMathDialect
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
