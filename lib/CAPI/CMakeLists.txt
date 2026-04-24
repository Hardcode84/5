# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_public_c_api_library(HCFrontCAPI
  HCFront.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRHCFrontDialect
)

add_mlir_public_c_api_library(HCCAPI
  HC.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRHCDialect
  MLIRHCFrontToHC
  MLIRHCFrontTransforms
  MLIRHCTransforms
)
