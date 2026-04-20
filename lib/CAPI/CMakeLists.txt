# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_public_c_api_library(HCFrontCAPI
  HCFront.cpp

  LINK_LIBS PUBLIC
  MLIRHCFrontDialect
)
