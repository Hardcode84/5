// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Front/IR/HCFrontDialect.h"

#include "hc/Front/IR/HCFrontOps.h"
#include "hc/Front/IR/HCFrontTypes.h"

using namespace mlir::hc::front;

#include "hc/Front/IR/HCFrontOpsDialect.cpp.inc"

void HCFrontDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Front/IR/HCFrontOps.cpp.inc"
      >();
  registerTypes();
}
