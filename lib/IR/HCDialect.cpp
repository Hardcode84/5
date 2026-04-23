// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCDialect.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

using namespace mlir::hc;

#include "hc/IR/HCOpsDialect.cpp.inc"

void HCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/IR/HCOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}
