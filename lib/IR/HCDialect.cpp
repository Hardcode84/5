// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCDialect.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include <cassert>

using namespace mlir::hc;

#include "hc/IR/HCOpsDialect.cpp.inc"

void HCDialect::initialize() {
  if (!symbolStore)
    symbolStore = std::make_unique<sym::Store>();
  addOperations<
#define GET_OP_LIST
#include "hc/IR/HCOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}

sym::Store &HCDialect::getSymbolStore() {
  assert(symbolStore && "hc symbolic store must be initialized");
  return *symbolStore;
}

const sym::Store &HCDialect::getSymbolStore() const {
  assert(symbolStore && "hc symbolic store must be initialized");
  return *symbolStore;
}
