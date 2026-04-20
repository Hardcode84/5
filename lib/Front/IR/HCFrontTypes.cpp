// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Front/IR/HCFrontTypes.h"

#include "hc/Front/IR/HCFrontDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::hc::front;

#define GET_TYPEDEF_CLASSES
#include "hc/Front/IR/HCFrontOpsTypes.cpp.inc"

void HCFrontDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Front/IR/HCFrontOpsTypes.cpp.inc"
      >();
}
