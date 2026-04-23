// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCAttrs.h"

#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::hc;

#define GET_ATTRDEF_CLASSES
#include "hc/IR/HCAttrs.cpp.inc"

void HCDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/IR/HCAttrs.cpp.inc"
      >();
}
