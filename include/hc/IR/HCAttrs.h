// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCATTRS_H
#define HC_IR_HCATTRS_H

#include "hc/IR/HCSymbols.h"
#include "mlir/IR/Attributes.h"

namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir

namespace mlir::hc {

class ShapeAttr;

FailureOr<ShapeAttr> parseInlineShapeAttr(AsmParser &parser);
void printInlineShapeAttr(AsmPrinter &printer, ShapeAttr attr);

} // namespace mlir::hc

#define GET_ATTRDEF_CLASSES
#include "hc/IR/HCAttrs.h.inc"

#endif // HC_IR_HCATTRS_H
