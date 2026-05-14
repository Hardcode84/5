// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCATTRS_H
#define HC_IR_HCATTRS_H

#include "hc/IR/HCSymbols.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir

namespace mlir::hc {

class ShapeAttr;

FailureOr<ShapeAttr> parseInlineShapeAttr(AsmParser &parser);
void printInlineShapeAttr(AsmPrinter &printer, ShapeAttr attr);

} // namespace mlir::hc

// The enum header is included before the attrdef header so that the
// generated `ReduceKindAttr` class can refer to the underlying enum.
#include "hc/IR/HCEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/IR/HCAttrs.h.inc"

namespace mlir::hc {

// Compute the storage_size expression a shaped value effectively
// addresses: substitutes `layout.shape_syms` with `originalShape`'s
// dim entries in `layout.storage_size` and returns the canonical
// `ExprAttr`. When `layout` is null the result is the product of the
// shape's dim entries (the default identity layout's storage size).
// Returns failure when the layout's shape_syms arity doesn't match
// the shape's rank or when ixsimpl composition fails.
//
// Promoted to a public helper so any verifier or pass that needs to
// compare a shaped type's address footprint (e.g. `hc.as_layout`'s
// storage-size-equivalence check, or `hc-flatten-with-layouts`'s
// 1-D collapse) reaches the same canonical handle via hash-consing.
mlir::FailureOr<ExprAttr> computeStorageSizeExpr(mlir::MLIRContext *ctx,
                                                 LayoutAttr layout,
                                                 ShapeAttr originalShape);

} // namespace mlir::hc

#endif // HC_IR_HCATTRS_H
