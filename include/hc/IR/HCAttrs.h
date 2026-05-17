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

// Enum header before attrdef header: generated `ReduceKindAttr` refers to
// the underlying enum.
#include "hc/IR/HCEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/IR/HCAttrs.h.inc"

namespace mlir::hc {

/// Storage extent in elements addressed by a shaped value. Substitutes
/// `layout.shape_syms` with `originalShape` dims into `layout.storage_size`;
/// null `layout` yields the dim product (identity-layout default). Fails on
/// rank mismatch or ixsimpl composition failure.
mlir::FailureOr<ExprAttr> computeStorageSizeExpr(mlir::MLIRContext *ctx,
                                                 LayoutAttr layout,
                                                 ShapeAttr originalShape);

/// Linear access offset for `(layout, originalShape)` at per-axis
/// `indexExprs`. `shape_syms` bind positionally to shape entries,
/// `index_syms` to `indexExprs`. Null `layout` uses identity row-major
/// over dims. Fails on rank mismatch or ixsimpl composition failure.
mlir::FailureOr<ExprAttr>
composeAccessOffsetExpr(mlir::MLIRContext *ctx, LayoutAttr layout,
                        ShapeAttr originalShape,
                        mlir::ArrayRef<ExprAttr> indexExprs);

} // namespace mlir::hc

#endif // HC_IR_HCATTRS_H
