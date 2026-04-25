// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPES_H
#define HC_IR_HCTYPES_H

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCTypesInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "hc/IR/HCTypes.h.inc"

namespace mlir::hc {

IdxType getUnpinnedIdxType(MLIRContext *ctx);
PredType getUnpinnedPredType(MLIRContext *ctx);

} // namespace mlir::hc

#endif // HC_IR_HCTYPES_H
