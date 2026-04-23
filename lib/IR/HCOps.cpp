// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"

using namespace mlir;
using namespace mlir::hc;

#define GET_OP_CLASSES
#include "hc/IR/HCOps.cpp.inc"

LogicalResult HCSymbolOp::verify() {
  // A bare `!hc.idx` is the inferred form of an unbound capture, not a
  // user-declared symbol; `hc.symbol` must carry a concrete expression in its
  // result type so the bound name is visible at one place.
  IdxType type = llvm::dyn_cast<IdxType>(getResult().getType());
  if (!type)
    return emitOpError("result must be an `!hc.idx` with a pinned expression");
  if (!type.getExpr())
    return emitOpError("result must pin a symbolic expression "
                       "(e.g. `!hc.idx<\"M\">`)");
  return success();
}
