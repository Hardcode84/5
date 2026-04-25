// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Front/IR/HCFrontOps.h"

#include "hc/Front/IR/HCFrontDialect.h"

#define GET_OP_CLASSES
#include "hc/Front/IR/HCFrontOps.cpp.inc"

mlir::LogicalResult mlir::hc::front::InlinedRegionOp::verify() {
  Region &body = getBody();
  if (body.empty() || !body.hasOneBlock())
    return emitOpError("body must be single-block");

  ReturnOp found;
  for (Operation &child : body.front()) {
    auto ret = dyn_cast<ReturnOp>(&child);
    if (!ret)
      continue;
    if (found)
      return emitOpError("body has multiple top-level returns");
    found = ret;
  }
  if (!found)
    return emitOpError("body missing top-level `hc_front.return`");

  if (found.getValues().size() != getNumResults())
    return emitOpError("result arity mismatch: region declares ")
           << getNumResults() << ", body returns " << found.getValues().size();
  return success();
}
