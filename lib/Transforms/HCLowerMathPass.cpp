// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-math`: rewrite NumPy-named `hc.builtin_call`
// carriers into upstream `math.<op>`. Collect first, mutate after.
// Diagnose at the original op so failure points at `np.<func>(...)`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERMATH
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Single dispatch table for the NumPy ufunc family. Float-only --
// upstream `math.*` rejects non-float element types.
static LogicalResult lowerBuiltinCall(HCBuiltinCallOp op) {
  StringRef name = op.getName();
  OperandRange args = op.getArgs();
  if (name == "numpy.sqrt" || name == "numpy.exp") {
    if (args.size() != 1)
      return op.emitOpError("`hc.builtin_call \"")
             << name << "\"` expects exactly one operand; got " << args.size();
    Value arg = args.front();
    Type argType = arg.getType();
    if (!isa<FloatType>(getElementTypeOrSelf(argType)))
      return op.emitOpError("`hc.builtin_call \"")
             << name
             << "\"`: only float-element operand is supported today; got "
             << argType;
    OpBuilder builder(op);
    Value replacement;
    if (name == "numpy.sqrt")
      replacement = math::SqrtOp::create(builder, op.getLoc(), arg).getResult();
    else
      replacement = math::ExpOp::create(builder, op.getLoc(), arg).getResult();
    op.getResult().replaceAllUsesWith(replacement);
    op.erase();
    return success();
  }
  return op.emitOpError("`hc.builtin_call \"")
         << name << "\"`: no lowering registered";
}

struct HCLowerMathPass : public hc::impl::HCLowerMathBase<HCLowerMathPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<HCBuiltinCallOp> calls;
    getOperation()->walk([&](HCBuiltinCallOp op) { calls.push_back(op); });
    for (HCBuiltinCallOp op : calls)
      if (failed(lowerBuiltinCall(op)))
        return signalPassFailure();
  }
};

} // namespace
