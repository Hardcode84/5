// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-math`: rewrite the NumPy-named `hc.builtin_call`
// carriers into upstream `math.<op>`. See the pass description in
// `include/hc/Transforms/Passes.td` and the carrier-op contract on
// `hc.builtin_call` in `include/hc/IR/HCOps.td`.
//
// Mirrors `-hc-lower-pow` in shape: collect first, mutate after, fail
// the pass with a single localised diagnostic per unsupported case so
// the original `np.<func>(...)` source is what gets fingerprinted
// rather than a downstream LLVM lowering reporting an unknown op.

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

// Single dispatch surface for the NumPy ufunc family carried by
// `hc.builtin_call`. The op definition keeps the name string opaque on
// purpose (so the carrier scales with the wider numpy surface), but
// every supported lowering lives here in one table — new entries are a
// one-row edit. The float-only check matches what upstream `math.*`
// accepts; the math dialect ops themselves take any
// math-compatible operand type uniformly (scalar / vector / shaped),
// so we don't have to dispatch on that here.
static LogicalResult lowerBuiltinCall(HCBuiltinCallOp op) {
  StringRef name = op.getName();
  OperandRange args = op.getArgs();
  // Unary float family covers what the langref blesses today: `sqrt`,
  // `exp`. Binary math (`maximum`, `minimum`) and other ufuncs add
  // their own arms here when they're wired in.
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
