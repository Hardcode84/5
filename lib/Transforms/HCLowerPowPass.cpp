// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-pow`: rewrite `hc.pow lhs, rhs` into an
// `hc.mul` chain via binary squaring. Only positive integer-literal
// exponents supported; everything else diagnoses at the original `**`.
// Generic `math.pow` codegen needs a separate HC math op.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERPOW
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Binary squaring high-to-low; `exp == 1` returns `lhs` unchanged.
static Value emitIntegerPow(OpBuilder &builder, Location loc, Value lhs,
                            int64_t exp, Type resultType) {
  assert(exp >= 1 && "emitIntegerPow expects a positive exponent");
  int highBit = 0;
  while ((int64_t(1) << (highBit + 1)) <= exp)
    ++highBit;
  Value result = lhs;
  for (int bit = highBit - 1; bit >= 0; --bit) {
    result = HCMulOp::create(builder, loc, resultType, result, result);
    if ((exp >> bit) & 1)
      result = HCMulOp::create(builder, loc, resultType, result, lhs);
  }
  return result;
}

static LogicalResult lowerPow(HCPowOp op) {
  Value rhs = op.getRhs();
  auto constOp = rhs.getDefiningOp<HCConstOp>();
  if (!constOp)
    return op.emitOpError(
        "unsupported `hc.pow` rhs: only positive integer-literal exponents "
        "are supported today; got a non-constant rhs. Runtime exponents "
        "need a dedicated `math.pow`-style lowering that doesn't exist "
        "yet");
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return op.emitOpError("unsupported `hc.pow` rhs: only positive "
                          "integer-literal exponents are supported today; got "
                          "rhs constant ")
           << constOp.getValue();
  int64_t exp = intAttr.getInt();
  if (exp < 1)
    return op.emitOpError("unsupported `hc.pow` rhs: only positive "
                          "integer-literal exponents are supported today; got ")
           << exp;
  OpBuilder builder(op);
  Value lhs = op.getLhs();
  Type resultType = op.getResult().getType();
  Value replacement =
      exp == 1 ? lhs
               : emitIntegerPow(builder, op.getLoc(), lhs, exp, resultType);
  op.getResult().replaceAllUsesWith(replacement);
  op.erase();
  return success();
}

struct HCLowerPowPass : public hc::impl::HCLowerPowBase<HCLowerPowPass> {
  using Base::Base;

  void runOnOperation() override {
    // Collect first; erase-in-walk invalidates the iterator.
    SmallVector<HCPowOp> pows;
    getOperation()->walk([&](HCPowOp op) { pows.push_back(op); });
    for (HCPowOp op : pows) {
      if (failed(lowerPow(op)))
        return signalPassFailure();
    }
  }
};

} // namespace
