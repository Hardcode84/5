// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-pow`: rewrite `hc.pow lhs, rhs` into the
// `hc.mul` chain the rest of the pipeline understands. See the pass
// description in `include/hc/Transforms/Passes.td` and the carrier-op
// contract on `hc.pow` in `include/hc/IR/HCOps.td`.
//
// Today only positive integer-literal exponents are supported — the
// realistic in-pipeline shape (`** 2` for squared distances, `** N`
// for fixed small N) — via binary squaring on the bit walk of the
// exponent. Non-constant, non-integer, zero, and negative exponents
// each get their own diagnostic so the failure fingers the original
// `**`. Generic `math.pow` codegen would need a new HC math op and is
// intentionally out of scope here.

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

// Binary exponentiation: walk bits of `exp` high-to-low, squaring the
// accumulator on each step and multiplying by `lhs` on every set bit.
// `exp == 1` returns `lhs` unchanged (loop runs zero times). Counts: K
// mults for K in {2,3} → 1,2; K in {4..7} → 2,3,3,4 — cheaper than the
// naive K-1 chain for K >= 4.
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
    // Collect first, mutate after — erasing inside the walk would
    // invalidate the iterator the walk is driving. Same shape as the
    // sibling `-hc-lower-strip-layout` pass.
    SmallVector<HCPowOp> pows;
    getOperation()->walk([&](HCPowOp op) { pows.push_back(op); });
    for (HCPowOp op : pows) {
      if (failed(lowerPow(op)))
        return signalPassFailure();
    }
  }
};

} // namespace
