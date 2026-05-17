// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-verify-bare-carriers`: launch-body entry gate. Two checks:
// no semantic `!hc.tensor` / `!hc.vector` survives past decompose,
// and every bare carrier has a static shape. Hoisted out of
// `hc-lower-launch-body` so the pair fires once instead of per
// launch-body invocation.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCVERIFYBARECARRIERS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static bool isSemanticShapedType(Type type) {
  return isa<hc::TensorType, hc::VectorType>(type);
}

static LogicalResult assertNoSemanticShapedSurvives(Operation *rootOp) {
  WalkResult walk = rootOp->walk([&](Operation *op) {
    auto bail = [&](Type type, StringRef role) -> WalkResult {
      op->emitOpError("semantic shaped type ")
          << type << " survived past hc-decompose-shaped-values on " << role
          << "; decompose must split !hc.tensor / !hc.vector into bare "
             "(data, mask) pairs before hc-lower-launch-body runs";
      return WalkResult::interrupt();
    };
    for (Value v : op->getOperands())
      if (isSemanticShapedType(v.getType()))
        return bail(v.getType(), "operand");
    for (Type t : op->getResultTypes())
      if (isSemanticShapedType(t))
        return bail(t, "result");
    return WalkResult::advance();
  });
  return success(!walk.wasInterrupted());
}

static bool isStaticShape(ShapeAttr shape) {
  if (!shape)
    return true;
  for (Attribute attr : shape.getDims()) {
    auto expr = dyn_cast<ExprAttr>(attr);
    if (!expr)
      return false;
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()));
    if (!value || *value < 0)
      return false;
  }
  return true;
}

static LogicalResult assertBareCarriersAreStaticShape(Operation *rootOp) {
  WalkResult walk = rootOp->walk([&](Operation *op) {
    auto checkType = [&](Type type, StringRef role) -> WalkResult {
      auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
      if (!shaped || !isa<BareTensorType, BareVectorType>(type))
        return WalkResult::advance();
      if (isStaticShape(shaped.getSymbolicShape()))
        return WalkResult::advance();
      op->emitOpError("bare carrier ")
          << type << " has a non-literal shape on " << role
          << "; hc-lower-launch-body needs every dim resolved to an integer "
             "literal to allocate the workgroup tile, so bind the free "
             "symbol(s) via hc.compile(symbols={...}) (add the symbol to the "
             "kernel decorator's `literals=` set if it isn't already)";
      return WalkResult::interrupt();
    };
    for (Value v : op->getOperands())
      if (WalkResult r = checkType(v.getType(), "operand"); r.wasInterrupted())
        return r;
    for (Type t : op->getResultTypes())
      if (WalkResult r = checkType(t, "result"); r.wasInterrupted())
        return r;
    return WalkResult::advance();
  });
  return success(!walk.wasInterrupted());
}

struct HCVerifyBareCarriersPass
    : public hc::impl::HCVerifyBareCarriersBase<HCVerifyBareCarriersPass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(assertNoSemanticShapedSurvives(getOperation())))
      return signalPassFailure();
    if (failed(assertBareCarriersAreStaticShape(getOperation())))
      signalPassFailure();
  }
};

} // namespace
