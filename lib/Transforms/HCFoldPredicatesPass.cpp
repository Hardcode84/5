// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-fold-predicates`: walk each `hc.predicate` and
// resolve by inspecting `$value`'s producer. See `doc/lowering.md`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCFOLDPREDICATES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Always-true / always-false short-circuit before producer dispatch.
// `m_One` / `m_Zero` match scalar i1 and `vector<Nxi1>` splats uniformly.
static bool tryFoldTrivialMask(HCPredicateOp op) {
  Value mask = op.getMask();
  if (matchPattern(mask, m_One())) {
    op.replaceAllUsesWith(op.getValue());
    op.erase();
    return true;
  }
  if (matchPattern(mask, m_Zero())) {
    // Always-false → load elided; drop producer if `isOpTriviallyDead`
    // (accepts MemRead-only loads with no remaining uses).
    Value value = op.getValue();
    op.replaceAllUsesWith(op.getPassthrough());
    op.erase();
    if (Operation *producer = value.getDefiningOp())
      if (isOpTriviallyDead(producer))
        producer->erase();
    return true;
  }
  return false;
}

// Allow-list — unknown producer is a diagnostic, not passthrough.
static LogicalResult foldPredicate(HCPredicateOp op, const DominanceInfo &dom) {
  if (tryFoldTrivialMask(op))
    return success();

  Value value = op.getValue();
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return op.emitOpError(
        "predicate value is a block argument; supported producers are "
        "hc.ptr_load and vector.extract");

  if (auto load = dyn_cast<HCPtrLoadOp>(producer)) {
    // Mask / passthrough must dominate the load — predicated clone
    // replaces the load in place, not at the predicate use site.
    Value mask = op.getMask();
    Value pass = op.getPassthrough();
    if (!dom.dominates(mask, load))
      return op.emitOpError("mask does not dominate the hc.ptr_load producer");
    if (!dom.dominates(pass, load))
      return op.emitOpError(
          "passthrough does not dominate the hc.ptr_load producer");

    OpBuilder b(load);
    auto pred = HCPtrLoadPredOp::create(b, load.getLoc(), value.getType(),
                                        load.getSource(), mask, pass);
    op.replaceAllUsesWith(pred.getResult());
    op.erase();
    // Per-use cloning; original survives only if non-predicated uses remain.
    if (load.use_empty())
      load.erase();
    return success();
  }

  if (isa<vector::ExtractOp>(producer)) {
    // Load stays unconditional; only the extracted lane is gated.
    OpBuilder b(op);
    auto sel = arith::SelectOp::create(b, op.getLoc(), op.getMask(), value,
                                       op.getPassthrough());
    op.replaceAllUsesWith(sel.getResult());
    op.erase();
    return success();
  }

  if (isa<HCPtrLoadPredOp>(producer))
    return op.emitOpError(
        "value is already produced by hc.ptr_load_pred; double-predicating "
        "is not supported (combine masks at the producer)");

  return op.emitOpError("unsupported producer for predicate hoist: '")
         << producer->getName()
         << "'; supported producers are hc.ptr_load and vector.extract";
}

struct HCFoldPredicatesPass
    : public hc::impl::HCFoldPredicatesBase<HCFoldPredicatesPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    DominanceInfo dom(root);

    // Collect first; folding mutates IR so walk-in-place would invalidate
    // the iterator. Diagnostics short-circuit; partial rewrites are fine
    // after the failure signal.
    SmallVector<HCPredicateOp> worklist;
    root->walk([&](HCPredicateOp op) { worklist.push_back(op); });

    for (HCPredicateOp op : worklist) {
      if (failed(foldPredicate(op, dom)))
        return signalPassFailure();
    }
  }
};

} // namespace
