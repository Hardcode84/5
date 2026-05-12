// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-fold-predicates`, the producer-hoist lowering for
// `hc.predicate` documented in `doc/lowering.md`. Runs after every emitter
// that can plant predicates (primarily `hc-lower-generic`); walks each
// `hc.predicate` and resolves it by inspecting `$value`'s producer.

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

// Trivial mask folds run before producer dispatch so always-true / always-false
// masks never reach the allow-list. Vector splat constants match too — the
// upstream `m_One` / `m_Zero` matchers handle scalar i1 and vector<Nxi1>
// uniformly.
static bool tryFoldTrivialMask(HCPredicateOp op) {
  Value mask = op.getMask();
  if (matchPattern(mask, m_One())) {
    op.replaceAllUsesWith(op.getValue());
    op.erase();
    return true;
  }
  if (matchPattern(mask, m_Zero())) {
    // The user's intent on an always-false mask is "load elided" — if the
    // predicate was the value's only consumer, drop the producer too.
    // `isOpTriviallyDead` handles the side-effect bookkeeping for us (it
    // accepts MemRead-only loads with no remaining uses).
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

// Producer dispatch — the allow-list is deliberate. Extending it is a
// per-producer choice; the default action on unrecognised producers is a
// diagnostic, not silent passthrough.
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
    // Mask / passthrough must dominate the load's site because the
    // predicated clone physically replaces the load there, not at the
    // hc.predicate use site. If the user computed the mask after the
    // load (rare — usually they precede the load by construction), the
    // schedule has to be fixed up at the source.
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
    // Per-use cloning: each hc.predicate gets its own predicated load.
    // The original unpredicated load survives only if some non-predicated
    // use is still live; otherwise it is dead and goes away here.
    if (load.use_empty())
      load.erase();
    return success();
  }

  if (isa<vector::ExtractOp>(producer)) {
    // The vector load behind the extract stays unconditional — only the
    // extracted lane is gated. arith.select at the predicate site is the
    // exact semantic.
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

    // Collect first, rewrite second — folding mutates the IR (erases ops,
    // inserts new ones at the producer's site) so iterating a walk in place
    // would invalidate the iterator. Diagnostics short-circuit the worklist;
    // partial rewrites are fine to leave on the IR since the pass already
    // signalled failure.
    SmallVector<HCPredicateOp> worklist;
    root->walk([&](HCPredicateOp op) { worklist.push_back(op); });

    for (HCPredicateOp op : worklist) {
      if (failed(foldPredicate(op, dom)))
        return signalPassFailure();
    }
  }
};

} // namespace
