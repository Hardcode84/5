// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-infer-generic-bounds`: fill placeholder `iter_bounds`
// on `hc.generic` from operand shape metadata via per-axis identity
// offsets. See the pass description in `include/hc/Transforms/Passes.td`
// and the design in `doc/layouts.md`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCINFERGENERICBOUNDS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// One identity binding for an iter sym: which (operand, axis) supplied
// it, and the dim expression carried by the operand's shape at that
// axis. The operand is kept around so conflict diagnostics can name
// the producer instead of just the iter symbol.
struct ImpliedBound {
  ExprAttr dim;
  Value operand;
  size_t axis;
  StringRef role;
  size_t roleIdx;
};

// Try to read an operand's symbolic shape entry at `axis`. Returns null
// for non-shaped operands (e.g. `!hc.undef`) or non-`#hc.expr` shape
// entries. Either case means the operand can't bind anything here, and
// we move on to the next pair.
static ExprAttr operandDimExpr(Value operand, size_t axis) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(operand.getType());
  if (!shaped)
    return {};
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape || axis >= shape.getDims().size())
    return {};
  return dyn_cast<ExprAttr>(shape.getDims()[axis]);
}

// Identity-only matcher: true iff `offset` is structurally the bare
// symbol `iterSym` under ixsimpl-canonical form. Built via the
// hash-consed compose API so pointer equality is the comparison;
// affine and scaled patterns deliberately fall through the floor for
// now — a richer matcher is the obvious next axis to grow.
static bool offsetIsIdentitySym(sym::Store &store, ExprAttr offset,
                                StringRef iterSym) {
  auto bareHandle = sym::composeExprSym(store, iterSym);
  if (failed(bareHandle))
    return false;
  return offset.getValue() == *bareHandle;
}

// Iter syms whose bound is still a placeholder `hc.undef` — only those
// need inference, the others were already bound by the producer.
static SmallVector<size_t> findPlaceholderIters(OperandRange iterBounds) {
  SmallVector<size_t> placeholders;
  for (auto [iterIdx, bound] : llvm::enumerate(iterBounds))
    if (isa_and_present<HCUndefValueOp>(bound.getDefiningOp()))
      placeholders.push_back(iterIdx);
  return placeholders;
}

// Collect every identity binding for every placeholder sym in one pass
// over the offsets for a single role (ins or outs). The per-iter list
// is grown so the diagnose step can flag conflicts and name both
// producers in the diagnostic.
static void collectImpliedBoundsForRole(
    OperandRange operands, ArrayAttr offsetsAttr, StringRef role,
    ArrayAttr iterSyms, ArrayRef<size_t> placeholderIters, sym::Store &store,
    MutableArrayRef<SmallVector<ImpliedBound, 2>> bindings) {
  for (auto [opIdx, operand, perOperandAttr] :
       llvm::enumerate(operands, offsetsAttr.getAsRange<ArrayAttr>())) {
    for (auto [axis, axisAttr] :
         llvm::enumerate(perOperandAttr.getAsRange<ExprAttr>())) {
      for (auto [slot, iterIdx] : llvm::enumerate(placeholderIters)) {
        StringRef sym = cast<StringAttr>(iterSyms[iterIdx]).getValue();
        if (!offsetIsIdentitySym(store, axisAttr, sym))
          continue;
        ExprAttr dim = operandDimExpr(operand, axis);
        if (!dim)
          continue;
        bindings[slot].push_back({dim, operand, axis, role, opIdx});
      }
    }
  }
}

// Diagnose first, mutate after — keeps the IR untouched on failure.
// An iter sym with no identity occurrence in any operand offset, or
// with conflicting dim expressions across operands, is a hard error.
static LogicalResult
diagnoseImpliedBoundConflicts(HCGenericOp op, ArrayAttr iterSyms,
                              ArrayRef<size_t> placeholderIters,
                              ArrayRef<SmallVector<ImpliedBound, 2>> bindings) {
  for (auto [slot, iterIdx] : llvm::enumerate(placeholderIters)) {
    StringRef sym = cast<StringAttr>(iterSyms[iterIdx]).getValue();
    ArrayRef<ImpliedBound> seen = bindings[slot];
    if (seen.empty())
      return op.emitOpError("iter sym '")
             << sym
             << "' has no identity occurrence in any operand offset; "
                "cannot infer bound from operand shapes";
    ImpliedBound first = seen.front();
    for (ImpliedBound other : seen.drop_front()) {
      if (other.dim.getValue() == first.dim.getValue())
        continue;
      return op.emitOpError("iter sym '")
             << sym << "' has conflicting implied bounds: " << first.role
             << " #" << first.roleIdx << " axis " << first.axis << " implies "
             << first.dim << ", " << other.role << " #" << other.roleIdx
             << " axis " << other.axis << " implies " << other.dim;
    }
  }
  return success();
}

// Materialise an `!hc.idx<dim>` SSA via empty-binding `hc.idx_apply`
// for each placeholder. No explicit operand bindings: the dim
// expression's free symbols are kernel / launch-context names whose
// runtime SSA isn't known here. They stay ambient and the launch-body
// lowering binds them through its existing walk.
static void
materializeInferredBounds(HCGenericOp op, OperandRange iterBounds,
                          ArrayRef<size_t> placeholderIters,
                          ArrayRef<SmallVector<ImpliedBound, 2>> bindings) {
  OpBuilder builder(op);
  for (auto [slot, iterIdx] : llvm::enumerate(placeholderIters)) {
    ImpliedBound bound = bindings[slot].front();
    auto idxType = IdxType::get(op.getContext(), bound.dim);
    auto materialized =
        HCIdxApplyOp::create(builder, op.getLoc(), idxType, ValueRange{},
                             builder.getStrArrayAttr({}));
    op.setOperand(
        static_cast<unsigned>(iterBounds.getBeginOperandIndex() + iterIdx),
        materialized.getResult());
  }
}

static LogicalResult inferOnGeneric(HCGenericOp op, sym::Store &store) {
  ArrayAttr iterSyms = op.getIterSymsAttr();
  OperandRange iterBounds = op.getIterBounds();
  SmallVector<size_t> placeholderIters = findPlaceholderIters(iterBounds);
  if (placeholderIters.empty())
    return success();

  SmallVector<SmallVector<ImpliedBound, 2>> bindings(placeholderIters.size());
  collectImpliedBoundsForRole(op.getIns(), op.getInsOffsetsAttr(), "ins",
                              iterSyms, placeholderIters, store, bindings);
  collectImpliedBoundsForRole(op.getOuts(), op.getOutsOffsetsAttr(), "outs",
                              iterSyms, placeholderIters, store, bindings);

  if (failed(diagnoseImpliedBoundConflicts(op, iterSyms, placeholderIters,
                                           bindings)))
    return failure();

  materializeInferredBounds(op, iterBounds, placeholderIters, bindings);
  return success();
}

struct HCInferGenericBoundsPass
    : public hc::impl::HCInferGenericBoundsBase<HCInferGenericBoundsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
    WalkResult walk = root->walk([&](HCGenericOp op) -> WalkResult {
      if (failed(inferOnGeneric(op, store)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
