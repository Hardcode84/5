// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-materialize-bound-exprs`, the HC-to-HC boundary that severs
// launch/bound symbolic SSA values from their producer chains before scope
// normalization.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCMATERIALIZEBOUNDEXPRS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static bool isBoundSymbolName(StringRef name) {
  // The richer kernel metadata contract is tracked separately. Today generated
  // launch and scope symbols are reserved with a `$` prefix, while user/problem
  // symbols such as `M`/`N` must remain symbolic until specialization.
  return name.starts_with("$");
}

template <typename AttrT> static bool dependsOnlyOnBoundSymbols(AttrT attr) {
  bool ok = true;
  sym::walkSymbolNames(attr.getValue(),
                       [&](StringRef name) { ok &= isBoundSymbolName(name); });
  return ok;
}

static bool shouldMaterializeType(Type type) {
  if (auto idx = dyn_cast_or_null<IdxType>(type))
    return idx.getExpr() && dependsOnlyOnBoundSymbols(idx.getExpr());
  if (auto pred = dyn_cast_or_null<PredType>(type))
    return pred.getPred() && dependsOnlyOnBoundSymbols(pred.getPred());
  return false;
}

static bool shouldMaterializeValue(Value value) {
  if (!shouldMaterializeType(value.getType()) || value.use_empty())
    return false;
  if (auto result = dyn_cast<OpResult>(value))
    return !isa<HCMaterializeBoundExprOp>(result.getOwner());
  return true;
}

static bool isScopeToken(Type type) {
  return isa<WorkitemType, SubgroupType>(type);
}

static bool isLaunchGeometryOp(Operation *op) {
  return isa<HCGroupIdOp, HCLocalIdOp, HCSubgroupIdOp, HCGroupShapeOp,
             HCGroupSizeOp, HCWorkOffsetOp, HCWorkShapeOp, HCWaveSizeOp>(op);
}

static LogicalResult rejectLiveScopeTokenGeometry(Operation *root) {
  WalkResult walkStatus = root->walk([&](Operation *op) -> WalkResult {
    if (!isLaunchGeometryOp(op) || op->use_empty())
      return WalkResult::advance();
    ValueRange operands = op->getOperands();
    if (operands.empty() || !isScopeToken(operands.front().getType()))
      return WalkResult::advance();
    op->emitOpError("still has live results depending on a workitem/subgroup "
                    "scope token after bound-expression materialization");
    return WalkResult::interrupt();
  });
  return failure(walkStatus.wasInterrupted());
}

static Operation *nearestHCCallable(Operation *op) {
  while (op) {
    if (isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(op))
      return op;
    op = op->getParentOp();
  }
  return nullptr;
}

static bool isNestedUnderHCCallable(Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return nearestHCCallable(result.getOwner()) != nullptr;
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && nearestHCCallable(arg.getOwner()->getParentOp()) != nullptr;
}

static void materializeValue(Value value, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = value.getLoc();
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *owner = result.getOwner();
    builder.setInsertionPointAfter(owner);
    loc = owner->getLoc();
  } else {
    auto arg = cast<BlockArgument>(value);
    Block *block = arg.getOwner();
    builder.setInsertionPointToStart(block);
    if (Operation *parent = block->getParentOp())
      loc = parent->getLoc();
  }

  auto materialized =
      HCMaterializeBoundExprOp::create(builder, loc, value.getType());
  value.replaceAllUsesExcept(materialized.getResult(), materialized);
}

struct HCMaterializeBoundExprsPass
    : public hc::impl::HCMaterializeBoundExprsBase<
          HCMaterializeBoundExprsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    SmallVector<Value> values;
    root->walk([&](Operation *op) {
      for (OpResult result : op->getResults())
        if (isNestedUnderHCCallable(result) && shouldMaterializeValue(result))
          values.push_back(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            if (isNestedUnderHCCallable(arg) && shouldMaterializeValue(arg))
              values.push_back(arg);
    });

    OpBuilder builder(root->getContext());
    for (Value value : values)
      materializeValue(value, builder);
    if (failed(rejectLiveScopeTokenGeometry(root)))
      signalPassFailure();
  }
};

} // namespace

// `createHCMaterializeBoundExprsPass()` is emitted by tablegen.
