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
#include "llvm/ADT/StringSet.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCMATERIALIZEBOUNDEXPRS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct BoundSymbolSet {
  llvm::StringSet<> symbols;
  bool hasDeclarations = false;
};

static BoundSymbolSet collectBoundSymbols(Operation *root) {
  BoundSymbolSet boundSymbols;
  root->walk([&](HCKernelOp kernel) {
    ArrayAttr symbols = kernel.getBoundSymbolsAttr();
    if (!symbols)
      return;
    boundSymbols.hasDeclarations = true;
    for (StringAttr symbol : symbols.getAsRange<StringAttr>())
      boundSymbols.symbols.insert(symbol.getValue());
  });
  return boundSymbols;
}

static bool isBoundSymbolName(StringRef name,
                              const BoundSymbolSet &boundSymbols) {
  if (boundSymbols.hasDeclarations)
    return boundSymbols.symbols.count(name);
  // Legacy and helper-only IR can still be run without a kernel metadata
  // anchor. Generated launch and scope symbols use a reserved `$` prefix;
  // user/problem symbols such as `M`/`N` remain symbolic until specialization.
  return name.starts_with("$");
}

template <typename AttrT>
static bool dependsOnlyOnBoundSymbols(AttrT attr,
                                      const BoundSymbolSet &boundSymbols) {
  bool ok = true;
  sym::walkSymbolNames(attr.getValue(), [&](StringRef name) {
    ok &= isBoundSymbolName(name, boundSymbols);
  });
  return ok;
}

static std::string firstUndeclaredSymbol(Attribute attr,
                                         const BoundSymbolSet &boundSymbols) {
  std::string undeclared;
  if (!boundSymbols.hasDeclarations)
    return undeclared;
  auto checkName = [&](StringRef name) {
    if (undeclared.empty() && !isBoundSymbolName(name, boundSymbols))
      undeclared = name.str();
  };
  if (auto expr = dyn_cast<ExprAttr>(attr))
    sym::walkSymbolNames(expr.getValue(), checkName);
  if (auto pred = dyn_cast<PredAttr>(attr))
    sym::walkSymbolNames(pred.getValue(), checkName);
  return undeclared;
}

static bool shouldMaterializeType(Type type,
                                  const BoundSymbolSet &boundSymbols) {
  if (auto idx = dyn_cast_or_null<IdxType>(type))
    return idx.getExpr() &&
           dependsOnlyOnBoundSymbols(idx.getExpr(), boundSymbols);
  if (auto pred = dyn_cast_or_null<PredType>(type))
    return pred.getPred() &&
           dependsOnlyOnBoundSymbols(pred.getPred(), boundSymbols);
  return false;
}

static bool shouldMaterializeValue(Value value,
                                   const BoundSymbolSet &boundSymbols) {
  if (!shouldMaterializeType(value.getType(), boundSymbols) ||
      value.use_empty())
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

static LogicalResult
verifyMaterializedExprSymbols(Operation *root,
                              const BoundSymbolSet &boundSymbols) {
  WalkResult status =
      root->walk([&](HCMaterializeBoundExprOp op) -> WalkResult {
        Type result = op.getResult().getType();
        Attribute expr;
        if (auto idx = dyn_cast<IdxType>(result))
          expr = idx.getExpr();
        else if (auto pred = dyn_cast<PredType>(result))
          expr = pred.getPred();
        if (!expr)
          return WalkResult::advance();
        std::string undeclared = firstUndeclaredSymbol(expr, boundSymbols);
        if (undeclared.empty())
          return WalkResult::advance();
        op->emitOpError("references undeclared bound symbol '")
            << undeclared << "'";
        return WalkResult::interrupt();
      });
  return failure(status.wasInterrupted());
}

struct HCMaterializeBoundExprsPass
    : public hc::impl::HCMaterializeBoundExprsBase<
          HCMaterializeBoundExprsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    BoundSymbolSet boundSymbols = collectBoundSymbols(root);
    SmallVector<Value> values;
    root->walk([&](Operation *op) {
      for (OpResult result : op->getResults())
        if (isNestedUnderHCCallable(result) &&
            shouldMaterializeValue(result, boundSymbols))
          values.push_back(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            if (isNestedUnderHCCallable(arg) &&
                shouldMaterializeValue(arg, boundSymbols))
              values.push_back(arg);
    });

    OpBuilder builder(root->getContext());
    for (Value value : values)
      materializeValue(value, builder);
    if (failed(verifyMaterializedExprSymbols(root, boundSymbols)))
      signalPassFailure();
    if (failed(rejectLiveScopeTokenGeometry(root)))
      signalPassFailure();
  }
};

} // namespace

// `createHCMaterializeBoundExprsPass()` is emitted by tablegen.
