// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-materialize-bound-exprs`: sever `!hc.idx<expr>` /
// `!hc.pred<pred>` SSA from producer chains when the carried expression
// depends only on bound (launch geometry / kernel-ABI) symbols. Replace
// with empty-binding `hc.idx_apply` / `hc.pred_apply` of the same type;
// the launch-body lowering binds the ambient symbols later. Cutting the
// chain here lets scope normalization erase the now-unused workitem /
// subgroup / group producers.

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
  // No kernel anchor: `$`-prefixed names are bound (launch / scope);
  // user / problem symbols stay symbolic until specialization.
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
  if (auto result = dyn_cast<OpResult>(value)) {
    // Empty-binding apply is the fixed point -- don't re-sever.
    Operation *owner = result.getOwner();
    if (auto idx = dyn_cast<HCIdxApplyOp>(owner))
      return !idx.getOperands().empty() || !idx.getSymbols().empty();
    if (auto pred = dyn_cast<HCPredApplyOp>(owner))
      return !pred.getOperands().empty() || !pred.getSymbols().empty();
  }
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

  // Type-only replacement; launch-body lowering binds free symbols ambiently.
  Type type = value.getType();
  ArrayAttr emptySymbols = builder.getStrArrayAttr({});
  Value replacement;
  if (isa<IdxType>(type)) {
    replacement =
        HCIdxApplyOp::create(builder, loc, type, ValueRange{}, emptySymbols)
            .getResult();
  } else {
    replacement =
        HCPredApplyOp::create(builder, loc, type, ValueRange{}, emptySymbols)
            .getResult();
  }
  value.replaceAllUsesExcept(replacement, replacement.getDefiningOp());
}

// First unbound ambient symbol in `exprAttr`; empty if all are bound.
// First-hit only -- diagnostics are per-symbol.
static std::string
firstUndeclaredAmbientSymbolName(Attribute exprAttr, ArrayAttr explicitSymbols,
                                 const BoundSymbolSet &boundSymbols) {
  if (!exprAttr)
    return {};
  llvm::StringSet<> bound;
  for (Attribute attr : explicitSymbols)
    if (auto str = dyn_cast<StringAttr>(attr))
      bound.insert(str.getValue());
  std::string undeclared;
  auto checkName = [&](StringRef name) {
    if (!undeclared.empty())
      return;
    if (bound.contains(name))
      return;
    if (!isBoundSymbolName(name, boundSymbols))
      undeclared = name.str();
  };
  if (auto expr = dyn_cast<ExprAttr>(exprAttr))
    sym::walkSymbolNames(expr.getValue(), checkName);
  if (auto pred = dyn_cast<PredAttr>(exprAttr))
    sym::walkSymbolNames(pred.getValue(), checkName);
  return undeclared;
}

// Every ambient symbol on a severed apply op must appear on the
// enclosing kernel's `bound_symbols`. Explicit-binding syms exempt.
static LogicalResult
verifyMaterializedExprSymbols(Operation *root,
                              const BoundSymbolSet &boundSymbols) {
  auto check = [&](Operation *op, Attribute exprAttr,
                   ArrayAttr explicitSymbols) -> WalkResult {
    std::string undeclared = firstUndeclaredAmbientSymbolName(
        exprAttr, explicitSymbols, boundSymbols);
    if (undeclared.empty())
      return WalkResult::advance();
    op->emitOpError("references undeclared bound symbol '")
        << undeclared << "'";
    return WalkResult::interrupt();
  };

  WalkResult status = root->walk([&](Operation *op) -> WalkResult {
    if (auto idx = dyn_cast<HCIdxApplyOp>(op)) {
      auto idxType = dyn_cast<IdxType>(idx.getResult().getType());
      return check(op, idxType ? idxType.getExpr() : Attribute{},
                   idx.getSymbolsAttr());
    }
    if (auto pred = dyn_cast<HCPredApplyOp>(op)) {
      auto predType = dyn_cast<PredType>(pred.getResult().getType());
      return check(op, predType ? predType.getPred() : Attribute{},
                   pred.getSymbolsAttr());
    }
    return WalkResult::advance();
  });
  return failure(status.wasInterrupted());
}

// Pre-collect -- rewrite phase mutates IR; walk-in-place would invalidate.
static SmallVector<Value>
collectValuesToMaterialize(Operation *root,
                           const BoundSymbolSet &boundSymbols) {
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
  return values;
}

struct HCMaterializeBoundExprsPass
    : public hc::impl::HCMaterializeBoundExprsBase<
          HCMaterializeBoundExprsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    BoundSymbolSet boundSymbols = collectBoundSymbols(root);
    SmallVector<Value> values = collectValuesToMaterialize(root, boundSymbols);

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
