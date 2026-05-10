// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-materialize-bound-exprs`, the HC-to-HC boundary that severs
// launch/bound symbolic SSA values from their producer chains before scope
// normalization. Replaces every reachable `!hc.idx<expr>` / `!hc.pred<pred>`
// SSA value whose carried expression depends only on bound (launch geometry
// or kernel-declared ABI) symbols with a fresh `hc.idx_apply` / `hc.pred_apply`
// of the same type. The replacement carries no operand bindings — the listed
// symbols stay ambient and the launch-body lowering resolves them later from
// launch context. Cutting the SSA chain here lets scope normalization erase
// the now-unused workitem / subgroup / group producer operations without
// dragging dependent index / predicate values along.

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
  if (auto result = dyn_cast<OpResult>(value)) {
    // Already-severed values (an empty-symbol `idx_apply` /
    // `pred_apply` carrying just a type) are the post-pass fixed
    // point — re-severing them would loop without progress.
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

  // Empty operand / symbol lists: the new value carries the type
  // only, severing the SSA chain just like the legacy
  // `materialize_bound_expr`. Lowering binds the type's free symbols
  // ambiently from the surrounding launch context.
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

// Validates that every severed apply op's residual ambient symbols
// are declared on the enclosing kernel's `bound_symbols` attribute.
// Any free symbol the op explicitly binds via its `symbols` list is
// resolved structurally and doesn't need to appear in the kernel's
// declaration; only ambient names go through this check.
static LogicalResult
verifyMaterializedExprSymbols(Operation *root,
                              const BoundSymbolSet &boundSymbols) {
  auto check = [&](Operation *op, Attribute exprAttr,
                   ArrayAttr explicitSymbols) -> WalkResult {
    if (!exprAttr)
      return WalkResult::advance();
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
