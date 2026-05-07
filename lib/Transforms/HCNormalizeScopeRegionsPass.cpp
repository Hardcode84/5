// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-normalize-scope-regions`, the HC-to-HC normalization that
// starts turning collective regions into per-workitem SPMD IR.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCNORMALIZESCOPEREGIONS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

using RemovedArgMap = llvm::StringMap<SmallVector<unsigned>>;

static bool isScopeToken(Type type) {
  return isa<WorkitemType, SubgroupType>(type);
}

static std::optional<LaunchContextMetadata>
launchMetadataFromCallable(Operation *callable) {
  Region &body = callable->getRegion(0);
  if (body.empty())
    return std::nullopt;
  for (BlockArgument arg : body.front().getArguments())
    if (std::optional<LaunchContextMetadata> metadata =
            getLaunchContextMetadata(arg.getType()))
      return metadata;
  std::optional<LaunchContextMetadata> nestedMetadata;
  body.walk([&](Operation *op) {
    if (nestedMetadata)
      return WalkResult::interrupt();
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      for (BlockArgument arg : region.front().getArguments()) {
        nestedMetadata = getLaunchContextMetadata(arg.getType());
        if (nestedMetadata)
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (nestedMetadata)
    return nestedMetadata;
  return std::nullopt;
}

static bool hasSuffix(ArrayRef<Attribute> dims, ArrayRef<Attribute> suffix) {
  return dims.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), dims.rbegin());
}

static Type dropWorkitemSuffix(Type type, ArrayRef<Attribute> suffix) {
  if (suffix.empty())
    return type;
  if (auto tuple = dyn_cast<TupleType>(type)) {
    SmallVector<Type> elements;
    bool changed = false;
    elements.reserve(tuple.size());
    for (Type element : tuple.getTypes()) {
      Type converted = dropWorkitemSuffix(element, suffix);
      changed |= converted != element;
      elements.push_back(converted);
    }
    return changed ? TupleType::get(type.getContext(), elements) : type;
  }

  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped || !isa<mlir::hc::VectorType, BareVectorType>(type))
    return type;
  ArrayRef<Attribute> dims = shaped.getSymbolicShape().getDims();
  if (!hasSuffix(dims, suffix))
    return type;

  SmallVector<Attribute> localDims(dims.drop_back(suffix.size()));
  ShapeAttr localShape = ShapeAttr::get(type.getContext(), localDims);
  if (auto vector = dyn_cast<mlir::hc::VectorType>(type))
    return mlir::hc::VectorType::get(type.getContext(), vector.getElementType(),
                                     localShape);
  auto bareVector = cast<BareVectorType>(type);
  return BareVectorType::get(type.getContext(), bareVector.getElementType(),
                             localShape);
}

static void dropWorkitemSuffixFromCallable(Operation *callable,
                                           ArrayRef<Attribute> suffix) {
  auto rewriteType = [&](Type type) {
    return dropWorkitemSuffix(type, suffix);
  };

  if (auto func = dyn_cast<HCFuncOp>(callable)) {
    if (TypeAttr fnTypeAttr = func.getFunctionTypeAttr()) {
      auto fnType = cast<FunctionType>(fnTypeAttr.getValue());
      SmallVector<Type> inputs;
      SmallVector<Type> results;
      llvm::transform(fnType.getInputs(), std::back_inserter(inputs),
                      rewriteType);
      llvm::transform(fnType.getResults(), std::back_inserter(results),
                      rewriteType);
      func.setFunctionTypeAttr(
          TypeAttr::get(FunctionType::get(func.getContext(), inputs, results)));
    }
  }

  callable->walk([&](Operation *op) {
    if (op != callable && isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(op))
      return WalkResult::skip();
    for (OpResult result : op->getResults())
      result.setType(rewriteType(result.getType()));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(rewriteType(arg.getType()));
    return WalkResult::advance();
  });
}

static SmallVector<unsigned> eraseUnusedScopeTokenArgs(HCFuncOp func) {
  if (func.getBody().empty())
    return {};

  Block &body = func.getBody().front();
  SmallVector<unsigned> erased;
  for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
    if (isScopeToken(arg.getType()) && arg.use_empty())
      erased.push_back(index);
  }
  if (erased.empty())
    return erased;

  if (TypeAttr fnTypeAttr = func.getFunctionTypeAttr()) {
    auto fnType = cast<FunctionType>(fnTypeAttr.getValue());
    SmallVector<Type> inputs;
    llvm::SmallDenseSet<unsigned> erasedSet(erased.begin(), erased.end());
    for (auto [index, type] : llvm::enumerate(fnType.getInputs()))
      if (!erasedSet.contains(index))
        inputs.push_back(type);
    func.setFunctionTypeAttr(TypeAttr::get(
        FunctionType::get(func.getContext(), inputs, fnType.getResults())));
  }

  for (unsigned index : llvm::reverse(erased))
    body.eraseArgument(index);
  return erased;
}

static void eraseCallOperands(ArrayRef<unsigned> erasedIndices, HCCallOp call,
                              OpBuilder &builder) {
  if (erasedIndices.empty())
    return;

  llvm::SmallDenseSet<unsigned> erasedSet(erasedIndices.begin(),
                                          erasedIndices.end());
  SmallVector<Value> args;
  for (auto [index, arg] : llvm::enumerate(call.getArgs()))
    if (!erasedSet.contains(index))
      args.push_back(arg);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(call);
  auto replacement =
      HCCallOp::create(builder, call.getLoc(), call.getResultTypes(),
                       call.getCalleeAttr(), args);
  replacement->setAttrs(call->getAttrs());
  call.replaceAllUsesWith(replacement->getResults());
  call.erase();
}

static bool dropUnusedScopeTokenArgsOnce(Operation *root) {
  RemovedArgMap erasedByCallee;
  root->walk([&](HCFuncOp func) {
    SmallVector<unsigned> erased = eraseUnusedScopeTokenArgs(func);
    if (!erased.empty())
      erasedByCallee[func.getName()] = std::move(erased);
  });
  if (erasedByCallee.empty())
    return false;

  OpBuilder builder(root->getContext());
  SmallVector<HCCallOp> calls;
  root->walk([&](HCCallOp call) { calls.push_back(call); });
  for (HCCallOp call : calls) {
    auto it = erasedByCallee.find(call.getCallee());
    if (it != erasedByCallee.end())
      eraseCallOperands(it->second, call, builder);
  }
  return true;
}

static void dropUnusedScopeTokenArgs(Operation *root) {
  while (dropUnusedScopeTokenArgsOnce(root)) {
  }
}

static LogicalResult requireNoLiveBlockArguments(HCWorkitemRegionOp op) {
  Block &body = op.getBody().front();
  for (BlockArgument arg : body.getArguments()) {
    for (Operation &nested : body) {
      if (llvm::any_of(nested.getOperands(),
                       [&](Value operand) { return operand == arg; }))
        return op.emitOpError("cannot flatten workitem region while its scope "
                              "token is still live; run "
                              "`hc-materialize-bound-exprs` first");
    }
  }
  return success();
}

static LogicalResult flattenResultlessWorkitemRegion(HCWorkitemRegionOp op) {
  SmallVector<Value> yieldedValues;
  if (op->getNumResults() != 0) {
    ValueRange yielded = op.getYieldedResultValues();
    if (yielded.size() != op->getNumResults())
      return op.emitOpError("result-producing workitem region must yield one "
                            "value per region result");
    llvm::append_range(yieldedValues, yielded);
  }
  if (op.getBody().empty())
    return success();

  if (failed(requireNoLiveBlockArguments(op)))
    return failure();

  Block &body = op.getBody().front();
  while (!body.empty()) {
    Operation &nested = body.front();
    if (isa<HCYieldOp>(nested)) {
      nested.erase();
      continue;
    }
    nested.moveBefore(op);
  }
  if (!yieldedValues.empty())
    op->replaceAllUsesWith(yieldedValues);
  op.erase();
  return success();
}

struct HCNormalizeScopeRegionsPass
    : public hc::impl::HCNormalizeScopeRegionsBase<
          HCNormalizeScopeRegionsPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<Operation *> callables;
    getOperation()->walk([&](Operation *op) {
      if (isa<HCKernelOp, HCFuncOp>(op))
        callables.push_back(op);
    });
    for (Operation *callable : callables) {
      std::optional<LaunchContextMetadata> metadata =
          launchMetadataFromCallable(callable);
      if (metadata && metadata->groupShape)
        dropWorkitemSuffixFromCallable(callable,
                                       metadata->groupShape.getDims());
    }
    dropUnusedScopeTokenArgs(getOperation());

    SmallVector<HCSubgroupRegionOp> subgroupRegions;
    SmallVector<HCWorkitemRegionOp> workitemRegions;
    getOperation()->walk([&](Operation *op) {
      if (auto subgroup = dyn_cast<HCSubgroupRegionOp>(op))
        subgroupRegions.push_back(subgroup);
      else if (auto workitem = dyn_cast<HCWorkitemRegionOp>(op))
        workitemRegions.push_back(workitem);
    });

    for (HCSubgroupRegionOp subgroup : subgroupRegions) {
      subgroup.emitOpError(
          "subgroup region normalization is not supported yet");
      return signalPassFailure();
    }
    for (HCWorkitemRegionOp workitem : workitemRegions) {
      if (failed(flattenResultlessWorkitemRegion(workitem)))
        return signalPassFailure();
    }
  }
};

} // namespace

// `createHCNormalizeScopeRegionsPass()` is emitted by tablegen.
