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

namespace mlir::hc {
#define GEN_PASS_DEF_HCNORMALIZESCOPEREGIONS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static LogicalResult requireNoLiveBlockArguments(HCWorkitemRegionOp op) {
  Block &body = op.getBody().front();
  for (BlockArgument arg : body.getArguments()) {
    if (arg.use_empty())
      continue;
    return op.emitOpError("cannot flatten workitem region while its scope "
                          "token is still live; run "
                          "`hc-materialize-bound-exprs` first");
  }
  return success();
}

static LogicalResult flattenResultlessWorkitemRegion(HCWorkitemRegionOp op) {
  if (op->getNumResults() != 0)
    return op.emitOpError("result-producing workitem region normalization "
                          "requires distributed value projection support");
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
  op.erase();
  return success();
}

struct HCNormalizeScopeRegionsPass
    : public hc::impl::HCNormalizeScopeRegionsBase<
          HCNormalizeScopeRegionsPass> {
  using Base::Base;

  void runOnOperation() override {
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
