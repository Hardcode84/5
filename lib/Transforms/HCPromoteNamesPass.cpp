// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -hc-promote-names: rebuild IR so every `hc.assign` / `hc.name_load` is
// lowered into a real SSA edge.
//
// Algorithm, in one paragraph: walk each callable (kernel / func /
// intrinsic) bottom-up; for every region-carrying op that opts in via
// `NameStoreRegionOpInterface`, rewrite it so the Python-level names
// `hc.assign`-ed inside cross the boundary via the op's native carrying
// mechanism (iter_args for `hc.for_range`, results for `hc.if`, etc);
// once all regions inside a callable have been rewritten, sweep its body
// linearly to replace every remaining `hc.name_load` with its reaching
// `hc.assign` value and erase both.
//
// This file only lands the flat-body sweep and the pass skeleton. Region
// promotion lands in a follow-up commit within the same bead.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCOpsInterfaces.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCPROMOTENAMES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Linear-block promotion: scan the single block of `region` top-to-bottom,
// collapsing `hc.assign` writes into a `name -> Value` map and replacing
// each `hc.name_load` with the most recent `hc.assign`'s value. Any
// `hc.name_load` that reaches here with no prior `hc.assign` is a
// frontend-stage bug; diagnose and fail the pass.
//
// Region-carrying ops inside the block are left untouched at this stage:
// the full algorithm (a follow-up commit) processes them bottom-up first,
// after which their inner `hc.assign` / `hc.name_load` either have been
// erased or float up to block-level boundary ops that this flat sweep
// then cleans up. Until that lands, the presence of a
// `NameStoreRegionOpInterface` op with live name-store ops inside is
// diagnosed as "unsupported" so the pass never silently leaves
// half-lowered IR.
static LogicalResult promoteFlatBlock(Block &block) {
  llvm::StringMap<Value> bindings;
  SmallVector<Operation *> toErase;
  for (Operation &op : block) {
    if (auto assign = dyn_cast<HCAssignOp>(&op)) {
      bindings[assign.getName()] = assign.getValue();
      toErase.push_back(assign);
      continue;
    }
    if (auto load = dyn_cast<HCNameLoadOp>(&op)) {
      auto it = bindings.find(load.getName());
      if (it == bindings.end())
        return load.emitOpError("read of name '")
               << load.getName()
               << "' that has no reaching `hc.assign` in the enclosing "
                  "function body; the frontend must emit an assign before "
                  "every read, or the promotion must see a prior iter_arg / "
                  "region result";
      load.getResult().replaceAllUsesWith(it->second);
      toErase.push_back(load);
      continue;
    }
    // Region-carrying opt-in: if any `hc.assign` / `hc.name_load` survives
    // inside, region promotion has not been implemented yet for that op
    // kind, so keep the pass honest by diagnosing rather than producing a
    // half-lowered module.
    if (isa<NameStoreRegionOpInterface>(&op)) {
      bool hasNameStoreOps = false;
      op.walk([&](Operation *inner) {
        if (isa<HCAssignOp, HCNameLoadOp>(inner)) {
          hasNameStoreOps = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasNameStoreOps)
        return op.emitOpError(
            "region-carrying op still contains `hc.assign` / `hc.name_load` "
            "after promotion; region-level promotion is not yet implemented "
            "for this op kind");
    }
  }
  for (Operation *op : toErase)
    op->erase();
  return success();
}

// Common entry for a callable's body: `HCKernelOp` / `HCFuncOp` /
// `HCIntrinsicOp` all hold a `SizedRegion<1>`, so we take the unique entry
// block and promote in-place.
static LogicalResult promoteCallable(Operation *op, Region &body) {
  if (body.empty())
    return success();
  return promoteFlatBlock(body.front());
}

struct HCPromoteNamesPass
    : public hc::impl::HCPromoteNamesBase<HCPromoteNamesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](Operation *op) {
      Region *body = nullptr;
      if (auto k = dyn_cast<HCKernelOp>(op))
        body = &k.getBody();
      else if (auto f = dyn_cast<HCFuncOp>(op))
        body = &f.getBody();
      else if (auto i = dyn_cast<HCIntrinsicOp>(op))
        body = &i.getBody();
      if (!body)
        return WalkResult::advance();
      if (failed(promoteCallable(op, *body)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hc::createPromoteNamesPass() {
  return std::make_unique<HCPromoteNamesPass>();
}
