// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-interpret-intrinsic-recipes`. Walks the sibling
// `__hc_intrinsic_lowerings__` module, applies every transform sequence
// whose `hc.target` matches against the outer payload, then erases the
// module. Surviving `hc.call_intrinsic` is a hard error.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCINTERPRETINTRINSICRECIPES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

constexpr StringRef kLoweringsModuleSym = "__hc_intrinsic_lowerings__";
constexpr StringRef kSequenceTargetAttrName = "hc.target";

static ModuleOp findLoweringsModule(ModuleOp root) {
  // Direct children only; nested kernels keep their own modules.
  ModuleOp result;
  for (Operation &op : *root.getBody()) {
    auto child = dyn_cast<ModuleOp>(op);
    if (!child)
      continue;
    StringAttr name = child.getSymNameAttr();
    if (!name || name.getValue() != kLoweringsModuleSym)
      continue;
    result = child;
    break;
  }
  return result;
}

static bool sequenceMatchesTarget(transform::NamedSequenceOp seq,
                                  StringRef target) {
  if (target.empty())
    return true;
  auto attr = seq->getAttrOfType<StringAttr>(kSequenceTargetAttrName);
  return attr && attr.getValue() == target;
}

static SmallVector<transform::NamedSequenceOp>
collectMatchingSequences(ModuleOp loweringsModule, StringRef target) {
  SmallVector<transform::NamedSequenceOp> sequences;
  for (Operation &op : *loweringsModule.getBody()) {
    auto seq = dyn_cast<transform::NamedSequenceOp>(op);
    if (seq && sequenceMatchesTarget(seq, target))
      sequences.push_back(seq);
  }
  return sequences;
}

class HCInterpretIntrinsicRecipesPass
    : public hc::impl::HCInterpretIntrinsicRecipesBase<
          HCInterpretIntrinsicRecipesPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp root = getOperation();
    StringRef targetRef(this->target);

    ModuleOp loweringsModule = findLoweringsModule(root);
    if (loweringsModule) {
      if (failed(applyAllRecipes(root, loweringsModule, targetRef)))
        return signalPassFailure();
      // Recipes spent -- clear before downstream sees transform IR.
      loweringsModule.erase();
    }

    if (failed(diagnoseUncovered(root, targetRef)))
      return signalPassFailure();

    // `Symbol`-trait ops survive DCE; sweep dead intrinsic decls here.
    eraseUnusedIntrinsics(root);
  }

private:
  LogicalResult applyAllRecipes(ModuleOp root, ModuleOp loweringsModule,
                                StringRef target) {
    SmallVector<transform::NamedSequenceOp> sequences =
        collectMatchingSequences(loweringsModule, target);
    transform::TransformOptions options;
    for (transform::NamedSequenceOp seq : sequences) {
      if (failed(transform::applyTransformNamedSequence(
              root.getOperation(), seq.getOperation(), loweringsModule,
              options))) {
        // Upstream already diagnosed; attach the recipe context.
        seq.emitError("failed to apply intrinsic lowering recipe '")
            << seq.getSymName() << "' for target '" << target << "'";
        return failure();
      }
    }
    return success();
  }

  LogicalResult diagnoseUncovered(ModuleOp root, StringRef target) {
    SmallVector<HCCallIntrinsicOp> uncovered;
    root.walk([&](HCCallIntrinsicOp call) { uncovered.push_back(call); });
    if (uncovered.empty())
      return success();
    for (HCCallIntrinsicOp call : uncovered) {
      auto diag = call.emitError("no intrinsic lowering recipe matched ")
                  << "@" << call.getCallee();
      if (!target.empty())
        diag << " for target '" << target << "'";
    }
    return failure();
  }

  void eraseUnusedIntrinsics(ModuleOp root) {
    // One-walk reference scan; per-decl `symbolKnownUseEmpty` is O(n*k).
    llvm::StringSet<> liveCallees;
    root.walk(
        [&](HCCallIntrinsicOp call) { liveCallees.insert(call.getCallee()); });
    SmallVector<HCIntrinsicOp> dead;
    for (Operation &op : *root.getBody()) {
      auto intrinsic = dyn_cast<HCIntrinsicOp>(op);
      if (!intrinsic)
        continue;
      if (liveCallees.contains(intrinsic.getSymName()))
        continue;
      dead.push_back(intrinsic);
    }
    for (HCIntrinsicOp intrinsic : dead)
      intrinsic.erase();
  }
};

} // namespace
