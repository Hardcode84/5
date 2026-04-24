// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -hc-front-fold-region-defs: erase the `hc_front.name + hc_front.call
// (+ hc_front.return)` trail that the Python frontend emits for
// `@group.workitems def inner(...): ...; inner()` patterns. The
// triad is a ghost invocation of a nested collective `def`: there is
// no callable named `inner` at `hc` level (the region op *is* the
// lowering), and any `hc_front.return %v` inside the region falls
// through to terminate the enclosing func, so the call's result — if
// any — is already dead.
//
// Without this pass the converter bails at the `ref.kind = "local"`
// callee because a local identifier has no target op. The folder
// runs before `-convert-hc-front-to-hc` (see pass description /
// `doc/lowering.md`); a surviving `ref.kind = "local"` call past the
// folder is still an error — the converter's diagnostic is the
// operator-facing signal that the pipeline ordering is wrong.
//
// Match shape (all siblings of a `hc_front.func`/`hc_front.kernel`/
// `hc_front.intrinsic` body):
//
//   hc_front.workitem_region ... attributes {name = "X"} { ... }
//   %n = hc_front.name "X" {ref = {kind = "local"}}
//   %c = hc_front.call %n()
//   [hc_front.return %c]          // optional; any uses of %c must be
//                                 // exactly this one return, else we
//                                 // punt and let the converter fire.
//
// `hc_front.subgroup_region` is handled identically.

#include "hc/Front/Transforms/Passes.h"

#include "hc/Front/IR/HCFrontDialect.h"
#include "hc/Front/IR/HCFrontOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::hc::front {
#define GEN_PASS_DEF_HCFRONTFOLDREGIONDEFS
#include "hc/Front/Transforms/Passes.h.inc"
} // namespace mlir::hc::front

using namespace mlir;
namespace hc_front = mlir::hc::front;

namespace {

// Check that `%call`'s result is dead or used by exactly one
// same-block `hc_front.return`. "Bound to a local" (`x = inner()`)
// falls through — the folder is structural, not semantic, so it
// leaves that case to the converter (which will report the stale
// `ref.kind = "local"`). `dead` is an out-param set on every success;
// it is initialized here so a future caller that forgets to seed it
// still reads a defined value on any `nullptr` return.
Operation *tailReturnOrNull(hc_front::CallOp call, bool &dead) {
  dead = false;
  if (call->getUses().empty()) {
    dead = true;
    return nullptr;
  }
  if (!call->hasOneUse())
    return nullptr;
  Operation *user = *call->getUsers().begin();
  auto ret = dyn_cast<hc_front::ReturnOp>(user);
  if (!ret)
    return nullptr;
  if (ret->getBlock() != call->getBlock())
    return nullptr;
  if (ret.getValues().size() != 1 ||
      ret.getValues().front() != call.getResult())
    return nullptr;
  return ret;
}

template <typename RegionOpT> void foldAfterRegion(RegionOpT regionOp) {
  std::optional<StringRef> regionNameOpt = regionOp.getName();
  if (!regionNameOpt || regionNameOpt->empty())
    return;
  StringRef regionN = *regionNameOpt;

  // Scan forward through the same block for the ghost triad. The
  // Python frontend emits the `hc_front.name` immediately after the
  // region, but we scan forward so unrelated sibling ops (e.g. a
  // later `hc_front.assign` or a second region op) between the region
  // and the ghost name don't prevent the match. The pattern is
  // selected by full predicate, not by uniqueness: even if several
  // `hc_front.name` ops share the region's string name, the first one
  // that also has `ref.kind = "local"` and is the sole callee of a
  // tail-or-unused call is the one we erase.
  Block *block = regionOp->getBlock();
  for (Operation &op : llvm::make_early_inc_range(llvm::make_range(
           std::next(Block::iterator(regionOp)), block->end()))) {
    auto nameOp = dyn_cast<hc_front::NameOp>(&op);
    if (!nameOp)
      continue;
    if (nameOp.getName() != regionN)
      continue;
    auto ref = nameOp->getAttrOfType<DictionaryAttr>("ref");
    if (!ref)
      continue;
    auto kind = ref.getAs<StringAttr>("kind");
    if (!kind || kind.getValue() != "local")
      continue;

    // The matching `hc_front.name` must be used as the callee of
    // exactly one `hc_front.call`; otherwise the local is plumbed
    // elsewhere and the fold is unsafe.
    if (!nameOp->hasOneUse())
      continue;
    auto call = dyn_cast<hc_front::CallOp>(*nameOp->getUsers().begin());
    if (!call || call.getCallee() != nameOp.getResult())
      continue;
    if (call->getBlock() != block)
      continue;
    // Don't fold if the local is also passed as a call argument
    // (higher-order use); callee is the canonical position. `continue`
    // rather than `return` so a later same-named `name` op can still
    // match if the emitter ever produces one.
    bool localAsArg = false;
    for (Value arg : call.getArguments()) {
      if (arg.getDefiningOp() == nameOp.getOperation()) {
        localAsArg = true;
        break;
      }
    }
    if (localAsArg)
      continue;

    bool callResultDead = false;
    Operation *tailReturn = tailReturnOrNull(call, callResultDead);
    if (!tailReturn && !callResultDead)
      continue;

    if (tailReturn)
      tailReturn->erase();
    call.erase();
    nameOp.erase();
    return;
  }
}

struct HCFrontFoldRegionDefsPass
    : public hc_front::impl::HCFrontFoldRegionDefsBase<
          HCFrontFoldRegionDefsPass> {
  using HCFrontFoldRegionDefsBase::HCFrontFoldRegionDefsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Collect region ops first, fold second; mutating during the
    // walk is tempting but the fold erases sibling ops and can
    // invalidate the walker's position.
    SmallVector<hc_front::WorkitemRegionOp> witRegions;
    SmallVector<hc_front::SubgroupRegionOp> sgRegions;
    module.walk([&](Operation *op) {
      if (auto w = dyn_cast<hc_front::WorkitemRegionOp>(op))
        witRegions.push_back(w);
      else if (auto s = dyn_cast<hc_front::SubgroupRegionOp>(op))
        sgRegions.push_back(s);
    });
    for (auto w : witRegions) {
      if (!w->getBlock())
        continue;
      foldAfterRegion(w);
    }
    for (auto s : sgRegions) {
      if (!s->getBlock())
        continue;
      foldAfterRegion(s);
    }
  }
};

} // namespace

// `createHCFrontFoldRegionDefsPass()` is emitted by tablegen (friend of
// the impl::HCFrontFoldRegionDefsBase CRTP). See `Passes.td` — no
// `let constructor`, so the generated factory is the only one.
