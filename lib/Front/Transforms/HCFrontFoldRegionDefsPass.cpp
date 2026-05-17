// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -hc-front-fold-region-defs: erase the `hc_front.name + hc_front.call
// (+ hc_front.return)` trail that the Python frontend emits for
// `@group.workitems def inner(...): ...; inner()` patterns. The
// triad is a ghost invocation of a nested collective `def`: there is
// no callable named `inner` at `hc` level (the region op *is* the
// lowering). When the ghost call is immediately returned, stamp the
// region so conversion can turn the inner return into a region yield
// plus an ordinary callable-level return.
//
// Without this pass the converter bails at the `ref.kind = "local"`
// callee because a local identifier has no target op. The folder
// runs before `-convert-hc-front-to-hc` (see pass description /
// `doc/lowering.md`); a surviving `ref.kind = "local"` call past the
// folder is still an error -- the converter's diagnostic is the
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
// falls through -- the folder is structural, not semantic, so it
// leaves that case to the converter (which will report the stale
// `ref.kind = "local"`). `dead` is an out-param set on every success;
// it is initialized here so a future caller that forgets to seed it
// still reads a defined value on any `nullptr` return.
static Operation *tailReturnOrNull(hc_front::CallOp call, bool &dead) {
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

// `op` is a `hc_front.name` naming the region (`local` ref kind, name
// matches the region's). Returns the name op for that match, or a null
// op handle to mean "skip; not a candidate".
static hc_front::NameOp matchNameForRegion(Operation &op, StringRef regionN) {
  auto nameOp = dyn_cast<hc_front::NameOp>(&op);
  if (!nameOp || nameOp.getName() != regionN)
    return {};
  auto ref = nameOp->getAttrOfType<DictionaryAttr>("ref");
  if (!ref)
    return {};
  auto kind = ref.getAs<StringAttr>("kind");
  if (!kind || kind.getValue() != "local")
    return {};
  return nameOp;
}

// Resolve the single, same-block call op that consumes `nameOp`'s result
// as its callee. Returns null when the local is plumbed elsewhere (more
// than one use, used as a call argument, used outside the block, ...).
static hc_front::CallOp matchSingleCallUse(hc_front::NameOp nameOp,
                                           Block *block) {
  if (!nameOp->hasOneUse())
    return {};
  auto call = dyn_cast<hc_front::CallOp>(*nameOp->getUsers().begin());
  if (!call || call.getCallee() != nameOp.getResult())
    return {};
  if (call->getBlock() != block)
    return {};
  return call;
}

// Higher-order rejection: callee position is the only fold-safe use of
// the local. Mirroring use as a call argument anywhere makes the local
// observable elsewhere, so we bail.
static bool localPassedAsCallArg(hc_front::NameOp nameOp,
                                 hc_front::CallOp call) {
  for (Value arg : call.getArguments())
    if (arg.getDefiningOp() == nameOp.getOperation())
      return true;
  return false;
}

// `return inner()` is only folded when the ghost trail (region, name,
// call, return) is the entire enclosing tail with no intervening sibling
// ops. Otherwise conversion would reorder unrelated siblings.
template <typename RegionOpT>
static bool
tailTrailAlignedForReturn(RegionOpT regionOp, hc_front::NameOp nameOp,
                          hc_front::CallOp call, Operation *tailReturn) {
  Block *block = regionOp->getBlock();
  if (&*std::next(Block::iterator(regionOp)) != nameOp.getOperation())
    return false;
  if (&*std::next(Block::iterator(nameOp)) != call.getOperation())
    return false;
  if (&*std::next(Block::iterator(call)) != tailReturn)
    return false;
  return std::next(Block::iterator(tailReturn)) == block->end();
}

// Try to match and fold one ghost triad starting at `op`. Returns true
// when a fold actually happened (and the caller should stop scanning),
// false to mean "no match here, keep scanning". The mutation side-effect
// is intentional: matching and rewriting are intertwined enough that
// splitting them again would just bring back the search/state plumbing.
template <typename RegionOpT>
static bool tryFoldGhostTriad(RegionOpT regionOp, Operation &op,
                              StringRef regionN, Block *block) {
  hc_front::NameOp nameOp = matchNameForRegion(op, regionN);
  if (!nameOp)
    return false;
  hc_front::CallOp call = matchSingleCallUse(nameOp, block);
  if (!call)
    return false;
  if (localPassedAsCallArg(nameOp, call))
    return false;

  bool callResultDead = false;
  Operation *tailReturn = tailReturnOrNull(call, callResultDead);
  if (!tailReturn && !callResultDead)
    return false;

  if (tailReturn) {
    if (!tailTrailAlignedForReturn(regionOp, nameOp, call, tailReturn))
      return false;
    regionOp.setTailReturnAttr(UnitAttr::get(regionOp.getContext()));
    tailReturn->erase();
  }
  call.erase();
  nameOp.erase();
  return true;
}

template <typename RegionOpT> static void foldAfterRegion(RegionOpT regionOp) {
  std::optional<StringRef> regionNameOpt = regionOp.getName();
  if (!regionNameOpt || regionNameOpt->empty())
    return;
  StringRef regionN = *regionNameOpt;

  // Scan forward through the same block for the ghost triad. The
  // Python frontend emits the `hc_front.name` immediately after the
  // region, but bare unused calls may appear after unrelated sibling
  // ops, so we scan forward. Tail-return folds are stricter (see
  // `tailTrailAlignedForReturn` -- conversion emits the callable return
  // at the region site, so no intervening siblings can be skipped).
  // The pattern is selected by full predicate, not by uniqueness: even
  // if several `hc_front.name` ops share the region's string name, the
  // first valid ghost use wins.
  Block *block = regionOp->getBlock();
  for (Operation &op : llvm::make_early_inc_range(llvm::make_range(
           std::next(Block::iterator(regionOp)), block->end())))
    if (tryFoldGhostTriad(regionOp, op, regionN, block))
      return;
}

struct HCFrontFoldRegionDefsPass
    : public hc_front::impl::HCFrontFoldRegionDefsBase<
          HCFrontFoldRegionDefsPass> {
  using HCFrontFoldRegionDefsBase::HCFrontFoldRegionDefsBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    // Collect region ops first, fold second; mutating during the
    // walk is tempting but the fold erases sibling ops and can
    // invalidate the walker's position.
    SmallVector<hc_front::WorkitemRegionOp> witRegions;
    SmallVector<hc_front::SubgroupRegionOp> sgRegions;
    root->walk([&](Operation *op) {
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
// the impl::HCFrontFoldRegionDefsBase CRTP). See `Passes.td` -- no
// `let constructor`, so the generated factory is the only one.
