// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -hc-front-inline: turn every `hc_front.call` dispatched to a
// `ref.kind = "inline"` helper into an `hc_front.inlined_region` cloning
// the callee body, then erase the (now-unreferenced) helper
// `hc_front.func`s. The region is a name-scope boundary inside
// `hc_front`; alpha-renaming happens at conversion time, not here.
//
// Algorithm:
//  1. Collect top-level `hc_front.func`s whose `ref.kind == "inline"`
//     into a name -> FuncOp map.
//  2. Seed a worklist with every `hc_front.call` reachable from the
//     module whose callee's `hc_front.name` carries the inline
//     classification (including calls inside inline helpers
//     themselves -- those are inlined first, so each clone of a
//     helper's body is already free of inline calls by the time it
//     lands at a real use site).
//  3. At each call site: look up the callee, clone its body into a
//     fresh `hc_front.inlined_region`, match the region's result
//     types to the callee's explicit `hc_front.return` operands, and replace
//     the call with the region. A cycle is a hard error -- inline helpers
//     must not recurse.
//  4. Erase every inlinable `hc_front.func`; post-condition: no
//     `ref.kind == "inline"` symbol survives to the
//     `-convert-hc-front-to-hc` boundary.
//
// Everything the converter needs to finish the job lives on the region
// (`parameters` attribute + operands + cloned body); we don't thread
// any out-of-band info here.

#include "hc/Front/Transforms/Passes.h"

#include "hc/Front/IR/HCFrontDialect.h"
#include "hc/Front/IR/HCFrontOps.h"
#include "hc/Front/IR/HCFrontTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::hc::front {
#define GEN_PASS_DEF_HCFRONTINLINE
#include "hc/Front/Transforms/Passes.h.inc"
} // namespace mlir::hc::front

using namespace mlir;
namespace hc_front = mlir::hc::front;

namespace {

// Check whether a `hc_front.call`'s callee is a `ref.kind = "inline"`
// name op. Absent defining op / absent ref / different kind all
// collapse to false -- non-inline calls are outside this pass's scope.
static bool isInlineCall(hc_front::CallOp call) {
  auto nameOp = call.getCallee().getDefiningOp<hc_front::NameOp>();
  if (!nameOp)
    return false;
  auto ref = nameOp->getAttrOfType<DictionaryAttr>("ref");
  if (!ref)
    return false;
  auto kind = ref.getAs<StringAttr>("kind");
  return kind && kind.getValue() == "inline";
}

// Inline helpers are plain Python functions: exactly one return from
// the helper itself. Only the top block's direct children count --
// nested regions (a previously-inlined call's `hc_front.inlined_region`
// or a `hc_front.if` body, etc.) have their own returns that aren't
// *this* function's return.
static FailureOr<hc_front::ReturnOp> findSingleReturn(hc_front::FuncOp func) {
  hc_front::ReturnOp found;
  Block &entry = func.getBody().front();
  for (Operation &op : entry) {
    auto r = dyn_cast<hc_front::ReturnOp>(&op);
    if (!r)
      continue;
    if (found)
      return func.emitOpError(
          "inlinable func must have exactly one top-level `hc_front.return`");
    found = r;
  }
  if (!found) {
    func.emitOpError("inlinable func missing top-level `hc_front.return`");
    return failure();
  }
  return found;
}

class Inliner {
public:
  Inliner(llvm::StringMap<hc_front::FuncOp> &funcs, hc_front::ValueType valueTy)
      : inlinableFuncs(funcs), valueTy(valueTy) {}

  // Inline every call reachable from `root`. `root` itself is walked --
  // if it is an inlinable `hc_front.func`, calls inside get inlined
  // before anyone clones its body, so the clone is already
  // inline-free.
  LogicalResult inlineReachable(Operation *root, ArrayRef<StringRef> chain) {
    SmallVector<hc_front::CallOp> sites;
    root->walk([&](hc_front::CallOp call) {
      if (isInlineCall(call))
        sites.push_back(call);
    });
    for (hc_front::CallOp call : sites) {
      // A previous site may have cloned-and-inlined a subtree that
      // contained `call`, so skip if it's already been detached.
      if (!call->getBlock())
        continue;
      if (failed(inlineAt(call, chain)))
        return failure();
    }
    return success();
  }

private:
  llvm::StringMap<hc_front::FuncOp> &inlinableFuncs;
  hc_front::ValueType valueTy;

  // Resolve and arity-check the call against the registered inlinable
  // funcs. Diagnoses missing func / cycle / param-attr / arity-mismatch
  // up front so the rest of `inlineAt` can stay on the happy path.
  FailureOr<hc_front::FuncOp>
  resolveAndValidateCallee(hc_front::CallOp call, StringRef calleeName,
                           ArrayRef<StringRef> chain) {
    auto funcIt = inlinableFuncs.find(calleeName);
    if (funcIt == inlinableFuncs.end())
      return call.emitOpError("no inlinable `hc_front.func` named `")
             << calleeName << "'";
    for (StringRef caller : chain)
      if (caller == calleeName)
        return call.emitOpError("recursive inline through `")
               << calleeName << "'";
    hc_front::FuncOp func = funcIt->second;

    auto params = func->getAttrOfType<ArrayAttr>("parameters");
    if (!params)
      return func.emitOpError("inlinable func missing `parameters` attribute");
    if (call.getArguments().size() != params.size())
      return call.emitOpError("inline call arity ")
             << call.getArguments().size() << " does not match callee `"
             << calleeName << "' parameter count " << params.size();
    return func;
  }

  // Mint a fresh `hc_front.inlined_region` and clone the callee's body
  // into it. `emplaceBlock()` creates an empty block with no args; the
  // inlined body doesn't use block args -- parameter bindings flow via
  // operands + the `parameters` attribute, consumed at conversion time.
  hc_front::InlinedRegionOp cloneFuncBodyIntoRegion(hc_front::CallOp call,
                                                    hc_front::FuncOp func,
                                                    StringRef calleeName,
                                                    ArrayAttr params,
                                                    unsigned nResults) {
    MLIRContext *ctx = call.getContext();
    SmallVector<Type> resultTypes(nResults, valueTy);
    OpBuilder builder(call);
    StringAttr calleeAttr = StringAttr::get(ctx, calleeName);
    auto regionOp = hc_front::InlinedRegionOp::create(
        builder, call.getLoc(), resultTypes, calleeAttr, call.getArguments());
    regionOp->setAttr("parameters", params);

    Block *destBlock = &regionOp.getBody().emplaceBlock();
    OpBuilder cloneBuilder(destBlock, destBlock->begin());
    IRMapping mapping;
    Block &srcBlock = func.getBody().front();
    for (Operation &srcOp : srcBlock)
      cloneBuilder.clone(srcOp, mapping);
    return regionOp;
  }

  // Wire the call's SSA users through to the new region's result. If
  // the callee has multiple explicit return operands, the converter
  // packages them into a tuple at result #0; either way, result #0 is
  // the single carrier the user-facing SSA sees.
  LogicalResult wireCallResultUses(hc_front::CallOp call,
                                   hc_front::InlinedRegionOp regionOp,
                                   StringRef calleeName, unsigned nResults) {
    if (nResults == 0) {
      if (!call.getResult().use_empty())
        return call.emitOpError("inline callee `")
               << calleeName
               << "' returns no value but the call result is used";
      return success();
    }
    call.getResult().replaceAllUsesWith(regionOp.getResult(0));
    return success();
  }

  LogicalResult inlineAt(hc_front::CallOp call, ArrayRef<StringRef> chain) {
    auto nameOp = call.getCallee().getDefiningOp<hc_front::NameOp>();
    StringRef calleeName = nameOp.getName();

    FailureOr<hc_front::FuncOp> funcOr =
        resolveAndValidateCallee(call, calleeName, chain);
    if (failed(funcOr))
      return failure();
    hc_front::FuncOp func = *funcOr;

    FailureOr<hc_front::ReturnOp> retOr = findSingleReturn(func);
    if (failed(retOr))
      return failure();
    // Region-result arity follows the explicit `hc_front.return` operand
    // list. A tuple operand is one first-class value; tuple destructuring is
    // handled later by conversion through `hc.getitem`.
    unsigned nResults = retOr->getValues().size();
    auto params = func->getAttrOfType<ArrayAttr>("parameters");

    hc_front::InlinedRegionOp regionOp =
        cloneFuncBodyIntoRegion(call, func, calleeName, params, nResults);

    if (failed(wireCallResultUses(call, regionOp, calleeName, nResults)))
      return failure();
    call.erase();

    // `nameOp` is the original call's callee -- only erase if it has
    // no more uses.
    Operation *calleeNameOp = nameOp.getOperation();
    if (calleeNameOp->use_empty())
      calleeNameOp->erase();

    // Recurse into the freshly-cloned body so any nested inline calls
    // inside the callee expand at this site too.
    SmallVector<StringRef> newChain(chain.begin(), chain.end());
    newChain.push_back(calleeName);
    return inlineReachable(regionOp, newChain);
  }
};

// True iff `op` is a `hc_front.func` carrying `ref.kind = "inline"`.
static bool isInlinableFuncOp(Operation &op) {
  auto funcOp = dyn_cast<hc_front::FuncOp>(op);
  if (!funcOp)
    return false;
  auto refAttr = funcOp->getAttrOfType<DictionaryAttr>("ref");
  if (!refAttr)
    return false;
  auto kindAttr = refAttr.getAs<StringAttr>("kind");
  return kindAttr && kindAttr.getValue() == "inline";
}

// First pass: gather every top-level inline helper into a name -> FuncOp
// map. Duplicate names are a hard error -- the inliner uses string-keyed
// lookup, so a duplicate would silently pick one and drop the other.
static LogicalResult
collectInlinableFuncs(Operation *root,
                      llvm::StringMap<hc_front::FuncOp> &inlinableFuncs) {
  for (Region &region : root->getRegions()) {
    for (Block &block : region) {
      for (Operation &topOp : block) {
        if (!isInlinableFuncOp(topOp))
          continue;
        auto funcOp = cast<hc_front::FuncOp>(topOp);
        StringRef name = funcOp.getName();
        auto [_, inserted] = inlinableFuncs.try_emplace(name, funcOp);
        if (!inserted)
          return funcOp.emitOpError("duplicate inlinable func name `")
                 << name << "'; names must be unique within a module";
      }
    }
  }
  return success();
}

// Third pass: inline at every real call site across the module. Inline
// helpers themselves were processed earlier (their bodies cloned at use
// sites) and are skipped here to avoid re-walking.
static LogicalResult inlineAtAllRealSites(Inliner &inliner, Operation *root) {
  for (Region &region : root->getRegions())
    for (Block &block : region)
      for (Operation &topOp : block) {
        if (isInlinableFuncOp(topOp))
          continue;
        if (failed(inliner.inlineReachable(&topOp, /*chain=*/{})))
          return failure();
      }
  return success();
}

struct HCFrontInlinePass
    : public hc_front::impl::HCFrontInlineBase<HCFrontInlinePass> {
  using HCFrontInlineBase::HCFrontInlineBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();

    llvm::StringMap<hc_front::FuncOp> inlinableFuncs;
    if (failed(collectInlinableFuncs(root, inlinableFuncs))) {
      signalPassFailure();
      return;
    }
    if (inlinableFuncs.empty())
      return;

    auto valueTy = hc_front::ValueType::get(ctx);
    Inliner inliner(inlinableFuncs, valueTy);

    // Inline the bodies of the marker funcs themselves first. Cloning
    // a helper later sees no `hc_front.call`s to other inline helpers
    // in its body -- nested inline calls are already gone.
    for (auto &entry : inlinableFuncs) {
      hc_front::FuncOp func = entry.second;
      SmallVector<StringRef, 2> chain = {func.getName()};
      if (failed(inliner.inlineReachable(func, chain))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(inlineAtAllRealSites(inliner, root))) {
      signalPassFailure();
      return;
    }

    // Erase the marker funcs. Any `hc_front.name` that referred to
    // them was already dropped as a dead def when the call it
    // produced was replaced; if something still references the symbol
    // (a hand-written IR quirk) the erase will complain via the
    // verifier downstream, which is the correct signal.
    for (auto &entry : inlinableFuncs)
      entry.second.erase();
  }
};

} // namespace

// `createHCFrontInlinePass()` is emitted by tablegen (friend of the
// impl::HCFrontInlineBase CRTP). See `Passes.td` -- no `let constructor`,
// so the generated factory is the only one.
