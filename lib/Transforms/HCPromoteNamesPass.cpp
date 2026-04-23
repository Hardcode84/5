// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -hc-promote-names: rebuild IR so every `hc.assign` / `hc.name_load` is
// lowered into a real SSA edge.
//
// Algorithm, in one paragraph: walk each callable (kernel / func /
// intrinsic) bottom-up; for every region-carrying op that opts in via
// `NameStoreRegionOpInterface`, rebuild it so the Python-level names
// `hc.assign`-ed inside cross the boundary via the op's native carrying
// mechanism (iter_args for `hc.for_range`, results for `hc.if`, etc).
// Each promoted op is flanked with transient `hc.name_load` (before) and
// `hc.assign` (after) ops that snapshot the outer binding in and push the
// new binding back out; the enclosing region (or the final flat sweep on
// the callable body) then resolves those. Once every region-carrying op
// under a callable has been promoted, the top-level block is swept
// linearly to collapse the remaining `hc.name_load` / `hc.assign` pairs
// into direct SSA edges.
//
// Implemented in this file: flat-body sweep, `hc.for_range` promotion,
// `hc.if` promotion, nested-scope sweep for `hc.workitem_region` /
// `hc.subgroup_region`. The two region ops introduce a new Python-
// style nested name scope — writes inside shadow any outer binding
// and don't leak out, but reads **do** capture outer bindings: an
// unbound read falls back to an outer-scope snapshot materialized
// lazily by the pass. Plumbing a write back out (the "return acc"
// pattern) needs an ODS-level result extension on those ops and
// lives in a follow-up bead.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCOpsInterfaces.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCPROMOTENAMES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

using NameSet = llvm::SmallSetVector<StringAttr, 4>;
using SnapMap = llvm::SmallDenseMap<StringAttr, Value>;

// Callback that materializes a capture-from-outer-scope snapshot for
// `name`. Returned `Value` seeds the binding for the first in-scope
// `hc.name_load` of `name`; subsequent reads hit the binding directly.
// A null `SnapFactory` disables capture — the scan errors on any
// unresolved read, which is what we want at the top level of a
// callable body (nothing to capture from) and inside `hc.for_range` /
// `hc.if` scans where the caller pre-seeds the binding.
using SnapFactory = llvm::function_ref<Value(StringAttr)>;

// Gathers the top-level `hc.assign` / `hc.name_load` ops in `block`.
// "Top-level" = direct children only; nested region ops at this point
// are assumed to already be promoted (bottom-up walk precondition) and
// their inner name-store ops have either been erased or hoisted out as
// transient snap/writeback pairs at `block`.
static void collectTopLevelNames(Block &block, NameSet &reads,
                                 NameSet &writes) {
  for (Operation &op : block) {
    if (auto assign = dyn_cast<HCAssignOp>(&op))
      writes.insert(assign.getNameAttr());
    else if (auto load = dyn_cast<HCNameLoadOp>(&op))
      reads.insert(load.getNameAttr());
  }
}

// Two-phase linear scan over `block`'s non-terminator ops:
//
//   Phase 1 (preflight): walk the block with a presence-only binding
//   set. Every `hc.assign` adds its name; every `hc.name_load` must
//   be bound (via the seeded keys, a prior in-block assign, or
//   `capture`); every `NameStoreRegionOpInterface` op must have an
//   assign/load-free subtree. The first violation emits a diagnostic
//   and returns `failure()` before any IR mutation — so a failing
//   block stays recoverable by the enclosing pipeline and no half-
//   rewritten module escapes.
//
//   Phase 2 (commit): preflight proved every read resolves, so this
//   sweep can't emit diagnostics. `hc.assign` binds, `hc.name_load`
//   either resolves from `binding` or triggers a real (IR-mutating)
//   `capture()` call, and residual ops get erased in one pass at the
//   end. `NameStoreRegionOpInterface` ops are already known clean
//   from preflight; the commit loop skips them.
//
// The `capture` factory is a capture-from-outer-scope hook: invoked
// only on truly unbound reads, its returned Value seeds the binding
// so repeated reads share the same snapshot. `capture = nullptr`
// disables outer capture — used at the callable top level (no outer
// to reach for) and inside the `hc.for_range` / `hc.if` scans (the
// caller pre-seeds `binding` for every reachable name).
static LogicalResult scanAndPromoteBlock(Block &block,
                                         llvm::StringMap<Value> &binding,
                                         SnapFactory capture = nullptr) {
  {
    llvm::StringSet<> bound;
    for (const auto &kv : binding)
      bound.insert(kv.getKey());
    for (Operation &op : block.without_terminator()) {
      if (auto assign = dyn_cast<HCAssignOp>(&op)) {
        bound.insert(assign.getName());
        continue;
      }
      if (auto load = dyn_cast<HCNameLoadOp>(&op)) {
        if (bound.contains(load.getName()))
          continue;
        if (capture) {
          bound.insert(load.getName());
          continue;
        }
        return load.emitOpError("read of name '")
               << load.getName()
               << "' that has no reaching `hc.assign` in the enclosing "
                  "scope; the frontend must emit an assign before every "
                  "read, or the promotion must see a prior iter_arg / "
                  "region result";
      }
      if (isa<NameStoreRegionOpInterface>(&op)) {
        bool stale = false;
        op.walk([&](Operation *inner) {
          if (isa<HCAssignOp, HCNameLoadOp>(inner)) {
            stale = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        // Two failure modes produce this: (a) a new op kind joined the
        // `NameStoreRegionOpInterface` without a matching case in
        // `promoteRegionOp`'s dispatch, or (b) an existing promoter
        // left residual name-store ops in the body. Both are pass
        // bugs, not frontend bugs — diagnose, don't silently lower.
        if (stale)
          return op.emitOpError(
              "region-carrying op still contains `hc.assign` / "
              "`hc.name_load` after promotion; either a new "
              "NameStoreRegionOpInterface op kind joined the interface "
              "without a matching case in `promoteRegionOp`, or a "
              "promoter left residual name-store ops in the body");
      }
    }
  }

  SmallVector<Operation *> toErase;
  for (Operation &op : block.without_terminator()) {
    if (auto assign = dyn_cast<HCAssignOp>(&op)) {
      binding[assign.getName()] = assign.getValue();
      toErase.push_back(&op);
      continue;
    }
    if (auto load = dyn_cast<HCNameLoadOp>(&op)) {
      auto it = binding.find(load.getName());
      Value resolved;
      if (it != binding.end()) {
        resolved = it->second;
      } else {
        assert(capture && "preflight must have admitted this load via "
                          "the capture factory");
        resolved = capture(load.getNameAttr());
        binding[load.getName()] = resolved;
      }
      load.getResult().replaceAllUsesWith(resolved);
      toErase.push_back(&op);
    }
  }
  for (Operation *o : toErase)
    o->erase();
  return success();
}

// Ensures `block` has an `hc.yield` terminator; if not (e.g. a
// just-emplaced empty else-region block), append an empty yield at
// `loc`. This is the minimal prelude every branch/body needs before the
// post-scan yield-rebuild step.
static void ensureYieldTerminator(Block &block, Location loc) {
  if (!block.empty() && isa<HCYieldOp>(block.back()))
    return;
  OpBuilder b(&block, block.end());
  HCYieldOp::create(b, loc);
}

// Drops `block`'s current `hc.yield` and replaces it with a new yield
// whose operands are the existing yield's operands followed by
// `binding[name]` for each `name` in `carried`. Callers use this to
// extend a region-op terminator with the carried-name values produced
// during the block's linear scan.
//
// Missing a carried name is a caller-side invariant break, not a user
// input problem — `promoteForRange` / `promoteIf` both seed `binding`
// with every carried name (snap value or iter_arg) before the scan
// runs, and the scan only overwrites entries, never deletes them.
// A plain `assert` would compile out under NDEBUG and leave
// `it->second` dereferencing an end iterator, so we abort loudly
// instead.
static void rewriteYieldWithCarried(Block &block,
                                    const llvm::StringMap<Value> &binding,
                                    ArrayRef<StringAttr> carried,
                                    Location loc) {
  auto oldYield = cast<HCYieldOp>(block.getTerminator());
  SmallVector<Value> values(oldYield.getValues().begin(),
                            oldYield.getValues().end());
  for (StringAttr name : carried) {
    auto it = binding.find(name.getValue());
    if (it == binding.end())
      llvm::report_fatal_error(llvm::Twine("hc-promote-names: carried name '") +
                               name.getValue() +
                               "' missing from binding at yield rebuild "
                               "(pass invariant violation)");
    values.push_back(it->second);
  }
  OpBuilder b(oldYield);
  HCYieldOp::create(b, loc, values);
  oldYield.erase();
}

// Inserts an `hc.name_load` for every name in `snapshot` at the current
// insertion point; records each snap Value in `out`. The pass uses these
// "outer-scope snapshots" to seed the branch/loop bindings so a name
// that's read in the body (possibly before any in-body write) has a
// value to bind to.
static void materializeSnapshots(OpBuilder &builder, Location loc, Type undefTy,
                                 const NameSet &snapshot, SnapMap &out) {
  for (StringAttr name : snapshot) {
    auto snap = HCNameLoadOp::create(builder, loc, undefTy, name);
    out[name] = snap.getResult();
  }
}

// After a region op has been rebuilt with extra carried-name results,
// every one of those results needs to flow back into the enclosing
// scope's name store. One transient `hc.assign` per carried name at the
// op's insertion point does the job — the enclosing flat sweep then
// resolves them into direct SSA uses.
static void writebackCarriedResults(OpBuilder &builder, Location loc,
                                    Operation *newOp,
                                    ArrayRef<StringAttr> carried,
                                    unsigned carriedResultsStart) {
  for (auto [i, name] : llvm::enumerate(carried)) {
    HCAssignOp::create(builder, loc, name,
                       newOp->getResult(carriedResultsStart + i));
  }
}

// Promote `op`, a `hc.for_range`, so every `hc.assign` / `hc.name_load`
// inside its body is rewritten in terms of iter_args and `hc.yield`.
// The old op is replaced with a new one with extended iter_inits /
// iter_results; its old body is transferred intact and then linearly
// scanned. Carried-name results get flanked with outer-scope
// snapshots (before) and writebacks (after); the enclosing flat sweep
// fuses those into direct SSA uses.
static LogicalResult promoteForRange(HCForRangeOp op) {
  Block &oldBody = op.getBody().front();
  NameSet readSet;
  NameSet writeSet;
  collectTopLevelNames(oldBody, readSet, writeSet);
  if (readSet.empty() && writeSet.empty())
    return success();

  NameSet snapshot = readSet;
  for (StringAttr n : writeSet)
    snapshot.insert(n);
  SmallVector<StringAttr> carried(writeSet.begin(), writeSet.end());

  MLIRContext *ctx = op.getContext();
  Type undefTy = UndefType::get(ctx);
  Location loc = op.getLoc();

  OpBuilder builder(op);
  SnapMap snapValues;
  materializeSnapshots(builder, loc, undefTy, snapshot, snapValues);

  // Build extended iter_inits / result-types; the snap value for each
  // carried name seeds the iter_init.
  SmallVector<Value> newIterInits(op.getIterInits().begin(),
                                  op.getIterInits().end());
  for (StringAttr name : carried)
    newIterInits.push_back(snapValues[name]);

  SmallVector<Type> newResultTypes(op.getIterResults().getTypes().begin(),
                                   op.getIterResults().getTypes().end());
  for (size_t i = 0, e = carried.size(); i < e; ++i)
    newResultTypes.push_back(undefTy);

  auto newOp = HCForRangeOp::create(builder, loc, newResultTypes, op.getLower(),
                                    op.getUpper(), op.getStep(), newIterInits);

  // Take the old body. The new op was built with an empty region; after
  // `takeBody` it holds the old body block — which still has only the
  // original (1 + oldIterInits.size()) block arguments. Append one
  // block argument per carried name so the block-arg count matches the
  // new iter_init count.
  newOp.getBody().takeBody(op.getBody());
  Block &body = newOp.getBody().front();
  llvm::SmallDenseMap<StringAttr, BlockArgument> iterArgFor;
  for (StringAttr name : carried)
    iterArgFor[name] = body.addArgument(undefTy, loc);

  // Seed the scan: every carried name starts out as its iter_arg; every
  // read-only snapshot name as its outer-scope snap value. A `hc.assign`
  // encountered during the scan overwrites the binding; the yield
  // rebuild picks up whatever the last write left.
  //
  // `binding` is seeded with `readSet ∪ writeSet`, which by construction
  // covers every top-level `hc.name_load` in `body`, so the scan cannot
  // emit the reaching-def diagnostic here. A `failure()` return would
  // indicate a post-order invariant break (inner NameStoreRegionOpInterface
  // op not yet promoted) — `scanAndPromoteBlock`'s stale-op check is
  // the explicit backstop for that.
  llvm::StringMap<Value> binding;
  for (StringAttr name : snapshot)
    binding[name.getValue()] = snapValues[name];
  for (StringAttr name : carried)
    binding[name.getValue()] = iterArgFor[name];

  if (failed(scanAndPromoteBlock(body, binding)))
    return failure();

  rewriteYieldWithCarried(body, binding, carried, loc);

  builder.setInsertionPointAfter(newOp);
  writebackCarriedResults(builder, loc, newOp, carried,
                          /*carriedResultsStart=*/op.getIterResults().size());

  // Preserve users of pre-existing results. The carried results sit
  // after them in `newOp`; the writeback loop above already handled
  // those.
  for (auto [oldR, newR] :
       llvm::zip(op.getIterResults(),
                 newOp.getIterResults().take_front(op.getIterResults().size())))
    oldR.replaceAllUsesWith(newR);

  op.erase();
  return success();
}

// Promote `op`, a `hc.if`, so every `hc.assign` / `hc.name_load` inside
// either branch is rewritten in terms of op results and `hc.yield`
// values. Semantics match Python's: a name assigned in one branch but
// not the other retains its outer binding in the silent branch (i.e.
// that branch yields the snapshot value).
static LogicalResult promoteIf(HCIfOp op) {
  Block &thenBlock = op.getThenRegion().front();
  Region &elseRegion = op.getElseRegion();

  // Per-branch read/write sets. `hc.if` promotion needs these split
  // because the snapshot policy depends on branch symmetry: a name
  // written in both branches is fully redefined on every path and
  // doesn't need an outer-scope snap, while a name written in only one
  // branch still needs the outer value to fall back to on the silent
  // branch.
  NameSet thenReads, thenWrites, elseReads, elseWrites;
  collectTopLevelNames(thenBlock, thenReads, thenWrites);
  bool hasElse = !elseRegion.empty();
  if (hasElse)
    collectTopLevelNames(elseRegion.front(), elseReads, elseWrites);
  if (thenReads.empty() && thenWrites.empty() && elseReads.empty() &&
      elseWrites.empty())
    return success();

  NameSet carriedSet;
  for (StringAttr n : thenWrites)
    carriedSet.insert(n);
  for (StringAttr n : elseWrites)
    carriedSet.insert(n);
  SmallVector<StringAttr> carried(carriedSet.begin(), carriedSet.end());

  // `snapshot` is the set of names that need an outer-scope snap:
  //   - every name read in any branch (the branch may read it before
  //     any in-branch write), plus
  //   - every carried name that isn't written on every path (the
  //     silent-branch yield must fall back to the outer value).
  //
  // A name written symmetrically in both branches (and not read) needs
  // no snap — each branch's own write provides the yield value, so an
  // outer snap would be a spurious `hc.name_load` that the flat sweep
  // would then fail to resolve when no outer binding exists.
  NameSet snapshot = thenReads;
  for (StringAttr n : elseReads)
    snapshot.insert(n);
  for (StringAttr n : carriedSet) {
    bool symmetric =
        hasElse && thenWrites.count(n) > 0 && elseWrites.count(n) > 0;
    if (!symmetric)
      snapshot.insert(n);
  }

  MLIRContext *ctx = op.getContext();
  Type undefTy = UndefType::get(ctx);
  Location loc = op.getLoc();

  OpBuilder builder(op);
  SnapMap snapValues;
  materializeSnapshots(builder, loc, undefTy, snapshot, snapValues);

  SmallVector<Type> newResultTypes(op.getResultTypes().begin(),
                                   op.getResultTypes().end());
  for (size_t i = 0, e = carried.size(); i < e; ++i)
    newResultTypes.push_back(undefTy);

  auto newOp = HCIfOp::create(builder, loc, newResultTypes, op.getCond());
  newOp.getThenRegion().takeBody(op.getThenRegion());
  // If the old op had an else region, move it as-is; otherwise synthesize
  // an empty else block when carried names demand a symmetric yield. If
  // neither applies (no carried names, no old else), leave the new else
  // region empty — the verifier is fine with that when `newOp` has no
  // results.
  if (!elseRegion.empty())
    newOp.getElseRegion().takeBody(elseRegion);
  else if (!carried.empty())
    newOp.getElseRegion().emplaceBlock();

  // Per-branch scan. `binding` is seeded from the shared `snapshot`
  // set, which includes every in-branch read (see the `snapshot`
  // construction above), so `scanAndPromoteBlock` cannot hit the
  // reaching-def diagnostic for either branch. A `failure()` here
  // would mean a post-order invariant break — treated as a bug via
  // the stale-op check inside the scan.
  auto processBranch = [&](Block &block) -> LogicalResult {
    llvm::StringMap<Value> binding;
    for (StringAttr name : snapshot)
      binding[name.getValue()] = snapValues[name];
    ensureYieldTerminator(block, loc);
    if (failed(scanAndPromoteBlock(block, binding)))
      return failure();
    rewriteYieldWithCarried(block, binding, carried, loc);
    return success();
  };

  if (failed(processBranch(newOp.getThenRegion().front())))
    return failure();
  if (!newOp.getElseRegion().empty()) {
    if (failed(processBranch(newOp.getElseRegion().front())))
      return failure();
  }

  builder.setInsertionPointAfter(newOp);
  writebackCarriedResults(builder, loc, newOp, carried,
                          /*carriedResultsStart=*/op.getNumResults());

  for (auto [oldR, newR] : llvm::zip(
           op.getResults(), newOp.getResults().take_front(op.getNumResults())))
    oldR.replaceAllUsesWith(newR);

  op.erase();
  return success();
}

// Promote `op`, a `hc.workitem_region` or `hc.subgroup_region`, as a
// Python-style nested scope. Writes inside shadow the outer name
// store and don't leak back out; reads **do** capture outer bindings
// — an unresolved read inside triggers a lazy `hc.name_load` at the
// region op's insertion point, which the outer flat sweep then
// resolves against the outer name store. Propagating a write back to
// the outer scope needs an ODS-level result extension on these ops
// — tracked in bead 5-2lf.
//
// Shape differs from `promoteForRange` / `promoteIf` on purpose:
// those are **transparent** carriers (names flow in AND out via
// iter_args / results), so their shape is "snapshot → scan → rebuild
// with carriers → writeback". These are **barriers** in the
// write-out direction (5-2lf will flip that for an explicit result
// list), so their shape collapses to "scan with capture factory";
// there's nothing to carry out yet, hence no rebuild and no
// writeback. When 5-2lf lands, the shape here should converge with
// the for/if path by reusing `writebackCarriedResults` for the new
// result list.
//
// By the time we reach here, the post-order walk has already promoted
// every inner NameStoreRegionOpInterface op; their transient
// snap/writeback pairs sit at this body's top level and get resolved
// by the same flat sweep, capturing upward through the region op if
// needed.
static LogicalResult promoteNestedScope(Operation *op) {
  Region &body = op->getRegion(0);
  if (body.empty())
    return success();

  MLIRContext *ctx = op->getContext();
  Type undefTy = UndefType::get(ctx);
  Location loc = op->getLoc();
  OpBuilder outerBuilder(op);
  auto capture = [&](StringAttr name) -> Value {
    return HCNameLoadOp::create(outerBuilder, loc, undefTy, name).getResult();
  };

  llvm::StringMap<Value> binding;
  return scanAndPromoteBlock(body.front(), binding, capture);
}

// Per-op-kind dispatch. The `NameStoreRegionOpInterface` marker is
// deliberately method-free: each op kind has a genuinely distinct
// promotion shape (iter_args for `for_range`, results for `if`,
// capture-only for the nested scopes), so hoisting the policy into
// interface methods would fragment the algorithm across ODS. Policy
// lives here; the interface just lists who opts in. A new op joining
// the interface needs a matching `promoteXxx` helper plus one more
// case below — the stale-interface diagnostic in `scanAndPromoteBlock`
// exists to catch anyone who forgets the second half.
static LogicalResult promoteRegionOp(Operation *op) {
  if (auto forOp = dyn_cast<HCForRangeOp>(op))
    return promoteForRange(forOp);
  if (auto ifOp = dyn_cast<HCIfOp>(op))
    return promoteIf(ifOp);
  if (isa<HCWorkitemRegionOp, HCSubgroupRegionOp>(op))
    return promoteNestedScope(op);
  return success();
}

// Collects every `NameStoreRegionOpInterface` op in `region`'s subtree
// in post-order (innermost first) and promotes each in turn. Post-order
// ensures children are promoted before parents, so by the time we
// process an outer op, its inner region ops are already clean and their
// transient snap/writeback ops sit at the outer op's body level, ready
// to be folded into its iter_args / results.
static LogicalResult promoteRegion(Region &region) {
  SmallVector<Operation *> work;
  region.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<NameStoreRegionOpInterface>(op))
      work.push_back(op);
  });
  for (Operation *op : work)
    if (failed(promoteRegionOp(op)))
      return failure();
  return success();
}

static LogicalResult promoteCallable(Region &body) {
  if (body.empty())
    return success();
  if (failed(promoteRegion(body)))
    return failure();
  // Final flat sweep: at this point all region-carrying ops under `body`
  // have been promoted (their snap/writeback pairs surfaced here), so a
  // linear pass with an empty initial binding resolves everything.
  llvm::StringMap<Value> binding;
  return scanAndPromoteBlock(body.front(), binding);
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
      if (failed(promoteCallable(*body)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

// `createHCPromoteNamesPass()` is emitted by tablegen (friend of the
// `impl::HCPromoteNamesBase` CRTP). See `Passes.td` — no `let
// constructor`, so the generated factory is the only one.
