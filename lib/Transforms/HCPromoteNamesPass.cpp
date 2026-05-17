// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -hc-promote-names: rewrite `hc.assign` / `hc.name_load` into SSA.
// Bottom-up per callable: each `NameStoreRegionOpInterface` op carries
// names across via iter_args / results / region_return; transient
// snap/writeback pairs surface at the parent block, a final flat sweep
// fuses them into direct uses. Nested-scope reads capture outward,
// writes stay local unless lifted by an explicit region_return.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCOpsInterfaces.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "hc-promote-names"

namespace mlir::hc {
#define GEN_PASS_DEF_HCPROMOTENAMES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

using NameSet = llvm::SmallSetVector<StringAttr, 4>;
using SnapMap = llvm::SmallDenseMap<StringAttr, Value>;

// Materializes an outer-scope snap for an unbound read; null disables
// capture (top-level body and pre-seeded for_range/if scans).
using SnapFactory = llvm::function_ref<Value(StringAttr)>;

struct TopLevelNameFacts {
  NameSet reads;
  NameSet writes;
  NameSet snapshot;
};

// Direct children only; nested region ops already promoted by the time
// we run. `snapshot` = read-first names only; write-first carries don't
// need an outer binding.
static TopLevelNameFacts collectTopLevelNameFacts(Block &block) {
  TopLevelNameFacts facts;
  llvm::SmallDenseSet<StringAttr> firstSeen;
  for (Operation &op : block) {
    StringAttr name;
    bool isRead;
    if (auto assign = dyn_cast<HCAssignOp>(&op)) {
      name = assign.getNameAttr();
      isRead = false;
      facts.writes.insert(name);
    } else if (auto load = dyn_cast<HCNameLoadOp>(&op)) {
      name = load.getNameAttr();
      isRead = true;
      facts.reads.insert(name);
    } else {
      continue;
    }
    if (firstSeen.insert(name).second && isRead)
      facts.snapshot.insert(name);
  }
  return facts;
}

// Catches a promoter that left name-store ops behind, or a new
// interface op without a `promoteRegionOp` dispatch case.
static bool hasStaleNameStoreOps(Operation &op) {
  bool stale = false;
  op.walk([&](Operation *inner) {
    if (isa<HCAssignOp, HCNameLoadOp>(inner)) {
      stale = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return stale;
}

// Presence-only check; diagnoses first violation without mutating IR.
static LogicalResult preflightNameBindings(Block &block,
                                           const llvm::StringMap<Value> &seeded,
                                           bool haveCapture) {
  llvm::StringSet<> bound;
  for (const auto &kv : seeded)
    bound.insert(kv.getKey());
  for (Operation &op : block.without_terminator()) {
    if (auto assign = dyn_cast<HCAssignOp>(&op)) {
      bound.insert(assign.getName());
      continue;
    }
    if (auto load = dyn_cast<HCNameLoadOp>(&op)) {
      if (bound.contains(load.getName()))
        continue;
      if (haveCapture) {
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
    if (isa<NameStoreRegionOpInterface>(&op) && hasStaleNameStoreOps(op))
      return op.emitOpError(
          "region-carrying op still contains `hc.assign` / "
          "`hc.name_load` after promotion; either a new "
          "NameStoreRegionOpInterface op kind joined the interface "
          "without a matching case in `promoteRegionOp`, or a "
          "promoter left residual name-store ops in the body");
  }
  return success();
}

// Preflight guarantees every load resolves; unbound here = desync,
// fatal abort.
static void commitNameRewrites(Block &block, llvm::StringMap<Value> &binding,
                               SnapFactory capture) {
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
        if (!capture)
          llvm::report_fatal_error(
              llvm::Twine("hc-promote-names: load of name '") + load.getName() +
              "' reached commit phase unbound with no capture factory "
              "(preflight/commit desync, pass invariant violation)");
        resolved = capture(load.getNameAttr());
        binding[load.getName()] = resolved;
      }
      load.getResult().replaceAllUsesWith(resolved);
      toErase.push_back(&op);
    }
  }
  for (Operation *o : toErase)
    o->erase();
}

// Per-call atomicity: every in-block mutation lands or none do. Caller
// owns the wider IR — for_range/if/nested-scope already restructured
// before calling, so they pre-seed `binding` and treat failure here as
// a fatal invariant break.
static LogicalResult scanAndPromoteBlock(Block &block,
                                         llvm::StringMap<Value> &binding,
                                         SnapFactory capture = nullptr) {
  if (failed(preflightNameBindings(block, binding, /*haveCapture=*/!!capture)))
    return failure();
  commitNameRewrites(block, binding, capture);
  return success();
}

// Empty yield on a fresh else-region; prelude to yield-rebuild.
static void ensureYieldTerminator(Block &block, Location loc) {
  if (!block.empty() && isa<HCYieldOp>(block.back()))
    return;
  OpBuilder b(&block, block.end());
  HCYieldOp::create(b, loc);
}

// Caller pre-seeds every carried name; miss here = fatal invariant
// break (release-safe abort, not `assert`).
static Value resolveCarriedValue(const llvm::StringMap<Value> &binding,
                                 StringAttr name, const char *where) {
  auto it = binding.find(name.getValue());
  if (it == binding.end())
    llvm::report_fatal_error(llvm::Twine("hc-promote-names: carried name '") +
                             name.getValue() + "' missing from binding at " +
                             where + " (pass invariant violation)");
  return it->second;
}

// Appends `binding[name]` for each carried name to the existing yield.
static void rewriteYieldWithCarried(Block &block,
                                    const llvm::StringMap<Value> &binding,
                                    ArrayRef<StringAttr> carried,
                                    Location loc) {
  auto oldYield = cast<HCYieldOp>(block.getTerminator());
  SmallVector<Value> values(oldYield.getValues().begin(),
                            oldYield.getValues().end());
  for (StringAttr name : carried)
    values.push_back(resolveCarriedValue(binding, name, "yield rebuild"));
  OpBuilder b(oldYield);
  HCYieldOp::create(b, loc, values);
  oldYield.erase();
}

// Nested-scope region_return → yield. One value per carried name.
static void rewriteRegionReturnToYield(Block &block,
                                       const llvm::StringMap<Value> &binding,
                                       ArrayRef<StringAttr> carried) {
  auto term = cast<HCRegionReturnOp>(block.back());
  SmallVector<Value> values;
  values.reserve(carried.size());
  for (StringAttr name : carried)
    values.push_back(
        resolveCarriedValue(binding, name, "region_return rewrite"));
  OpBuilder b(term);
  HCYieldOp::create(b, term.getLoc(), values);
  term.erase();
}

// Outer-scope snaps to seed branch/loop bindings before any in-body write.
static void materializeSnapshots(OpBuilder &builder, Location loc, Type undefTy,
                                 const NameSet &snapshot, SnapMap &out) {
  for (StringAttr name : snapshot) {
    auto snap = HCNameLoadOp::create(builder, loc, undefTy, name);
    out[name] = snap.getResult();
  }
}

// One transient `hc.assign` per carried result; outer flat sweep
// resolves to direct uses.
static void writebackCarriedResults(OpBuilder &builder, Location loc,
                                    Operation *newOp,
                                    ArrayRef<StringAttr> carried,
                                    unsigned carriedResultsStart) {
  for (auto [i, name] : llvm::enumerate(carried)) {
    HCAssignOp::create(builder, loc, name,
                       newOp->getResult(carriedResultsStart + i));
  }
}

// Matcher for the IV self-bind pattern described in the `hc.assign`
// ODS description (see "Induction-variable self-bind" in HCOps.td).
// Consumes as many leading `hc.assign "<n>", %iv` ops as match and
// returns the captured name -> %iv map. `promoteForRange` uses the
// map in two ways: (1) to pre-seed the in-body binding scan so reads
// of `<n>` resolve directly to %iv; (2) to exclude those names from
// the snapshot / carried sets so no outer snap or iter_arg is ever
// manufactured for a counter that only exists inside the loop.
static llvm::SmallDenseMap<StringAttr, Value>
extractIvSelfBinds(Block &body, BlockArgument ivArg) {
  llvm::SmallDenseMap<StringAttr, Value> binds;
  while (!body.empty()) {
    auto assign = dyn_cast<HCAssignOp>(&body.front());
    if (!assign || assign.getValue() != ivArg)
      break;
    binds[assign.getNameAttr()] = ivArg;
    assign.erase();
  }
  return binds;
}

// Drops IV self-bind names from every name-fact set. The leading
// `hc.assign "<n>", %iv` ops are erased by `extractIvSelfBinds`
// before `collectTopLevelNameFacts` runs, but the user may still
// have `i = expr` shadow writes or `use(i)` reads in the body; those
// stay loop-local instead of becoming iter_args.
//
// Shadow semantics: the shadow write is consumed by the body scan
// (`binding["i"]` gets overwritten and subsequent reads resolve to
// the shadowed value), but is *not* promoted to an iter_arg /
// iter_result. The outer scope does not see the shadowed value
// after the loop — that diverges from Python's leaking-loop-variable
// semantics, but making it leak would require synthesizing an outer
// snap for the IV name, which doesn't exist at this layer. The
// Python driver is expected to reject IV-name shadowing before it
// reaches MLIR; this block is the belt to that suspenders.
static void dropIvSelfBindNamesFromFacts(
    TopLevelNameFacts &facts,
    const llvm::SmallDenseMap<StringAttr, Value> &ivSelfBinds) {
  for (auto &kv : ivSelfBinds) {
    facts.reads.remove(kv.first);
    facts.writes.remove(kv.first);
    facts.snapshot.remove(kv.first);
  }
}

// Builds the extended iter_init list for a `hc.for_range` rebuild.
// Read-first carriers seed from their outer snap; write-first
// carriers seed from `hc.undef_value`. The first in-body assign
// overwrites the iter_arg before any load sees it, and zero-trip
// leaves the placeholder flowing out as the op's result — matching
// "name stays undefined if the loop never ran".
static SmallVector<Value> buildExtendedIterInits(OpBuilder &builder,
                                                 Location loc, Type undefTy,
                                                 HCForRangeOp op,
                                                 ArrayRef<StringAttr> carried,
                                                 const SnapMap &snapValues) {
  SmallVector<Value> newIterInits(op.getIterInits().begin(),
                                  op.getIterInits().end());
  for (StringAttr name : carried) {
    auto it = snapValues.find(name);
    if (it != snapValues.end()) {
      newIterInits.push_back(it->second);
      continue;
    }
    LDBG() << "write-first `hc.for_range` carrier '" << name.getValue()
           << "' at " << loc;
    auto placeholder = HCUndefValueOp::create(builder, loc, undefTy);
    newIterInits.push_back(placeholder.getResult());
  }
  return newIterInits;
}

// Seeds the body-scan `binding` with every name guaranteed to have
// a value: IV self-bind names resolve to %iv directly; every
// carried name starts out as its iter_arg; every snapshot name as
// its outer-scope snap value. `binding` covers `snapshot ∪ carried
// ∪ ivSelfBinds`, which is every top-level name that can appear in
// a `hc.name_load` within the loop body, so the scan can never hit
// the reaching-def diagnostic.
static llvm::StringMap<Value> seedForRangeBodyBinding(
    const llvm::SmallDenseMap<StringAttr, Value> &ivSelfBinds,
    const NameSet &snapshot, const SnapMap &snapValues,
    ArrayRef<StringAttr> carried,
    const llvm::SmallDenseMap<StringAttr, BlockArgument> &iterArgFor) {
  llvm::StringMap<Value> binding;
  for (auto &kv : ivSelfBinds)
    binding[kv.first.getValue()] = kv.second;
  for (StringAttr name : snapshot)
    binding[name.getValue()] = snapValues.lookup(name);
  for (StringAttr name : carried)
    binding[name.getValue()] = iterArgFor.lookup(name);
  return binding;
}

// Promote `op`, a `hc.for_range`, so every `hc.assign` / `hc.name_load`
// inside its body is rewritten in terms of iter_args and `hc.yield`.
// The old op is replaced with a new one with extended iter_inits /
// iter_results; its old body is transferred intact and then linearly
// scanned. Carried-name results get flanked with outer-scope
// snapshots (before) and writebacks (after); the enclosing flat sweep
// fuses those into direct SSA uses.
//
// Leading `hc.assign "<n>", %iv-block-arg` ops are handled as a
// separate, pre-scan step — see `extractIvSelfBinds`. They are
// pre-seeded into the body scan's binding, kept out of
// snapshot / carried, and erased from the body before
// `collectTopLevelNameFacts` sees them.
static LogicalResult promoteForRange(HCForRangeOp op) {
  Block &oldBody = op.getBody().front();
  BlockArgument ivArg = oldBody.getArgument(0);
  auto ivSelfBinds = extractIvSelfBinds(oldBody, ivArg);

  TopLevelNameFacts facts = collectTopLevelNameFacts(oldBody);
  dropIvSelfBindNamesFromFacts(facts, ivSelfBinds);
  if (facts.reads.empty() && facts.writes.empty() && ivSelfBinds.empty())
    return success();

  SmallVector<StringAttr> carried(facts.writes.begin(), facts.writes.end());

  Type undefTy = UndefType::get(op.getContext());
  Location loc = op.getLoc();

  OpBuilder builder(op);
  SnapMap snapValues;
  materializeSnapshots(builder, loc, undefTy, facts.snapshot, snapValues);

  SmallVector<Value> newIterInits =
      buildExtendedIterInits(builder, loc, undefTy, op, carried, snapValues);

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

  // By this point we've already restructured the outer IR (new op +
  // `takeBody`), so a soft `return failure()` would leak a torn
  // module; preflight guarantees no diagnostic remains, so any
  // failure here is a pass-invariant break — abort fatally.
  llvm::StringMap<Value> binding = seedForRangeBodyBinding(
      ivSelfBinds, facts.snapshot, snapValues, carried, iterArgFor);
  if (failed(scanAndPromoteBlock(body, binding)))
    llvm::report_fatal_error(
        "hc-promote-names: scan of `hc.for_range` body failed after the "
        "outer IR was already restructured (pass invariant violation)");

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

// Bundles the per-branch facts that the rest of `promoteIf` consumes.
struct IfBranchFacts {
  TopLevelNameFacts thenFacts;
  TopLevelNameFacts elseFacts;
  bool hasElse;
};

// True iff neither branch reads or writes any name — nothing to
// promote and the caller can early-out.
static bool isNoOpIf(const IfBranchFacts &f) {
  return f.thenFacts.reads.empty() && f.thenFacts.writes.empty() &&
         f.elseFacts.reads.empty() && f.elseFacts.writes.empty();
}

// Union of writes across both branches; this is exactly the set of
// names that need to be carried out via the rebuilt op's results.
static SmallVector<StringAttr> computeIfCarried(const IfBranchFacts &f,
                                                NameSet &carriedSet) {
  for (StringAttr n : f.thenFacts.writes)
    carriedSet.insert(n);
  for (StringAttr n : f.elseFacts.writes)
    carriedSet.insert(n);
  return SmallVector<StringAttr>(carriedSet.begin(), carriedSet.end());
}

// Creates the replacement `hc.if` with `op`'s original results
// followed by one `!hc.undef` slot per carried name, and transfers
// both branch regions over. An empty else region gets emplaced when
// carried names demand a symmetric yield; otherwise we leave it
// empty (the verifier is fine with that when `newOp` has no
// results).
static HCIfOp buildIfShellOp(OpBuilder &builder, Location loc, Type undefTy,
                             HCIfOp op, ArrayRef<StringAttr> carried) {
  SmallVector<Type> newResultTypes(op.getResultTypes().begin(),
                                   op.getResultTypes().end());
  for (size_t i = 0, e = carried.size(); i < e; ++i)
    newResultTypes.push_back(undefTy);

  Region &elseRegion = op.getElseRegion();
  auto newOp = HCIfOp::create(builder, loc, newResultTypes, op.getCond());
  newOp.getThenRegion().takeBody(op.getThenRegion());
  if (!elseRegion.empty())
    newOp.getElseRegion().takeBody(elseRegion);
  else if (!carried.empty())
    newOp.getElseRegion().emplaceBlock();
  return newOp;
}

// Computes the set of names that need an outer-scope snap for `hc.if`
// promotion:
//   - every name read in any branch (the branch may read it before
//     any in-branch write), plus
//   - every carried name that isn't written on every path (the
//     silent-branch yield must fall back to the outer value).
//
// A name written symmetrically in both branches (and not read) needs
// no snap — each branch's own write provides the yield value, so an
// outer snap would be a spurious `hc.name_load` that the flat sweep
// would then fail to resolve when no outer binding exists.
static NameSet computeIfSnapshot(const TopLevelNameFacts &thenFacts,
                                 const TopLevelNameFacts &elseFacts,
                                 const NameSet &carriedSet, bool hasElse) {
  NameSet snapshot = thenFacts.reads;
  for (StringAttr n : elseFacts.reads)
    snapshot.insert(n);
  for (StringAttr n : carriedSet) {
    bool symmetric = hasElse && thenFacts.writes.count(n) > 0 &&
                     elseFacts.writes.count(n) > 0;
    if (!symmetric)
      snapshot.insert(n);
  }
  return snapshot;
}

// Scans one branch of a `hc.if` and replaces its yield with the
// carried-name values. `binding` is seeded from the shared snapshot
// (every in-branch read is in `snapshot`), so `scanAndPromoteBlock`
// can't hit the reaching-def diagnostic. The only remaining failure
// mode is a stale-NameStoreRegionOpInterface check — a pass bug,
// and the outer IR has already been restructured here, so any
// failure is treated as a fatal invariant break.
static void scanAndRebuildIfBranch(Block &block, const NameSet &snapshot,
                                   const SnapMap &snapValues,
                                   ArrayRef<StringAttr> carried, Location loc) {
  llvm::StringMap<Value> binding;
  for (StringAttr name : snapshot)
    binding[name.getValue()] = snapValues.lookup(name);
  ensureYieldTerminator(block, loc);
  if (failed(scanAndPromoteBlock(block, binding)))
    llvm::report_fatal_error(
        "hc-promote-names: scan of `hc.if` branch failed after the outer "
        "IR was already restructured (pass invariant violation)");
  rewriteYieldWithCarried(block, binding, carried, loc);
}

// Promote `op`, a `hc.if`, so every `hc.assign` / `hc.name_load` inside
// either branch is rewritten in terms of op results and `hc.yield`
// values. Semantics match Python's: a name assigned in one branch but
// not the other retains its outer binding in the silent branch (i.e.
// that branch yields the snapshot value).
static LogicalResult promoteIf(HCIfOp op) {
  // Per-branch name facts. `hc.if` promotion needs these split
  // because the snapshot policy depends on branch symmetry: a name
  // written in both branches is fully redefined on every path and
  // doesn't need an outer-scope snap, while a name written in only one
  // branch still needs the outer value to fall back to on the silent
  // branch.
  IfBranchFacts f;
  f.thenFacts = collectTopLevelNameFacts(op.getThenRegion().front());
  f.hasElse = !op.getElseRegion().empty();
  if (f.hasElse)
    f.elseFacts = collectTopLevelNameFacts(op.getElseRegion().front());
  if (isNoOpIf(f))
    return success();

  NameSet carriedSet;
  SmallVector<StringAttr> carried = computeIfCarried(f, carriedSet);

  NameSet snapshot =
      computeIfSnapshot(f.thenFacts, f.elseFacts, carriedSet, f.hasElse);

  Type undefTy = UndefType::get(op.getContext());
  Location loc = op.getLoc();

  OpBuilder builder(op);
  SnapMap snapValues;
  materializeSnapshots(builder, loc, undefTy, snapshot, snapValues);

  HCIfOp newOp = buildIfShellOp(builder, loc, undefTy, op, carried);

  scanAndRebuildIfBranch(newOp.getThenRegion().front(), snapshot, snapValues,
                         carried, loc);
  if (!newOp.getElseRegion().empty())
    scanAndRebuildIfBranch(newOp.getElseRegion().front(), snapshot, snapValues,
                           carried, loc);

  builder.setInsertionPointAfter(newOp);
  writebackCarriedResults(builder, loc, newOp, carried,
                          /*carriedResultsStart=*/op.getNumResults());

  for (auto [oldR, newR] : llvm::zip(
           op.getResults(), newOp.getResults().take_front(op.getNumResults())))
    oldR.replaceAllUsesWith(newR);

  op.erase();
  return success();
}

// Rebuilds `op` (a `HCWorkitemRegionOp` or `HCSubgroupRegionOp`) with
// the supplied result types, copying the op-kind-specific `captures`
// attribute through the concrete accessor. Reading `captures` by
// accessor keeps the pass tied to the ODS surface — a rename in
// HCOps.td breaks the build here, not at runtime.
static Operation *rebuildNestedScopeOp(OpBuilder &builder, Location loc,
                                       Operation *op,
                                       ArrayRef<Type> newResultTypes) {
  if (auto wi = dyn_cast<HCWorkitemRegionOp>(op))
    return HCWorkitemRegionOp::create(builder, loc, newResultTypes,
                                      wi.getCapturesAttr());
  if (auto sg = dyn_cast<HCSubgroupRegionOp>(op))
    return HCSubgroupRegionOp::create(builder, loc, newResultTypes,
                                      sg.getCapturesAttr());
  llvm::report_fatal_error(
      "hc-promote-names: promoteNestedScope dispatched on an op kind it "
      "doesn't know how to rebuild (pass invariant violation)");
}

// Promote `op`, a `hc.workitem_region` or `hc.subgroup_region`, as a
// Python-style nested scope. Two shapes, picked by inspecting the
// body's terminator:
//
//   * No `hc.region_return` — body is side-effect only. Scan with a
//     lazy outer-capture factory; no rebuild, no writeback. Writes
//     shadow locally (the `hc.assign`s are erased along with their
//     name); reads hit local bindings first and fall back to a
//     boundary `hc.name_load` that the outer flat sweep resolves.
//     `promoteForRange` / `promoteIf` are transparent carriers (names
//     flow in *and out* via iter_args / results); this shape is a
//     **one-way barrier** — reads cross, writes don't.
//
//   * Body ends with `hc.region_return <names>` — the frontend's
//     "return acc" hand-off. Rebuild the op with one `!hc.undef`
//     result per carried name, take the old body, scan it (same lazy
//     capture), resolve each carried name against the final binding,
//     swap the terminator for `hc.yield`, and writeback the new
//     results as outer-level `hc.assign`s. From the outer scope's
//     perspective this matches the for_range / if promotion shape —
//     the same two-phase (scan → writeback) protocol, and the same
//     flat-sweep fusion downstream.
//
// By the time we reach here the post-order walk has already promoted
// every inner `NameStoreRegionOpInterface` op; their transient snap/
// writeback pairs sit at this body's top level and get resolved by
// the same scan, capturing upward through the region op if needed.
static LogicalResult promoteNestedScope(Operation *op) {
  Region &body = op->getRegion(0);
  if (body.empty())
    return success();
  Block &bodyBlock = body.front();

  Type undefTy = UndefType::get(op->getContext());
  Location loc = op->getLoc();

  OpBuilder outerBuilder(op);
  auto capture = [&](StringAttr name) -> Value {
    return HCNameLoadOp::create(outerBuilder, loc, undefTy, name).getResult();
  };

  // Scan-only shape: no terminator or a non-return terminator means
  // there's nothing to surface. Writes stay local, reads capture via
  // the lazy factory, and the op stays result-less.
  HCRegionReturnOp regionReturn;
  if (!bodyBlock.empty())
    regionReturn = dyn_cast<HCRegionReturnOp>(bodyBlock.back());
  if (!regionReturn) {
    llvm::StringMap<Value> binding;
    return scanAndPromoteBlock(bodyBlock, binding, capture);
  }

  // "Return acc" shape. Collect the names (frontend-ordered) and
  // rebuild the op with a matching `!hc.undef` result per name.
  // Types stay `!hc.undef` here — type inference later narrows them
  // via the yield operands. `HCRegionReturnOp::verify` already
  // checked each element is a `StringAttr`, so the cast is sound at
  // this point; if a later non-verified producer slips in, the cast
  // asserts, which is preferable to silently dropping entries.
  SmallVector<StringAttr> carried;
  carried.reserve(regionReturn.getNames().size());
  for (Attribute a : regionReturn.getNames())
    carried.push_back(cast<StringAttr>(a));

  SmallVector<Type> newResultTypes(carried.size(), undefTy);
  Operation *newOp =
      rebuildNestedScopeOp(outerBuilder, loc, op, newResultTypes);
  newOp->getRegion(0).takeBody(body);
  Block &newBody = newOp->getRegion(0).front();

  // Rewind the outer builder so capture snaps land *before* `newOp`,
  // not after. `create` left the builder positioned past `newOp`.
  outerBuilder.setInsertionPoint(newOp);

  // Outer IR has already been restructured here (`takeBody` moved
  // the body out of the old op), so a soft `return failure()` would
  // leave a torn module. Preflight guarantees no user-facing
  // diagnostic remains — the only way the scan can fail now is a
  // pass-invariant break, which we surface loudly.
  llvm::StringMap<Value> binding;
  if (failed(scanAndPromoteBlock(newBody, binding, capture)))
    llvm::report_fatal_error(
        "hc-promote-names: scan of nested-scope region body failed after "
        "the outer IR was already restructured (pass invariant violation)");

  // A carried name the body never references (no in-body assign, no
  // in-body load) is still legal — the frontend is asking for the
  // outer binding to be threaded through. The scan can't have seen
  // it, so we materialize the outer snap ourselves, using the same
  // capture factory and while `outerBuilder` is still positioned
  // before `newOp` (scan snaps stacked there too). Post-loop
  // invariant: every `carried` name has a `binding` entry —
  // `rewriteRegionReturnToYield` asserts on that.
  for (StringAttr name : carried) {
    if (binding.find(name.getValue()) == binding.end())
      binding[name.getValue()] = capture(name);
  }

  // Terminator rewrite: `hc.region_return ["n1", "n2"]` →
  // `hc.yield %v1, %v2`. `scanAndPromoteBlock` skipped the
  // terminator, so the body's `back()` is still the original
  // `hc.region_return`.
  rewriteRegionReturnToYield(newBody, binding, carried);

  // Writeback: each new result lands as `hc.assign "<name>", %result`
  // at the enclosing scope, where the outer flat sweep fuses it into
  // downstream reads — the same shape `promoteForRange` / `promoteIf`
  // leave behind.
  outerBuilder.setInsertionPointAfter(newOp);
  writebackCarriedResults(outerBuilder, loc, newOp, carried,
                          /*carriedResultsStart=*/0);

  op->erase();
  return success();
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
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<HCForRangeOp>(promoteForRange)
      .Case<HCIfOp>(promoteIf)
      .Case<HCWorkitemRegionOp, HCSubgroupRegionOp>(
          [](Operation *op) { return promoteNestedScope(op); })
      .Default([](Operation *) { return success(); });
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
    Operation *root = getOperation();
    WalkResult result = root->walk([&](Operation *op) {
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
