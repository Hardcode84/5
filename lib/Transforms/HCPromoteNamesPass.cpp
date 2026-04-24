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
// new binding back out; snapshots are driven by first use in block order
// (read-first names only), not by `reads ∪ writes`. The enclosing region
// (or the final flat sweep on the callable body) then resolves those. Once
// every region-carrying op under a callable has been promoted, the top-level
// block is swept linearly to collapse the remaining `hc.name_load` /
// `hc.assign` pairs into direct SSA edges.
//
// Implemented in this file: flat-body sweep, `hc.for_range` promotion,
// `hc.if` promotion, nested-scope sweep for `hc.workitem_region` /
// `hc.subgroup_region`. The nested-scope sweep is Python-style —
// writes shadow without leaking, reads capture the outer binding,
// and an `hc.region_return` terminator lifts named bindings as
// region results. `promoteNestedScope` documents the two body
// shapes in detail; this banner stops at the policy level.

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

struct TopLevelNameFacts {
  NameSet reads;
  NameSet writes;
  NameSet snapshot;
};

// Gathers the top-level `hc.assign` / `hc.name_load` facts in `block`.
// "Top-level" = direct children only; nested region ops at this point
// are assumed to already be promoted (bottom-up walk precondition) and
// their inner name-store ops have either been erased or hoisted out as
// transient snap/writeback pairs at `block`.
//
// `snapshot` is deliberately not `reads`: it records only names whose
// first top-level mention is a read, in block order. A write-first name
// can be carried without loading an outer binding that may not exist.
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

// Two-phase linear scan over `block`'s non-terminator ops:
//
//   Phase 1 (preflight): walk the block with a presence-only binding
//   set. Every `hc.assign` adds its name; every `hc.name_load` must
//   be bound (via the seeded keys, a prior in-block assign, or
//   `capture`); every `NameStoreRegionOpInterface` op must have an
//   assign/load-free subtree. The first violation emits a diagnostic
//   and returns `failure()` before any IR mutation.
//
//   Phase 2 (commit): preflight proved every read resolves, so this
//   sweep can't emit diagnostics. `hc.assign` binds, `hc.name_load`
//   either resolves from `binding` or triggers a real (IR-mutating)
//   `capture()` call, and residual ops get erased in one pass at the
//   end. `NameStoreRegionOpInterface` ops are already known clean
//   from preflight; the commit loop skips them.
//
// Atomicity guarantee: **within a single call**, either every
// in-`block` mutation runs or none do. That is the guarantee the
// callable-level flat sweep in `promoteCallable` relies on — a
// failing user body bubbles out as a clean pass failure. It is
// **not** a pipeline-wide guarantee: `promoteForRange` / `promoteIf`
// restructure the outer IR (`takeBody`, result-extended clone)
// before calling this function, so if the scan failed there, the
// module would still be torn. Those callers sidestep the problem by
// seeding `binding` with every reachable name and then treating any
// scan failure as a pass-invariant break (fatal abort), not a
// recoverable failure — see their call sites for the rationale.
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
  // Phase 1 (preflight): validate every reachable name has a binding
  // without mutating any IR. Diagnostics fire here, not in phase 2.
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

  // Phase 2 (commit): preflight passed, so every load is guaranteed
  // resolvable. The only failure mode here would be a preflight/commit
  // desync (future-us rewrites one without the other) — handled with a
  // loud, release-safe abort rather than a stripped-under-NDEBUG assert.
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

// Resolves `name` against `binding` or aborts with a consistent
// diagnostic. Both terminator-rebuild helpers share this: a carried
// name that isn't in the binding by the time we build the new
// yield is always a caller-side invariant break (the caller seeded
// the binding with every reachable name before the scan, and the
// scan only overwrites entries). `llvm::report_fatal_error` keeps
// the abort loud in release builds; a plain `assert` would compile
// out and leave `it->second` dereferencing `end()`.
static Value resolveCarriedValue(const llvm::StringMap<Value> &binding,
                                 StringAttr name, const char *where) {
  auto it = binding.find(name.getValue());
  if (it == binding.end())
    llvm::report_fatal_error(llvm::Twine("hc-promote-names: carried name '") +
                             name.getValue() + "' missing from binding at " +
                             where + " (pass invariant violation)");
  return it->second;
}

// Drops `block`'s current `hc.yield` and replaces it with a new yield
// whose operands are the existing yield's operands followed by
// `binding[name]` for each `name` in `carried`. Callers use this to
// extend a region-op terminator with the carried-name values produced
// during the block's linear scan.
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

// Swaps `block`'s `hc.region_return <names>` terminator for an
// `hc.yield %v1, ...` with one value per `carried` name. Mirrors
// `rewriteYieldWithCarried`, minus the "append to existing yield"
// step — nested-scope regions don't stack yields. Callers pre-seed
// `binding` so every carried name resolves; `resolveCarriedValue`
// aborts on an invariant break.
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
  // IV self-bind names may still appear as reads in the body (the
  // user's code says `for i in ...: use(i)`); drop them from both
  // sets so snapshot / iter_init computation treats them as loop-
  // local, not carried.
  //
  // `facts.writes` membership is the IV-name-shadow case: the user wrote
  // `for i in ...: i = expr`. The leading self-bind is already
  // stripped by `extractIvSelfBinds`, but a remaining in-body
  // `hc.assign "i", %expr` still lands here in `facts.writes`. Dropping
  // it means the shadow write is consumed by the body scan
  // (`binding["i"]` gets overwritten and subsequent `hc.name_load
  // "i"` reads resolve to %expr) but is *not* promoted to an
  // iter_arg / iter_result. The shadow stays loop-local — the outer
  // scope does not see the shadowed value after the loop.
  //
  // That diverges from Python's leaking-loop-variable semantics, but
  // making it leak here would require synthesizing an outer snap for
  // the IV name, which is not a thing that exists at this layer
  // (the IV is a block arg, not a name in the outer store). The
  // Python driver is expected to reject IV-name shadowing before it
  // reaches MLIR; this block is the belt to that suspenders.
  for (auto &kv : ivSelfBinds) {
    facts.reads.remove(kv.first);
    facts.writes.remove(kv.first);
    facts.snapshot.remove(kv.first);
  }
  if (facts.reads.empty() && facts.writes.empty() && ivSelfBinds.empty())
    return success();

  SmallVector<StringAttr> carried(facts.writes.begin(), facts.writes.end());

  MLIRContext *ctx = op.getContext();
  Type undefTy = UndefType::get(ctx);
  Location loc = op.getLoc();

  OpBuilder builder(op);
  SnapMap snapValues;
  materializeSnapshots(builder, loc, undefTy, facts.snapshot, snapValues);

  // iter_inits: read-first carriers seed from their outer snap; write-first
  // carriers seed from `hc.undef_value`. The first in-body assign overwrites
  // the iter_arg before any load sees it, and zero-trip leaves the placeholder
  // flowing out as the op's result, matching "name stays undefined if the loop
  // never ran".
  SmallVector<Value> newIterInits(op.getIterInits().begin(),
                                  op.getIterInits().end());
  for (StringAttr name : carried) {
    auto it = snapValues.find(name);
    if (it != snapValues.end()) {
      newIterInits.push_back(it->second);
      continue;
    }
    auto placeholder = HCUndefValueOp::create(builder, loc, undefTy);
    newIterInits.push_back(placeholder.getResult());
  }

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

  // Seed the scan: IV self-bind names resolve to %iv directly; every
  // carried name starts out as its iter_arg; every read-only
  // snapshot name as its outer-scope snap value. A `hc.assign`
  // encountered during the scan overwrites the binding; the yield
  // rebuild picks up whatever the last write left.
  //
  // `binding` is seeded with `facts.snapshot ∪ carried ∪ ivSelfBinds`;
  // together these cover every top-level name that can appear in a
  // `hc.name_load` within `body` (snapshot = read-first names,
  // carried = write-set = every assigned name), so the scan cannot
  // emit the reaching-def diagnostic here. The only remaining
  // failure mode is the stale-NameStoreRegionOpInterface check — a
  // pass bug, not a frontend bug. By this point we've already
  // restructured the outer IR (new op + `takeBody`), so a soft
  // `return failure()` would leak a torn module; abort fatally
  // instead.
  llvm::StringMap<Value> binding;
  for (auto &kv : ivSelfBinds)
    binding[kv.first.getValue()] = kv.second;
  for (StringAttr name : facts.snapshot)
    binding[name.getValue()] = snapValues[name];
  for (StringAttr name : carried)
    binding[name.getValue()] = iterArgFor[name];

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

// Promote `op`, a `hc.if`, so every `hc.assign` / `hc.name_load` inside
// either branch is rewritten in terms of op results and `hc.yield`
// values. Semantics match Python's: a name assigned in one branch but
// not the other retains its outer binding in the silent branch (i.e.
// that branch yields the snapshot value).
static LogicalResult promoteIf(HCIfOp op) {
  Block &thenBlock = op.getThenRegion().front();
  Region &elseRegion = op.getElseRegion();

  // Per-branch name facts. `hc.if` promotion needs these split
  // because the snapshot policy depends on branch symmetry: a name
  // written in both branches is fully redefined on every path and
  // doesn't need an outer-scope snap, while a name written in only one
  // branch still needs the outer value to fall back to on the silent
  // branch.
  TopLevelNameFacts thenFacts = collectTopLevelNameFacts(thenBlock);
  TopLevelNameFacts elseFacts;
  bool hasElse = !elseRegion.empty();
  if (hasElse)
    elseFacts = collectTopLevelNameFacts(elseRegion.front());
  if (thenFacts.reads.empty() && thenFacts.writes.empty() &&
      elseFacts.reads.empty() && elseFacts.writes.empty())
    return success();

  NameSet carriedSet;
  for (StringAttr n : thenFacts.writes)
    carriedSet.insert(n);
  for (StringAttr n : elseFacts.writes)
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
  NameSet snapshot = thenFacts.reads;
  for (StringAttr n : elseFacts.reads)
    snapshot.insert(n);
  for (StringAttr n : carriedSet) {
    bool symmetric = hasElse && thenFacts.writes.count(n) > 0 &&
                     elseFacts.writes.count(n) > 0;
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
  // reaching-def diagnostic for either branch. The only remaining
  // failure mode is the stale-NameStoreRegionOpInterface check — a
  // pass bug, and the outer IR has already been restructured here,
  // so any failure is treated as a fatal invariant break to avoid
  // leaking a torn module.
  auto processBranch = [&](Block &block) {
    llvm::StringMap<Value> binding;
    for (StringAttr name : snapshot)
      binding[name.getValue()] = snapValues[name];
    ensureYieldTerminator(block, loc);
    if (failed(scanAndPromoteBlock(block, binding)))
      llvm::report_fatal_error(
          "hc-promote-names: scan of `hc.if` branch failed after the outer "
          "IR was already restructured (pass invariant violation)");
    rewriteYieldWithCarried(block, binding, carried, loc);
  };

  processBranch(newOp.getThenRegion().front());
  if (!newOp.getElseRegion().empty())
    processBranch(newOp.getElseRegion().front());

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

  MLIRContext *ctx = op->getContext();
  Type undefTy = UndefType::get(ctx);
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

  // Only the op-kind choice needs a typed case; the rest of the
  // surgery is type-erased through `Operation *`. Reading the
  // `captures` attr through the concrete op accessor (rather than
  // by string key) keeps the pass tied to the ODS surface — a
  // rename in HCOps.td breaks the build here, not at runtime.
  Operation *newOp = nullptr;
  if (auto wi = dyn_cast<HCWorkitemRegionOp>(op)) {
    newOp = HCWorkitemRegionOp::create(outerBuilder, loc, newResultTypes,
                                       wi.getCapturesAttr());
  } else if (auto sg = dyn_cast<HCSubgroupRegionOp>(op)) {
    newOp = HCSubgroupRegionOp::create(outerBuilder, loc, newResultTypes,
                                       sg.getCapturesAttr());
  } else {
    llvm::report_fatal_error(
        "hc-promote-names: promoteNestedScope dispatched on an op kind it "
        "doesn't know how to rebuild (pass invariant violation)");
  }
  newOp->getRegion(0).takeBody(body);
  Block &newBody = newOp->getRegion(0).front();

  // Rewind the outer builder so capture snaps land *before* `newOp`,
  // not after. `create` left the builder positioned past `newOp`.
  outerBuilder.setInsertionPoint(newOp);

  // Same lazy-capture scan as the scan-only shape; reads unbound
  // locally still reach an outer-scope snapshot.
  //
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
    ModuleOp moduleOp = getOperation();
    WalkResult result = moduleOp.walk([&](Operation *op) {
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
