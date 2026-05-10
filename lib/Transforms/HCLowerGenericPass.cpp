// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-generic`: scalar (U = 1) baseline of the
// unroll-and-merge codegen frame for `hc.generic`. Lowers each op to
// `scf.parallel` over the parallel iters with an optional nested
// `scf.for` over the reduction iters whose `iter_args` carry the per-
// output accumulators. Per-operand offset expressions surface as
// `hc.idx_apply` carrying the original `ExprAttr` and binding the
// op's iter syms to the loop induction vars; the launch-body
// lowering downstream walks the expression and substitutes the
// listed operands plus any ambient kernel-bound symbols (shape dims,
// stride params). Pointer access lowers to `hc.ptr_offset` +
// `hc.ptr_load` (ins, outs init) / `hc.ptr_store` (outs final).

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERGENERIC
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Materialize a Value usable from index-consuming ops. Builtin
// `index` values pass through; `!hc.idx<...>` SSA crosses via
// `unrealized_conversion_cast`, which the launch-body lowering /
// reconcile-unrealized-casts collapse downstream once both ends are
// concrete.
static Value castIdxToIndex(OpBuilder &builder, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), v)
      .getResult(0);
}

// Pre-flight check: returns `true` when this op matches the v0
// codegen scope (all operands typed-ptr, every offset rank-1, no
// `scf.if` in body, every iter bound resolved). Mismatch leaves the
// op in place — we'd rather skip than partially lower. Opaque
// pointers (no element type on the carrier) also bail: the v0 load
// emission needs the element type to spell the result, and the
// pointer's `$elementType` is the only available source.
static bool isV0Candidate(HCGenericOp op) {
  auto goodPtr = [](Value v) {
    auto p = dyn_cast<PtrType>(v.getType());
    return p && p.getElementType();
  };
  for (Value v : op.getIns())
    if (!goodPtr(v))
      return false;
  for (Value v : op.getOuts())
    if (!goodPtr(v))
      return false;
  ArrayAttr insOff = op.getInsOffsetsAttr();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (Attribute perOp : insOff)
    if (cast<ArrayAttr>(perOp).size() != 1)
      return false;
  for (Attribute perOp : outsOff)
    if (cast<ArrayAttr>(perOp).size() != 1)
      return false;
  for (Value bound : op.getIterBounds())
    if (auto def = bound.getDefiningOp())
      if (isa<HCUndefValueOp>(def))
        return false;
  for (Operation &nested : op.getBody().front())
    if (isa<scf::IfOp>(nested))
      return false;
  return true;
}

// Pull the single composed offset expression for an operand at
// position `idx` from an `ins_offsets` / `outs_offsets` array.
// Caller has already checked rank-1 via `isV0Candidate`.
static ExprAttr getOperandOffset(ArrayAttr arrayAttr, size_t idx) {
  return cast<ExprAttr>(cast<ArrayAttr>(arrayAttr[idx])[0]);
}

// Materialize one offset expression as `index`-typed SSA. Filters
// the loop-scope binding map down to the symbols that actually
// occur in the offset expression — `hc.idx_apply`'s verifier
// requires every listed symbol to be free in the expression — then
// emits the apply and casts the resulting `!hc.idx<expr>` to
// `index` for the consumer's `hc.ptr_offset`. Symbols not in the
// loop-scope map (kernel ABI shape dims, stride params, etc.) stay
// ambient and are bound by the launch-body lowering's launch-walk.
static Value emitOffset(OpBuilder &builder, Location loc, ExprAttr offsetExpr,
                        const llvm::StringMap<Value> &loopScope) {
  llvm::StringSet<> freeSyms;
  sym::walkSymbolNames(offsetExpr.getValue(),
                       [&](StringRef name) { freeSyms.insert(name); });

  // Sort the binding order lexicographically: `llvm::StringMap`
  // iteration is unordered, and `hc.idx_apply`'s textual form pins
  // the operand list, so leaving it to map order would make
  // round-trip tests fragile across rebuilds.
  SmallVector<StringRef> names;
  for (const auto &entry : loopScope)
    if (freeSyms.contains(entry.getKey()))
      names.push_back(entry.getKey());
  llvm::sort(names);

  SmallVector<Attribute> symAttrs;
  SmallVector<Value> operands;
  for (StringRef name : names) {
    symAttrs.push_back(builder.getStringAttr(name));
    operands.push_back(loopScope.lookup(name));
  }

  auto idxTy = IdxType::get(builder.getContext(), offsetExpr);
  Value applied = HCIdxApplyOp::create(builder, loc, idxTy, operands,
                                       builder.getArrayAttr(symAttrs))
                      .getResult();
  return castIdxToIndex(builder, loc, applied);
}

// Body cloner. Replaces block-arg references with their loaded /
// carry values via the IR mapping, drops the `hc.yield` (its values
// are surfaced through `yieldedOut` for the caller to feed into the
// SCF loop's terminator), and emits the cloned ops at the builder's
// current insertion point.
static LogicalResult cloneBody(OpBuilder &builder, HCGenericOp op,
                               ValueRange insVals, ValueRange outsVals,
                               SmallVectorImpl<Value> &yieldedOut) {
  Block &src = op.getBody().front();
  IRMapping mapping;
  size_t pos = 0;
  for (Value v : insVals)
    mapping.map(src.getArgument(pos++), v);
  for (Value v : outsVals)
    mapping.map(src.getArgument(pos++), v);
  auto yield = dyn_cast<HCYieldOp>(src.back());
  if (!yield)
    return op.emitOpError("body must end with `hc.yield`");
  for (Operation &nested : src) {
    if (&nested == yield.getOperation())
      break;
    builder.clone(nested, mapping);
  }
  yieldedOut.clear();
  yieldedOut.reserve(yield.getValues().size());
  for (Value v : yield.getValues())
    yieldedOut.push_back(mapping.lookupOrDefault(v));
  return success();
}

// Single-operand load helper: emit the offset (as `hc.idx_apply` +
// cast to index), then `hc.ptr_offset` + `hc.ptr_load`. Used at both
// the outer (parallel-iter scope, for outs init) and inner (full-
// iter scope, for ins per reduction iteration) levels.
static Value emitPtrLoad(OpBuilder &builder, Location loc, Value ptr,
                         ExprAttr expr, const llvm::StringMap<Value> &scope) {
  Value off = emitOffset(builder, loc, expr, scope);
  Value addr =
      HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off).getResult();
  Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
  return HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
}

// Load every input operand using the supplied scope. The scope
// must have bindings for every iter sym referenced in the ins
// offsets; callers extend the parallel-iter scope with reduction-
// iter induction vars before invoking this from the reduction
// nest's innermost level. Symbols beyond the iter-sym set (e.g.
// kernel shape dims) stay ambient through `idx_apply`'s unlisted
// symbols.
static SmallVector<Value> emitInsLoads(OpBuilder &builder, HCGenericOp op,
                                       const llvm::StringMap<Value> &scope) {
  Location loc = op.getLoc();
  ArrayAttr insOff = op.getInsOffsetsAttr();
  SmallVector<Value> out;
  for (auto [i, inOperand] : llvm::enumerate(op.getIns()))
    out.push_back(emitPtrLoad(builder, loc, inOperand,
                              getOperandOffset(insOff, i), scope));
  return out;
}

// Load every output's initial accumulator at the parallel-iter
// scope (the outs offset references parallel iters only by op
// invariant, so the reduction iters need not be in scope here).
static SmallVector<Value>
emitOutsInitLoads(OpBuilder &builder, HCGenericOp op,
                  const llvm::StringMap<Value> &scope) {
  Location loc = op.getLoc();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  SmallVector<Value> out;
  for (auto [i, outOperand] : llvm::enumerate(op.getOuts()))
    out.push_back(emitPtrLoad(builder, loc, outOperand,
                              getOperandOffset(outsOff, i), scope));
  return out;
}

// Emit `hc.ptr_store` for each output at the loop-scope offset
// using the supplied accumulator values. Recomputes the address
// (rather than reusing the load's address) so the final write
// doesn't depend on hoisting the address through the inner
// reduction loop — keeps the IR straightforward and CSE handles
// the duplication later.
static void emitOutsStores(OpBuilder &builder, HCGenericOp op,
                           ValueRange finals,
                           const llvm::StringMap<Value> &scope) {
  Location loc = op.getLoc();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (auto [i, outOperand] : llvm::enumerate(op.getOuts())) {
    Value off = emitOffset(builder, loc, getOperandOffset(outsOff, i), scope);
    Value addr = HCPtrOffsetOp::create(builder, loc, outOperand.getType(),
                                       outOperand, off)
                     .getResult();
    HCPtrStoreOp::create(builder, loc, finals[i], addr);
  }
}

// Inner reduction nest. Builds a chain of `scf.for` ops one per
// reduction iter, threading the outs accumulators through
// `iter_args` at every level. The innermost loop body clones the
// `hc.generic` body once with the current iter induction vars.
// Returns the outermost reduction loop's results (the final
// accumulator values).
static FailureOr<SmallVector<Value>>
emitReductionNest(OpBuilder &builder, HCGenericOp op, ArrayRef<size_t> redIdx,
                  ValueRange initAccs, llvm::StringMap<Value> scope) {
  Location loc = op.getLoc();
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  ArrayAttr symsAttr = op.getIterSymsAttr();
  ValueRange bounds = op.getIterBounds();

  std::function<FailureOr<SmallVector<Value>>(size_t, ValueRange,
                                              llvm::StringMap<Value> &)>
      build = [&](size_t depth, ValueRange iterArgs,
                  llvm::StringMap<Value> &localScope)
      -> FailureOr<SmallVector<Value>> {
    if (depth == redIdx.size()) {
      SmallVector<Value> insVals = emitInsLoads(builder, op, localScope);
      SmallVector<Value> outsArgs(iterArgs.begin(), iterArgs.end());
      SmallVector<Value> yielded;
      if (failed(cloneBody(builder, op, insVals, outsArgs, yielded)))
        return failure();
      return yielded;
    }
    size_t iter = redIdx[depth];
    StringRef name = cast<StringAttr>(symsAttr[iter]).getValue();
    Value bound = castIdxToIndex(builder, loc, bounds[iter]);
    auto forOp = scf::ForOp::create(builder, loc, c0, bound, c1, iterArgs);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    auto prev = localScope.lookup(name);
    localScope[name] = forOp.getInductionVar();
    auto inner = build(depth + 1, forOp.getRegionIterArgs(), localScope);
    if (prev)
      localScope[name] = prev;
    else
      localScope.erase(name);
    if (failed(inner))
      return failure();
    scf::YieldOp::create(builder, loc, *inner);
    return SmallVector<Value>(forOp.getResults().begin(),
                              forOp.getResults().end());
  };

  return build(0, initAccs, scope);
}

// Top-level: emit the parallel iters as an outer `scf.parallel`,
// load the initial accumulators inside, run the reduction nest,
// store the finals. Pure-parallel collapses to a single
// `scf.parallel` with no inner reduction; pure-reduction collapses
// to a bare reduction nest with no outer parallel. Iter-sym to
// induction-var bindings live in `scope` and feed every offset's
// `hc.idx_apply`.
static LogicalResult lowerOne(HCGenericOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  ArrayAttr symsAttr = op.getIterSymsAttr();
  ArrayAttr kindsAttr = op.getIterKindsAttr();
  ValueRange bounds = op.getIterBounds();

  SmallVector<size_t> parIdx, redIdx;
  for (auto [i, kindAttr] :
       llvm::enumerate(kindsAttr.getAsRange<IterKindAttr>())) {
    if (kindAttr.getValue() == IterKind::Parallel)
      parIdx.push_back(i);
    else
      redIdx.push_back(i);
  }

  auto buildPerParallel =
      [&](OpBuilder &b, llvm::StringMap<Value> &localScope) -> LogicalResult {
    SmallVector<Value> initOuts = emitOutsInitLoads(b, op, localScope);
    SmallVector<Value> finals;
    if (redIdx.empty()) {
      SmallVector<Value> insVals = emitInsLoads(b, op, localScope);
      if (failed(cloneBody(b, op, insVals, initOuts, finals)))
        return failure();
    } else {
      auto reduced = emitReductionNest(b, op, redIdx, initOuts, localScope);
      if (failed(reduced))
        return failure();
      finals = std::move(*reduced);
    }
    emitOutsStores(b, op, finals, localScope);
    return success();
  };

  if (parIdx.empty()) {
    llvm::StringMap<Value> scope;
    if (failed(buildPerParallel(builder, scope)))
      return failure();
  } else {
    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
    SmallVector<Value> lowers(parIdx.size(), c0);
    SmallVector<Value> uppers;
    SmallVector<Value> steps(parIdx.size(), c1);
    for (size_t pi : parIdx)
      uppers.push_back(castIdxToIndex(builder, loc, bounds[pi]));

    LogicalResult bodyStatus = success();
    scf::ParallelOp::create(builder, loc, lowers, uppers, steps,
                            [&](OpBuilder &b, Location, ValueRange ivs) {
                              llvm::StringMap<Value> scope;
                              for (auto [k, pi] : llvm::enumerate(parIdx)) {
                                StringRef name =
                                    cast<StringAttr>(symsAttr[pi]).getValue();
                                scope[name] = ivs[k];
                              }
                              bodyStatus = buildPerParallel(b, scope);
                            });
    if (failed(bodyStatus))
      return failure();
  }

  // v0 candidates are all-ptr-outs (zero SSA results), so erasing
  // is sufficient — no `replaceAllUsesWith` needed. The candidate
  // check would have rejected a value-typed out, but we keep the
  // belt-and-suspenders check so a future relaxation of
  // `isV0Candidate` doesn't silently lose results.
  if (op.getNumResults() != 0)
    return op.emitOpError("v0 lowering only handles all-ptr outs (zero SSA "
                          "results)");
  op.erase();
  return success();
}

struct HCLowerGenericPass
    : public hc::impl::HCLowerGenericBase<HCLowerGenericPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<HCGenericOp> work;
    getOperation()->walk([&](HCGenericOp op) { work.push_back(op); });
    for (HCGenericOp op : work) {
      if (!isV0Candidate(op))
        continue;
      if (failed(lowerOne(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

// `createHCLowerGenericPass()` is emitted by tablegen.
