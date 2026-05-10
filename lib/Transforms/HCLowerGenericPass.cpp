// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-generic`: scalar (U = 1) baseline of the
// unroll-and-merge codegen frame for `hc.generic`. Lowers each op to
// `scf.parallel` over the parallel iters with an optional nested
// `scf.for` over the reduction iters whose `iter_args` carry the per-
// output accumulators. Per-operand offset expressions are evaluated
// structurally through the ixsimpl AST and emitted as
// `arith.{constant, addi, muli, remsi}`; pointer access lowers to
// `hc.ptr_offset` + `hc.ptr_load` (ins, outs init) /
// `hc.ptr_store` (outs final). See the pass description in
// `include/hc/Transforms/Passes.td` and the design in
// `doc/layouts.md`.

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

#include "ixsimpl.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERGENERIC
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Evaluate an ixsimpl expression node to a `Value` of `index` type.
// Recursive, structural. Returns failure for tags this v0 doesn't
// model (`IXS_FLOOR`, `IXS_CEIL`, `IXS_PIECEWISE`, `IXS_MAX`,
// `IXS_MIN`, `IXS_RAT` with non-unit denominator, `IXS_MUL` with a
// negative or oversized exponent, `IXS_SYM` with no scope binding),
// in which case the caller leaves the `hc.generic` op in place for a
// follow-up slice to handle. The `index` type matches what
// `arith.{constant, addi, muli, remsi}` and the SCF loop induction
// vars use, so the produced Value lands directly into
// `hc.ptr_offset`'s `$index` slot without further casts.
static FailureOr<Value> emitExpr(OpBuilder &builder, Location loc,
                                 const ixs_node *raw,
                                 const llvm::StringMap<Value> &syms) {
  if (!raw)
    return failure();
  ixs_node *node = const_cast<ixs_node *>(raw);
  switch (ixs_node_tag(node)) {
  case IXS_INT:
    return arith::ConstantIndexOp::create(builder, loc, ixs_node_int_val(node))
        .getResult();
  case IXS_RAT: {
    if (ixs_node_rat_den(node) != 1)
      return failure();
    return arith::ConstantIndexOp::create(builder, loc, ixs_node_rat_num(node))
        .getResult();
  }
  case IXS_SYM: {
    auto it = syms.find(ixs_node_sym_name(node));
    if (it == syms.end())
      return failure();
    return it->second;
  }
  case IXS_ADD: {
    int64_t coeff = ixs_node_int_val(ixs_node_add_coeff(node));
    Value acc = arith::ConstantIndexOp::create(builder, loc, coeff).getResult();
    uint32_t n = ixs_node_add_nterms(node);
    for (uint32_t i = 0; i < n; ++i) {
      auto term = emitExpr(builder, loc, ixs_node_add_term(node, i), syms);
      if (failed(term))
        return failure();
      int64_t tc = ixs_node_int_val(ixs_node_add_term_coeff(node, i));
      Value scaled = *term;
      if (tc != 1) {
        Value cv = arith::ConstantIndexOp::create(builder, loc, tc).getResult();
        scaled = arith::MulIOp::create(builder, loc, cv, *term).getResult();
      }
      acc = arith::AddIOp::create(builder, loc, acc, scaled).getResult();
    }
    return acc;
  }
  case IXS_MUL: {
    int64_t coeff = ixs_node_int_val(ixs_node_mul_coeff(node));
    Value acc = arith::ConstantIndexOp::create(builder, loc, coeff).getResult();
    uint32_t n = ixs_node_mul_nfactors(node);
    for (uint32_t i = 0; i < n; ++i) {
      // Negative or oversized exponents mean rationals or large
      // power expansions the linear-index lowering doesn't model;
      // bail and let the op survive.
      int32_t exp = ixs_node_mul_factor_exp(node, i);
      if (exp <= 0 || exp > 8)
        return failure();
      auto base =
          emitExpr(builder, loc, ixs_node_mul_factor_base(node, i), syms);
      if (failed(base))
        return failure();
      Value pow = *base;
      for (int32_t e = 1; e < exp; ++e)
        pow = arith::MulIOp::create(builder, loc, pow, *base).getResult();
      acc = arith::MulIOp::create(builder, loc, acc, pow).getResult();
    }
    return acc;
  }
  case IXS_MOD: {
    auto lhs = emitExpr(builder, loc, ixs_node_binary_lhs(node), syms);
    auto rhs = emitExpr(builder, loc, ixs_node_binary_rhs(node), syms);
    if (failed(lhs) || failed(rhs))
      return failure();
    return arith::RemSIOp::create(builder, loc, *lhs, *rhs).getResult();
  }
  default:
    return failure();
  }
}

// Sniff an SSA value for a "this Value carries the runtime binding
// for symbolic name N" relationship. Today the only carrier we
// recognize is `!hc.idx<expr>` whose `expr` is a bare `IXS_SYM` node
// — that's the type the rest of the dialect uses to pin a symbolic
// name to a concrete SSA value. Returns the bound name or empty.
static StringRef boundNameOf(Value v) {
  auto idx = dyn_cast<IdxType>(v.getType());
  if (!idx)
    return {};
  ExprAttr expr = idx.getExpr();
  if (!expr)
    return {};
  ixs_node *node = const_cast<ixs_node *>(expr.getValue().raw());
  if (!node || ixs_node_tag(node) != IXS_SYM)
    return {};
  return ixs_node_sym_name(node);
}

// Walk parents of `op` and collect every reachable `!hc.idx<bare-sym>`
// SSA Value as a potential outer-scope binding for that sym name.
// Innermost binding wins (we walk inside-out and skip names already
// in the map). The returned values may need a cast to `index` before
// they enter the offset evaluator's arithmetic — see `castIdxToIndex`.
static llvm::StringMap<Value> collectOuterScope(Operation *root) {
  llvm::StringMap<Value> out;
  Operation *cursor = root;
  while (cursor) {
    Block *block = cursor->getBlock();
    if (block) {
      for (BlockArgument arg : block->getArguments()) {
        StringRef name = boundNameOf(arg);
        if (!name.empty() && !out.count(name))
          out[name] = arg;
      }
      for (Operation &predOp : *block) {
        if (&predOp == cursor)
          break;
        for (OpResult r : predOp.getResults()) {
          StringRef name = boundNameOf(r);
          if (!name.empty() && !out.count(name))
            out[name] = r;
        }
      }
    }
    cursor = cursor->getParentOp();
  }
  return out;
}

// Materialize a Value usable from `arith` ops at `index` type. SSA
// Values typed `!hc.idx<...>` cross the boundary via
// `unrealized_conversion_cast`; the lower-to-llvm pass at the end
// of the pipeline collapses these casts when both endpoints are
// concrete. Pre-bound bindings whose Value is already `index` pass
// through unchanged.
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

// Pull the single composed offset expression for an operand at index
// `idx` from an `ins_offsets` / `outs_offsets` array. Caller has
// already checked rank-1 via `isV0Candidate`.
static const ixs_node *getOperandOffset(ArrayAttr arrayAttr, size_t idx) {
  auto perOp = cast<ArrayAttr>(arrayAttr[idx]);
  return cast<ExprAttr>(perOp[0]).getValue().raw();
}

// Body cloner. Replaces block-arg references with their loaded /
// carry values via the IR mapping, drops the `hc.yield` (its values
// are surfaced through `yieldedOut` for the caller to feed into the
// SCF loop's terminator), and emits the cloned ops at the builder's
// current insertion point. Mirrors the `HCLowerLaunchBodyPass`
// pattern for `hc.for_range` / `hc.if`.
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

// Single-operand load helper: evaluate the offset expression at the
// current scope, emit `hc.ptr_offset` + `hc.ptr_load` for it. Used
// at both the outer (parallel-iter scope, for outs init) and inner
// (full-iter scope, for ins per reduction iteration) levels.
static FailureOr<Value> emitPtrLoad(OpBuilder &builder, Location loc, Value ptr,
                                    const ixs_node *expr,
                                    const llvm::StringMap<Value> &scope) {
  auto offValue = emitExpr(builder, loc, expr, scope);
  if (failed(offValue))
    return failure();
  Value typedOff = castIdxToIndex(builder, loc, *offValue);
  Value addr = HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, typedOff)
                   .getResult();
  Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
  return HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
}

// Load every input operand using the supplied scope. The scope must
// have bindings for every iter sym referenced in the ins offsets;
// callers extend the parallel-iter scope with reduction-iter
// induction vars before invoking this from the reduction nest's
// innermost level.
static FailureOr<SmallVector<Value>>
emitInsLoads(OpBuilder &builder, HCGenericOp op,
             const llvm::StringMap<Value> &scope) {
  Location loc = op.getLoc();
  ArrayAttr insOff = op.getInsOffsetsAttr();
  SmallVector<Value> out;
  for (auto [i, inOperand] : llvm::enumerate(op.getIns())) {
    auto loaded = emitPtrLoad(builder, loc, inOperand,
                              getOperandOffset(insOff, i), scope);
    if (failed(loaded))
      return failure();
    out.push_back(*loaded);
  }
  return out;
}

// Load every output's initial accumulator at the parallel-iter
// scope (the outs offset references parallel iters only by op
// invariant, so the reduction iters need not be in scope here).
static FailureOr<SmallVector<Value>>
emitOutsInitLoads(OpBuilder &builder, HCGenericOp op,
                  const llvm::StringMap<Value> &scope) {
  Location loc = op.getLoc();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  SmallVector<Value> out;
  for (auto [i, outOperand] : llvm::enumerate(op.getOuts())) {
    auto loaded = emitPtrLoad(builder, loc, outOperand,
                              getOperandOffset(outsOff, i), scope);
    if (failed(loaded))
      return failure();
    out.push_back(*loaded);
  }
  return out;
}

// Emit `hc.ptr_store` for each output at the loop-scope offset using
// the supplied accumulator values. Recomputes the address (rather
// than reusing the load's address) so the final write doesn't
// depend on hoisting the address through the inner reduction loop —
// keeps the IR straightforward and CSE handles the duplication
// later.
static LogicalResult emitOutsStores(OpBuilder &builder, HCGenericOp op,
                                    ValueRange finals,
                                    const llvm::StringMap<Value> &scope) {
  Location loc = op.getLoc();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (auto [i, outOperand] : llvm::enumerate(op.getOuts())) {
    auto offValue = emitExpr(builder, loc, getOperandOffset(outsOff, i), scope);
    if (failed(offValue))
      return failure();
    Value typedOff = castIdxToIndex(builder, loc, *offValue);
    Value addr = HCPtrOffsetOp::create(builder, loc, outOperand.getType(),
                                       outOperand, typedOff)
                     .getResult();
    HCPtrStoreOp::create(builder, loc, finals[i], addr);
  }
  return success();
}

// Inner reduction nest. Builds a chain of `scf.for` ops one per
// reduction iter, threading the outs accumulators through `iter_args`
// at every level. The innermost loop body clones the `hc.generic`
// body once with the current iter induction vars. Returns the
// outermost reduction loop's results (the final accumulator values).
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
      // Innermost: emit per-iteration ins loads and clone the body
      // with `iter_args` providing the carry slot.
      auto insVals = emitInsLoads(builder, op, localScope);
      if (failed(insVals))
        return failure();
      SmallVector<Value> outsArgs(iterArgs.begin(), iterArgs.end());
      SmallVector<Value> yielded;
      if (failed(cloneBody(builder, op, *insVals, outsArgs, yielded)))
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
// to a bare reduction nest with no outer parallel.
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

  llvm::StringMap<Value> outerScope = collectOuterScope(op);
  // Pre-cast each outer-scope binding to `index` once. The cast
  // sits at the op's original insertion point, so it dominates all
  // emitted loops; subsequent `emitExpr` SYM lookups reuse it. For
  // bindings whose Value is already `index`-typed the cast is the
  // identity and `castIdxToIndex` returns the original Value.
  for (auto &entry : outerScope)
    entry.second = castIdxToIndex(builder, loc, entry.second);

  // Helper: emit the per-parallel-iteration body. `localScope` carries
  // the parallel-iter sym -> induction var bindings (plus any outer
  // scope). The outs init load uses this scope; the reduction nest
  // (if any) extends scope with the reduction iter vars and runs
  // the ins loads + body clone there.
  auto buildPerParallel =
      [&](OpBuilder &b, llvm::StringMap<Value> &localScope) -> LogicalResult {
    auto initOuts = emitOutsInitLoads(b, op, localScope);
    if (failed(initOuts))
      return failure();
    SmallVector<Value> finals;
    if (redIdx.empty()) {
      // Pure-parallel: ins live in the same scope as the outs init,
      // body clones once with both as block args.
      auto insVals = emitInsLoads(b, op, localScope);
      if (failed(insVals))
        return failure();
      if (failed(cloneBody(b, op, *insVals, *initOuts, finals)))
        return failure();
    } else {
      // Reduction nest: the inner level emits the ins loads and body
      // clone once the reduction iter vars are in scope.
      auto reduced = emitReductionNest(b, op, redIdx, *initOuts, localScope);
      if (failed(reduced))
        return failure();
      finals = std::move(*reduced);
    }
    return emitOutsStores(b, op, finals, localScope);
  };

  if (parIdx.empty()) {
    // Pure-reduction: no outer parallel, build the reduction nest
    // directly at the original insertion point.
    llvm::StringMap<Value> scope = outerScope;
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
                              llvm::StringMap<Value> scope = outerScope;
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
