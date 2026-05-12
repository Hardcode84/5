// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-load-store-to-generic`: rewrite the per-tile memory
// ops `hc.load` / `hc.vload` / `hc.store` into the body-driven
// `hc.generic` surface that owns the post-flatten compute pipeline.
// See the pass description in `include/hc/Transforms/Passes.td` and
// the design in `doc/layouts.md`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOADSTORETOGENERIC
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Materialize one bound dim as `!hc.idx<dim>` SSA via an
// `hc.idx_apply` with no listed symbols (free shape names stay
// ambient). Mirrors the helper in `hc-shaped-compute-to-generic` —
// the rewriter has full structural knowledge of the iteration space,
// so the bounds-inference pass is a no-op if it ever runs after.
static Value materializeIdxBound(OpBuilder &builder, Location loc,
                                 ExprAttr dim) {
  auto idxTy = IdxType::get(builder.getContext(), dim);
  return HCIdxApplyOp::create(builder, loc, idxTy, ValueRange{},
                              builder.getStrArrayAttr({}));
}

// Build a `tuple<idx<...>, ...>` SSA tuple from the per-axis bounds —
// every shaped allocator (`hc.zeros` / `hc.vzeros`) takes its result
// shape through one of these.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// Pull each dim off a shaped operand. Bails on non-shaped types
// (`!hc.undef`) and on shapes that carry non-`#hc.expr` dim entries —
// either case means the rewriter can't materialize bounds and the op
// stays for later inference / downstream diagnostics.
static FailureOr<SmallVector<ExprAttr>> getOperandShape(Type t) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
  if (!shaped)
    return failure();
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return failure();
  SmallVector<ExprAttr> dims;
  dims.reserve(shape.getDims().size());
  for (Attribute a : shape.getDims()) {
    auto e = dyn_cast<ExprAttr>(a);
    if (!e)
      return failure();
    dims.push_back(e);
  }
  return dims;
}

// Per-axis decomposition of an access op's index operand. `base` is the
// lower bound of the tile walk on that axis (a scalar `!hc.idx<expr>`
// contributes `base = expr`; a slice contributes `base = lower` or `0`
// when the slice's lower is absent). `step` is the per-iter stride; a
// scalar idx is always step 1, a slice contributes `step = step_expr`
// or `1` when absent. Both are stored as `ExprAttr` so callers can
// build the per-axis offset structurally without re-parsing.
struct AxisIndex {
  ExprAttr base;
  ExprAttr step;
};

// Extract `(base, step)` for one access op index operand. Mirrors
// `extractAccessIndexExpr` in `hc-flatten-with-layouts` on the slice
// branch — accepts default-step slices (`step = 1`) and pinned
// `!hc.idx<expr>` lower / step. Non-pinned slice parts (`!hc.undef` or
// other), raw `index`, and untyped `!hc.idx` fail the rewrite: there's
// no symbolic name to bind into a `lower + step*iter` offset, and
// silently lowering would emit an offset the launch-body would walk
// without the stride contribution.
static FailureOr<AxisIndex>
extractAxisIndex(MLIRContext *ctx, sym::Store &store, Type indexType) {
  auto litOne = sym::composeExprInt(store, 1);
  if (failed(litOne))
    return failure();
  ExprAttr stepOne = ExprAttr::get(ctx, *litOne);
  if (auto idx = llvm::dyn_cast<IdxType>(indexType)) {
    if (ExprAttr expr = idx.getExpr())
      return AxisIndex{expr, stepOne};
    return failure();
  }
  if (auto slice = llvm::dyn_cast<SliceType>(indexType)) {
    ExprAttr base;
    if (Type lowerTy = slice.getLowerType()) {
      auto lowerIdx = llvm::dyn_cast<IdxType>(lowerTy);
      if (!lowerIdx || !lowerIdx.getExpr())
        return failure();
      base = lowerIdx.getExpr();
    } else {
      auto zero = sym::composeExprInt(store, 0);
      if (failed(zero))
        return failure();
      base = ExprAttr::get(ctx, *zero);
    }
    ExprAttr step;
    if (Type stepTy = slice.getStepType()) {
      auto stepIdx = llvm::dyn_cast<IdxType>(stepTy);
      if (!stepIdx || !stepIdx.getExpr())
        return failure();
      step = stepIdx.getExpr();
    } else {
      step = stepOne;
    }
    return AxisIndex{base, step};
  }
  return failure();
}

// Compose `base + step * iterSym` for the per-axis offset on the
// memory-side operand. Folds the trivial `step == 1` to `base +
// iterSym` so the printed offset stays the form scalar-idx callers
// already produce — without the fold, otherwise-identical loads land
// on two distinct hash-consed sums and obscure the diff. ixsimpl
// hash-conses, so building via the store keeps the printed offset
// canonical and shares storage with other identical sums elsewhere in
// the IR.
static FailureOr<ExprAttr> composeBasePlusStepIter(MLIRContext *ctx,
                                                   sym::Store &store,
                                                   AxisIndex axis,
                                                   StringAttr iterSym) {
  auto symHandle = sym::composeExprSym(store, iterSym.getValue());
  if (failed(symHandle))
    return failure();
  sym::ExprHandle term = *symHandle;
  std::optional<int64_t> stepLit =
      sym::getIntegerLiteralValue(axis.step.getValue());
  if (!stepLit || *stepLit != 1) {
    auto mul = sym::composeExprBinary(store, axis.step.getValue(),
                                      sym::ExprBinaryOp::Mul, term);
    if (failed(mul))
      return failure();
    term = *mul;
  }
  auto sum = sym::composeExprBinary(store, axis.base.getValue(),
                                    sym::ExprBinaryOp::Add, term);
  if (failed(sum))
    return failure();
  return ExprAttr::get(ctx, *sum);
}

// Build an `ArrayAttr<#hc.expr>` from a list of bare iter sym names —
// identity offsets on the value-side operand. Mirrors the helper
// in `hc-shaped-compute-to-generic`.
static ArrayAttr offsetArrayFromIterSyms(MLIRContext *ctx, sym::Store &store,
                                         ArrayRef<StringAttr> names) {
  SmallVector<Attribute> exprs;
  exprs.reserve(names.size());
  for (StringAttr name : names) {
    auto handle = sym::composeExprSym(store, name.getValue());
    assert(succeeded(handle) && "iter sym name must compose to an expression");
    exprs.push_back(ExprAttr::get(ctx, *handle));
  }
  return ArrayAttr::get(ctx, exprs);
}

// Build the per-axis offset attribute on the memory-side operand:
// `[base_0 + step_0*i_0, base_1 + step_1*i_1, ...]`. Empty `axes`
// (whole-tensor access, e.g. `hc.store %dst[]`) collapses to identity
// over the iter syms — same shape, no addressing addend. Sizes must
// match post-pre-checks; this helper just assembles the array.
static FailureOr<ArrayAttr>
composeMemoryOffsetArray(MLIRContext *ctx, sym::Store &store,
                         ArrayRef<AxisIndex> axes,
                         ArrayRef<StringAttr> iterSyms) {
  SmallVector<Attribute> exprs;
  exprs.reserve(iterSyms.size());
  if (axes.empty()) {
    for (StringAttr name : iterSyms) {
      auto handle = sym::composeExprSym(store, name.getValue());
      if (failed(handle))
        return failure();
      exprs.push_back(ExprAttr::get(ctx, *handle));
    }
  } else {
    for (auto [axis, name] : llvm::zip_equal(axes, iterSyms)) {
      auto sum = composeBasePlusStepIter(ctx, store, axis, name);
      if (failed(sum))
        return failure();
      exprs.push_back(*sum);
    }
  }
  return ArrayAttr::get(ctx, exprs);
}

// Synthesise a fresh value-typed init of the given shape. The body
// of an all-parallel `hc.generic` never reads its outs carry, so
// any zero-cost initialisation is fine — `hc.zeros` for tensors,
// `hc.vzeros` for vectors. Result type drives the choice; the verifier
// only enforces workgroup scope on the tensor variants.
static Value emitValueInit(OpBuilder &builder, Location loc, Type resultTy,
                           Value shape) {
  if (isa<mlir::hc::VectorType, BareVectorType>(resultTy))
    return HCVZerosOp::create(builder, loc, resultTy, shape, TypeAttr());
  return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr());
}

// Common shape:
//   * iter syms `i_0`, ..., `i_{r-1}` — distinct from the source-level
//     symbol names that index operands carry. Conflicts with a body
//     name `i_0` are theoretically possible but very unlikely in
//     practice; the verifier catches it and a proper collision-free
//     namer is its own follow-up (matches the same trade-off the
//     reduce rewriter takes).
//   * all parallel iters; one bound per axis materialised from the
//     tile shape.
//   * value-side operand carries identity offsets, memory-side
//     carries `[base_k + i_k]`.
struct CommonRewriteData {
  SmallVector<StringAttr> iterSyms;
  SmallVector<Value> iterBounds;
  ArrayAttr iterSymsAttr;
  ArrayAttr iterKindsAttr;
};

static CommonRewriteData buildCommon(OpBuilder &builder, Location loc,
                                     ArrayRef<ExprAttr> tileShape) {
  MLIRContext *ctx = builder.getContext();
  CommonRewriteData out;
  out.iterSyms.reserve(tileShape.size());
  out.iterBounds.reserve(tileShape.size());
  SmallVector<Attribute> symAttrs;
  symAttrs.reserve(tileShape.size());
  SmallVector<Attribute> kindAttrs;
  kindAttrs.reserve(tileShape.size());
  for (auto [k, dim] : llvm::enumerate(tileShape)) {
    auto sym = StringAttr::get(ctx, ("i_" + Twine(k)).str());
    out.iterSyms.push_back(sym);
    symAttrs.push_back(sym);
    out.iterBounds.push_back(materializeIdxBound(builder, loc, dim));
    kindAttrs.push_back(IterKindAttr::get(ctx, IterKind::Parallel));
  }
  out.iterSymsAttr = ArrayAttr::get(ctx, symAttrs);
  out.iterKindsAttr = ArrayAttr::get(ctx, kindAttrs);
  return out;
}

// Pull the per-axis `(base, step)` off an op's index operands. v0
// handles pinned `!hc.idx<expr>` scalar indices and slices whose
// lower/step (when present) are pinned `!hc.idx<expr>`. Anything else
// fails the whole rewrite — the launch-body lowering still owns those
// shapes.
static FailureOr<SmallVector<AxisIndex>>
collectAxisIndices(MLIRContext *ctx, sym::Store &store, ValueRange indices) {
  SmallVector<AxisIndex> axes;
  axes.reserve(indices.size());
  for (Value idx : indices) {
    auto axis = extractAxisIndex(ctx, store, idx.getType());
    if (failed(axis))
      return failure();
    axes.push_back(*axis);
  }
  return axes;
}

// Element / pointee type the body block-arg carries for a given
// operand. Mirrors `genericOperandElement` in `lib/IR/HCOps.cpp` for
// the shaped + ptr cases the rewriter actually emits — kept local
// because hardcoding the shape here is fine while we only emit
// shaped + ptr/buffer operands.
static Type bodyArgElementType(Type t) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t))
    return shaped.getSymbolicElementType();
  return {};
}

// ----- hc.load / hc.vload -----------------------------------------------

// Common load rewriter for both `hc.load` and `hc.vload`. The two ops
// have identical operand layouts (source, indices, shape) and the
// only difference is which value-init op the result type wants.
template <typename OpT>
static LogicalResult rewriteLoadLike(OpT op, sym::Store &store) {
  Type resultTy = op.getResult().getType();
  auto tileShape = getOperandShape(resultTy);
  if (failed(tileShape))
    return failure();

  // Source operand is `getBuffer` on `hc.load` and `getSource` on
  // `hc.vload`; the type is what tells us the body element type.
  Value source;
  if constexpr (std::is_same_v<OpT, HCLoadOp>)
    source = op.getBuffer();
  else
    source = op.getSource();

  // Empty index list is a legal shape (`hc.load %t[], shape ...`):
  // it means the access addresses the operand at the tile origin,
  // which is just the iter syms with no addressing addend. Non-empty
  // lists must rank-match the tile shape — anything else is
  // inconsistent IR the access op's own checks would have caught.
  ValueRange indices = op.getIndices();
  if (!indices.empty() && indices.size() != tileShape->size())
    return failure();
  MLIRContext *ctx = op.getContext();
  auto axes = collectAxisIndices(ctx, store, indices);
  if (failed(axes))
    return failure();

  Type srcElem = bodyArgElementType(source.getType());
  Type resElem = bodyArgElementType(resultTy);
  if (!srcElem || !resElem || srcElem != resElem)
    return failure();

  Location loc = op.getLoc();
  OpBuilder builder(op);
  CommonRewriteData common = buildCommon(builder, loc, *tileShape);
  Value shapeTuple = buildShapeTuple(builder, loc, common.iterBounds);
  Value initOut = emitValueInit(builder, loc, resultTy, shapeTuple);

  auto inOff = composeMemoryOffsetArray(ctx, store, *axes, common.iterSyms);
  if (failed(inOff))
    return failure();
  ArrayAttr outOff = offsetArrayFromIterSyms(ctx, store, common.iterSyms);
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {*inOff});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  SmallVector<Value> insArr{source};
  SmallVector<Value> outsArr{initOut};
  // No ambient sym SSA captured at this surface; the offset attrs still
  // carry the free names and ambient-context resolution handles them.
  // `hc-flatten-with-layouts` is where we bind them as a real dataflow
  // edge — see HCGenericOp's `ambient_idxs` doc.
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{resultTy}, common.iterSymsAttr,
      ValueRange(common.iterBounds), common.iterKindsAttr, ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  BlockArgument bv = body->addArgument(srcElem, loc);
  body->addArgument(resElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  HCYieldOp::create(bodyBuilder, loc, ValueRange{bv});

  op->replaceAllUsesWith(generic.getResults());
  op->erase();
  return success();
}

// ----- hc.store ---------------------------------------------------------

static LogicalResult rewriteStore(HCStoreOp op, sym::Store &store) {
  // Tensor / bare_tensor dst is workgroup-shared LDS storage; the IR
  // models it as a value-typed operand even though the runtime
  // semantic is in-place mutation. A clean rewrite would need to
  // produce a new SSA tensor result and propagate it through every
  // subsequent use of `%dst`, which crosses op boundaries we can't
  // resolve in a local rewrite. Buffer dst falls through cleanly
  // because polymorphic outs already model ptr/buffer in-place writes
  // without an SSA result. Tensor-dst stores are a separate slice.
  Value dst = op.getDest();
  if (!isa<BufferType>(dst.getType()))
    return failure();

  Value src = op.getSource();
  Type srcTy = src.getType();
  auto tileShape = getOperandShape(srcTy);
  if (failed(tileShape))
    return failure();

  ValueRange indices = op.getIndices();
  if (!indices.empty() && indices.size() != tileShape->size())
    return failure();
  MLIRContext *ctx = op.getContext();
  auto axes = collectAxisIndices(ctx, store, indices);
  if (failed(axes))
    return failure();

  Type srcElem = bodyArgElementType(srcTy);
  Type dstElem = bodyArgElementType(dst.getType());
  if (!srcElem || !dstElem || srcElem != dstElem)
    return failure();

  // Masked path: the mask operand rides as an extra ins slot with
  // identity offsets (same shape as `src`, both tile-local). The body
  // loads the mask element alongside the src element and terminates
  // with `hc.yield_predicated` instead of `hc.yield`; the lowering
  // routes that through `hc.ptr_store_pred` at the dst's ins-slot
  // offset, so masked-out lanes leave the existing dst contents in
  // place. Mask must be the same shape as src (the `hc.store`
  // verifier already enforces this); we don't carry separate axes
  // because the mask's offsets are identity by construction.
  Value mask = op.getMask();
  Type maskElem;
  if (mask) {
    auto maskShape = getOperandShape(mask.getType());
    if (failed(maskShape))
      return failure();
    if (maskShape->size() != tileShape->size())
      return failure();
    maskElem = bodyArgElementType(mask.getType());
    if (!maskElem)
      return failure();
  }

  Location loc = op.getLoc();
  OpBuilder builder(op);
  CommonRewriteData common = buildCommon(builder, loc, *tileShape);

  ArrayAttr inOff = offsetArrayFromIterSyms(ctx, store, common.iterSyms);
  auto outOffArr = composeMemoryOffsetArray(ctx, store, *axes, common.iterSyms);
  if (failed(outOffArr))
    return failure();
  SmallVector<Attribute> insOffArr{inOff};
  SmallVector<Value> insArr{src};
  if (mask) {
    insOffArr.push_back(inOff);
    insArr.push_back(mask);
  }
  ArrayAttr insOffsets = ArrayAttr::get(ctx, insOffArr);
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {*outOffArr});

  SmallVector<Value> outsArr{dst};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{}, common.iterSymsAttr,
      ValueRange(common.iterBounds), common.iterKindsAttr, ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  BlockArgument sv = body->addArgument(srcElem, loc);
  BlockArgument mv;
  if (mask)
    mv = body->addArgument(maskElem, loc);
  body->addArgument(dstElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  if (mask)
    HCYieldPredicatedOp::create(bodyBuilder, loc, ValueRange{sv},
                                ValueRange{mv});
  else
    HCYieldOp::create(bodyBuilder, loc, ValueRange{sv});

  op->erase();
  return success();
}

struct HCLoadStoreToGenericPass
    : public hc::impl::HCLoadStoreToGenericBase<HCLoadStoreToGenericPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
    // Collect first, mutate after — `op->erase()` inside the walk
    // would invalidate the iterator the walk is driving.
    SmallVector<HCLoadOp> loads;
    SmallVector<HCVLoadOp> vloads;
    SmallVector<HCStoreOp> stores;
    root->walk([&](Operation *op) {
      if (auto l = dyn_cast<HCLoadOp>(op))
        loads.push_back(l);
      else if (auto v = dyn_cast<HCVLoadOp>(op))
        vloads.push_back(v);
      else if (auto s = dyn_cast<HCStoreOp>(op))
        stores.push_back(s);
    });
    for (HCLoadOp l : loads)
      (void)rewriteLoadLike(l, store);
    for (HCVLoadOp v : vloads)
      (void)rewriteLoadLike(v, store);
    for (HCStoreOp s : stores)
      (void)rewriteStore(s, store);
  }
};

} // namespace
