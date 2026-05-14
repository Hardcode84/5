// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-strip-layout`: rewrite the user-marked
// boundary op `hc.strip_layout` into the body-driven `hc.generic`
// surface. See the pass description in
// `include/hc/Transforms/Passes.td` and the bare-result contract on
// `hc.strip_layout` in `include/hc/IR/HCOps.td`.
//
// Shape parity with `hc-load-store-to-generic`'s `rewriteLoadLike` —
// emit one parallel iter per result axis, source on `ins` with
// identity per-axis offsets (flatten composes them through the
// source's layout into the gather offset post-flatten), bare init
// on `outs` with identity per-axis offsets, body is a single
// `hc.yield` of the loaded element. The helper functions
// (`materializeIdxBound`, `buildShapeTuple`, `offsetArrayFromIterSyms`)
// duplicate the same shapes that live in three sibling pass files;
// extracting them is a follow-up tidy that's noted in the
// generic-funneling cluster.

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
#define GEN_PASS_DEF_HCLOWERSTRIPLAYOUT
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Materialize one bound dim as `!hc.idx<dim>` SSA via an
// `hc.idx_apply` with no listed symbols (free shape names stay
// ambient). Mirrors the helper in the sibling generic-funneling
// passes.
static Value materializeIdxBound(OpBuilder &builder, Location loc,
                                 ExprAttr dim) {
  auto idxTy = IdxType::get(builder.getContext(), dim);
  return HCIdxApplyOp::create(builder, loc, idxTy, ValueRange{},
                              builder.getStrArrayAttr({}));
}

// Build a `tuple<idx<...>, ...>` SSA tuple for the shape operand the
// nullary allocators consume.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// Build the per-axis offset attribute from iter sym names — `[#hc.expr
// <"i_0">, #hc.expr<"i_1">, ...]`. ixsimpl hash-conses these so the
// printed form stays canonical across emitters.
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

// Synthesise a zero-init of the strip's result type. The verifier
// enforces `result.layout == null` (the whole point of the op is to
// drop the layout), so the nullary allocator's optional layout attr
// is left absent. Flavor preserved: `hc.vzeros` for vector /
// bare_vector results, `hc.zeros` for tensor / bare_tensor.
static Value emitBareInit(OpBuilder &builder, Location loc, Type resultTy,
                          Value shape) {
  if (isa<mlir::hc::VectorType, mlir::hc::BareVectorType>(resultTy))
    return HCVZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                              /*layout=*/LayoutAttr{});
  return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                           /*layout=*/LayoutAttr{});
}

static LogicalResult lowerStripLayout(HCStripLayoutOp op, sym::Store &store) {
  Value src = op.getValue();
  Type srcTy = src.getType();
  Type resTy = op.getResult().getType();

  auto srcShaped = dyn_cast<SymbolicallyShapedTypeInterface>(srcTy);
  auto resShaped = dyn_cast<SymbolicallyShapedTypeInterface>(resTy);
  if (!srcShaped || !resShaped)
    return failure();

  // Defensive no-op: source already matches the result. The user
  // reached for `as_layout(value, None)` against a producer that
  // already gave them a bare-with-no-layout carrier, so the strip
  // collapses to a forward — no allocator, no generic. Common in
  // helpers that don't know upfront whether the caller's
  // ``group.vload`` carried an explicit layout.
  if (srcTy == resTy) {
    op.getResult().replaceAllUsesWith(src);
    op.erase();
    return success();
  }

  // Result shape drives the iter range and the init's shape tuple.
  // The op's verifier already requires operand and result shape to
  // agree; we read from the result side so a missing op-side shape
  // bails this rewrite cleanly without surfacing the operand /
  // result divergence elsewhere.
  ShapeAttr resShape = resShaped.getSymbolicShape();
  if (!resShape)
    return failure();
  SmallVector<ExprAttr> tileShape;
  tileShape.reserve(resShape.getDims().size());
  for (Attribute dim : resShape.getDims()) {
    auto e = dyn_cast<ExprAttr>(dim);
    if (!e)
      return failure();
    tileShape.push_back(e);
  }

  Type srcElem = srcShaped.getSymbolicElementType();
  Type resElem = resShaped.getSymbolicElementType();
  if (!srcElem || !resElem || srcElem != resElem)
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // Iter syms `i_0`, ..., `i_{r-1}` (parallel) — matches the naming
  // the load-side rewriter uses; collisions with body-local names are
  // structurally unlikely and the verifier catches them. Bounds come
  // from the result shape's per-axis dim exprs, materialised as
  // `!hc.idx<dim>` via empty-binding `hc.idx_apply`.
  SmallVector<StringAttr> iterSyms;
  SmallVector<Value> iterBounds;
  SmallVector<Attribute> iterSymAttrs;
  SmallVector<Attribute> iterKindAttrs;
  iterSyms.reserve(tileShape.size());
  iterBounds.reserve(tileShape.size());
  iterSymAttrs.reserve(tileShape.size());
  iterKindAttrs.reserve(tileShape.size());
  for (auto [k, dim] : llvm::enumerate(tileShape)) {
    auto sym = StringAttr::get(ctx, ("i_" + Twine(k)).str());
    iterSyms.push_back(sym);
    iterSymAttrs.push_back(sym);
    iterBounds.push_back(materializeIdxBound(builder, loc, dim));
    iterKindAttrs.push_back(IterKindAttr::get(ctx, IterKind::Parallel));
  }

  // Allocate bare destination matching the result type. The bare
  // allocators don't take a layout attribute on emission — the
  // result type's layout is null by `hc.strip_layout`'s verifier.
  Value shapeTuple = buildShapeTuple(builder, loc, iterBounds);
  Value initOut = emitBareInit(builder, loc, resTy, shapeTuple);

  // Per-axis offsets on both operands are identity (the iter syms).
  // The source's layout (if any) provides the actual gather offset —
  // flatten substitutes the iter exprs into the layout's `offset` at
  // `index_syms[k]` and yields the linear storage offset post-1D.
  // For a layout-less source (the verifier allows this; the strip is
  // a layout-and/or-flavor change) flatten composes the identity
  // layout, which collapses the per-axis exprs into the linear
  // dim-product offset — a trivial element copy.
  ArrayAttr inOff = offsetArrayFromIterSyms(ctx, store, iterSyms);
  ArrayAttr outOff = offsetArrayFromIterSyms(ctx, store, iterSyms);
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {inOff});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  SmallVector<Value> insArr{src};
  SmallVector<Value> outsArr{initOut};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{resTy},
      ArrayAttr::get(ctx, iterSymAttrs), ValueRange(iterBounds),
      ArrayAttr::get(ctx, iterKindAttrs), ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  // Body: yield the loaded source element. The generic body takes one
  // block arg per operand (source elem, then init elem); the yield
  // forwards the source's value, leaving the init's value unread.
  Block *body = new Block();
  BlockArgument bv = body->addArgument(srcElem, loc);
  body->addArgument(resElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  HCYieldOp::create(bodyBuilder, loc, ValueRange{bv});

  op.getResult().replaceAllUsesWith(generic.getResult(0));
  op.erase();
  return success();
}

struct HCLowerStripLayoutPass
    : public hc::impl::HCLowerStripLayoutBase<HCLowerStripLayoutPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
    // Collect first, mutate after — erasing inside the walk would
    // invalidate the iterator the walk is driving.
    SmallVector<HCStripLayoutOp> strips;
    root->walk([&](HCStripLayoutOp op) { strips.push_back(op); });
    for (HCStripLayoutOp op : strips)
      (void)lowerStripLayout(op, store);
  }
};

} // namespace
