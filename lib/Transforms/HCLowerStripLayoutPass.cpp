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
// is left absent. Flavor preserved: `hc.vzeros` for bare_vector
// results, `hc.zeros` for bare_tensor. Semantic carriers are
// rejected by the contract gate in `hc-decompose-shaped-values` and
// never reach this pass.
static Value emitBareInit(OpBuilder &builder, Location loc, Type resultTy,
                          Value shape) {
  if (isa<mlir::hc::BareVectorType>(resultTy))
    return HCVZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                              /*layout=*/LayoutAttr{});
  assert(isa<mlir::hc::BareTensorType>(resultTy) &&
         "strip_layout result type must be a bare shaped carrier");
  return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                           /*layout=*/LayoutAttr{});
}

// Extract the per-axis ExprAttr dim list from a shaped type's symbolic
// shape, or fail if the shape is absent or any dim is not an ExprAttr.
static FailureOr<SmallVector<ExprAttr>>
collectTileShape(SymbolicallyShapedTypeInterface shaped) {
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return failure();
  SmallVector<ExprAttr> tile;
  tile.reserve(shape.getDims().size());
  for (Attribute dim : shape.getDims()) {
    auto e = dyn_cast<ExprAttr>(dim);
    if (!e)
      return failure();
    tile.push_back(e);
  }
  return tile;
}

// Per-axis iter spec for the generic op: sym name, bound SSA, and the
// matching attributes for the op's `iter_syms` / `iter_kinds` arrays.
// All axes are Parallel for the strip rewrite.
struct ParallelIterSpace {
  SmallVector<StringAttr> syms;
  SmallVector<Value> bounds;
  SmallVector<Attribute> symAttrs;
  SmallVector<Attribute> kindAttrs;
};

// Iter syms `i_0`, ..., `i_{r-1}` (parallel) — matches the naming the
// load-side rewriter uses; collisions with body-local names are
// structurally unlikely and the verifier catches them. Bounds come
// from the result shape's per-axis dim exprs, materialised as
// `!hc.idx<dim>` via empty-binding `hc.idx_apply`.
static ParallelIterSpace buildParallelIterSpace(OpBuilder &builder,
                                                Location loc,
                                                ArrayRef<ExprAttr> tile) {
  MLIRContext *ctx = builder.getContext();
  ParallelIterSpace space;
  space.syms.reserve(tile.size());
  space.bounds.reserve(tile.size());
  space.symAttrs.reserve(tile.size());
  space.kindAttrs.reserve(tile.size());
  for (auto [k, dim] : llvm::enumerate(tile)) {
    auto sym = StringAttr::get(ctx, ("i_" + Twine(k)).str());
    space.syms.push_back(sym);
    space.symAttrs.push_back(sym);
    space.bounds.push_back(materializeIdxBound(builder, loc, dim));
    space.kindAttrs.push_back(IterKindAttr::get(ctx, IterKind::Parallel));
  }
  return space;
}

// Build the element-copy generic: one parallel iter per axis, identity
// per-axis offsets on both source and init, body yields the source
// element so the init's value goes unread.
static HCGenericOp
emitElementCopyGeneric(OpBuilder &builder, Location loc, Type resTy,
                       Type srcElem, Type resElem, Value src, Value initOut,
                       const ParallelIterSpace &iters, ArrayAttr insOffsets,
                       ArrayAttr outsOffsets) {
  MLIRContext *ctx = builder.getContext();
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{resTy},
      ArrayAttr::get(ctx, iters.symAttrs), ValueRange(iters.bounds),
      ArrayAttr::get(ctx, iters.kindAttrs), ValueRange(src),
      ValueRange(initOut), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  BlockArgument bv = body->addArgument(srcElem, loc);
  body->addArgument(resElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  HCYieldOp::create(bodyBuilder, loc, ValueRange{bv});
  return generic;
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
  FailureOr<SmallVector<ExprAttr>> tileShape = collectTileShape(resShaped);
  if (failed(tileShape))
    return failure();

  Type srcElem = srcShaped.getSymbolicElementType();
  Type resElem = resShaped.getSymbolicElementType();
  if (!srcElem || !resElem || srcElem != resElem)
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  OpBuilder builder(op);

  ParallelIterSpace iters = buildParallelIterSpace(builder, loc, *tileShape);

  // Allocate bare destination matching the result type. The bare
  // allocators don't take a layout attribute on emission — the
  // result type's layout is null by `hc.strip_layout`'s verifier.
  Value shapeTuple = buildShapeTuple(builder, loc, iters.bounds);
  Value initOut = emitBareInit(builder, loc, resTy, shapeTuple);

  // Per-axis offsets on both operands are identity (the iter syms).
  // The source's layout (if any) provides the actual gather offset —
  // flatten substitutes the iter exprs into the layout's `offset` at
  // `index_syms[k]` and yields the linear storage offset post-1D.
  // For a layout-less source (the verifier allows this; the strip is
  // a layout-and/or-flavor change) flatten composes the identity
  // layout, which collapses the per-axis exprs into the linear
  // dim-product offset — a trivial element copy.
  ArrayAttr inOff = offsetArrayFromIterSyms(ctx, store, iters.syms);
  ArrayAttr outOff = offsetArrayFromIterSyms(ctx, store, iters.syms);
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {inOff});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  HCGenericOp generic =
      emitElementCopyGeneric(builder, loc, resTy, srcElem, resElem, src,
                             initOut, iters, insOffsets, outsOffsets);

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
