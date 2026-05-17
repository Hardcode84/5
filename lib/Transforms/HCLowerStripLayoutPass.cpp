// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-strip-layout`: rewrite `hc.strip_layout` into
// `hc.generic` — one parallel iter per result axis, source on `ins`
// with identity per-axis offsets, bare init on `outs`, body yields
// the loaded element.

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

// Empty-binding `hc.idx_apply`; free shape names stay ambient.
static Value materializeIdxBound(OpBuilder &builder, Location loc,
                                 ExprAttr dim) {
  auto idxTy = IdxType::get(builder.getContext(), dim);
  return HCIdxApplyOp::create(builder, loc, idxTy, ValueRange{},
                              builder.getStrArrayAttr({}));
}

// Shape operand for the nullary allocators.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// Per-axis offset attr from iter sym names; hash-consed so canonical.
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

// Zero-init the strip's result; `result.layout == null` by verifier so
// allocator's layout attr stays absent. `hc.vzeros` / `hc.zeros` per flavor.
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

// Fail if shape absent or any dim is not an `ExprAttr`.
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

// All axes Parallel for the strip rewrite.
struct ParallelIterSpace {
  SmallVector<StringAttr> syms;
  SmallVector<Value> bounds;
  SmallVector<Attribute> symAttrs;
  SmallVector<Attribute> kindAttrs;
};

// Iter syms `i_0`..`i_{r-1}`; verifier catches body-local collisions.
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

// Body yields source element; init value goes unread.
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

  // src matches result — forward, no allocator/generic.
  if (srcTy == resTy) {
    op.getResult().replaceAllUsesWith(src);
    op.erase();
    return success();
  }

  // Read result shape; verifier already enforces operand-result parity.
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

  // Bare allocator; `result.layout == null` by verifier.
  Value shapeTuple = buildShapeTuple(builder, loc, iters.bounds);
  Value initOut = emitBareInit(builder, loc, resTy, shapeTuple);

  // Identity per-axis offsets; flatten substitutes iter syms into the
  // source's layout (or identity layout when none).
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
    // Collect first; erase-in-walk invalidates the iterator.
    SmallVector<HCStripLayoutOp> strips;
    root->walk([&](HCStripLayoutOp op) { strips.push_back(op); });
    for (HCStripLayoutOp op : strips)
      (void)lowerStripLayout(op, store);
  }
};

} // namespace
