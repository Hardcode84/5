// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-shaped-compute-to-generic`: rewrite `hc.matmul` and
// `hc.reduce` into `hc.generic`. See `doc/layouts.md`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCSHAPEDCOMPUTETOGENERIC
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

// Shape operand for `hc.zeros` / `hc.full`.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// Per-axis offset attr from iter sym names; hash-consed via compose API.
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

// Failure = no-op on pre-inference fragments.
static FailureOr<SmallVector<ExprAttr>> getOperandShape(Value v) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(v.getType());
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

// Sum: float + int. Max / min: float only (int needs sign decision).
static bool reduceComboSupported(ReduceKind kind, Type elem) {
  bool isFloat = isa<FloatType>(elem);
  bool isInt = isa<IntegerType>(elem);
  switch (kind) {
  case ReduceKind::Sum:
    return isFloat || isInt;
  case ReduceKind::Max:
  case ReduceKind::Min:
    return isFloat;
  }
  return false;
}

// Identity per kind: sum → 0, max → -inf, min → +inf. Interface dispatch
// covers bare and semantic carriers uniformly. Caller pre-validates with
// `reduceComboSupported`.
static Value emitReduceIdentityFill(OpBuilder &builder, Location loc,
                                    ReduceKind kind, Type resultTy,
                                    Value shape) {
  Type elem =
      cast<SymbolicallyShapedTypeInterface>(resultTy).getSymbolicElementType();
  if (kind == ReduceKind::Sum) {
    if (auto intTy = dyn_cast<IntegerType>(elem)) {
      auto zero = HCConstOp::create(builder, loc, elem,
                                    builder.getIntegerAttr(intTy, 0));
      return HCFullOp::create(builder, loc, resultTy, zero, shape, TypeAttr(),
                              /*layout=*/LayoutAttr{});
    }
    return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                             /*layout=*/LayoutAttr{});
  }
  auto floatTy = cast<FloatType>(elem);
  bool negative = (kind == ReduceKind::Max);
  APFloat ident = APFloat::getInf(floatTy.getFloatSemantics(), negative);
  auto fill =
      HCConstOp::create(builder, loc, elem, builder.getFloatAttr(elem, ident));
  return HCFullOp::create(builder, loc, resultTy, fill, shape, TypeAttr(),
                          /*layout=*/LayoutAttr{});
}

// Caller pre-validates; element type matches `reduceComboSupported`.
static Value emitReduceCombine(OpBuilder &builder, Location loc,
                               ReduceKind kind, Value acc, Value cur) {
  switch (kind) {
  case ReduceKind::Sum:
    return HCAddOp::create(builder, loc, acc.getType(), acc, cur);
  case ReduceKind::Max:
    return arith::MaximumFOp::create(builder, loc, acc, cur);
  case ReduceKind::Min:
    return arith::MinimumFOp::create(builder, loc, acc, cur);
  }
  llvm_unreachable("reduceComboSupported gating skipped");
}

// `hc.astype` if needed; block arg is element-typed.
static Value promoteScalar(OpBuilder &builder, Location loc, Value v,
                           Type target) {
  if (v.getType() == target)
    return v;
  return HCAsTypeOp::create(builder, loc, target, v, TypeAttr::get(target));
}

// M / N / K from a rank-2 matmul (operand and result shapes agree).
struct MatmulShape {
  ExprAttr mDim;
  ExprAttr nDim;
  ExprAttr kDim;
};

// Rank-2 + shape agreement (K between operands, M/N across in+out).
// Disagreement → failure; caller leaves op for downstream diagnostics.
static FailureOr<MatmulShape> validateMatmulShape(HCMatmulOp op) {
  auto lhsShape = getOperandShape(op.getLhs());
  auto rhsShape = getOperandShape(op.getRhs());
  auto outShape = getOperandShape(op.getResult());
  if (failed(lhsShape) || failed(rhsShape) || failed(outShape))
    return failure();
  if (lhsShape->size() != 2 || rhsShape->size() != 2 || outShape->size() != 2)
    return failure();
  ExprAttr mDim = (*lhsShape)[0];
  ExprAttr kDim = (*lhsShape)[1];
  ExprAttr kDim2 = (*rhsShape)[0];
  ExprAttr nDim = (*rhsShape)[1];
  ExprAttr mOut = (*outShape)[0];
  ExprAttr nOut = (*outShape)[1];
  // `ExprHandle` defines `==` only.
  if (!(kDim.getValue() == kDim2.getValue()) ||
      !(mDim.getValue() == mOut.getValue()) ||
      !(nDim.getValue() == nOut.getValue()))
    return failure();
  return MatmulShape{mDim, nDim, kDim};
}

// Uniform arith family only (mixed FP / int needs sign convention).
static bool matmulElementsSupported(Type lhsElem, Type rhsElem, Type outElem) {
  if (!isa<FloatType, IntegerType>(outElem))
    return false;
  return isa<FloatType>(outElem) == isa<FloatType>(lhsElem) &&
         isa<FloatType>(outElem) == isa<FloatType>(rhsElem);
}

// Rank-2 + uniform arith only; mixed inputs `hc.astype`-promote to
// output element type. Failure leaves op for downstream diagnostics.
static LogicalResult rewriteMatmul(HCMatmulOp op, sym::Store &store) {
  FailureOr<MatmulShape> shapeDims = validateMatmulShape(op);
  if (failed(shapeDims))
    return failure();
  ExprAttr mDim = shapeDims->mDim;
  ExprAttr nDim = shapeDims->nDim;
  ExprAttr kDim = shapeDims->kDim;

  // Interface dispatch covers bare and semantic carriers uniformly.
  auto lhsTy = dyn_cast<SymbolicallyShapedTypeInterface>(op.getLhs().getType());
  auto rhsTy = dyn_cast<SymbolicallyShapedTypeInterface>(op.getRhs().getType());
  auto outTy =
      dyn_cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
  if (!lhsTy || !rhsTy || !outTy)
    return failure();
  Type lhsElem = lhsTy.getSymbolicElementType();
  Type rhsElem = rhsTy.getSymbolicElementType();
  Type outElem = outTy.getSymbolicElementType();
  if (!matmulElementsSupported(lhsElem, rhsElem, outElem))
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  OpBuilder builder(op);

  Value mBound = materializeIdxBound(builder, loc, mDim);
  Value nBound = materializeIdxBound(builder, loc, nDim);
  Value kBound = materializeIdxBound(builder, loc, kDim);
  Value shape = buildShapeTuple(builder, loc, {mBound, nBound});
  // Sum identity matches the matmul accumulator for both float and int.
  Value fill =
      emitReduceIdentityFill(builder, loc, ReduceKind::Sum, outTy, shape);

  StringAttr iSym = StringAttr::get(ctx, "i");
  StringAttr jSym = StringAttr::get(ctx, "j");
  StringAttr kSym = StringAttr::get(ctx, "k");
  ArrayAttr iterSyms = ArrayAttr::get(ctx, {iSym, jSym, kSym});
  ArrayAttr iterKinds =
      ArrayAttr::get(ctx, {IterKindAttr::get(ctx, IterKind::Parallel),
                           IterKindAttr::get(ctx, IterKind::Parallel),
                           IterKindAttr::get(ctx, IterKind::Reduction)});
  ArrayAttr lhsOff = offsetArrayFromIterSyms(ctx, store, {iSym, kSym});
  ArrayAttr rhsOff = offsetArrayFromIterSyms(ctx, store, {kSym, jSym});
  ArrayAttr outOff = offsetArrayFromIterSyms(ctx, store, {iSym, jSym});
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {lhsOff, rhsOff});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  SmallVector<Value> iterBoundsArr{mBound, nBound, kBound};
  SmallVector<Value> insArr{op.getLhs(), op.getRhs()};
  SmallVector<Value> outsArr{fill};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{outTy}, iterSyms,
      ValueRange(iterBoundsArr), iterKinds, ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  // Body: prod = lhs * rhs (astype-promoted), acc += prod.
  Block *body = new Block();
  BlockArgument av = body->addArgument(lhsElem, loc);
  BlockArgument bv = body->addArgument(rhsElem, loc);
  BlockArgument cv = body->addArgument(outElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  Value avp = promoteScalar(bodyBuilder, loc, av, outElem);
  Value bvp = promoteScalar(bodyBuilder, loc, bv, outElem);
  Value prod = HCMulOp::create(bodyBuilder, loc, outElem, avp, bvp);
  Value acc = HCAddOp::create(bodyBuilder, loc, outElem, cv, prod);
  HCYieldOp::create(bodyBuilder, loc, ValueRange{acc});

  op->replaceAllUsesWith(generic.getResults());
  op->erase();
  return success();
}

// Input minus `axis` equals output; rank and per-axis dims verified.
struct ReduceShape {
  SmallVector<ExprAttr> valShape;
  SmallVector<ExprAttr> outShape;
  uint64_t axis;
};

// Accepts: keepdims=false, valid axis, rank-1-smaller, positional dim agree.
static FailureOr<ReduceShape> validateReduceShape(HCReduceOp op) {
  if (op.getKeepdims())
    return failure();
  auto vs = getOperandShape(op.getValue());
  auto os = getOperandShape(op.getResult());
  if (failed(vs) || failed(os))
    return failure();
  uint64_t axis = op.getAxis();
  if (axis >= vs->size())
    return failure();
  if (os->size() + 1 != vs->size())
    return failure();
  for (size_t i = 0, j = 0; i < vs->size(); ++i) {
    if (i == axis)
      continue;
    if (!((*vs)[i].getValue() == (*os)[j].getValue()))
      return failure();
    ++j;
  }
  return ReduceShape{std::move(*vs), std::move(*os), axis};
}

// One parallel iter (`i_<n>`) per non-axis position; reduction sym separate.
static void buildReduceParallelIters(OpBuilder &builder, Location loc,
                                     MLIRContext *ctx,
                                     ArrayRef<ExprAttr> valShape, uint64_t axis,
                                     SmallVectorImpl<StringAttr> &syms,
                                     SmallVectorImpl<Value> &bounds) {
  syms.reserve(valShape.size() - 1);
  bounds.reserve(valShape.size() - 1);
  for (size_t i = 0; i < valShape.size(); ++i) {
    if (i == axis)
      continue;
    syms.push_back(StringAttr::get(ctx, ("i_" + Twine(syms.size())).str()));
    bounds.push_back(materializeIdxBound(builder, loc, valShape[i]));
  }
}

// Parallel iters at non-axis positions; reduction iter at `axis`.
static ArrayAttr buildReduceInputOffsets(MLIRContext *ctx, sym::Store &store,
                                         size_t inputRank, uint64_t axis,
                                         ArrayRef<StringAttr> parallelSyms,
                                         StringAttr reductionSym) {
  SmallVector<Attribute> inAxisExprs;
  inAxisExprs.reserve(inputRank);
  size_t parallelCursor = 0;
  for (size_t i = 0; i < inputRank; ++i) {
    StringRef name = (i == axis) ? reductionSym.getValue()
                                 : parallelSyms[parallelCursor++].getValue();
    auto handle = sym::composeExprSym(store, name);
    assert(succeeded(handle) && "iter sym name must compose to an expression");
    inAxisExprs.push_back(ExprAttr::get(ctx, *handle));
  }
  return ArrayAttr::get(ctx, inAxisExprs);
}

// Accepts keepdims=false; floats fully, int sum only. Failure leaves op.
static LogicalResult rewriteReduce(HCReduceOp op, sym::Store &store) {
  FailureOr<ReduceShape> shape = validateReduceShape(op);
  if (failed(shape))
    return failure();
  ArrayRef<ExprAttr> valShape = shape->valShape;
  uint64_t axis = shape->axis;

  // Interface dispatch covers bare and semantic carriers.
  auto valTy =
      dyn_cast<SymbolicallyShapedTypeInterface>(op.getValue().getType());
  auto outTy =
      dyn_cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
  if (!valTy || !outTy)
    return failure();
  // Rank-0 result needs a different output carrier; skipped.
  if (valTy.getSymbolicElementType() != outTy.getSymbolicElementType())
    return failure();
  Type elem = outTy.getSymbolicElementType();
  if (!reduceComboSupported(op.getKind(), elem))
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // Bounds in iter-sym order: parallels (non-axis), then `axis` reduction.
  SmallVector<StringAttr> parallelSyms;
  SmallVector<Value> parallelBounds;
  buildReduceParallelIters(builder, loc, ctx, valShape, axis, parallelSyms,
                           parallelBounds);
  StringAttr reductionSym = StringAttr::get(ctx, "r");
  Value reductionBound = materializeIdxBound(builder, loc, valShape[axis]);

  Value shapeTuple = buildShapeTuple(builder, loc, parallelBounds);
  Value fill =
      emitReduceIdentityFill(builder, loc, op.getKind(), outTy, shapeTuple);

  SmallVector<Attribute> iterSymList(parallelSyms.begin(), parallelSyms.end());
  iterSymList.push_back(reductionSym);
  ArrayAttr iterSyms = ArrayAttr::get(ctx, iterSymList);
  SmallVector<Attribute> iterKindList(
      parallelSyms.size(), IterKindAttr::get(ctx, IterKind::Parallel));
  iterKindList.push_back(IterKindAttr::get(ctx, IterKind::Reduction));
  ArrayAttr iterKinds = ArrayAttr::get(ctx, iterKindList);

  ArrayAttr inOffset = buildReduceInputOffsets(
      ctx, store, valShape.size(), axis, parallelSyms, reductionSym);
  ArrayAttr outOffset = offsetArrayFromIterSyms(ctx, store, parallelSyms);
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {inOffset});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOffset});

  SmallVector<Value> iterBoundsAll(parallelBounds.begin(),
                                   parallelBounds.end());
  iterBoundsAll.push_back(reductionBound);
  SmallVector<Value> insArr{op.getValue()};
  SmallVector<Value> outsArr{fill};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{outTy}, iterSyms,
      ValueRange(iterBoundsAll), iterKinds, ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  BlockArgument vIn = body->addArgument(elem, loc);
  BlockArgument vAcc = body->addArgument(elem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  Value combined = emitReduceCombine(bodyBuilder, loc, op.getKind(), vAcc, vIn);
  HCYieldOp::create(bodyBuilder, loc, ValueRange{combined});

  op->replaceAllUsesWith(generic.getResults());
  op->erase();
  return success();
}

struct HCShapedComputeToGenericPass
    : public hc::impl::HCShapedComputeToGenericBase<
          HCShapedComputeToGenericPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
    // Collect first; erase-in-walk invalidates the iterator.
    SmallVector<HCMatmulOp> matmuls;
    SmallVector<HCReduceOp> reduces;
    root->walk([&](Operation *op) {
      if (auto m = dyn_cast<HCMatmulOp>(op))
        matmuls.push_back(m);
      else if (auto r = dyn_cast<HCReduceOp>(op))
        reduces.push_back(r);
    });
    for (HCMatmulOp m : matmuls)
      (void)rewriteMatmul(m, store);
    for (HCReduceOp r : reduces)
      (void)rewriteReduce(r, store);
  }
};

} // namespace
