// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-shaped-compute-to-generic`: rewrite the source-level
// `hc.matmul` and `hc.reduce` ops into the body-driven `hc.generic`
// surface that the post-flatten compute pipeline understands. See the
// pass description in `include/hc/Transforms/Passes.td` and the
// design in `doc/layouts.md`.

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

// Wrap a single dim attr (taken from an operand's symbolic shape) as
// an `hc.idx_apply` carrying `!hc.idx<dim>` so later bound-resolution
// sees a fully-typed source. No listed symbols: the dim's free
// names (shape syms) are ambient and get bound by the launch-body
// lowering. The rewriter has full structural knowledge of the
// iteration space, so the bounds-inference pass would be a no-op if
// it ran after this one.
static Value materializeIdxBound(OpBuilder &builder, Location loc,
                                 ExprAttr dim) {
  auto idxTy = IdxType::get(builder.getContext(), dim);
  return HCIdxApplyOp::create(builder, loc, idxTy, ValueRange{},
                              builder.getStrArrayAttr({}));
}

// Build a `tuple<idx<...>, idx<...>, ...>` SSA value from the
// per-axis materialized bounds. `hc.zeros` / `hc.full` need a
// concrete shape operand of this exact shape.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// Build the per-axis offset attribute from a list of iter sym names:
// `[#hc.expr<"i">, #hc.expr<"j">, ...]`. ixsimpl hash-conses
// expressions, so building via `composeExprSym` keeps the printed
// form canonical and shares storage across uses.
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

// Pull the symbolic shape off a tensor/vector operand. Returns
// failure rather than emitting — callers fold it into a leave-as-is
// outcome so this pass is a no-op on pre-inference fragments.
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

// (kind, elementType) pairs handled in v0. Sum works on float and
// integer; max / min are float only — integer max/min identity needs
// a signed-vs-unsigned slot decision and the matching combinator
// op, which is a separate piece of work.
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

// Identity fill matching the reduce kind. Sum -> 0 (`hc.zeros` for
// floats, `hc.full <0>` for ints); max -> -inf; min -> +inf.
// Caller pre-validates with `reduceComboSupported`. Result type is
// the HC semantic tensor; `mlir::hc::TensorType` is spelled out
// because the unqualified `TensorType` collides with builtin
// `mlir::TensorType` under both `using namespace mlir` and
// `using namespace mlir::hc`.
static Value emitReduceIdentityFill(OpBuilder &builder, Location loc,
                                    ReduceKind kind,
                                    mlir::hc::TensorType resultTy,
                                    Value shape) {
  Type elem = resultTy.getElementType();
  if (kind == ReduceKind::Sum) {
    if (auto intTy = dyn_cast<IntegerType>(elem)) {
      auto zero = HCConstOp::create(builder, loc, elem,
                                    builder.getIntegerAttr(intTy, 0));
      return HCFullOp::create(builder, loc, resultTy, zero, shape, TypeAttr());
    }
    return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr());
  }
  auto floatTy = cast<FloatType>(elem);
  bool negative = (kind == ReduceKind::Max);
  APFloat ident = APFloat::getInf(floatTy.getFloatSemantics(), negative);
  auto fill =
      HCConstOp::create(builder, loc, elem, builder.getFloatAttr(elem, ident));
  return HCFullOp::create(builder, loc, resultTy, fill, shape, TypeAttr());
}

// Combinator used by `hc.reduce` body. Caller pre-validates with
// `reduceComboSupported`, so element type is float for max/min and
// either float or integer for sum.
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

// Insert an `hc.astype` if the source's element type differs from
// `target`. The block arg type doubles as the source element type
// because `hc.generic` lowers per-element work in scalar form.
static Value promoteScalar(OpBuilder &builder, Location loc, Value v,
                           Type target) {
  if (v.getType() == target)
    return v;
  return HCAsTypeOp::create(builder, loc, target, v, TypeAttr::get(target));
}

// Rewrite a single `hc.matmul lhs, rhs -> out` into an `hc.zeros` +
// `hc.generic` pair. v0 only handles rank-2 operands and a uniform
// arith family on inputs and output (all-float or all-int). Mixed
// input element types promote through `hc.astype` to the output
// element type. Returns failure (and leaves the op alone) on
// anything unsupported so the original op survives for downstream
// diagnostics.
static LogicalResult rewriteMatmul(HCMatmulOp op, sym::Store &store) {
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
  // K must agree between operands; M and N must agree across inputs
  // and output. If the symbolic shapes disagree, the original op
  // wouldn't have type-checked — bail to leave the IR untouched.
  // ExprHandle defines `==` only; pre-C++20 doesn't synthesize `!=`.
  if (!(kDim.getValue() == kDim2.getValue()) ||
      !(mDim.getValue() == mOut.getValue()) ||
      !(nDim.getValue() == nOut.getValue()))
    return failure();

  auto lhsTy = cast<mlir::hc::TensorType>(op.getLhs().getType());
  auto rhsTy = cast<mlir::hc::TensorType>(op.getRhs().getType());
  auto outTy = cast<mlir::hc::TensorType>(op.getResult().getType());
  Type lhsElem = lhsTy.getElementType();
  Type rhsElem = rhsTy.getElementType();
  Type outElem = outTy.getElementType();
  if (!isa<FloatType, IntegerType>(outElem))
    return failure();
  // Mixed FP / int won't promote cleanly via `hc.astype` for the
  // matmul body without picking a sign convention; punt.
  if (isa<FloatType>(outElem) != isa<FloatType>(lhsElem) ||
      isa<FloatType>(outElem) != isa<FloatType>(rhsElem))
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  OpBuilder builder(op);

  Value mBound = materializeIdxBound(builder, loc, mDim);
  Value nBound = materializeIdxBound(builder, loc, nDim);
  Value kBound = materializeIdxBound(builder, loc, kDim);
  Value shape = buildShapeTuple(builder, loc, {mBound, nBound});
  // Sum identity matches the matmul accumulator for both float and
  // integer flavours.
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
      ValueRange(outsArr), insOffsets, outsOffsets);

  // Body: %p = lhs * rhs (with astype-promotion to the accumulator
  // element type), %s = acc + %p, yield %s.
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

// Rewrite a single `hc.reduce val, kind, axis -> out` into an
// identity fill + `hc.generic`. v0 only handles `keepdims = false`,
// floats fully (sum / max / min) and integer sum. Other shapes
// leave the op alone for downstream diagnostics.
static LogicalResult rewriteReduce(HCReduceOp op, sym::Store &store) {
  if (op.getKeepdims())
    return failure();
  auto valShape = getOperandShape(op.getValue());
  auto outShape = getOperandShape(op.getResult());
  if (failed(valShape) || failed(outShape))
    return failure();
  uint64_t axis = op.getAxis();
  if (axis >= valShape->size())
    return failure();
  // For keepdims = false the result rank is one less than the input;
  // the result shape is the input shape with `axis` dropped.
  if (outShape->size() + 1 != valShape->size())
    return failure();
  for (size_t i = 0, j = 0; i < valShape->size(); ++i) {
    if (i == axis)
      continue;
    if (!((*valShape)[i].getValue() == (*outShape)[j].getValue()))
      return failure();
    ++j;
  }

  auto valTy = cast<mlir::hc::TensorType>(op.getValue().getType());
  auto outTy = dyn_cast<mlir::hc::TensorType>(op.getResult().getType());
  // Reduction-to-scalar (rank 0 result with non-tensor result type)
  // would need a different output carrier; leave it for now.
  if (!outTy)
    return failure();
  if (valTy.getElementType() != outTy.getElementType())
    return failure();
  Type elem = outTy.getElementType();
  if (!reduceComboSupported(op.getKind(), elem))
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // Materialize iter bounds in iter-sym order: parallel iters track
  // every non-`axis` input dimension in input order, then a single
  // reduction iter for `axis`. Naming uses `i_<n>` for parallels
  // and `r` for the reduction so the printed IR reads cleanly even
  // for large ranks.
  SmallVector<StringAttr> parallelSyms;
  SmallVector<Value> parallelBounds;
  parallelSyms.reserve(outShape->size());
  parallelBounds.reserve(outShape->size());
  for (size_t i = 0; i < valShape->size(); ++i) {
    if (i == axis)
      continue;
    parallelSyms.push_back(
        StringAttr::get(ctx, ("i_" + Twine(parallelSyms.size())).str()));
    parallelBounds.push_back(materializeIdxBound(builder, loc, (*valShape)[i]));
  }
  StringAttr reductionSym = StringAttr::get(ctx, "r");
  Value reductionBound = materializeIdxBound(builder, loc, (*valShape)[axis]);

  Value shape = buildShapeTuple(builder, loc, parallelBounds);
  Value fill = emitReduceIdentityFill(builder, loc, op.getKind(), outTy, shape);

  SmallVector<Attribute> iterSymList(parallelSyms.begin(), parallelSyms.end());
  iterSymList.push_back(reductionSym);
  ArrayAttr iterSyms = ArrayAttr::get(ctx, iterSymList);
  SmallVector<Attribute> iterKindList(
      parallelSyms.size(), IterKindAttr::get(ctx, IterKind::Parallel));
  iterKindList.push_back(IterKindAttr::get(ctx, IterKind::Reduction));
  ArrayAttr iterKinds = ArrayAttr::get(ctx, iterKindList);

  // Per-axis offsets for the input: parallel iters fill non-axis
  // positions in input order; the reduction iter fills `axis`.
  SmallVector<Attribute> inAxisExprs;
  inAxisExprs.reserve(valShape->size());
  size_t parallelCursor = 0;
  for (size_t i = 0; i < valShape->size(); ++i) {
    StringRef name = (i == axis) ? reductionSym.getValue()
                                 : parallelSyms[parallelCursor++].getValue();
    auto handle = sym::composeExprSym(store, name);
    assert(succeeded(handle) && "iter sym name must compose to an expression");
    inAxisExprs.push_back(ExprAttr::get(ctx, *handle));
  }
  ArrayAttr inOffset = ArrayAttr::get(ctx, inAxisExprs);
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
      ValueRange(outsArr), insOffsets, outsOffsets);

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
    // Collect first, mutate after — `op->erase()` inside the walk
    // would invalidate the iterator the walk is driving.
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
