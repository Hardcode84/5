// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCSymbols.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::hc;

namespace {

static sym::Store &symbolStore(MLIRContext *ctx) {
  return ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
}

static std::optional<ExprAttr> idxExprAttr(Type type) {
  auto idx = dyn_cast_or_null<IdxType>(type);
  if (!idx)
    return std::nullopt;
  if (ExprAttr expr = idx.getExpr())
    return expr;
  return std::nullopt;
}

template <typename HandleT>
static LogicalResult emitComposeError(Operation *op, StringRef kind,
                                      FailureOr<HandleT> handle,
                                      const std::string &diag) {
  if (succeeded(handle))
    return success();
  op->emitOpError("failed to infer symbolic ") << kind << ": " << diag;
  return failure();
}

static FailureOr<ExprAttr> composeExprAttr(ExprAttr lhs, sym::ExprBinaryOp op,
                                           ExprAttr rhs, Operation *diagOp) {
  MLIRContext *ctx = diagOp->getContext();
  std::string diag;
  FailureOr<sym::ExprHandle> handle =
      sym::composeExprBinary(symbolStore(ctx), sym::ExprHandle(lhs.getNode()),
                             op, sym::ExprHandle(rhs.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<ExprAttr> composeNegExprAttr(ExprAttr value,
                                              Operation *diagOp) {
  MLIRContext *ctx = diagOp->getContext();
  std::string diag;
  FailureOr<sym::ExprHandle> handle = sym::composeExprNeg(
      symbolStore(ctx), sym::ExprHandle(value.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<ExprAttr> composeCeilExprAttr(ExprAttr value,
                                               Operation *diagOp) {
  MLIRContext *ctx = diagOp->getContext();
  std::string diag;
  FailureOr<sym::ExprHandle> handle = sym::composeExprCeil(
      symbolStore(ctx), sym::ExprHandle(value.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<PredAttr> composePredAttr(ExprAttr lhs, sym::PredCmpOp op,
                                           ExprAttr rhs, Operation *diagOp) {
  MLIRContext *ctx = diagOp->getContext();
  std::string diag;
  FailureOr<sym::PredHandle> handle =
      sym::composePredCmp(symbolStore(ctx), sym::ExprHandle(lhs.getNode()), op,
                          sym::ExprHandle(rhs.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "predicate", handle, diag)))
    return failure();
  return PredAttr::get(ctx, *handle);
}

// Hash-consed leaves: pointer equality on canonical handle drives downstream
// comparison.
static FailureOr<ExprAttr> composeIntExprAttr(int64_t value,
                                              Operation *diagOp) {
  MLIRContext *ctx = diagOp->getContext();
  std::string diag;
  FailureOr<sym::ExprHandle> handle =
      sym::composeExprInt(symbolStore(ctx), value, &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<ExprAttr> composeSymExprAttr(StringRef name,
                                              Operation *diagOp) {
  MLIRContext *ctx = diagOp->getContext();
  std::string diag;
  FailureOr<sym::ExprHandle> handle =
      sym::composeExprSym(symbolStore(ctx), name, &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<Type> inferIntegerAttrAsIdx(IntegerAttr value, Operation *op) {
  // IXS_INT is int64. Wider IntegerAttrs must not truncate.
  std::optional<int64_t> intValue = value.getValue().trySExtValue();
  if (!intValue) {
    SmallString<32> rendered;
    value.getValue().toStringSigned(rendered);
    op->emitOpError("integer constant ")
        << rendered << " does not fit in int64 for symbolic inference";
    return failure();
  }
  FailureOr<ExprAttr> expr = composeIntExprAttr(*intValue, op);
  if (failed(expr))
    return failure();
  return Type(IdxType::get(op->getContext(), *expr));
}

static LogicalResult pushInferred(SmallVectorImpl<Type> &resultTypes,
                                  FailureOr<Type> type) {
  if (failed(type))
    return failure();
  resultTypes.push_back(*type);
  return success();
}

static LogicalResult requireOperandCount(Operation *op, ArrayRef<Type> operands,
                                         unsigned expected) {
  if (operands.size() == expected)
    return success();
  return op->emitOpError("expected ")
         << expected << " operand type fact(s), got " << operands.size();
}

// Caller guarantees >=1 operand is IdxType.
static LogicalResult
inferIndexBinaryIdxArm(Type lhs, Type rhs, bool lhsIdx, bool rhsIdx,
                       sym::ExprBinaryOp opKind, Operation *op,
                       SmallVectorImpl<Type> &resultTypes) {
  if (lhsIdx != rhsIdx) {
    resultTypes.push_back({});
    return success();
  }
  std::optional<ExprAttr> lhsExpr = idxExprAttr(lhs);
  std::optional<ExprAttr> rhsExpr = idxExprAttr(rhs);
  if (!lhsExpr || !rhsExpr) {
    resultTypes.push_back(getUnpinnedIdxType(op->getContext()));
    return success();
  }
  FailureOr<ExprAttr> expr = composeExprAttr(*lhsExpr, opKind, *rhsExpr, op);
  if (failed(expr))
    return failure();
  resultTypes.push_back(IdxType::get(op->getContext(), *expr));
  return success();
}

// Same-rank NumPy broadcast. Frontend `unit_axes` aligns ranks beforehand.
static FailureOr<SmallVector<Attribute>>
broadcastShapeDims(ArrayRef<Attribute> lhs, ArrayRef<Attribute> rhs,
                   Operation *diagOp) {
  if (lhs.size() != rhs.size())
    return failure();
  FailureOr<ExprAttr> one = composeIntExprAttr(1, diagOp);
  if (failed(one))
    return failure();
  SmallVector<Attribute> out;
  out.reserve(lhs.size());
  for (auto [l, r] : llvm::zip_equal(lhs, rhs)) {
    auto lExpr = dyn_cast<ExprAttr>(l);
    auto rExpr = dyn_cast<ExprAttr>(r);
    if (!lExpr || !rExpr)
      return failure();
    if (lExpr == *one) {
      out.push_back(rExpr);
      continue;
    }
    if (rExpr == *one) {
      out.push_back(lExpr);
      continue;
    }
    if (lExpr.getValue() == rExpr.getValue()) {
      out.push_back(lExpr);
      continue;
    }
    return failure();
  }
  return out;
}

// Cross-flavor binaries diagnose; no silent cast.
static bool sameShapedFlavor(Type lhs, Type rhs) {
  if (isa<mlir::hc::TensorType>(lhs))
    return isa<mlir::hc::TensorType>(rhs);
  if (isa<mlir::hc::BareTensorType>(lhs))
    return isa<mlir::hc::BareTensorType>(rhs);
  if (isa<mlir::hc::VectorType>(lhs))
    return isa<mlir::hc::VectorType>(rhs);
  if (isa<mlir::hc::BareVectorType>(lhs))
    return isa<mlir::hc::BareVectorType>(rhs);
  return false;
}

// Layout dropped: broadcast has no result-side layout.
static Type rebuildShapedType(Type flavor, Type elementType, ShapeAttr shape) {
  MLIRContext *ctx = flavor.getContext();
  if (isa<mlir::hc::TensorType>(flavor))
    return mlir::hc::TensorType::get(ctx, elementType, shape, LayoutAttr{});
  if (isa<mlir::hc::BareTensorType>(flavor))
    return mlir::hc::BareTensorType::get(ctx, elementType, shape, LayoutAttr{});
  if (isa<mlir::hc::VectorType>(flavor))
    return mlir::hc::VectorType::get(ctx, elementType, shape, LayoutAttr{});
  return mlir::hc::BareVectorType::get(ctx, elementType, shape, LayoutAttr{});
}

// Same flavor + broadcast. Element via joinHCTypes (undef -> concrete).
static FailureOr<Type> inferShapedBinaryResult(Type lhs, Type rhs,
                                               Operation *op) {
  if (!sameShapedFlavor(lhs, rhs))
    return failure();

  auto lhsShaped = cast<SymbolicallyShapedTypeInterface>(lhs);
  auto rhsShaped = cast<SymbolicallyShapedTypeInterface>(rhs);
  ShapeAttr lhsShape = lhsShaped.getSymbolicShape();
  ShapeAttr rhsShape = rhsShaped.getSymbolicShape();
  if (!lhsShape || !rhsShape)
    return failure();
  FailureOr<SmallVector<Attribute>> dims =
      broadcastShapeDims(lhsShape.getDims(), rhsShape.getDims(), op);
  if (failed(dims))
    return failure();

  Type elementType = joinHCTypes(lhsShaped.getSymbolicElementType(),
                                 rhsShaped.getSymbolicElementType());
  if (!elementType)
    return failure();
  ShapeAttr shape = ShapeAttr::get(op->getContext(), *dims);
  return rebuildShapedType(lhs, elementType, shape);
}

static LogicalResult inferIndexBinary(ArrayRef<Type> operands,
                                      sym::ExprBinaryOp opKind, Operation *op,
                                      SmallVectorImpl<Type> &resultTypes) {
  if (failed(requireOperandCount(op, operands, 2)))
    return failure();
  Type lhs = operands[0];
  Type rhs = operands[1];
  if (!lhs || !rhs) {
    resultTypes.push_back({});
    return success();
  }

  bool lhsIdx = isa<IdxType>(lhs);
  bool rhsIdx = isa<IdxType>(rhs);
  if (lhsIdx || rhsIdx)
    return inferIndexBinaryIdxArm(lhs, rhs, lhsIdx, rhsIdx, opKind, op,
                                  resultTypes);

  if (lhs == rhs && (lhs.isIntOrIndexOrFloat() ||
                     isa<mlir::hc::TensorType, mlir::hc::VectorType>(lhs))) {
    resultTypes.push_back(lhs);
    return success();
  }
  FailureOr<Type> broadcasted = inferShapedBinaryResult(lhs, rhs, op);
  if (succeeded(broadcasted)) {
    resultTypes.push_back(*broadcasted);
    return success();
  }
  resultTypes.push_back({});
  return success();
}

static LogicalResult inferIndexCmpIdxArm(Type lhs, Type rhs, bool lhsIdx,
                                         bool rhsIdx, sym::PredCmpOp predKind,
                                         Operation *op,
                                         SmallVectorImpl<Type> &resultTypes) {
  if (lhsIdx != rhsIdx) {
    resultTypes.push_back({});
    return success();
  }
  std::optional<ExprAttr> lhsExpr = idxExprAttr(lhs);
  std::optional<ExprAttr> rhsExpr = idxExprAttr(rhs);
  if (!lhsExpr || !rhsExpr) {
    resultTypes.push_back(getUnpinnedPredType(op->getContext()));
    return success();
  }
  FailureOr<PredAttr> pred = composePredAttr(*lhsExpr, predKind, *rhsExpr, op);
  if (failed(pred))
    return failure();
  resultTypes.push_back(PredType::get(op->getContext(), *pred));
  return success();
}

static LogicalResult inferIndexCmp(ArrayRef<Type> operands,
                                   sym::PredCmpOp predKind, Operation *op,
                                   SmallVectorImpl<Type> &resultTypes) {
  if (failed(requireOperandCount(op, operands, 2)))
    return failure();
  Type lhs = operands[0];
  Type rhs = operands[1];
  if (!lhs || !rhs) {
    resultTypes.push_back({});
    return success();
  }

  bool lhsIdx = isa<IdxType>(lhs);
  bool rhsIdx = isa<IdxType>(rhs);
  if (lhsIdx || rhsIdx)
    return inferIndexCmpIdxArm(lhs, rhs, lhsIdx, rhsIdx, predKind, op,
                               resultTypes);

  if (lhs == rhs && lhs.isIntOrIndexOrFloat()) {
    resultTypes.push_back(IntegerType::get(op->getContext(), 1));
    return success();
  }
  resultTypes.push_back({});
  return success();
}

static ShapeAttr getSymbolicShape(Type type) {
  if (auto shaped = dyn_cast_or_null<SymbolicallyShapedTypeInterface>(type))
    return shaped.getSymbolicShape();
  return {};
}

static Type getSymbolicElementType(Type type) {
  if (auto shaped = dyn_cast_or_null<SymbolicallyShapedTypeInterface>(type))
    return shaped.getSymbolicElementType();
  return {};
}

static Type getShapedValueElementType(Type type) {
  if (!isa<mlir::hc::TensorType, mlir::hc::VectorType, mlir::hc::BareTensorType,
           mlir::hc::BareVectorType>(type))
    return {};
  return getSymbolicElementType(type);
}

static Type inferBufferDim(Type bufferType, int64_t axis, Operation *op) {
  ShapeAttr shape = getSymbolicShape(bufferType);
  if (!shape || axis < 0 ||
      axis >= static_cast<int64_t>(shape.getDims().size()))
    return getUnpinnedIdxType(op->getContext());
  auto dim = dyn_cast<ExprAttr>(shape.getDims()[axis]);
  if (!dim)
    return getUnpinnedIdxType(op->getContext());
  return IdxType::get(op->getContext(), dim);
}

static FailureOr<ExprAttr> defaultZeroExpr(Operation *op) {
  return composeIntExprAttr(0, op);
}

static FailureOr<ExprAttr> defaultOneExpr(Operation *op) {
  return composeIntExprAttr(1, op);
}

static FailureOr<ExprAttr> composeSubExprAttr(ExprAttr lhs, ExprAttr rhs,
                                              Operation *op) {
  return composeExprAttr(lhs, sym::ExprBinaryOp::Sub, rhs, op);
}

static FailureOr<ExprAttr> composeDivExprAttr(ExprAttr lhs, ExprAttr rhs,
                                              Operation *op) {
  return composeExprAttr(lhs, sym::ExprBinaryOp::Div, rhs, op);
}

// Null type = elided; non-null must pin to ExprAttr.
static FailureOr<std::optional<ExprAttr>> optionalIdxExprAttr(Type t) {
  if (!t)
    return std::optional<ExprAttr>();
  std::optional<ExprAttr> expr = idxExprAttr(t);
  if (!expr)
    return failure();
  return expr;
}

// stop - start. Elided start -> 0, elided stop -> baseDim.
static FailureOr<ExprAttr> composeSliceViewExtent(ExprAttr baseDim,
                                                  std::optional<ExprAttr> lower,
                                                  std::optional<ExprAttr> upper,
                                                  Operation *op) {
  FailureOr<ExprAttr> zero = defaultZeroExpr(op);
  if (failed(zero))
    return failure();
  ExprAttr start = lower.value_or(*zero);
  ExprAttr stop = upper.value_or(baseDim);
  return composeSubExprAttr(stop, start, op);
}

// ceil(extent / step); step==1 short-circuits.
static FailureOr<ExprAttr> scaleSliceViewExtent(ExprAttr extent, ExprAttr step,
                                                Operation *op) {
  FailureOr<ExprAttr> one = defaultOneExpr(op);
  if (failed(one))
    return failure();
  if (step == *one)
    return extent;
  FailureOr<ExprAttr> scaled = composeDivExprAttr(extent, step, op);
  if (failed(scaled))
    return failure();
  return composeCeilExprAttr(*scaled, op);
}

static FailureOr<ExprAttr> inferSliceViewDim(ExprAttr baseDim, SliceType slice,
                                             Operation *op) {
  Type lowerType = slice.getLowerType();
  Type upperType = slice.getUpperType();
  Type stepType = slice.getStepType();
  if (!lowerType && !upperType && !stepType)
    return baseDim;

  FailureOr<std::optional<ExprAttr>> lower = optionalIdxExprAttr(lowerType);
  FailureOr<std::optional<ExprAttr>> upper = optionalIdxExprAttr(upperType);
  FailureOr<std::optional<ExprAttr>> step = optionalIdxExprAttr(stepType);
  if (failed(lower) || failed(upper) || failed(step))
    return failure();

  FailureOr<ExprAttr> extent =
      composeSliceViewExtent(baseDim, *lower, *upper, op);
  if (failed(extent))
    return failure();
  if (!stepType)
    return *extent;
  return scaleSliceViewExtent(*extent, **step, op);
}

namespace {
// Parallel target/replacement arrays in raw form for ixs_subs_multi.
struct LayoutSubBuilder {
  MLIRContext *ctx;
  sym::Store &store;
  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;

  LayoutSubBuilder(MLIRContext *c, sym::Store &s) : ctx(c), store(s) {}

  LogicalResult pushSymPair(StringRef name, sym::ExprHandle replacement) {
    FailureOr<sym::ExprHandle> symHandle = sym::composeExprSym(store, name);
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(const_cast<ixs_node *>(replacement.raw()));
    return success();
  }

  FailureOr<ExprAttr> apply(ExprAttr in) const {
    if (!in)
      return ExprAttr{};
    if (targets.empty())
      return in;
    sym::Session session(store);
    ixs_node *out =
        ixs_subs_multi(session.raw(), const_cast<ixs_node *>(in.getNode()),
                       static_cast<uint32_t>(targets.size()),
                       const_cast<ixs_node **>(targets.data()),
                       const_cast<ixs_node **>(replacements.data()));
    if (!out)
      return failure();
    return ExprAttr::get(ctx, sym::ExprHandle(out));
  }
};
} // namespace

// All per-axis arrays must rank-align with baseDims.
static LogicalResult verifyBufferViewLayoutRanks(
    ArrayRef<Attribute> shapeSyms, ArrayRef<Attribute> indexSyms,
    ArrayRef<Attribute> baseDims, ArrayRef<bool> keepAxis,
    ArrayRef<ExprAttr> sliceLowerExpr, ArrayRef<ExprAttr> sliceStepExpr,
    ArrayRef<bool> sliceShapeChanged) {
  size_t rank = baseDims.size();
  if (shapeSyms.size() != rank || indexSyms.size() != rank ||
      keepAxis.size() != rank || sliceLowerExpr.size() != rank ||
      sliceStepExpr.size() != rank || sliceShapeChanged.size() != rank)
    return failure();
  return success();
}

// Non-trivial slice axes: index_sym |-> lower + step*index_sym.
// Caller has confirmed (lower, step) != (0, 1).
static LogicalResult appendSliceIndexRebind(LayoutSubBuilder &builder,
                                            StringRef indexName,
                                            ExprAttr lowerExpr,
                                            ExprAttr stepExpr) {
  FailureOr<sym::ExprHandle> indexSymHandle =
      sym::composeExprSym(builder.store, indexName);
  if (failed(indexSymHandle))
    return failure();
  FailureOr<sym::ExprHandle> stepMul =
      sym::composeExprBinary(builder.store, stepExpr.getValue(),
                             sym::ExprBinaryOp::Mul, *indexSymHandle);
  if (failed(stepMul))
    return failure();
  FailureOr<sym::ExprHandle> rebound = sym::composeExprBinary(
      builder.store, lowerExpr.getValue(), sym::ExprBinaryOp::Add, *stepMul);
  if (failed(rebound))
    return failure();
  builder.targets.push_back(const_cast<ixs_node *>(indexSymHandle->raw()));
  builder.replacements.push_back(const_cast<ixs_node *>(rebound->raw()));
  return success();
}

// Kept axis: shape-sym/index-sym rebinds per slice policy. Slot pre-pushed.
static LogicalResult composeKeptAxisSubstitutions(
    LayoutSubBuilder &builder, Attribute baseDim, Attribute shapeSymAttr,
    Attribute indexSymAttr, ExprAttr lowerExpr, ExprAttr stepExpr,
    bool sliceShapeChanged, ExprAttr zeroAttr, ExprAttr oneAttr) {
  // Pass-through axes (more axes than subscripts): no substitution.
  if (!lowerExpr || !stepExpr)
    return success();
  auto dimExpr = dyn_cast<ExprAttr>(baseDim);
  if (!dimExpr)
    return failure();
  if (sliceShapeChanged) {
    StringRef shapeName = cast<StringAttr>(shapeSymAttr).getValue();
    if (failed(builder.pushSymPair(shapeName, dimExpr.getValue())))
      return failure();
  }
  if (lowerExpr == zeroAttr && stepExpr == oneAttr)
    return success();
  StringRef indexName = cast<StringAttr>(indexSymAttr).getValue();
  return appendSliceIndexRebind(builder, indexName, lowerExpr, stepExpr);
}

// Scalar-indexed axis: shape_sym |-> dim, index_sym |-> scalar. Drops from
// residual.
static LogicalResult composeScalarAxisSubstitutions(LayoutSubBuilder &builder,
                                                    Attribute baseDim,
                                                    Attribute shapeSymAttr,
                                                    Attribute indexSymAttr,
                                                    ExprAttr scalarExpr) {
  auto dimExpr = dyn_cast<ExprAttr>(baseDim);
  if (!dimExpr || !scalarExpr)
    return failure();
  StringRef shapeName = cast<StringAttr>(shapeSymAttr).getValue();
  StringRef indexName = cast<StringAttr>(indexSymAttr).getValue();
  if (failed(builder.pushSymPair(shapeName, dimExpr.getValue())))
    return failure();
  return builder.pushSymPair(indexName, scalarExpr.getValue());
}

// Rewrite each ExprAttr in params via builder; non-expr pass through.
static FailureOr<DictionaryAttr>
rewriteLayoutParams(DictionaryAttr params, const LayoutSubBuilder &builder,
                    MLIRContext *ctx) {
  if (!params)
    return DictionaryAttr::get(ctx, {});
  SmallVector<NamedAttribute> newParams;
  newParams.reserve(params.size());
  for (NamedAttribute kv : params) {
    auto exprAttr = dyn_cast<ExprAttr>(kv.getValue());
    if (!exprAttr) {
      newParams.push_back(kv);
      continue;
    }
    FailureOr<ExprAttr> subsed = builder.apply(exprAttr);
    if (failed(subsed))
      return failure();
    newParams.emplace_back(kv.getName(), *subsed);
  }
  return DictionaryAttr::get(ctx, newParams);
}

// Per-axis: push sub into builder; record residual slot for kept axes.
static LogicalResult composeBufferViewAxes(
    LayoutSubBuilder &builder, ArrayRef<Attribute> baseDims,
    ArrayRef<Attribute> shapeSyms, ArrayRef<Attribute> indexSyms,
    ArrayRef<bool> keepAxis, ArrayRef<ExprAttr> scalarValueExpr,
    ArrayRef<ExprAttr> sliceLowerExpr, ArrayRef<ExprAttr> sliceStepExpr,
    ArrayRef<bool> sliceShapeChanged, ExprAttr zeroAttr, ExprAttr oneAttr,
    SmallVectorImpl<Attribute> &remainShapeSyms,
    SmallVectorImpl<Attribute> &remainIndexSyms) {
  for (unsigned k = 0; k < baseDims.size(); ++k) {
    if (keepAxis[k]) {
      remainShapeSyms.push_back(shapeSyms[k]);
      remainIndexSyms.push_back(indexSyms[k]);
      if (failed(composeKeptAxisSubstitutions(
              builder, baseDims[k], shapeSyms[k], indexSyms[k],
              sliceLowerExpr[k], sliceStepExpr[k], sliceShapeChanged[k],
              zeroAttr, oneAttr)))
        return failure();
      continue;
    }
    if (failed(composeScalarAxisSubstitutions(builder, baseDims[k],
                                              shapeSyms[k], indexSyms[k],
                                              scalarValueExpr[k])))
      return failure();
  }
  return success();
}

// Per-axis layout rewrite for `hc.buffer_view`:
//   scalar:           index_syms[k]->scalar, shape_syms[k]->opDim; drop slot.
//   trivial slice:    keep slot, no sub.
//   non-trivial slice: index_syms[k]->lower+step*idx; if extent != opDim also
//                     shape_syms[k]->opDim. Both slots remain (rank invariant).
//   pass-through:     keep slot, no sub.
// Sub applies to every ExprAttr in params.
static FailureOr<LayoutAttr> composeBufferViewLayout(
    LayoutAttr sourceLayout, ArrayRef<Attribute> baseDims,
    ArrayRef<bool> keepAxis, ArrayRef<ExprAttr> scalarValueExpr,
    ArrayRef<ExprAttr> sliceLowerExpr, ArrayRef<ExprAttr> sliceStepExpr,
    ArrayRef<bool> sliceShapeChanged, Operation *op) {
  MLIRContext *ctx = op->getContext();
  ArrayRef<Attribute> shapeSyms = sourceLayout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = sourceLayout.getIndexSyms();
  if (failed(verifyBufferViewLayoutRanks(shapeSyms, indexSyms, baseDims,
                                         keepAxis, sliceLowerExpr,
                                         sliceStepExpr, sliceShapeChanged)))
    return failure();

  sym::Store &store = symbolStore(ctx);
  LayoutSubBuilder builder(ctx, store);
  // Hash-consed 0/1 leaves: pointer-eq skips `m -> 0 + 1 * m` on trivial
  // slices.
  FailureOr<sym::ExprHandle> zeroLeaf = sym::composeExprInt(store, 0);
  FailureOr<sym::ExprHandle> oneLeaf = sym::composeExprInt(store, 1);
  if (failed(zeroLeaf) || failed(oneLeaf))
    return failure();
  ExprAttr zeroAttr = ExprAttr::get(ctx, *zeroLeaf);
  ExprAttr oneAttr = ExprAttr::get(ctx, *oneLeaf);

  SmallVector<Attribute> remainShapeSyms;
  SmallVector<Attribute> remainIndexSyms;
  remainShapeSyms.reserve(baseDims.size());
  remainIndexSyms.reserve(baseDims.size());
  if (failed(composeBufferViewAxes(builder, baseDims, shapeSyms, indexSyms,
                                   keepAxis, scalarValueExpr, sliceLowerExpr,
                                   sliceStepExpr, sliceShapeChanged, zeroAttr,
                                   oneAttr, remainShapeSyms, remainIndexSyms)))
    return failure();

  FailureOr<ExprAttr> newStorage = builder.apply(sourceLayout.getStorageSize());
  if (failed(newStorage))
    return failure();
  FailureOr<ExprAttr> newOffset = builder.apply(sourceLayout.getOffset());
  if (failed(newOffset))
    return failure();
  FailureOr<DictionaryAttr> newParams =
      rewriteLayoutParams(sourceLayout.getParams(), builder, ctx);
  if (failed(newParams))
    return failure();

  return LayoutAttr::get(ctx, remainShapeSyms, remainIndexSyms, *newParams,
                         *newStorage, *newOffset);
}

// Per-axis classification for `inferBufferViewResult`:
//   keep:              survives in residual rank.
//   resultDim:         only when keep=true.
//   scalarValueExpr:   scalar idx's IdxType expr; empty on kept axes.
//   sliceLower/Step:   slice lower/step (default 0/1); empty otherwise.
//   sliceShapeChanged: sliced extent != opDim.
struct BufferViewAxisEntry {
  Attribute resultDim;
  bool keep = false;
  ExprAttr scalarValueExpr;
  ExprAttr sliceLowerExpr;
  ExprAttr sliceStepExpr;
  bool sliceShapeChanged = false;
};

// Per-source-axis arrays for `composeBufferViewLayout`. Rank-aligned with
// source. Vector-root collective suffix never appends.
struct BufferViewAxisArrays {
  SmallVector<Attribute> resultDims;
  SmallVector<bool> keepAxis;
  SmallVector<ExprAttr> scalarValueExpr;
  SmallVector<ExprAttr> sliceLowerExpr;
  SmallVector<ExprAttr> sliceStepExpr;
  SmallVector<bool> sliceShapeChanged;

  void reserve(size_t n) {
    resultDims.reserve(n);
    keepAxis.reserve(n);
    scalarValueExpr.reserve(n);
    sliceLowerExpr.reserve(n);
    sliceStepExpr.reserve(n);
    sliceShapeChanged.reserve(n);
  }

  void append(const BufferViewAxisEntry &e) {
    if (e.keep)
      resultDims.push_back(e.resultDim);
    keepAxis.push_back(e.keep);
    scalarValueExpr.push_back(e.scalarValueExpr);
    sliceLowerExpr.push_back(e.sliceLowerExpr);
    sliceStepExpr.push_back(e.sliceStepExpr);
    sliceShapeChanged.push_back(e.sliceShapeChanged);
  }
};

// Slice subscript: Python 0/1 defaults; extent via inferSliceViewDim.
// nullopt = bail (malformed bound); failure = sym-store error.
static FailureOr<std::optional<BufferViewAxisEntry>>
classifyBufferViewSliceAxis(ExprAttr baseDim, SliceType slice, Operation *op) {
  FailureOr<ExprAttr> dim = inferSliceViewDim(baseDim, slice, op);
  if (failed(dim))
    return std::optional<BufferViewAxisEntry>();
  FailureOr<std::optional<ExprAttr>> lower =
      optionalIdxExprAttr(slice.getLowerType());
  FailureOr<std::optional<ExprAttr>> step =
      optionalIdxExprAttr(slice.getStepType());
  if (failed(lower) || failed(step))
    return std::optional<BufferViewAxisEntry>();
  FailureOr<ExprAttr> zero = defaultZeroExpr(op);
  FailureOr<ExprAttr> one = defaultOneExpr(op);
  if (failed(zero) || failed(one))
    return failure();

  BufferViewAxisEntry entry;
  entry.resultDim = *dim;
  entry.keep = true;
  entry.sliceLowerExpr = lower->value_or(*zero);
  entry.sliceStepExpr = step->value_or(*one);
  entry.sliceShapeChanged = *dim != baseDim;
  return std::optional<BufferViewAxisEntry>(entry);
}

// Scalar subscript: only IdxType pins. Plain index/i64 vs layout-bearing
// source can't substitute -> bail.
static std::optional<BufferViewAxisEntry>
classifyBufferViewScalarAxis(Type indexType, LayoutAttr sourceLayout) {
  ExprAttr scalarExpr;
  if (auto idxExpr = idxExprAttr(indexType))
    scalarExpr = *idxExpr;
  if (sourceLayout && !scalarExpr)
    return std::nullopt;
  BufferViewAxisEntry entry;
  entry.scalarValueExpr = scalarExpr;
  return entry;
}

// Implicit pass-through (more source axes than subscripts): keep, inherit dim.
static BufferViewAxisEntry buildPassThroughAxisEntry(Attribute baseDim) {
  BufferViewAxisEntry entry;
  entry.resultDim = baseDim;
  entry.keep = true;
  return entry;
}

// Subscripts past source rank: legal only on vector root (collective suffix);
// refine no local dims, only type-check.
static bool acceptsExtraSubscript(bool vectorRoot, Type indexType) {
  if (!vectorRoot || !indexType || isHCUndefType(indexType))
    return false;
  return isa<SliceType, IdxType>(indexType) || indexType.isIntOrIndex();
}

// Dispatcher by indexType; appends to arrays. false = bail.
static FailureOr<bool> classifyOneBufferViewAxis(BufferViewAxisArrays &arrays,
                                                 ExprAttr baseDim,
                                                 Type indexType,
                                                 LayoutAttr sourceLayout,
                                                 Operation *op) {
  if (auto slice = dyn_cast<SliceType>(indexType)) {
    FailureOr<std::optional<BufferViewAxisEntry>> entry =
        classifyBufferViewSliceAxis(baseDim, slice, op);
    if (failed(entry))
      return failure();
    if (!*entry)
      return false;
    arrays.append(**entry);
    return true;
  }
  if (isa<IdxType>(indexType) || indexType.isIntOrIndex()) {
    std::optional<BufferViewAxisEntry> entry =
        classifyBufferViewScalarAxis(indexType, sourceLayout);
    if (!entry)
      return false;
    arrays.append(*entry);
    return true;
  }
  return false;
}

// Walk subscripts vs source rank; build per-axis arrays. nullopt = bail;
// failure() = hard error.
static FailureOr<std::optional<BufferViewAxisArrays>>
classifyBufferViewIndices(ArrayRef<Type> indexTypes,
                          ArrayRef<Attribute> baseDims, bool vectorRoot,
                          LayoutAttr sourceLayout, Operation *op) {
  BufferViewAxisArrays arrays;
  arrays.reserve(baseDims.size());
  unsigned axis = 0;
  for (Type indexType : indexTypes) {
    if (axis >= baseDims.size()) {
      if (!acceptsExtraSubscript(vectorRoot, indexType))
        return std::optional<BufferViewAxisArrays>();
      continue;
    }
    auto baseDim = dyn_cast<ExprAttr>(baseDims[axis]);
    if (!baseDim || !indexType || isHCUndefType(indexType))
      return std::optional<BufferViewAxisArrays>();
    FailureOr<bool> ok =
        classifyOneBufferViewAxis(arrays, baseDim, indexType, sourceLayout, op);
    if (failed(ok))
      return failure();
    if (!*ok)
      return std::optional<BufferViewAxisArrays>();
    ++axis;
  }
  for (; axis < baseDims.size(); ++axis)
    arrays.append(buildPassThroughAxisEntry(baseDims[axis]));
  return std::optional<BufferViewAxisArrays>(std::move(arrays));
}

// Result type selector. Rank-0 vector-root -> element type.
static Type buildBufferViewResultType(Type sourceType, bool vectorRoot,
                                      Type elementType,
                                      ArrayRef<Attribute> resultDims,
                                      LayoutAttr resultLayout,
                                      MLIRContext *ctx) {
  ShapeAttr resultShape = ShapeAttr::get(ctx, resultDims);
  if (isa<mlir::hc::BufferType>(sourceType))
    return mlir::hc::BufferType::get(ctx, elementType, resultShape,
                                     resultLayout);
  if (vectorRoot) {
    if (resultDims.empty())
      return elementType;
    if (isa<mlir::hc::BareVectorType>(sourceType))
      return mlir::hc::BareVectorType::get(ctx, elementType, resultShape,
                                           resultLayout);
    return mlir::hc::VectorType::get(ctx, elementType, resultShape,
                                     resultLayout);
  }
  if (isa<mlir::hc::BareTensorType>(sourceType))
    return mlir::hc::BareTensorType::get(ctx, elementType, resultShape,
                                         resultLayout);
  return mlir::hc::TensorType::get(ctx, elementType, resultShape, resultLayout);
}

// Interleave size-1 dims at unitAxes into keptDims.
// Output rank = keptDims + unitAxes. Sorted locally for single-pass merge.
static FailureOr<SmallVector<Attribute>>
interleaveUnitAxes(ArrayRef<Attribute> keptDims, ArrayRef<int64_t> unitAxes,
                   Operation *op) {
  if (unitAxes.empty())
    return SmallVector<Attribute>(keptDims.begin(), keptDims.end());
  FailureOr<ExprAttr> one = defaultOneExpr(op);
  if (failed(one))
    return failure();
  SmallVector<int64_t> sortedUnits(unitAxes.begin(), unitAxes.end());
  llvm::sort(sortedUnits);
  size_t outputRank = keptDims.size() + sortedUnits.size();
  SmallVector<Attribute> result;
  result.reserve(outputRank);
  size_t unitIdx = 0;
  size_t keptIdx = 0;
  for (size_t pos = 0; pos < outputRank; ++pos) {
    if (unitIdx < sortedUnits.size() &&
        static_cast<size_t>(sortedUnits[unitIdx]) == pos) {
      result.push_back(*one);
      ++unitIdx;
      continue;
    }
    result.push_back(keptDims[keptIdx++]);
  }
  return result;
}

// keepAxis covers source rank. Composition fail -> un-refined; null layout
// in -> empty layout out.
static FailureOr<LayoutAttr> maybeComposeBufferViewLayout(
    LayoutAttr sourceLayout, ArrayRef<Attribute> baseDims,
    const BufferViewAxisArrays &arrays, Type currentResultType, Operation *op) {
  if (!sourceLayout)
    return LayoutAttr{};
  FailureOr<LayoutAttr> composed = composeBufferViewLayout(
      sourceLayout, baseDims, arrays.keepAxis, arrays.scalarValueExpr,
      arrays.sliceLowerExpr, arrays.sliceStepExpr, arrays.sliceShapeChanged,
      op);
  if (failed(composed))
    return LayoutAttr{};
  return *composed;
}

// Post-classification merge: compose the residual layout, splice the
// `unit_axes` size-1 dims into the kept shape, and build the final
// result type. Pulled out of `inferBufferViewResult` to keep that
// function's branch-count under the lizard threshold.
static FailureOr<Type>
finalizeBufferViewResult(Type sourceType, const BufferViewAxisArrays &arrays,
                         ArrayRef<Attribute> baseDims,
                         ArrayRef<int64_t> unitAxes, bool vectorRoot,
                         Type elementType, LayoutAttr sourceLayout,
                         Type currentResultType, Operation *op) {
  FailureOr<LayoutAttr> resultLayoutOr = maybeComposeBufferViewLayout(
      sourceLayout, baseDims, arrays, currentResultType, op);
  if (failed(resultLayoutOr))
    return currentResultType;
  LayoutAttr resultLayout = *resultLayoutOr;

  // unit_axes interleave size-1 dims result-side only. Layout-bearing source +
  // unit-axis insertion would need zero-stride dims; bail conservatively.
  if (!unitAxes.empty() && resultLayout)
    return currentResultType;
  FailureOr<SmallVector<Attribute>> finalDims =
      interleaveUnitAxes(arrays.resultDims, unitAxes, op);
  if (failed(finalDims))
    return failure();

  return buildBufferViewResultType(sourceType, vectorRoot, elementType,
                                   *finalDims, resultLayout, op->getContext());
}

static FailureOr<Type> inferBufferViewResult(Type sourceType,
                                             ArrayRef<Type> indexTypes,
                                             ArrayRef<int64_t> unitAxes,
                                             Type currentResultType,
                                             Operation *op) {
  if (!sourceType || isHCUndefType(sourceType))
    return currentResultType;

  Type elementType = getSymbolicElementType(sourceType);
  ShapeAttr shape = getSymbolicShape(sourceType);
  if (!elementType || !shape)
    return currentResultType;

  LayoutAttr sourceLayout;
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(sourceType))
    sourceLayout = shaped.getSymbolicLayout();

  ArrayRef<Attribute> baseDims = shape.getDims();
  bool vectorRoot =
      isa<mlir::hc::VectorType, mlir::hc::BareVectorType>(sourceType);
  FailureOr<std::optional<BufferViewAxisArrays>> classification =
      classifyBufferViewIndices(indexTypes, baseDims, vectorRoot, sourceLayout,
                                op);
  if (failed(classification))
    return failure();
  if (!*classification)
    return currentResultType;
  return finalizeBufferViewResult(sourceType, **classification, baseDims,
                                  unitAxes, vectorRoot, elementType,
                                  sourceLayout, currentResultType, op);
}

// Producer-side `layout` attr baked in by inference; lets non-injective
// layouts (broadcasts, WMMA fragments) skip the bare->layout barrier that
// `hc.as_layout` enforces.
static LayoutAttr producerLayoutAttr(Operation *op) {
  if (!op)
    return {};
  return op->getAttrOfType<LayoutAttr>("layout");
}

static Type inferLoadLikeResult(Type sourceType, Type shapeType,
                                bool vectorResult, Operation *op) {
  ShapeAttr shape = getStaticShapeFromTupleType(shapeType);
  if (!shape || !sourceType)
    return {};
  Type elementType = getSymbolicElementType(sourceType);
  if (!elementType)
    return {};
  LayoutAttr layout = producerLayoutAttr(op);
  return vectorResult
             ? Type(mlir::hc::VectorType::get(op->getContext(), elementType,
                                              shape, layout))
             : Type(mlir::hc::TensorType::get(op->getContext(), elementType,
                                              shape, layout));
}

static Type inferAllocLikeResult(Type resultType, Type shapeType,
                                 TypeAttr dtype, Type fillType,
                                 bool vectorResult, Operation *op) {
  ShapeAttr shape = getStaticShapeFromTupleType(shapeType);
  if (!shape)
    return resultType;
  Type elementType = getShapedValueElementType(resultType);
  if (!elementType && dtype)
    elementType = dtype.getValue();
  if (!elementType && fillType && !isHCUndefType(fillType))
    elementType = fillType;
  if (!elementType)
    return resultType;
  LayoutAttr layout = producerLayoutAttr(op);
  return vectorResult
             ? Type(mlir::hc::VectorType::get(op->getContext(), elementType,
                                              shape, layout))
             : Type(mlir::hc::TensorType::get(op->getContext(), elementType,
                                              shape, layout));
}

static Type inferTupleResult(ArrayRef<Type> elementTypes, Operation *op) {
  if (llvm::any_of(elementTypes, [](Type type) { return !type; }))
    return {};
  return TupleType::get(op->getContext(), elementTypes);
}

static std::optional<int64_t> staticIntegerIndex(Type type) {
  std::optional<ExprAttr> expr = idxExprAttr(type);
  if (!expr)
    return std::nullopt;
  // Tuple getitem needs literal indexing, not symbolic eval.
  return sym::getIntegerLiteralValue(sym::ExprHandle(expr->getNode()));
}

// Tuple[idx] -> slot. Python neg-index. Failure: not static or out of [-size,
// size).
static FailureOr<size_t> resolveTupleGetItemIndex(ArrayRef<Type> indexTypes,
                                                  TupleType tuple,
                                                  Operation *op) {
  if (indexTypes.size() != 1) {
    op->emitOpError(
        "tuple getitem expects exactly one index after inference, got ")
        << indexTypes.size();
    return failure();
  }
  if (!indexTypes.front()) {
    op->emitOpError(
        "tuple getitem index must be a static integer after inference");
    return failure();
  }
  std::optional<int64_t> index = staticIntegerIndex(indexTypes.front());
  if (!index) {
    op->emitOpError(
        "tuple getitem index must be a static integer after inference, got ")
        << indexTypes.front();
    return failure();
  }
  int64_t normalized = *index;
  int64_t tupleSize = static_cast<int64_t>(tuple.size());
  if (normalized < 0 && normalized >= -tupleSize)
    normalized += tupleSize;
  if (normalized < 0 || normalized >= tupleSize) {
    op->emitOpError("tuple index ")
        << *index << " out of bounds for tuple of size " << tupleSize;
    return failure();
  }
  return static_cast<size_t>(normalized);
}

static FailureOr<Type>
inferGetItemResult(Type sourceType, ArrayRef<Type> indexTypes, Operation *op) {
  if (!sourceType)
    return Type{};
  if (isa<mlir::hc::VectorType>(sourceType))
    return inferBufferViewResult(sourceType, indexTypes, /*unitAxes=*/{},
                                 Type{}, op);
  auto tuple = dyn_cast<TupleType>(sourceType);
  if (!tuple) {
    if (isa<UndefType, mlir::hc::BufferType, mlir::hc::TensorType,
            mlir::hc::VectorType>(sourceType))
      return Type{};
    op->emitOpError("getitem base type ")
        << sourceType
        << " cannot be refined; expected tuple, buffer, tensor, or vector";
    return failure();
  }
  FailureOr<size_t> slot = resolveTupleGetItemIndex(indexTypes, tuple, op);
  if (failed(slot))
    return failure();
  return tuple.getType(*slot);
}

static bool allResultsAreUndef(Operation *op) {
  return llvm::all_of(op->getResultTypes(),
                      [](Type type) { return isa<UndefType>(type); });
}

static LogicalResult appendPrefixedIdxTypes(Operation *op, StringRef prefix,
                                            unsigned count,
                                            SmallVectorImpl<Type> &types) {
  MLIRContext *ctx = op->getContext();
  for (unsigned axis = 0; axis < count; ++axis) {
    // composeExprSym builds leaf directly: "$WG0" is a single symbol; parseExpr
    // would mis-tokenize.
    SmallString<32> name(prefix);
    llvm::raw_svector_ostream(name) << axis;
    FailureOr<ExprAttr> expr = composeSymExprAttr(name, op);
    if (failed(expr))
      return failure();
    types.push_back(IdxType::get(ctx, *expr));
  }
  return success();
}

static LogicalResult appendLaunchAxisIdxTypes(Operation *op,
                                              LaunchGeoMethod method,
                                              unsigned count,
                                              SmallVectorImpl<Type> &types) {
  return appendPrefixedIdxTypes(op, getLaunchGeoMethodInfo(method).symbolPrefix,
                                count, types);
}

static void appendShapeIdxTypes(MLIRContext *ctx, ShapeAttr shape,
                                unsigned count, SmallVectorImpl<Type> &types) {
  for (unsigned axis = 0; axis < count; ++axis) {
    if (shape && axis < shape.getDims().size()) {
      if (auto expr = dyn_cast<ExprAttr>(shape.getDims()[axis])) {
        types.push_back(IdxType::get(ctx, expr));
        continue;
      }
    }
    types.push_back(getUnpinnedIdxType(ctx));
  }
}

static FailureOr<Type> groupSizeType(ShapeAttr shape, Operation *op) {
  MLIRContext *ctx = op->getContext();
  if (!shape)
    return Type(getUnpinnedIdxType(ctx));
  if (shape.getDims().empty()) {
    FailureOr<ExprAttr> one = composeIntExprAttr(1, op);
    if (failed(one))
      return failure();
    return Type(IdxType::get(ctx, *one));
  }

  auto first = dyn_cast<ExprAttr>(shape.getDims().front());
  if (!first)
    return Type(getUnpinnedIdxType(ctx));
  ExprAttr product = first;
  for (Attribute dim : shape.getDims().drop_front()) {
    auto expr = dyn_cast<ExprAttr>(dim);
    if (!expr)
      return Type(getUnpinnedIdxType(ctx));
    FailureOr<ExprAttr> next =
        composeExprAttr(product, sym::ExprBinaryOp::Mul, expr, op);
    if (failed(next))
      return failure();
    product = *next;
  }
  return Type(IdxType::get(ctx, product));
}

// Launch-axis id family. nullopt when raw is not one.
static std::optional<LogicalResult>
tryInferLaunchAxisIdTypes(Operation *raw, unsigned count,
                          SmallVectorImpl<Type> &types) {
  if (isa<HCGroupIdOp>(raw))
    return appendLaunchAxisIdxTypes(raw, LaunchGeoMethod::GroupId, count,
                                    types);
  if (isa<HCLocalIdOp>(raw))
    return appendLaunchAxisIdxTypes(raw, LaunchGeoMethod::LocalId, count,
                                    types);
  if (isa<HCSubgroupIdOp>(raw))
    return appendLaunchAxisIdxTypes(raw, LaunchGeoMethod::SubgroupId, count,
                                    types);
  if (isa<HCWorkOffsetOp>(raw))
    return appendLaunchAxisIdxTypes(raw, LaunchGeoMethod::WorkOffset, count,
                                    types);
  return std::nullopt;
}

// Launch shape projectors: stamp shape attr from metadata across results.
static std::optional<LogicalResult>
tryInferLaunchShapeTypes(Operation *raw, MLIRContext *ctx,
                         const LaunchContextMetadata &metadata, unsigned count,
                         SmallVectorImpl<Type> &types) {
  if (isa<HCGroupShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, metadata.groupShape, count, types);
    return success();
  }
  if (isa<HCWorkShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, metadata.workShape, count, types);
    return success();
  }
  return std::nullopt;
}

// Launch scalar: one IdxType from product(group_shape) or subgroup_size.
static std::optional<LogicalResult>
tryInferLaunchScalarTypes(Operation *raw, MLIRContext *ctx,
                          const LaunchContextMetadata &metadata,
                          SmallVectorImpl<Type> &types) {
  if (isa<HCGroupSizeOp>(raw)) {
    FailureOr<Type> type = groupSizeType(metadata.groupShape, raw);
    if (failed(type))
      return failure();
    types.push_back(*type);
    return success();
  }
  if (isa<HCWaveSizeOp>(raw)) {
    if (ExprAttr size = metadata.subgroupSize)
      types.push_back(IdxType::get(ctx, size));
    else
      types.push_back(getUnpinnedIdxType(ctx));
    return success();
  }
  return std::nullopt;
}

template <typename OpT>
static LogicalResult inferLaunchGeometryTypes(OpT op,
                                              ArrayRef<Type> operandTypes,
                                              SmallVectorImpl<Type> &types) {
  std::optional<LaunchContextMetadata> metadata =
      getLaunchContextMetadata(operandTypes.empty() ? Type{} : operandTypes[0]);
  if (!metadata || !allResultsAreUndef(op.getOperation())) {
    types.append(op.getResultTypes().begin(), op.getResultTypes().end());
    return success();
  }

  MLIRContext *ctx = op.getContext();
  Operation *raw = op.getOperation();
  unsigned count = op->getNumResults();
  if (auto r = tryInferLaunchAxisIdTypes(raw, count, types))
    return *r;
  if (auto r = tryInferLaunchShapeTypes(raw, ctx, *metadata, count, types))
    return *r;
  if (auto r = tryInferLaunchScalarTypes(raw, ctx, *metadata, types))
    return *r;

  types.append(op.getResultTypes().begin(), op.getResultTypes().end());
  return success();
}

} // namespace

LogicalResult HCConstOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                      SmallVectorImpl<Type> &resultTypes) {
  Attribute value = getValue();
  if (auto integer = dyn_cast<IntegerAttr>(value))
    return pushInferred(resultTypes, inferIntegerAttrAsIdx(integer, *this));
  if (auto typed = dyn_cast<TypedAttr>(value)) {
    Type type = typed.getType();
    if (type && !isa<NoneType>(type)) {
      resultTypes.push_back(type);
      return success();
    }
  }
  resultTypes.push_back(getResult().getType());
  return success();
}

LogicalResult HCSymbolOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                       SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(getResult().getType());
  return success();
}

LogicalResult HCIdxApplyOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                         SmallVectorImpl<Type> &resultTypes) {
  // Result pinned by carried `!hc.idx<expr>` + declared syms.
  resultTypes.push_back(getResult().getType());
  return success();
}

LogicalResult HCPredApplyOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                          SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(getResult().getType());
  return success();
}

LogicalResult HCTupleOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(inferTupleResult(operandTypes, *this));
  return success();
}

LogicalResult HCCastOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                     SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(operandTypes.empty() ? Type{} : operandTypes.front());
  return success();
}

LogicalResult HCAddOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, sym::ExprBinaryOp::Add, *this,
                          resultTypes);
}

LogicalResult HCSubOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, sym::ExprBinaryOp::Sub, *this,
                          resultTypes);
}

LogicalResult HCMulOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, sym::ExprBinaryOp::Mul, *this,
                          resultTypes);
}

LogicalResult HCDivOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, sym::ExprBinaryOp::Div, *this,
                          resultTypes);
}

LogicalResult HCModOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, sym::ExprBinaryOp::Mod, *this,
                          resultTypes);
}

// `hc.pow`: ixsimpl has no Pow -> idx arm bails to {}; unfold pass yields mul
// chain. Scalar/shaped: same-type passthrough.
LogicalResult HCPowOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  if (failed(requireOperandCount(*this, operandTypes, 2)))
    return failure();
  Type lhs = operandTypes[0];
  Type rhs = operandTypes[1];
  if (!lhs || !rhs) {
    resultTypes.push_back({});
    return success();
  }
  if (isa<IdxType>(lhs) || isa<IdxType>(rhs)) {
    resultTypes.push_back({});
    return success();
  }
  if (lhs == rhs && (lhs.isIntOrIndexOrFloat() ||
                     isa<mlir::hc::TensorType, mlir::hc::VectorType>(lhs))) {
    resultTypes.push_back(lhs);
    return success();
  }
  resultTypes.push_back({});
  return success();
}

LogicalResult HCNegOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  Type value = operandTypes.empty() ? Type{} : operandTypes.front();
  if (!value) {
    resultTypes.push_back({});
    return success();
  }
  if (std::optional<ExprAttr> expr = idxExprAttr(value)) {
    FailureOr<ExprAttr> neg = composeNegExprAttr(*expr, *this);
    if (failed(neg))
      return failure();
    resultTypes.push_back(IdxType::get(getContext(), *neg));
    return success();
  }
  resultTypes.push_back(value);
  return success();
}

// `hc.builtin_call` elementwise homogeneous: result type = first operand's
// type. `!hc.undef` and `!hc.idx` arms fall through as `{}`.
LogicalResult
HCBuiltinCallOp::inferHCTypes(ArrayRef<Type> operandTypes,
                              SmallVectorImpl<Type> &resultTypes) {
  Type first = operandTypes.empty() ? Type{} : operandTypes.front();
  if (!first || isa<IdxType>(first)) {
    resultTypes.push_back({});
    return success();
  }
  resultTypes.push_back(first);
  return success();
}

LogicalResult HCCmpLtOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, sym::PredCmpOp::Lt, *this, resultTypes);
}

LogicalResult HCCmpLeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, sym::PredCmpOp::Le, *this, resultTypes);
}

LogicalResult HCCmpGtOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, sym::PredCmpOp::Gt, *this, resultTypes);
}

LogicalResult HCCmpGeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, sym::PredCmpOp::Ge, *this, resultTypes);
}

LogicalResult HCCmpEqOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, sym::PredCmpOp::Eq, *this, resultTypes);
}

LogicalResult HCCmpNeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, sym::PredCmpOp::Ne, *this, resultTypes);
}

LogicalResult HCGroupIdOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                        SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCLocalIdOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                        SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCSubgroupIdOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                           SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCGroupShapeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                           SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCGroupSizeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                          SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCWorkOffsetOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                           SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCWorkShapeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                          SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCWaveSizeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                         SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, operandTypes, resultTypes);
}

LogicalResult HCBufferDimOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                          SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(inferBufferDim(
      operandTypes.empty() ? Type{} : operandTypes[0], getAxis(), *this));
  return success();
}

LogicalResult HCBufferViewOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                           SmallVectorImpl<Type> &resultTypes) {
  Type sourceType = operandTypes.empty() ? Type{} : operandTypes.front();
  ArrayRef<Type> indexTypes =
      operandTypes.empty() ? ArrayRef<Type>{} : operandTypes.drop_front();
  ArrayRef<int64_t> unitAxes;
  if (auto attr = getUnitAxesAttr())
    unitAxes = attr.asArrayRef();
  return pushInferred(resultTypes,
                      inferBufferViewResult(sourceType, indexTypes, unitAxes,
                                            getResult().getType(), *this));
}

LogicalResult HCSliceExprOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                          SmallVectorImpl<Type> &resultTypes) {
  unsigned nextOperand = 0;
  auto partType = [&](Value part) -> Type {
    if (!part)
      return {};
    if (nextOperand >= operandTypes.size())
      return part.getType();
    return operandTypes[nextOperand++];
  };
  Type lowerType = partType(getLower());
  Type upperType = partType(getUpper());
  Type stepType = partType(getStep());
  resultTypes.push_back(
      SliceType::get(getContext(), lowerType, upperType, stepType));
  return success();
}

LogicalResult HCLoadOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                     SmallVectorImpl<Type> &resultTypes) {
  Type sourceType = operandTypes.empty() ? Type{} : operandTypes.front();
  // Shape is trailing operand after variadic indices.
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.back();
  resultTypes.push_back(inferLoadLikeResult(sourceType, shapeType,
                                            /*vectorResult=*/false, *this));
  return success();
}

LogicalResult HCVLoadOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  Type sourceType = operandTypes.empty() ? Type{} : operandTypes.front();
  // Shape is trailing operand after variadic indices.
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.back();
  resultTypes.push_back(
      inferLoadLikeResult(sourceType, shapeType, /*vectorResult=*/true, *this));
  return success();
}

LogicalResult HCGetItemOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                        SmallVectorImpl<Type> &resultTypes) {
  Type source = operandTypes.empty() ? Type{} : operandTypes.front();
  ArrayRef<Type> indices =
      operandTypes.empty() ? ArrayRef<Type>{} : operandTypes.drop_front();
  FailureOr<Type> result = inferGetItemResult(source, indices, *this);
  if (failed(result))
    return failure();
  resultTypes.push_back(*result);
  return success();
}

LogicalResult HCVecOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  Type value = operandTypes.empty() ? Type{} : operandTypes.front();
  LayoutAttr layout = getLayoutAttr();
  if (auto tensor = dyn_cast_or_null<mlir::hc::TensorType>(value)) {
    resultTypes.push_back(mlir::hc::VectorType::get(
        getContext(), tensor.getElementType(), tensor.getShape(), layout));
    return success();
  }
  if (auto bareTensor = dyn_cast_or_null<mlir::hc::BareTensorType>(value)) {
    // Bare carriers have no layout slot. Layout-bearing path already ran on
    // semantic type before decomposition.
    resultTypes.push_back(mlir::hc::BareVectorType::get(
        getContext(), bareTensor.getElementType(), bareTensor.getShape()));
    return success();
  }
  resultTypes.push_back(value);
  return success();
}

LogicalResult
HCWithInactiveOp::inferHCTypes(ArrayRef<Type> operandTypes,
                               SmallVectorImpl<Type> &resultTypes) {
  if (failed(requireOperandCount(*this, operandTypes, 2)))
    return failure();
  resultTypes.push_back(operandTypes.front());
  return success();
}

LogicalResult
HCCallIntrinsicOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                SmallVectorImpl<Type> &resultTypes) {
  (void)operandTypes;
  auto callee = SymbolTable::lookupNearestSymbolFrom<HCIntrinsicOp>(
      getOperation(), getCalleeAttr());
  if (!callee || !callee.getFunctionTypeAttr()) {
    resultTypes.append(getNumResults(), Type{});
    return success();
  }

  auto fnType = cast<FunctionType>(callee.getFunctionTypeAttr().getValue());
  if (fnType.getNumResults() != getNumResults())
    return emitOpError("callee function_type declares ")
           << fnType.getNumResults() << " result(s), but call has "
           << getNumResults();

  for (Type type : fnType.getResults())
    resultTypes.push_back(isHCUndefType(type) ? Type{} : type);
  return success();
}

// Apply op's layout to shaped operand's flavor, keep element+shape.
// Null on unknown flavors (buffer is caller's job).
static Type applyLayoutToCarrier(Type valueType, Type elementType,
                                 ShapeAttr shape, LayoutAttr layout,
                                 MLIRContext *ctx) {
  if (isa<mlir::hc::VectorType>(valueType))
    return mlir::hc::VectorType::get(ctx, elementType, shape, layout);
  if (isa<mlir::hc::BareVectorType>(valueType))
    return mlir::hc::BareVectorType::get(ctx, elementType, shape, layout);
  if (isa<mlir::hc::TensorType>(valueType))
    return mlir::hc::TensorType::get(ctx, elementType, shape, layout);
  if (isa<mlir::hc::BareTensorType>(valueType))
    return mlir::hc::BareTensorType::get(ctx, elementType, shape, layout);
  return {};
}

// Buffer-rooted as_layout: fresh !hc.buffer with op's layout.
// `shape=` operand wins over operand dims when present.
static Type inferAsLayoutBufferResult(BufferType buffer, Type shapeType,
                                      LayoutAttr layout, MLIRContext *ctx) {
  ShapeAttr resultShape = buffer.getShape();
  if (shapeType && !isHCUndefType(shapeType))
    if (ShapeAttr declared = getStaticShapeFromTupleType(shapeType))
      resultShape = declared;
  return mlir::hc::BufferType::get(ctx, buffer.getElementType(), resultShape,
                                   layout);
}

// Value operand: bake op's layout onto operand's element/shape. Fall back to
// operand type when shape is unrecoverable; verifier still pins storage_size.
static Type inferAsLayoutShapedResult(Type valueType, LayoutAttr layout,
                                      MLIRContext *ctx) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(valueType);
  if (!shaped)
    return valueType;
  Type elementType = shaped.getSymbolicElementType();
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!elementType || !shape)
    return valueType;
  Type relabelled =
      applyLayoutToCarrier(valueType, elementType, shape, layout, ctx);
  return relabelled ? relabelled : valueType;
}

LogicalResult HCAsLayoutOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                         SmallVectorImpl<Type> &resultTypes) {
  // operandTypes = [value, (shape?)]; shape only on buffer-rooted.
  Type valueType = operandTypes.empty() ? Type{} : operandTypes.front();
  Type shapeType =
      (getShape() && operandTypes.size() > 1) ? operandTypes[1] : Type{};

  // Pre-inference operand: preserve frontend-stamped refined result.
  if (!valueType || isHCUndefType(valueType)) {
    resultTypes.push_back(getResult().getType());
    return success();
  }

  if (auto buffer = dyn_cast<mlir::hc::BufferType>(valueType)) {
    resultTypes.push_back(inferAsLayoutBufferResult(buffer, shapeType,
                                                    getLayout(), getContext()));
    return success();
  }

  resultTypes.push_back(
      inferAsLayoutShapedResult(valueType, getLayout(), getContext()));
  return success();
}

// Strip preserves flavor; drops layout, keeps semantic-vs-bare.
// Null on unknown flavors (buffer): caller leaves un-refined.
static Type stripLayoutForCarrier(Type valueType, Type elementType,
                                  ShapeAttr shape, MLIRContext *ctx) {
  if (isa<mlir::hc::VectorType>(valueType))
    return mlir::hc::VectorType::get(ctx, elementType, shape, LayoutAttr{});
  if (isa<mlir::hc::BareVectorType>(valueType))
    return mlir::hc::BareVectorType::get(ctx, elementType, shape, LayoutAttr{});
  if (isa<mlir::hc::TensorType>(valueType))
    return mlir::hc::TensorType::get(ctx, elementType, shape, LayoutAttr{});
  if (isa<mlir::hc::BareTensorType>(valueType))
    return mlir::hc::BareTensorType::get(ctx, elementType, shape, LayoutAttr{});
  return {};
}

LogicalResult
HCStripLayoutOp::inferHCTypes(ArrayRef<Type> operandTypes,
                              SmallVectorImpl<Type> &resultTypes) {
  // Flavor must survive: mix would clash downstream (vector intrinsic vs loop
  // iter init). Buffer has no bare counterpart; verifier rejects, leave
  // un-refined.
  Type valueType = operandTypes.empty() ? Type{} : operandTypes.front();
  if (!valueType || isHCUndefType(valueType)) {
    resultTypes.push_back(getResult().getType());
    return success();
  }
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(valueType);
  if (!shaped) {
    resultTypes.push_back(getResult().getType());
    return success();
  }
  Type elementType = shaped.getSymbolicElementType();
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!elementType || !shape) {
    resultTypes.push_back(getResult().getType());
    return success();
  }
  Type stripped =
      stripLayoutForCarrier(valueType, elementType, shape, getContext());
  resultTypes.push_back(stripped ? stripped : getResult().getType());
  return success();
}

LogicalResult HCVZerosOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                       SmallVectorImpl<Type> &resultTypes) {
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.front();
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), Type{},
                                             /*vectorResult=*/true, *this));
  return success();
}

LogicalResult HCVOnesOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.front();
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), Type{},
                                             /*vectorResult=*/true, *this));
  return success();
}

LogicalResult HCVFullOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  Type fillType = operandTypes.empty() ? Type{} : operandTypes.front();
  Type shapeType = operandTypes.size() < 2 ? Type{} : operandTypes[1];
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), fillType,
                                             /*vectorResult=*/true, *this));
  return success();
}

LogicalResult HCZerosOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.front();
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), Type{},
                                             /*vectorResult=*/false, *this));
  return success();
}

LogicalResult HCOnesOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                     SmallVectorImpl<Type> &resultTypes) {
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.front();
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), Type{},
                                             /*vectorResult=*/false, *this));
  return success();
}

LogicalResult HCFullOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                     SmallVectorImpl<Type> &resultTypes) {
  Type fillType = operandTypes.empty() ? Type{} : operandTypes.front();
  Type shapeType = operandTypes.size() < 2 ? Type{} : operandTypes[1];
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), fillType,
                                             /*vectorResult=*/false, *this));
  return success();
}

LogicalResult HCEmptyOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.front();
  resultTypes.push_back(inferAllocLikeResult(getResult().getType(), shapeType,
                                             getDtypeAttr(), Type{},
                                             /*vectorResult=*/false, *this));
  return success();
}

// keepdims=false drops axis dim; keepdims=true replaces with 1.
static FailureOr<SmallVector<Attribute>>
collapseReduceShape(ArrayRef<Attribute> dims, uint64_t axis, bool keepdims,
                    Operation *diagOp) {
  SmallVector<Attribute> outDims;
  outDims.reserve(keepdims ? dims.size() : dims.size() - 1);
  if (!keepdims) {
    for (auto [idx, dim] : llvm::enumerate(dims))
      if (idx != axis)
        outDims.push_back(dim);
    return outDims;
  }
  FailureOr<ExprAttr> one = composeIntExprAttr(1, diagOp);
  if (failed(one))
    return failure();
  for (auto [idx, dim] : llvm::enumerate(dims))
    outDims.push_back(idx == axis ? Attribute(*one) : dim);
  return outDims;
}

// Mirror operand flavor on result. Layout dropped: reduced axis is in operand's
// index space; carrying its name list past collapse would dangle.
static Type rebuildReduceResultType(Type valueType, Type elem,
                                    ShapeAttr outShape) {
  MLIRContext *ctx = elem.getContext();
  if (isa<mlir::hc::TensorType>(valueType))
    return mlir::hc::TensorType::get(ctx, elem, outShape, LayoutAttr{});
  if (isa<mlir::hc::VectorType>(valueType))
    return mlir::hc::VectorType::get(ctx, elem, outShape, LayoutAttr{});
  if (isa<mlir::hc::BareTensorType>(valueType))
    return mlir::hc::BareTensorType::get(ctx, elem, outShape, LayoutAttr{});
  if (isa<mlir::hc::BareVectorType>(valueType))
    return mlir::hc::BareVectorType::get(ctx, elem, outShape, LayoutAttr{});
  return {};
}

// `hc.reduce` collapses $value's $axis dim. Refines result so
// `hc-shaped-compute-to-generic` validates before flatten.
LogicalResult HCReduceOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                       SmallVectorImpl<Type> &resultTypes) {
  Type valueType = operandTypes.empty() ? Type{} : operandTypes.front();
  auto shaped = dyn_cast_or_null<SymbolicallyShapedTypeInterface>(valueType);
  ShapeAttr shape = shaped ? shaped.getSymbolicShape() : ShapeAttr{};
  // Out-of-range axis is verifier's job; bail with typed diagnostic.
  if (!shape || getAxis() >= shape.getDims().size()) {
    resultTypes.push_back({});
    return success();
  }
  FailureOr<SmallVector<Attribute>> outDims =
      collapseReduceShape(shape.getDims(), getAxis(), getKeepdims(), *this);
  if (failed(outDims))
    return failure();
  ShapeAttr outShape = ShapeAttr::get(getContext(), *outDims);
  resultTypes.push_back(rebuildReduceResultType(
      valueType, shaped.getSymbolicElementType(), outShape));
  return success();
}

LogicalResult
HCForRangeOp::inferHCRegionArgTypes(RegionSuccessor successor,
                                    ValueRange nonSuccessorInputs,
                                    SmallVectorImpl<Type> &regionArgTypes) {
  if (successor.isParent())
    return success();
  for (Value input : nonSuccessorInputs) {
    Type type = input.getType();
    regionArgTypes.push_back(
        isa<UndefType>(type) ? Type(getUnpinnedIdxType(getContext())) : type);
  }
  return success();
}
