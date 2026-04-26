// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCSymbols.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallString.h"

#include <optional>

using namespace mlir;
using namespace mlir::hc;

namespace {

static sym::Store &symbolStore(MLIRContext *ctx) {
  return ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
}

template <typename AttrT, typename HandleT,
          FailureOr<HandleT> (*Parse)(sym::Store &, StringRef, std::string *)>
static FailureOr<AttrT> parseSymbolicAttr(MLIRContext *ctx, Twine text,
                                          Operation *diagOp, StringRef kind) {
  SmallString<64> storage;
  StringRef rendered = text.toStringRef(storage);
  std::string diag;
  FailureOr<HandleT> handle = Parse(symbolStore(ctx), rendered, &diag);
  if (failed(handle)) {
    diagOp->emitOpError("failed to infer symbolic ")
        << kind << " '" << rendered << "': " << diag;
    return failure();
  }
  return AttrT::get(ctx, *handle);
}

static FailureOr<ExprAttr> parseExprAttr(MLIRContext *ctx, Twine text,
                                         Operation *diagOp) {
  return parseSymbolicAttr<ExprAttr, sym::ExprHandle, sym::parseExpr>(
      ctx, text, diagOp, "expression");
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

static FailureOr<ExprAttr> composeExprAttr(MLIRContext *ctx, ExprAttr lhs,
                                           sym::ExprBinaryOp op, ExprAttr rhs,
                                           Operation *diagOp) {
  std::string diag;
  FailureOr<sym::ExprHandle> handle =
      sym::composeExprBinary(symbolStore(ctx), sym::ExprHandle(lhs.getNode()),
                             op, sym::ExprHandle(rhs.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<ExprAttr> composeNegExprAttr(MLIRContext *ctx, ExprAttr value,
                                              Operation *diagOp) {
  std::string diag;
  FailureOr<sym::ExprHandle> handle = sym::composeExprNeg(
      symbolStore(ctx), sym::ExprHandle(value.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<ExprAttr> composeCeilExprAttr(MLIRContext *ctx, ExprAttr value,
                                               Operation *diagOp) {
  std::string diag;
  FailureOr<sym::ExprHandle> handle = sym::composeExprCeil(
      symbolStore(ctx), sym::ExprHandle(value.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "expression", handle, diag)))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<PredAttr> composePredAttr(MLIRContext *ctx, ExprAttr lhs,
                                           sym::PredCmpOp op, ExprAttr rhs,
                                           Operation *diagOp) {
  std::string diag;
  FailureOr<sym::PredHandle> handle =
      sym::composePredCmp(symbolStore(ctx), sym::ExprHandle(lhs.getNode()), op,
                          sym::ExprHandle(rhs.getNode()), &diag);
  if (failed(emitComposeError(diagOp, "predicate", handle, diag)))
    return failure();
  return PredAttr::get(ctx, *handle);
}

static FailureOr<Type> inferIntegerAttrAsIdx(IntegerAttr value, Operation *op) {
  SmallString<32> text;
  value.getValue().toStringSigned(text);
  FailureOr<ExprAttr> expr = parseExprAttr(op->getContext(), text, op);
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
  if (lhsIdx || rhsIdx) {
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
    FailureOr<ExprAttr> expr =
        composeExprAttr(op->getContext(), *lhsExpr, opKind, *rhsExpr, op);
    if (failed(expr))
      return failure();
    resultTypes.push_back(IdxType::get(op->getContext(), *expr));
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
  if (lhsIdx || rhsIdx) {
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
    FailureOr<PredAttr> pred =
        composePredAttr(op->getContext(), *lhsExpr, predKind, *rhsExpr, op);
    if (failed(pred))
      return failure();
    resultTypes.push_back(PredType::get(op->getContext(), *pred));
    return success();
  }

  if (lhs == rhs && lhs.isIntOrIndexOrFloat()) {
    resultTypes.push_back(IntegerType::get(op->getContext(), 1));
    return success();
  }
  resultTypes.push_back({});
  return success();
}

static ShapeAttr getShapeAttrFromBuffer(Type type) {
  if (auto buffer = dyn_cast_or_null<mlir::hc::BufferType>(type))
    return buffer.getShape();
  return {};
}

static ShapeAttr getShapeAttrFromBufferOrTensor(Type type) {
  if (auto buffer = dyn_cast_or_null<mlir::hc::BufferType>(type))
    return buffer.getShape();
  if (auto tensor = dyn_cast_or_null<mlir::hc::TensorType>(type))
    return tensor.getShape();
  return {};
}

static Type getElementTypeFromBufferOrTensor(Type type) {
  if (auto buffer = dyn_cast_or_null<mlir::hc::BufferType>(type))
    return buffer.getElementType();
  if (auto tensor = dyn_cast_or_null<mlir::hc::TensorType>(type))
    return tensor.getElementType();
  return {};
}

static Type inferBufferDim(Type bufferType, int64_t axis, Operation *op) {
  ShapeAttr shape = getShapeAttrFromBuffer(bufferType);
  if (!shape || axis < 0 ||
      axis >= static_cast<int64_t>(shape.getDims().size()))
    return getUnpinnedIdxType(op->getContext());
  auto dim = dyn_cast<ExprAttr>(shape.getDims()[axis]);
  if (!dim)
    return getUnpinnedIdxType(op->getContext());
  return IdxType::get(op->getContext(), dim);
}

static ShapeAttr shapeFromTupleType(Type shapeType, Operation *op) {
  if (!shapeType || isHCUndefType(shapeType))
    return {};
  auto tuple = dyn_cast<TupleType>(shapeType);
  if (!tuple)
    return {};
  SmallVector<Attribute> dims;
  dims.reserve(tuple.size());
  for (Type dimType : tuple.getTypes()) {
    auto idx = dyn_cast<IdxType>(dimType);
    if (!idx || !idx.getExpr())
      return {};
    dims.push_back(idx.getExpr());
  }
  return ShapeAttr::get(op->getContext(), dims);
}

static FailureOr<ExprAttr> defaultZeroExpr(MLIRContext *ctx, Operation *op) {
  return parseExprAttr(ctx, "0", op);
}

static FailureOr<ExprAttr> defaultOneExpr(MLIRContext *ctx, Operation *op) {
  return parseExprAttr(ctx, "1", op);
}

static FailureOr<ExprAttr> composeSubExprAttr(MLIRContext *ctx, ExprAttr lhs,
                                              ExprAttr rhs, Operation *op) {
  return composeExprAttr(ctx, lhs, sym::ExprBinaryOp::Sub, rhs, op);
}

static FailureOr<ExprAttr> composeDivExprAttr(MLIRContext *ctx, ExprAttr lhs,
                                              ExprAttr rhs, Operation *op) {
  return composeExprAttr(ctx, lhs, sym::ExprBinaryOp::Div, rhs, op);
}

static FailureOr<ExprAttr> inferSliceViewDim(ExprAttr baseDim, SliceType slice,
                                             Operation *op) {
  MLIRContext *ctx = op->getContext();
  Type lowerType = slice.getLowerType();
  Type upperType = slice.getUpperType();
  Type stepType = slice.getStepType();

  if (!lowerType && !upperType && !stepType)
    return baseDim;

  std::optional<ExprAttr> lower =
      lowerType ? idxExprAttr(lowerType) : std::optional<ExprAttr>();
  std::optional<ExprAttr> upper =
      upperType ? idxExprAttr(upperType) : std::optional<ExprAttr>();
  std::optional<ExprAttr> step =
      stepType ? idxExprAttr(stepType) : std::optional<ExprAttr>();
  if ((lowerType && !lower) || (upperType && !upper) || (stepType && !step))
    return failure();

  FailureOr<ExprAttr> zero = defaultZeroExpr(ctx, op);
  if (failed(zero))
    return failure();
  ExprAttr start = lower.value_or(*zero);
  ExprAttr stop = upper.value_or(baseDim);
  FailureOr<ExprAttr> extent = composeSubExprAttr(ctx, stop, start, op);
  if (failed(extent))
    return failure();

  if (!stepType)
    return *extent;

  FailureOr<ExprAttr> one = defaultOneExpr(ctx, op);
  if (failed(one))
    return failure();
  if (*step == *one)
    return *extent;

  FailureOr<ExprAttr> scaled = composeDivExprAttr(ctx, *extent, *step, op);
  if (failed(scaled))
    return failure();
  return composeCeilExprAttr(ctx, *scaled, op);
}

static FailureOr<Type> inferBufferViewResult(Type sourceType,
                                             ArrayRef<Type> indexTypes,
                                             Type currentResultType,
                                             Operation *op) {
  if (!sourceType || isHCUndefType(sourceType))
    return currentResultType;

  Type elementType = getElementTypeFromBufferOrTensor(sourceType);
  ShapeAttr shape = getShapeAttrFromBufferOrTensor(sourceType);
  if (!elementType || !shape)
    return currentResultType;

  ArrayRef<Attribute> baseDims = shape.getDims();
  SmallVector<Attribute> resultDims;
  unsigned axis = 0;
  for (Type indexType : indexTypes) {
    if (axis >= baseDims.size())
      return currentResultType;
    auto baseDim = dyn_cast<ExprAttr>(baseDims[axis]);
    if (!baseDim)
      return currentResultType;

    if (!indexType || isHCUndefType(indexType))
      return currentResultType;
    if (auto slice = dyn_cast<SliceType>(indexType)) {
      FailureOr<ExprAttr> dim = inferSliceViewDim(baseDim, slice, op);
      if (failed(dim))
        return currentResultType;
      resultDims.push_back(*dim);
      ++axis;
      continue;
    }
    if (isa<IdxType>(indexType) || indexType.isIntOrIndex()) {
      ++axis;
      continue;
    }
    return currentResultType;
  }

  for (; axis < baseDims.size(); ++axis)
    resultDims.push_back(baseDims[axis]);

  ShapeAttr resultShape = ShapeAttr::get(op->getContext(), resultDims);
  if (isa<mlir::hc::BufferType>(sourceType))
    return Type(
        mlir::hc::BufferType::get(op->getContext(), elementType, resultShape));
  return Type(
      mlir::hc::TensorType::get(op->getContext(), elementType, resultShape));
}

static Type inferLoadLikeResult(Type sourceType, Type shapeType,
                                bool vectorResult, Operation *op) {
  ShapeAttr shape = shapeFromTupleType(shapeType, op);
  if (!shape || !sourceType)
    return {};
  Type elementType;
  if (auto buffer = dyn_cast<mlir::hc::BufferType>(sourceType))
    elementType = buffer.getElementType();
  else if (auto tensor = dyn_cast<mlir::hc::TensorType>(sourceType))
    elementType = tensor.getElementType();
  if (!elementType)
    return {};
  return vectorResult ? Type(mlir::hc::VectorType::get(op->getContext(),
                                                       elementType, shape))
                      : Type(mlir::hc::TensorType::get(op->getContext(),
                                                       elementType, shape));
}

static Type shapedElementType(Type type) {
  if (auto tensor = dyn_cast_or_null<mlir::hc::TensorType>(type))
    return tensor.getElementType();
  if (auto vector = dyn_cast_or_null<mlir::hc::VectorType>(type))
    return vector.getElementType();
  return {};
}

static Type inferAllocLikeResult(Type resultType, Type shapeType,
                                 TypeAttr dtype, Type fillType,
                                 bool vectorResult, Operation *op) {
  ShapeAttr shape = shapeFromTupleType(shapeType, op);
  if (!shape)
    return resultType;
  Type elementType = shapedElementType(resultType);
  if (!elementType && dtype)
    elementType = dtype.getValue();
  if (!elementType && fillType && !isHCUndefType(fillType))
    elementType = fillType;
  if (!elementType)
    return resultType;
  return vectorResult ? Type(mlir::hc::VectorType::get(op->getContext(),
                                                       elementType, shape))
                      : Type(mlir::hc::TensorType::get(op->getContext(),
                                                       elementType, shape));
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
  // Tuple getitem needs literal indexing, not general symbolic evaluation.
  return sym::getIntegerLiteralValue(sym::ExprHandle(expr->getNode()));
}

static FailureOr<Type>
inferGetItemResult(Type sourceType, ArrayRef<Type> indexTypes, Operation *op) {
  if (!sourceType)
    return Type{};
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
  int64_t originalIndex = *index;
  int64_t normalizedIndex = originalIndex;
  int64_t tupleSize = static_cast<int64_t>(tuple.size());
  if (normalizedIndex < 0 && normalizedIndex >= -tupleSize)
    normalizedIndex += tupleSize;
  if (normalizedIndex < 0 || normalizedIndex >= tupleSize) {
    op->emitOpError("tuple index ")
        << originalIndex << " out of bounds for tuple of size " << tupleSize;
    return failure();
  }
  return tuple.getType(static_cast<size_t>(normalizedIndex));
}

static bool allResultsAreUndef(Operation *op) {
  return llvm::all_of(op->getResultTypes(),
                      [](Type type) { return isa<UndefType>(type); });
}

static LogicalResult appendPrefixedIdxTypes(MLIRContext *ctx, Operation *op,
                                            StringRef prefix, unsigned count,
                                            SmallVectorImpl<Type> &types) {
  for (unsigned axis = 0; axis < count; ++axis) {
    FailureOr<ExprAttr> expr =
        parseExprAttr(ctx, Twine(prefix) + Twine(axis), op);
    if (failed(expr))
      return failure();
    types.push_back(IdxType::get(ctx, *expr));
  }
  return success();
}

static LogicalResult appendLaunchAxisIdxTypes(MLIRContext *ctx, Operation *op,
                                              LaunchGeoMethod method,
                                              unsigned count,
                                              SmallVectorImpl<Type> &types) {
  return appendPrefixedIdxTypes(
      ctx, op, getLaunchGeoMethodInfo(method).symbolPrefix, count, types);
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

static FailureOr<Type> groupSizeType(MLIRContext *ctx, ShapeAttr shape,
                                     Operation *op) {
  if (!shape)
    return Type(getUnpinnedIdxType(ctx));
  if (shape.getDims().empty()) {
    FailureOr<ExprAttr> one = parseExprAttr(ctx, "1", op);
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
        composeExprAttr(ctx, product, sym::ExprBinaryOp::Mul, expr, op);
    if (failed(next))
      return failure();
    product = *next;
  }
  return Type(IdxType::get(ctx, product));
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
  if (isa<HCGroupIdOp>(raw))
    return appendLaunchAxisIdxTypes(ctx, raw, LaunchGeoMethod::GroupId, count,
                                    types);
  if (isa<HCLocalIdOp>(raw))
    return appendLaunchAxisIdxTypes(ctx, raw, LaunchGeoMethod::LocalId, count,
                                    types);
  if (isa<HCSubgroupIdOp>(raw))
    return appendLaunchAxisIdxTypes(ctx, raw, LaunchGeoMethod::SubgroupId,
                                    count, types);
  if (isa<HCWorkOffsetOp>(raw))
    return appendLaunchAxisIdxTypes(ctx, raw, LaunchGeoMethod::WorkOffset,
                                    count, types);
  if (isa<HCGroupShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, metadata->groupShape, count, types);
    return success();
  }
  if (isa<HCWorkShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, metadata->workShape, count, types);
    return success();
  }
  if (isa<HCGroupSizeOp>(raw)) {
    FailureOr<Type> type = groupSizeType(ctx, metadata->groupShape, raw);
    if (failed(type))
      return failure();
    types.push_back(*type);
    return success();
  }
  if (isa<HCWaveSizeOp>(raw)) {
    if (ExprAttr size = metadata->subgroupSize) {
      types.push_back(IdxType::get(ctx, size));
      return success();
    }
    types.push_back(getUnpinnedIdxType(ctx));
    return success();
  }

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

LogicalResult HCNegOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  Type value = operandTypes.empty() ? Type{} : operandTypes.front();
  if (!value) {
    resultTypes.push_back({});
    return success();
  }
  if (std::optional<ExprAttr> expr = idxExprAttr(value)) {
    FailureOr<ExprAttr> neg = composeNegExprAttr(getContext(), *expr, *this);
    if (failed(neg))
      return failure();
    resultTypes.push_back(IdxType::get(getContext(), *neg));
    return success();
  }
  resultTypes.push_back(value);
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
  return pushInferred(resultTypes,
                      inferBufferViewResult(sourceType, indexTypes,
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
  // Shape is the trailing operand after the variadic index list.
  Type shapeType = operandTypes.empty() ? Type{} : operandTypes.back();
  resultTypes.push_back(inferLoadLikeResult(sourceType, shapeType,
                                            /*vectorResult=*/false, *this));
  return success();
}

LogicalResult HCVLoadOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  Type sourceType = operandTypes.empty() ? Type{} : operandTypes.front();
  // Shape is the trailing operand after the variadic index list.
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
  if (auto tensor = dyn_cast_or_null<mlir::hc::TensorType>(value)) {
    resultTypes.push_back(mlir::hc::VectorType::get(
        getContext(), tensor.getElementType(), tensor.getShape()));
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

LogicalResult HCAsLayoutOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                         SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(operandTypes.empty() ? Type{} : operandTypes.front());
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
