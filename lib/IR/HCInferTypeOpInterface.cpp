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

static Type inferLoadLikeResult(Type sourceType, ShapeAttr shape,
                                bool vectorResult, Operation *op) {
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

static std::optional<int64_t> staticIntegerIndex(Type type) {
  std::optional<ExprAttr> expr = idxExprAttr(type);
  if (!expr)
    return std::nullopt;
  // `hc.getitem` only needs literal tuple indices today. Keep broader
  // expression evaluation out of this local helper until it has a shared home.
  std::string rendered =
      symbolStore(type.getContext()).render((*expr).getNode());
  int64_t value;
  if (StringRef(rendered).getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static FailureOr<Type>
inferGetItemResult(Type sourceType, ArrayRef<Type> indexTypes, Operation *op) {
  auto tuple = dyn_cast_or_null<TupleType>(sourceType);
  if (!tuple || indexTypes.size() != 1)
    return Type{};
  std::optional<int64_t> index = staticIntegerIndex(indexTypes.front());
  if (!index)
    return Type{};
  int64_t originalIndex = *index;
  int64_t normalizedIndex = originalIndex;
  int64_t tupleSize = static_cast<int64_t>(tuple.size());
  if (normalizedIndex < 0)
    normalizedIndex += tupleSize;
  if (normalizedIndex < 0 || normalizedIndex >= tupleSize) {
    op->emitOpError("tuple index ")
        << originalIndex << " out of bounds for tuple of size " << tuple.size();
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
  auto group = dyn_cast_or_null<GroupType>(
      operandTypes.empty() ? Type{} : operandTypes[0]);
  if (!group || !allResultsAreUndef(op.getOperation())) {
    types.append(op.getResultTypes().begin(), op.getResultTypes().end());
    return success();
  }

  MLIRContext *ctx = op.getContext();
  Operation *raw = op.getOperation();
  unsigned count = op->getNumResults();
  if (isa<HCGroupIdOp>(raw))
    return appendPrefixedIdxTypes(ctx, raw, "$WG", count, types);
  if (isa<HCLocalIdOp>(raw))
    return appendPrefixedIdxTypes(ctx, raw, "$WI", count, types);
  if (isa<HCSubgroupIdOp>(raw))
    return appendPrefixedIdxTypes(ctx, raw, "$SG", count, types);
  if (isa<HCWorkOffsetOp>(raw))
    return appendPrefixedIdxTypes(ctx, raw, "$WO", count, types);
  if (isa<HCGroupShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, group.getGroupShape(), count, types);
    return success();
  }
  if (isa<HCWorkShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, group.getWorkShape(), count, types);
    return success();
  }
  if (isa<HCGroupSizeOp>(raw)) {
    FailureOr<Type> type = groupSizeType(ctx, group.getGroupShape(), raw);
    if (failed(type))
      return failure();
    types.push_back(*type);
    return success();
  }
  if (isa<HCWaveSizeOp>(raw)) {
    if (IntegerAttr size = group.getSubgroupSize()) {
      FailureOr<ExprAttr> expr = parseExprAttr(ctx, Twine(size.getInt()), raw);
      if (failed(expr))
        return failure();
      types.push_back(IdxType::get(ctx, *expr));
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

LogicalResult HCSliceExprOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                          SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(SliceType::get(getContext()));
  return success();
}

LogicalResult HCLoadOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                     SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(inferLoadLikeResult(
      operandTypes.empty() ? Type{} : operandTypes.front(), getShapeAttr(),
      /*vectorResult=*/false, *this));
  return success();
}

LogicalResult HCVLoadOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(inferLoadLikeResult(
      operandTypes.empty() ? Type{} : operandTypes.front(), getShapeAttr(),
      /*vectorResult=*/true, *this));
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
  resultTypes.push_back(operandTypes.empty() ? Type{} : operandTypes.front());
  return success();
}

LogicalResult HCAsLayoutOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                         SmallVectorImpl<Type> &resultTypes) {
  resultTypes.push_back(operandTypes.empty() ? Type{} : operandTypes.front());
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
