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

static Type unpinnedIdx(MLIRContext *ctx) {
  return IdxType::get(ctx, ExprAttr{});
}

static Type unpinnedPred(MLIRContext *ctx) {
  return PredType::get(ctx, PredAttr{});
}

template <typename AttrT, typename HandleT,
          FailureOr<HandleT> (*Parse)(sym::Store &, StringRef, std::string *)>
static FailureOr<AttrT> parseSymbolicAttr(MLIRContext *ctx, Twine text,
                                          Operation *diagOp, StringRef kind) {
  SmallString<64> storage;
  StringRef rendered = text.toStringRef(storage);
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  std::string diag;
  FailureOr<HandleT> handle = Parse(store, rendered, &diag);
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

static FailureOr<PredAttr> parsePredAttr(MLIRContext *ctx, Twine text,
                                         Operation *diagOp) {
  return parseSymbolicAttr<PredAttr, sym::PredHandle, sym::parsePred>(
      ctx, text, diagOp, "predicate");
}

// The symbolic library currently exposes parser/import entry points, not
// structural arithmetic builders. Keep expression composition isolated here so
// a later structural API can replace the render-parse bridge in one place.
static std::string renderExpr(ExprAttr expr) {
  auto *ctx = expr.getContext();
  return ctx->getOrLoadDialect<HCDialect>()->getSymbolStore().render(
      expr.getNode());
}

static std::optional<std::string> idxExprText(Type type) {
  auto idx = dyn_cast_or_null<IdxType>(type);
  if (!idx)
    return std::nullopt;
  if (ExprAttr expr = idx.getExpr())
    return renderExpr(expr);
  return std::nullopt;
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

static LogicalResult inferIndexBinary(ArrayRef<Type> operands, StringRef opText,
                                      Operation *op,
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
    std::optional<std::string> lhsExpr = idxExprText(lhs);
    std::optional<std::string> rhsExpr = idxExprText(rhs);
    if (!lhsExpr || !rhsExpr) {
      resultTypes.push_back(unpinnedIdx(op->getContext()));
      return success();
    }
    FailureOr<ExprAttr> expr = parseExprAttr(
        op->getContext(),
        Twine("(") + *lhsExpr + ") " + opText + " (" + *rhsExpr + ")", op);
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

static LogicalResult inferIndexCmp(ArrayRef<Type> operands, StringRef predText,
                                   Operation *op,
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
    std::optional<std::string> lhsExpr = idxExprText(lhs);
    std::optional<std::string> rhsExpr = idxExprText(rhs);
    if (!lhsExpr || !rhsExpr) {
      resultTypes.push_back(unpinnedPred(op->getContext()));
      return success();
    }
    FailureOr<PredAttr> pred =
        parsePredAttr(op->getContext(),
                      Twine(*lhsExpr) + " " + predText + " " + *rhsExpr, op);
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
    return unpinnedIdx(op->getContext());
  auto dim = dyn_cast<ExprAttr>(shape.getDims()[axis]);
  if (!dim)
    return unpinnedIdx(op->getContext());
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

template <typename OpT>
static LogicalResult inferLaunchGeometryTypes(OpT op,
                                              SmallVectorImpl<Type> &types) {
  types.append(op.getNumResults(), unpinnedIdx(op.getContext()));
  return success();
}

} // namespace

LogicalResult HCConstOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                      SmallVectorImpl<Type> &resultTypes) {
  Attribute value = getValue();
  if (auto integer = dyn_cast<IntegerAttr>(value))
    return pushInferred(resultTypes, inferIntegerAttrAsIdx(integer, *this));
  if (auto typed = dyn_cast<TypedAttr>(value)) {
    resultTypes.push_back(typed.getType());
    return success();
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
  return inferIndexBinary(operandTypes, "+", *this, resultTypes);
}

LogicalResult HCSubOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, "-", *this, resultTypes);
}

LogicalResult HCMulOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, "*", *this, resultTypes);
}

LogicalResult HCDivOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, "/", *this, resultTypes);
}

LogicalResult HCModOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  return inferIndexBinary(operandTypes, "%", *this, resultTypes);
}

LogicalResult HCNegOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                    SmallVectorImpl<Type> &resultTypes) {
  Type value = operandTypes.empty() ? Type{} : operandTypes.front();
  if (!value) {
    resultTypes.push_back({});
    return success();
  }
  if (std::optional<std::string> expr = idxExprText(value)) {
    FailureOr<ExprAttr> neg =
        parseExprAttr(getContext(), Twine("-(") + *expr + ")", *this);
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
  return inferIndexCmp(operandTypes, "<", *this, resultTypes);
}

LogicalResult HCCmpLeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, "<=", *this, resultTypes);
}

LogicalResult HCCmpGtOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, ">", *this, resultTypes);
}

LogicalResult HCCmpGeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, ">=", *this, resultTypes);
}

LogicalResult HCCmpEqOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, "==", *this, resultTypes);
}

LogicalResult HCCmpNeOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  return inferIndexCmp(operandTypes, "!=", *this, resultTypes);
}

LogicalResult HCGroupIdOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                        SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCLocalIdOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                        SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCSubgroupIdOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                           SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCGroupShapeOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                           SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCGroupSizeOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                          SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCWorkOffsetOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                           SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCWorkShapeOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                          SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
}

LogicalResult HCWaveSizeOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                         SmallVectorImpl<Type> &resultTypes) {
  return inferLaunchGeometryTypes(*this, resultTypes);
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
