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

// Leaf builders. These exist so the per-op inference helpers below
// can stay terse and so we go through the same hash-consed leaves
// any other producer in the dialect would — `composeExprInt(0)` and
// `composeExprSym("$WG0")` give back canonical handles, and pointer
// equality on those is the right comparison everywhere downstream.
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
  // ixsimpl's `IXS_INT` carries int64; reject wider IntegerAttrs up front
  // instead of letting them silently truncate. The previous textual round-
  // trip masked this — the integer was rendered with `toStringSigned` and
  // re-parsed, where ixsimpl would either accept the truncation or fail
  // with a generic diagnostic.
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
    FailureOr<ExprAttr> expr = composeExprAttr(*lhsExpr, opKind, *rhsExpr, op);
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
        composePredAttr(*lhsExpr, predKind, *rhsExpr, op);
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

static FailureOr<ExprAttr> inferSliceViewDim(ExprAttr baseDim, SliceType slice,
                                             Operation *op) {
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

  FailureOr<ExprAttr> zero = defaultZeroExpr(op);
  if (failed(zero))
    return failure();
  ExprAttr start = lower.value_or(*zero);
  ExprAttr stop = upper.value_or(baseDim);
  FailureOr<ExprAttr> extent = composeSubExprAttr(stop, start, op);
  if (failed(extent))
    return failure();

  if (!stepType)
    return *extent;

  FailureOr<ExprAttr> one = defaultOneExpr(op);
  if (failed(one))
    return failure();
  if (*step == *one)
    return *extent;

  FailureOr<ExprAttr> scaled = composeDivExprAttr(*extent, *step, op);
  if (failed(scaled))
    return failure();
  return composeCeilExprAttr(*scaled, op);
}

// Compose `sourceLayout` against a `hc.buffer_view`'s per-axis
// disposition. For every scalar-indexed axis k we substitute
// `index_syms[k]` (the layout's coordinate name for that axis) with
// the scalar's `IdxType` expression and `shape_syms[k]` (the layout's
// dim-alias name) with the operand's actual dim expression, then
// drop both names from the residual layout's slot lists. Slice axes
// and implicit pass-through axes keep their slots and contribute no
// substitution. The substitution also runs through every `ExprAttr`
// in the params dict so derived params (`row_stride = 4 + d1`) stay
// well-formed against the residual shape sym set.
//
// Slice-axis index rebinding (`index_syms[k] -> lower + step *
// index_syms[k]` for non-trivial slices) is intentionally not
// performed here: the current `hc.slice_expr` surface only emits
// `[0 : dim : 1]` from the frontend's `[:]` lowering, and any non-
// trivial slice would also need axis-extent rewriting that doesn't
// fit cleanly in the type-inference loop. Tracked separately on the
// slice-relayout follow-up.
static FailureOr<LayoutAttr>
composeBufferViewLayout(LayoutAttr sourceLayout, ArrayRef<Attribute> baseDims,
                        ArrayRef<bool> keepAxis,
                        ArrayRef<ExprAttr> scalarValueExpr, Operation *op) {
  MLIRContext *ctx = op->getContext();
  ArrayRef<Attribute> shapeSyms = sourceLayout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = sourceLayout.getIndexSyms();
  // `LayoutAttr::verify` already pins
  // `shape_syms.size() == index_syms.size()`; cross-check against
  // the operand's rank so we don't accidentally substitute against
  // a partially-typed source (rank-0 placeholder + layout, retype
  // intermediate, ...).
  if (shapeSyms.size() != baseDims.size())
    return failure();
  if (indexSyms.size() != baseDims.size())
    return failure();
  if (keepAxis.size() != baseDims.size())
    return failure();

  sym::Store &store = symbolStore(ctx);
  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  SmallVector<Attribute> remainShapeSyms;
  SmallVector<Attribute> remainIndexSyms;
  remainShapeSyms.reserve(baseDims.size());
  remainIndexSyms.reserve(baseDims.size());

  auto pushPair = [&](StringRef name,
                      sym::ExprHandle replacement) -> LogicalResult {
    auto symHandle = sym::composeExprSym(store, name);
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(const_cast<ixs_node *>(replacement.raw()));
    return success();
  };

  for (unsigned k = 0; k < baseDims.size(); ++k) {
    if (keepAxis[k]) {
      remainShapeSyms.push_back(shapeSyms[k]);
      remainIndexSyms.push_back(indexSyms[k]);
      continue;
    }
    auto dimExpr = dyn_cast<ExprAttr>(baseDims[k]);
    ExprAttr scalarExpr = scalarValueExpr[k];
    if (!dimExpr || !scalarExpr)
      return failure();
    StringRef shapeName = cast<StringAttr>(shapeSyms[k]).getValue();
    StringRef indexName = cast<StringAttr>(indexSyms[k]).getValue();
    if (failed(pushPair(shapeName, dimExpr.getValue())) ||
        failed(pushPair(indexName, scalarExpr.getValue())))
      return failure();
  }

  auto applySubs = [&](ExprAttr in) -> FailureOr<ExprAttr> {
    if (!in)
      return ExprAttr{};
    if (targets.empty())
      return in;
    sym::Session session(store);
    ixs_node *out =
        ixs_subs_multi(session.raw(), const_cast<ixs_node *>(in.getNode()),
                       static_cast<uint32_t>(targets.size()), targets.data(),
                       replacements.data());
    if (!out)
      return failure();
    return ExprAttr::get(ctx, sym::ExprHandle(out));
  };

  FailureOr<ExprAttr> newStorage = applySubs(sourceLayout.getStorageSize());
  if (failed(newStorage))
    return failure();
  FailureOr<ExprAttr> newOffset = applySubs(sourceLayout.getOffset());
  if (failed(newOffset))
    return failure();

  DictionaryAttr params = sourceLayout.getParams();
  SmallVector<NamedAttribute> newParams;
  if (params) {
    for (NamedAttribute kv : params) {
      auto exprAttr = dyn_cast<ExprAttr>(kv.getValue());
      if (!exprAttr) {
        newParams.push_back(kv);
        continue;
      }
      FailureOr<ExprAttr> subsed = applySubs(exprAttr);
      if (failed(subsed))
        return failure();
      newParams.emplace_back(kv.getName(), *subsed);
    }
  }
  DictionaryAttr newParamsAttr = DictionaryAttr::get(ctx, newParams);

  return LayoutAttr::get(ctx, remainShapeSyms, remainIndexSyms, newParamsAttr,
                         *newStorage, *newOffset);
}

static FailureOr<Type> inferBufferViewResult(Type sourceType,
                                             ArrayRef<Type> indexTypes,
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
  SmallVector<Attribute> resultDims;
  // Parallel arrays tracking per-source-axis disposition. `keepAxis[k]`
  // is true iff axis k survives in the residual rank (slice subscript
  // or implicit pass-through); for the scalar-consumed axes
  // `scalarValueExpr[k]` carries the index value's `IdxType` expression
  // so the layout composer can substitute it into the residual offset.
  // The vector-root "collective suffix" branch indexes beyond
  // `baseDims.size()` and doesn't populate either array — layout
  // composition is only meaningful within the source's logical rank.
  SmallVector<bool> keepAxis;
  SmallVector<ExprAttr> scalarValueExpr;
  keepAxis.reserve(baseDims.size());
  scalarValueExpr.reserve(baseDims.size());
  unsigned axis = 0;
  bool vectorRoot =
      isa<mlir::hc::VectorType, mlir::hc::BareVectorType>(sourceType);
  for (Type indexType : indexTypes) {
    if (axis >= baseDims.size()) {
      // Workitem/subgroup-lifted vectors carry collective suffix axes in
      // syntax even though the vector type stores only the participant-local
      // shape. Once local axes are consumed, typed suffix indices refine no
      // local result dimensions.
      if (!vectorRoot)
        return currentResultType;
      if (!indexType || isHCUndefType(indexType))
        return currentResultType;
      if (isa<SliceType, IdxType>(indexType) || indexType.isIntOrIndex())
        continue;
      return currentResultType;
    }
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
      keepAxis.push_back(true);
      scalarValueExpr.push_back({});
      ++axis;
      continue;
    }
    if (isa<IdxType>(indexType) || indexType.isIntOrIndex()) {
      keepAxis.push_back(false);
      // Only `IdxType` carries a symbolic expression. A plain `index`
      // or `i64` indexed against a layout-bearing source means we
      // can't substitute a name into the layout's offset; bail back
      // to the conservative no-refinement result.
      ExprAttr scalarExpr;
      if (auto idxExpr = idxExprAttr(indexType))
        scalarExpr = *idxExpr;
      if (sourceLayout && !scalarExpr)
        return currentResultType;
      scalarValueExpr.push_back(scalarExpr);
      ++axis;
      continue;
    }
    return currentResultType;
  }

  for (; axis < baseDims.size(); ++axis) {
    resultDims.push_back(baseDims[axis]);
    keepAxis.push_back(true);
    scalarValueExpr.push_back({});
  }

  LayoutAttr resultLayout;
  if (sourceLayout) {
    // Vector-root collective-suffix indices (beyond `baseDims.size()`)
    // would have already early-returned `continue`, so by the time we
    // reach the layout compose `keepAxis` covers exactly the source
    // rank. If composition fails (non-`IdxType` scalar against a
    // layout, malformed shape entry, ...) leave the result type
    // un-refined and let a later inference pass try again.
    FailureOr<LayoutAttr> composed = composeBufferViewLayout(
        sourceLayout, baseDims, keepAxis, scalarValueExpr, op);
    if (failed(composed))
      return currentResultType;
    resultLayout = *composed;
  }

  ShapeAttr resultShape = ShapeAttr::get(op->getContext(), resultDims);
  if (isa<mlir::hc::BufferType>(sourceType))
    return Type(mlir::hc::BufferType::get(op->getContext(), elementType,
                                          resultShape, resultLayout));
  if (vectorRoot) {
    if (resultDims.empty())
      return elementType;
    if (isa<mlir::hc::BareVectorType>(sourceType))
      return Type(mlir::hc::BareVectorType::get(op->getContext(), elementType,
                                                resultShape, resultLayout));
    return Type(mlir::hc::VectorType::get(op->getContext(), elementType,
                                          resultShape, resultLayout));
  }
  if (isa<mlir::hc::BareTensorType>(sourceType))
    return Type(mlir::hc::BareTensorType::get(op->getContext(), elementType,
                                              resultShape, resultLayout));
  return Type(mlir::hc::TensorType::get(op->getContext(), elementType,
                                        resultShape, resultLayout));
}

static Type inferLoadLikeResult(Type sourceType, Type shapeType,
                                bool vectorResult, Operation *op) {
  ShapeAttr shape = getStaticShapeFromTupleType(shapeType);
  if (!shape || !sourceType)
    return {};
  Type elementType = getSymbolicElementType(sourceType);
  if (!elementType)
    return {};
  return vectorResult ? Type(mlir::hc::VectorType::get(op->getContext(),
                                                       elementType, shape))
                      : Type(mlir::hc::TensorType::get(op->getContext(),
                                                       elementType, shape));
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
  if (isa<mlir::hc::VectorType>(sourceType))
    return inferBufferViewResult(sourceType, indexTypes, Type{}, op);
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

static LogicalResult appendPrefixedIdxTypes(Operation *op, StringRef prefix,
                                            unsigned count,
                                            SmallVectorImpl<Type> &types) {
  MLIRContext *ctx = op->getContext();
  for (unsigned axis = 0; axis < count; ++axis) {
    // Bypass the parser entirely: launch-axis names like "$WG0" are a
    // single symbol leaf, not an expression in need of tokenizing. A
    // bad character in `prefix` (which comes from a programmatic
    // LaunchGeoMethodInfo field) would silently produce a wrong tree
    // through parseExpr; composeExprSym builds the leaf directly.
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
  if (isa<HCGroupShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, metadata->groupShape, count, types);
    return success();
  }
  if (isa<HCWorkShapeOp>(raw)) {
    appendShapeIdxTypes(ctx, metadata->workShape, count, types);
    return success();
  }
  if (isa<HCGroupSizeOp>(raw)) {
    FailureOr<Type> type = groupSizeType(metadata->groupShape, raw);
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

LogicalResult HCIdxApplyOp::inferHCTypes(ArrayRef<Type> /*operandTypes*/,
                                         SmallVectorImpl<Type> &resultTypes) {
  // Result type is fully pinned by the carried `!hc.idx<expr>` and the
  // declared symbols list; operand types contribute nothing further.
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
  if (auto bareTensor = dyn_cast_or_null<mlir::hc::BareTensorType>(value)) {
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
