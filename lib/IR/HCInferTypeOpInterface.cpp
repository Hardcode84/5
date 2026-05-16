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

// Idx-vs-idx arm of `inferIndexBinary`. Caller has already checked
// that at least one operand is an `IdxType`; this resolves the
// expression-composition and pinning cases and appends a single
// result.
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

// Same-rank NumPy-style broadcast on `#hc.shape<>` dim arrays. Returns
// the broadcasted dim list, or failure when ranks differ or a pair of
// non-`1` non-equal dims meet. Literal `1` is the broadcast neutral
// element — the canonical `composeExprInt(1)` handle gives pointer
// equality on the hash-consed `ExprAttr` so the per-axis test is one
// comparison, no parser involved. Different-rank inputs aren't
// supported yet: the front pass already aligns ranks via
// `unit_axes` so they don't reach this helper, and NumPy's
// left-padding rule needs more careful element-typing.
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

// True iff `lhs` and `rhs` are the same HC shaped flavor (both
// tensor / both bare-tensor / both vector / both bare-vector). The
// elementwise rewriter only emits a result whose flavor matches the
// operands, so cross-flavor binaries land on the diagnostic surface
// instead of silently casting.
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

// Rebuild a shaped HC type with a fresh shape and element. Layout is
// always null — broadcasting on a non-identity layout has no obvious
// result-side layout to carry and the elementwise rewriter doesn't
// read layouts off the result. Caller has already verified that
// `flavor` is one of the four shaped HC kinds.
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

// Shaped binary inference: tensors / vectors of the same flavor.
// Broadcast-aware — `[A, 1, C] op [1, B, C]` yields `[A, B, C]`.
// Element type joins through `joinHCTypes` so a partly-inferred
// `!hc.undef` element on one side picks up the concrete element from
// the other.
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

// Idx-vs-idx arm of `inferIndexCmp`. Mirror of
// `inferIndexBinaryIdxArm` but yields a `PredType` result.
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

// Bound-type → optional pinned `ExprAttr`. A null type means the bound
// was elided (`std::nullopt`); a non-null type must carry a pinned
// `IdxType` expression or we fail outright.
static FailureOr<std::optional<ExprAttr>> optionalIdxExprAttr(Type t) {
  if (!t)
    return std::optional<ExprAttr>();
  std::optional<ExprAttr> expr = idxExprAttr(t);
  if (!expr)
    return failure();
  return expr;
}

// `stop - start`, with `start` defaulting to 0 and `stop` defaulting
// to the operand's base dim when the slice elides the bound.
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

// `ceil(extent / step)`, with a fast path when `step == 1`. Caller
// has already verified that a step was supplied.
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
// Accumulator for symbolic-leaf substitutions inside a
// `hc.buffer_view` layout rewrite. Holds the parallel
// `targets[i] -> replacements[i]` arrays in the raw form
// `ixs_subs_multi` expects, plus helpers to (a) extend the lists
// from named-symbol replacements and (b) apply them to an
// `ExprAttr`.
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

// Cross-check that every per-axis input array is rank-aligned with
// `baseDims` so we don't accidentally substitute against a
// partially-typed source (rank-0 placeholder + layout, retype
// intermediate, ...).
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

// Non-trivial slice axes substitute the index sym with
// `lower + step * index_sym`. Caller has already established that at
// least one of (lower, step) departs from (0, 1).
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

// Kept (sliced) axis: drive the shape-sym / index-sym rebinds dictated
// by the slice's per-axis policy. Caller has already pushed the
// shape/index slot into the residual layout's lists.
static LogicalResult composeKeptAxisSubstitutions(
    LayoutSubBuilder &builder, Attribute baseDim, Attribute shapeSymAttr,
    Attribute indexSymAttr, ExprAttr lowerExpr, ExprAttr stepExpr,
    bool sliceShapeChanged, ExprAttr zeroAttr, ExprAttr oneAttr) {
  // Pass-through axes (more axes than subscripts) have empty slice
  // exprs and structurally cannot have a shape change; they shortcut
  // to the trivial-slice "no substitution" branch.
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

// Scalar-indexed axis: substitute both the shape sym (operand dim)
// and the index sym (scalar value). The slot is dropped from the
// residual's lists.
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

// Rewrite every `ExprAttr` value inside `params` through `builder`;
// non-expr entries pass through untouched. Returns the rebuilt
// dictionary (empty when `params` was null) so the layout's params
// continue to refer to canonical handles after the rebind.
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

// Walk each axis, push the appropriate substitution into `builder`
// and (for kept axes) record the residual layout's slot in
// `remainShapeSyms`/`remainIndexSyms`. Caller has already verified
// rank alignment across all per-axis arrays.
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

// Compose `sourceLayout` against a `hc.buffer_view`'s per-axis
// disposition. The substitution rules per axis:
//
//   * Scalar-indexed axes: substitute `index_syms[k]` with the
//     scalar's `IdxType` expression and `shape_syms[k]` with the
//     operand's actual dim expression, then drop both names from the
//     residual layout's slot lists.
//   * Trivial-slice axes (`[:]`, `[0:dim:1]` — sliced extent equals
//     the operand's dim, lower defaults to 0, step defaults to 1):
//     keep both slots in the residual, no substitution. The original
//     index/shape sym names continue to refer to the operand's axis
//     k.
//   * Non-trivial slice axes (`[lo:hi:st]` where any of lower /
//     upper / step departs from the trivial defaults): substitute
//     `index_syms[k]` with `lower + step * index_syms[k]` and (if the
//     sliced extent differs from the operand's dim) substitute
//     `shape_syms[k]` with the operand's dim expression. Both slots
//     remain in the residual's lists to satisfy
//     `shape_syms.size() == index_syms.size() == rank`; the
//     substituted names are reachable through their handles in the
//     rebound expressions but no longer appear as bare references in
//     the offset / storage_size / params formulas.
//   * Implicit pass-through axes (more axes than subscript entries):
//     keep both slots, no substitution. Same as trivial slices.
//
// The substitution also runs through every `ExprAttr` in the params
// dict so derived params (`row_stride = 4 + d1`) stay well-formed
// against the residual shape sym set.
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
  // ixsimpl canonicalises `0` / `1` to a single hash-consed handle,
  // so structural-equality checks against these leaves let us skip
  // the `m -> 0 + 1 * m` self-substitution that would otherwise
  // bloat trivial-slice offsets.
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

// Per-axis classification result for `inferBufferViewResult`. The
// layout composer consumes these as parallel arrays; here they're
// bundled so the per-axis classifiers can return one struct.
//
//   * `keep` is true for slice-subscripted axes (and pass-throughs)
//     that survive in the residual rank, false for scalar-consumed
//     axes whose slot drops out.
//   * `resultDim` is populated only for kept axes.
//   * `scalarValueExpr` carries the scalar index's `IdxType`
//     expression for the layout-offset substitution; empty for kept
//     axes.
//   * `sliceLowerExpr` / `sliceStepExpr` capture the slice's lower /
//     step (defaulting to 0 / 1); empty for non-slice cases.
//   * `sliceShapeChanged` records whether the sliced extent departs
//     from the operand's dim.
struct BufferViewAxisEntry {
  Attribute resultDim;
  bool keep = false;
  ExprAttr scalarValueExpr;
  ExprAttr sliceLowerExpr;
  ExprAttr sliceStepExpr;
  bool sliceShapeChanged = false;
};

// Per-source-axis arrays consumed by `composeBufferViewLayout`. All
// vectors stay rank-aligned with the source's logical shape; the
// vector-root "collective suffix" branch never appends to any of
// them.
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

// Slice subscript handler. Defaults missing slice parts to 0 / 1
// per Python semantics and computes the residual extent through
// `inferSliceViewDim`. Returns `nullopt` to mean "bail to
// currentResultType" (malformed slice bound) and `failure()` only
// for hard sym-store errors.
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

// Scalar subscript handler. Only `IdxType` carries a symbolic
// expression; a plain `index` / `i64` indexed against a layout-bearing
// source means we can't substitute a name into the layout offset, so
// we bail.
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

// Implicit pass-through axis (more source axes than subscripts):
// keep the slot and inherit the operand's dim.
static BufferViewAxisEntry buildPassThroughAxisEntry(Attribute baseDim) {
  BufferViewAxisEntry entry;
  entry.resultDim = baseDim;
  entry.keep = true;
  return entry;
}

// Subscripts past the source's logical rank are only legal on a
// vector root (workitem/subgroup-lifted collective suffix). Once the
// local axes are consumed, the typed suffix indices refine no local
// result dimensions and must just type-check.
static bool acceptsExtraSubscript(bool vectorRoot, Type indexType) {
  if (!vectorRoot || !indexType || isHCUndefType(indexType))
    return false;
  return isa<SliceType, IdxType>(indexType) || indexType.isIntOrIndex();
}

// Dispatcher: pick the slice / scalar arm based on `indexType`'s
// type and append the resulting entry to `arrays`. Returns `false`
// to mean "bail to currentResultType".
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

// Walk every subscript against the source's logical rank, building
// the per-axis arrays consumed by the layout composer. Returns
// `nullopt` to mean "bail to currentResultType"; `failure()` for
// hard errors only.
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

// Result type selector. The shape comes from `arrays.resultDims`;
// the rank-0 vector-root case collapses to the element type.
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

// Interleave unit-size dims into `keptDims` at the output positions
// listed by `unitAxes`. Output rank = `keptDims.size() + unitAxes.size()`.
// Positions are pre-verified unique, sorted is NOT assumed; we sort a
// local copy so the merge runs in one pass. Returns failure only on
// internal one-expr lookup, which should not happen for well-formed
// inputs.
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

// Vector-root collective-suffix indices (beyond `baseDims.size()`) were
// dropped during classification, so by the time we reach the layout
// compose `keepAxis` covers exactly the source rank. If composition
// fails (non-`IdxType` scalar against a layout, malformed shape entry,
// ...) leave the result type un-refined and let a later inference pass
// try again — encoded by returning the input `currentResultType`. A
// null `sourceLayout` short-circuits with an empty result layout.
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

  // `unit_axes` interleave size-1 dims into the kept shape. Layout
  // composition above only walks consuming subscripts; the unit dims
  // exist only on the result side and don't enter the layout offset,
  // so it's safe to splice them in afterwards. Bail conservatively if
  // a layout-bearing source ever needs unit-axis insertion — the
  // current frontend never plants `unit_axes` on layout-bearing
  // tensors, so this branch is unreachable from real input; if it ever
  // does, we'd need to extend the layout to carry zero-stride dims at
  // the inserted positions.
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

// Optional `layout` attribute on load/alloc/`hc.vec` producer ops: when
// FrontToHC sees a `layout=` kwarg on the call site it stamps the
// captured `LayoutAttr` directly on the producing op rather than emitting
// a separate `hc.as_layout` overlay. Inference reads it here and bakes
// it into the result type so non-injective layouts (broadcasts, per-lane
// WMMA fragments — `layout.storage_size` < `product(shape)`) flow
// through without producing a bare→layout-bearing transition that the
// `hc.as_layout` storage_size verifier would reject.
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
  // Tuple getitem needs literal indexing, not general symbolic evaluation.
  return sym::getIntegerLiteralValue(sym::ExprHandle(expr->getNode()));
}

// Resolve a tuple `[idx]` subscript to a non-negative slot index,
// accounting for Python negative-index semantics. Emits an op error
// and returns failure when the subscript isn't a static integer or
// falls outside `[-size, size)`.
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

// Launch-axis id family (`group_id`, `local_id`, `subgroup_id`,
// `work_offset`). Returns `nullopt` when `raw` isn't one of those op
// kinds.
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

// Launch shape projectors (`group_shape`, `work_shape`). Reads the
// matching shape attribute off `metadata` and stamps it across all
// results.
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

// Launch scalar accessors (`group_size`, `wave_size`). Produces one
// `IdxType` from the relevant aggregate (product of `group_shape`,
// or `subgroup_size` directly).
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

// `hc.pow` is a carrier; the structural lowering (`-hc-lower-pow`)
// rewrites it into a mul chain before infer-types runs in the canonical
// schedule, so this only fires when a non-standard pipeline forces
// inference on a surviving pow. ixsimpl has no `Pow` primitive, so the
// idx arm bails to `{}` and lets the unfold pass produce the chain
// whose mul ops `inferIndexBinary` will then type-refine. The scalar /
// shaped arm mirrors `inferIndexBinary`'s fall-through (same-type
// numeric in → same-type out) so a stray `hc.pow %x, %x` between
// matching scalars still gets a result type during partial pipelines.
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
  LayoutAttr layout = getLayoutAttr();
  if (auto tensor = dyn_cast_or_null<mlir::hc::TensorType>(value)) {
    resultTypes.push_back(mlir::hc::VectorType::get(
        getContext(), tensor.getElementType(), tensor.getShape(), layout));
    return success();
  }
  if (auto bareTensor = dyn_cast_or_null<mlir::hc::BareTensorType>(value)) {
    // Bare carriers carry no layout; if the source has been decomposed we
    // can't honour `layout=` here anymore, but the original layout-bearing
    // path will already have run on the semantic type before decomposition.
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

// Apply the op's `layout` attribute to a (operand) shaped type's
// flavor, preserving element type and shape. Returns null when the
// flavor isn't one we know how to retype (buffer is handled by the
// caller, which also has to pull the new shape off the optional shape
// operand). Mirrors `stripLayoutForCarrier` on the strip side.
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

// Buffer-rooted as_layout: result is a fresh `!hc.buffer` carrying
// the op's layout. When the op has a `shape=` operand, the declared
// shape from the tuple type drives the result; without it, the
// operand's dims pass through so a no-op relabel stays well-typed.
static Type inferAsLayoutBufferResult(BufferType buffer, Type shapeType,
                                      LayoutAttr layout, MLIRContext *ctx) {
  ShapeAttr resultShape = buffer.getShape();
  if (shapeType && !isHCUndefType(shapeType))
    if (ShapeAttr declared = getStaticShapeFromTupleType(shapeType))
      resultShape = declared;
  return mlir::hc::BufferType::get(ctx, buffer.getElementType(), resultShape,
                                   layout);
}

// Value-semantic operand (tensor / vector, bare or not): bake the
// op's layout onto the operand's existing element type and shape.
// Returns the unchanged operand type when the shape couldn't be
// recovered structurally — the verifier still pins the storage_size
// invariant downstream.
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
  // operandTypes is [value, (shape?)]; the shape slot is only present
  // when the op carries a `shape=` operand (buffer-rooted only — the
  // verifier rejects shape= on tensor/vector).
  Type valueType = operandTypes.empty() ? Type{} : operandTypes.front();
  Type shapeType =
      (getShape() && operandTypes.size() > 1) ? operandTypes[1] : Type{};

  // Pre-inference operand → preserve any refined result type the
  // frontend may have stamped (e.g. a buffer-rooted `as_layout` whose
  // result was pre-typed from the shape kwarg + layout). Without this
  // a subsequent inference round would clobber the refined slot with
  // `!hc.undef`.
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

// Strip preserves carrier flavor — drops the layout slot, not the
// semantic-vs-bare distinction. Returns the rebuilt type for the
// carrier kinds we know how to strip; returns null for everything
// else (buffer, unknown), at which point the caller leaves the
// result un-refined.
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
  // Mixing the flavor at the strip boundary would clash with
  // downstream inference (e.g. an `hc.call_intrinsic` whose result
  // is `vector<...>` and the loop's iter init type would conflict).
  // Buffers don't have a bare counterpart in the v0 surface — the
  // verifier rejects them, and we mirror that here by leaving the
  // result un-refined so a later inference pass sees the existing
  // placeholder. Pre-inference (`!hc.undef`) operands fall through
  // to the same un-refined branch.
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

// `keepdims = false` drops the axis dim (rank shrinks by one);
// `keepdims = true` replaces it with literal `1`. Pulled out so the
// outer op-inference can stay a flat sequence of guards.
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

// `hc.reduce`'s ODS pins the operand to `HC_ShapedValueType`
// (`!hc.undef` / `!hc.tensor` / `!hc.vector`); bare carriers can't
// surface (`hc-decompose-shaped-values` either rewrites the reduce
// up-front or fails legality), so the dispatcher only mirrors the two
// semantic flavors. Layout is intentionally dropped — the reduced
// axis was part of the operand's index space, and carrying the
// operand layout's name list past the collapse would dangle a stale
// dim sym.
static Type rebuildReduceResultType(Type valueType, Type elem,
                                    ShapeAttr outShape) {
  MLIRContext *ctx = elem.getContext();
  if (isa<mlir::hc::TensorType>(valueType))
    return mlir::hc::TensorType::get(ctx, elem, outShape, LayoutAttr{});
  if (isa<mlir::hc::VectorType>(valueType))
    return mlir::hc::VectorType::get(ctx, elem, outShape, LayoutAttr{});
  return {};
}

// `hc.reduce` collapses `$value`'s `$axis` dim. Without an inference
// rule the result stays `!hc.undef` after `hc-infer-types`, which makes
// `hc-shaped-compute-to-generic` bail (it needs both input and output
// shapes to validate the reduce surface). The flatten pass then
// retypes the surviving reduce's operand to a single product dim and
// the verifier catches the out-of-range axis. Refining the result
// here keeps the reduce rewritable before flatten and lets the
// existing rewriter own the only post-rewrite shape contract.
//
// Anything not shaped (`!hc.undef`, non-shaped builtins) keeps the
// result unknown so the next inference barrier or the verifier
// handles it.
LogicalResult HCReduceOp::inferHCTypes(ArrayRef<Type> operandTypes,
                                       SmallVectorImpl<Type> &resultTypes) {
  Type valueType = operandTypes.empty() ? Type{} : operandTypes.front();
  auto shaped = dyn_cast_or_null<SymbolicallyShapedTypeInterface>(valueType);
  ShapeAttr shape = shaped ? shaped.getSymbolicShape() : ShapeAttr{};
  // Out-of-range axis is the verifier's job once the operand type is
  // concrete; we bail with no inference so the diagnostic fires
  // against the typed input instead of being masked by a refined
  // result.
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
