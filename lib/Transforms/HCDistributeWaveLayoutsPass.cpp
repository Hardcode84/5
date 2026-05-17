// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-distribute-wave-layouts`: wave-cooperative layout carriers
// → per-lane peers. Recognition: first index sym `lane` enters `offset` as an
// affine `lane * stride` term; `lane := $WI0` gives per-lane offset, `lane :=
// 0` gives the iter-only residue. Leading shape dim must equal `subgroup_size`.
// Scope is `hc.workitem_region` bodies only. See `doc/layouts.md`.

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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCDISTRIBUTEWAVELAYOUTS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// `offset = lane*laneStride + residueOffset`; perLaneStorage =
// storage_size/wave_size.
struct WaveFactorization {
  sym::ExprHandle residueOffset;
  int64_t laneStride;
  sym::ExprHandle perLaneStorage;
  StringRef laneIndexSymName;
};

// Substitute `targetName -> replacement` in `expr` via `ixs_subs_multi`.
static FailureOr<sym::ExprHandle> substituteOne(sym::Store &store,
                                                sym::ExprHandle expr,
                                                StringRef targetName,
                                                sym::ExprHandle replacement) {
  auto target = sym::composeExprSym(store, targetName);
  if (failed(target))
    return failure();
  sym::Session session(store);
  ixs_node *target_node = const_cast<ixs_node *>(target->raw());
  ixs_node *replacement_node = const_cast<ixs_node *>(replacement.raw());
  ixs_node *bound =
      ixs_subs_multi(session.raw(), const_cast<ixs_node *>(expr.raw()), 1,
                     &target_node, &replacement_node);
  if (!bound)
    return failure();
  return sym::ExprHandle(bound);
}

// Literal-int replacement variant of `substituteOne`.
static FailureOr<sym::ExprHandle> substituteInt(sym::Store &store,
                                                sym::ExprHandle expr,
                                                StringRef targetName,
                                                int64_t value) {
  auto replacement = sym::composeExprInt(store, value);
  if (failed(replacement))
    return failure();
  return substituteOne(store, expr, targetName, *replacement);
}

// Predicate analogue of `substituteOne`; type uniquing propagates to users.
static FailureOr<sym::PredHandle>
substituteOnePred(sym::Store &store, sym::PredHandle pred, StringRef targetName,
                  sym::ExprHandle replacement) {
  auto target = sym::composeExprSym(store, targetName);
  if (failed(target))
    return failure();
  sym::Session session(store);
  ixs_node *target_node = const_cast<ixs_node *>(target->raw());
  ixs_node *replacement_node = const_cast<ixs_node *>(replacement.raw());
  ixs_node *bound =
      ixs_subs_multi(session.raw(), const_cast<ixs_node *>(pred.raw()), 1,
                     &target_node, &replacement_node);
  if (!bound)
    return failure();
  return sym::PredHandle(bound);
}

// True when `expr` mentions `target` as one of its symbolic leaves.
static bool referencesSymbol(sym::ExprHandle expr, StringRef target) {
  bool found = false;
  sym::walkSymbolNames(expr, [&](StringRef name) {
    if (name == target)
      found = true;
  });
  return found;
}
static bool referencesSymbol(sym::PredHandle pred, StringRef target) {
  bool found = false;
  sym::walkSymbolNames(pred, [&](StringRef name) {
    if (name == target)
      found = true;
  });
  return found;
}

// Rank parity: index_syms / shape_syms / dims share the same non-zero arity.
static bool layoutShapeHaveMatchedRank(LayoutAttr layout, ShapeAttr shape) {
  if (!layout || !shape)
    return false;
  ArrayRef<Attribute> indexSyms = layout.getIndexSyms();
  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> dims = shape.getDims();
  if (indexSyms.empty() || shapeSyms.empty() || dims.empty())
    return false;
  return indexSyms.size() == dims.size() && shapeSyms.size() == dims.size();
}

// Lane sym name off the layout's leading index sym; leading dim must ==
// waveSize.
static std::optional<StringRef>
extractWaveLaneSym(LayoutAttr layout, ShapeAttr shape, int64_t waveSize) {
  if (!layoutShapeHaveMatchedRank(layout, shape))
    return std::nullopt;
  auto laneSymAttr = dyn_cast<StringAttr>(layout.getIndexSyms()[0]);
  auto firstDim = dyn_cast<ExprAttr>(shape.getDims()[0]);
  if (!laneSymAttr || !firstDim)
    return std::nullopt;
  auto firstDimInt = sym::getIntegerLiteralValue(firstDim.getValue());
  if (!firstDimInt || *firstDimInt != waveSize)
    return std::nullopt;
  return laneSymAttr.getValue();
}

// Probe affine factorization: residue@(lane:=0) lane-free; offset@(lane:=1)
// - residue must be positive constant stride.
static std::optional<std::pair<sym::ExprHandle, int64_t>>
probeLaneStride(sym::Store &store, sym::ExprHandle offset, StringRef laneSym) {
  auto residue = substituteInt(store, offset, laneSym, 0);
  if (failed(residue))
    return std::nullopt;
  if (referencesSymbol(*residue, laneSym))
    return std::nullopt;
  auto laneIs1 = substituteInt(store, offset, laneSym, 1);
  if (failed(laneIs1))
    return std::nullopt;
  auto negResidue = sym::composeExprNeg(store, *residue);
  if (failed(negResidue))
    return std::nullopt;
  auto strideExpr = sym::composeExprBinary(store, *laneIs1,
                                           sym::ExprBinaryOp::Add, *negResidue);
  if (failed(strideExpr))
    return std::nullopt;
  auto strideInt = sym::getIntegerLiteralValue(*strideExpr);
  if (!strideInt || *strideInt <= 0)
    return std::nullopt;
  return std::make_pair(*residue, *strideInt);
}

// `storage_size > 0`, divisible by `waveSize`; per-lane =
// storage_size/waveSize.
static std::optional<sym::ExprHandle>
computePerLaneStorage(sym::Store &store, ExprAttr storageAttr,
                      int64_t waveSize) {
  auto storageInt = sym::getIntegerLiteralValue(storageAttr.getValue());
  if (!storageInt || *storageInt <= 0)
    return std::nullopt;
  if (*storageInt % waveSize != 0)
    return std::nullopt;
  auto perLaneStorage = sym::composeExprInt(store, *storageInt / waveSize);
  if (failed(perLaneStorage))
    return std::nullopt;
  return *perLaneStorage;
}

// Factor `layout.offset` along `index_syms[0]`; nullopt on non-affine / parity
// mismatch.
static std::optional<WaveFactorization> tryFactorWaveLayout(LayoutAttr layout,
                                                            ShapeAttr shape,
                                                            int64_t waveSize,
                                                            sym::Store &store) {
  auto laneSym = extractWaveLaneSym(layout, shape, waveSize);
  if (!laneSym)
    return std::nullopt;

  auto strideOr =
      probeLaneStride(store, layout.getOffset().getValue(), *laneSym);
  if (!strideOr)
    return std::nullopt;

  auto perLaneStorage =
      computePerLaneStorage(store, layout.getStorageSize(), waveSize);
  if (!perLaneStorage)
    return std::nullopt;

  return WaveFactorization{strideOr->first, strideOr->second, *perLaneStorage,
                           *laneSym};
}

// Per-lane peer: drop leading shape dim and layout slot.
static Type distributedTypeMaybeBare(SymbolicallyShapedTypeInterface shaped,
                                     int64_t /*waveSize*/,
                                     StringRef /*waveSym*/,
                                     sym::Store & /*store*/,
                                     const WaveFactorization & /*factor*/) {
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return Type();
  ArrayRef<Attribute> dims = shape.getDims();
  if (dims.empty())
    return Type();
  MLIRContext *ctx = shaped.getContext();
  SmallVector<Attribute> perLaneDims(dims.drop_front());
  auto perLaneShape = ShapeAttr::get(ctx, perLaneDims);
  Type reshaped = shaped.cloneWithSymbolicShape(perLaneShape);
  auto reshapedShaped = dyn_cast<SymbolicallyShapedTypeInterface>(reshaped);
  if (!reshapedShaped)
    return Type();
  return reshapedShaped.cloneWithSymbolicLayout(LayoutAttr{});
}

// Nullary / fill alloc ops the rewrite handles.
static bool isShapedAllocOp(Operation *op) {
  return isa<HCVZerosOp, HCVOnesOp, HCVFullOp, HCZerosOp, HCOnesOp, HCFullOp>(
      op);
}

// Build `tuple<idx<dim>, ...>` from per-axis dim exprs; empty-binding
// `hc.idx_apply` each.
static Value buildShapeTuple(OpBuilder &builder, Location loc, MLIRContext *ctx,
                             ArrayRef<Attribute> dims) {
  SmallVector<Value> idxs;
  SmallVector<Type> idxTypes;
  idxs.reserve(dims.size());
  idxTypes.reserve(dims.size());
  for (Attribute dim : dims) {
    auto e = dyn_cast<ExprAttr>(dim);
    if (!e)
      return Value();
    auto idxTy = IdxType::get(ctx, e);
    Value v = HCIdxApplyOp::create(builder, loc, idxTy, ValueRange{},
                                   builder.getStrArrayAttr({}));
    idxs.push_back(v);
    idxTypes.push_back(idxTy);
  }
  auto tupleTy = TupleType::get(ctx, idxTypes);
  return HCTupleOp::create(builder, loc, tupleTy, idxs);
}

// Six alloc ops share accessor shape, distinct types; dispatch by kind.
static LogicalResult patchAllocShape(Operation *op, Value newShape,
                                     LayoutAttr newLayout) {
  if (auto vz = dyn_cast<HCVZerosOp>(op)) {
    vz.getShapeMutable().assign(newShape);
    vz.setLayoutAttr(newLayout);
  } else if (auto vo = dyn_cast<HCVOnesOp>(op)) {
    vo.getShapeMutable().assign(newShape);
    vo.setLayoutAttr(newLayout);
  } else if (auto vf = dyn_cast<HCVFullOp>(op)) {
    vf.getShapeMutable().assign(newShape);
    vf.setLayoutAttr(newLayout);
  } else if (auto z = dyn_cast<HCZerosOp>(op)) {
    z.getShapeMutable().assign(newShape);
    z.setLayoutAttr(newLayout);
  } else if (auto o = dyn_cast<HCOnesOp>(op)) {
    o.getShapeMutable().assign(newShape);
    o.setLayoutAttr(newLayout);
  } else if (auto f = dyn_cast<HCFullOp>(op)) {
    f.getShapeMutable().assign(newShape);
    f.setLayoutAttr(newLayout);
  } else {
    return failure();
  }
  return success();
}

// `shape` SSA tuple operand off any of the six alloc ops.
static Value getAllocShapeOperand(Operation *op) {
  if (auto vz = dyn_cast<HCVZerosOp>(op))
    return vz.getShape();
  if (auto vo = dyn_cast<HCVOnesOp>(op))
    return vo.getShape();
  if (auto vf = dyn_cast<HCVFullOp>(op))
    return vf.getShape();
  if (auto z = dyn_cast<HCZerosOp>(op))
    return z.getShape();
  if (auto o = dyn_cast<HCOnesOp>(op))
    return o.getShape();
  if (auto f = dyn_cast<HCFullOp>(op))
    return f.getShape();
  return Value{};
}

// Pinned `!hc.idx<dim>` exprs off a `tuple<idx<...>, ...>` type.
static std::optional<SmallVector<Attribute>>
pinnedDimsFromIdxTuple(Type tupleType) {
  auto tupleTy = dyn_cast<TupleType>(tupleType);
  if (!tupleTy)
    return std::nullopt;
  SmallVector<Attribute> dims;
  dims.reserve(tupleTy.size());
  for (Type t : tupleTy.getTypes()) {
    auto idxTy = dyn_cast<IdxType>(t);
    if (!idxTy)
      return std::nullopt;
    ExprAttr expr = idxTy.getExpr();
    if (!expr)
      return std::nullopt;
    dims.push_back(expr);
  }
  return dims;
}

// Per-axis dim exprs pinned in the alloc's `shape` tuple element types.
static std::optional<SmallVector<Attribute>>
shapeFromAllocOperand(Operation *op) {
  Value shape = getAllocShapeOperand(op);
  if (!shape)
    return std::nullopt;
  return pinnedDimsFromIdxTuple(shape.getType());
}

// Sync the alloc's `shape` operand and `layout` attr to its (cascaded) result
// type.
static LogicalResult harmoniseAllocOp(Operation *op) {
  Value result = op->getResult(0);
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(result.getType());
  if (!shaped)
    return failure();
  auto operandDims = shapeFromAllocOperand(op);
  if (!operandDims)
    return failure();
  ArrayRef<Attribute> resultDims = shaped.getSymbolicShape().getDims();
  if (operandDims->size() == resultDims.size() &&
      llvm::equal(*operandDims, resultDims))
    return failure();

  // Rebuild from result-type dims; per-lane peer is layout-less.
  OpBuilder builder(op);
  Value newShape =
      buildShapeTuple(builder, op->getLoc(), op->getContext(), resultDims);
  if (!newShape)
    return failure();
  return patchAllocShape(op, newShape, shaped.getSymbolicLayout());
}

// Iter sym name when the offset entry is exactly one sym leaf; nullopt
// otherwise.
static std::optional<StringRef> getBareIterSym(ExprAttr offsetEntry,
                                               sym::Store &store) {
  if (!offsetEntry)
    return std::nullopt;
  // Exactly one sym leaf, expression handle == reconstructed sym handle.
  StringRef found;
  unsigned count = 0;
  sym::walkSymbolNames(offsetEntry.getValue(), [&](StringRef n) {
    found = n;
    ++count;
  });
  if (count != 1)
    return std::nullopt;
  auto reconstructed = sym::composeExprSym(store, found);
  if (failed(reconstructed))
    return std::nullopt;
  if (reconstructed->raw() != offsetEntry.getValue().raw())
    return std::nullopt;
  return found;
}

// Recurse through `!hc.idx`/`!hc.pred`/`!hc.slice`/shaped carriers; `Type()` =
// bail.
static Type substituteInType(MLIRContext *ctx, sym::Store &store, Type ty,
                             StringRef targetName, sym::ExprHandle replacement);

static ExprAttr substituteExprAttr(MLIRContext *ctx, sym::Store &store,
                                   ExprAttr expr, StringRef targetName,
                                   sym::ExprHandle replacement) {
  if (!expr || !referencesSymbol(expr.getValue(), targetName))
    return expr;
  auto subs = substituteOne(store, expr.getValue(), targetName, replacement);
  if (failed(subs))
    return ExprAttr{};
  return ExprAttr::get(ctx, *subs);
}

static ShapeAttr substituteShapeAttr(MLIRContext *ctx, sym::Store &store,
                                     ShapeAttr shape, StringRef targetName,
                                     sym::ExprHandle replacement) {
  if (!shape)
    return shape;
  bool changed = false;
  SmallVector<Attribute> newDims;
  newDims.reserve(shape.getDims().size());
  for (Attribute dim : shape.getDims()) {
    auto e = dyn_cast<ExprAttr>(dim);
    if (!e) {
      newDims.push_back(dim);
      continue;
    }
    auto rewritten = substituteExprAttr(ctx, store, e, targetName, replacement);
    if (!rewritten)
      return ShapeAttr{};
    if (rewritten != e)
      changed = true;
    newDims.push_back(rewritten);
  }
  if (!changed)
    return shape;
  return ShapeAttr::get(ctx, newDims);
}

static LayoutAttr substituteLayoutAttr(MLIRContext *ctx, sym::Store &store,
                                       LayoutAttr layout, StringRef targetName,
                                       sym::ExprHandle replacement) {
  if (!layout)
    return layout;
  ExprAttr newOffset = substituteExprAttr(ctx, store, layout.getOffset(),
                                          targetName, replacement);
  ExprAttr newStorage = substituteExprAttr(ctx, store, layout.getStorageSize(),
                                           targetName, replacement);
  if (!newOffset || !newStorage)
    return LayoutAttr{};
  if (newOffset == layout.getOffset() && newStorage == layout.getStorageSize())
    return layout;
  return LayoutAttr::get(ctx, layout.getShapeSyms(), layout.getIndexSyms(),
                         layout.getParams(), newStorage, newOffset);
}

// `Type{}` propagates failure to the caller.
static Type substituteInIdx(MLIRContext *ctx, sym::Store &store, IdxType ty,
                            StringRef targetName, sym::ExprHandle replacement) {
  auto e = ty.getExpr();
  if (!e)
    return ty;
  auto newAttr = substituteExprAttr(ctx, store, e, targetName, replacement);
  if (!newAttr)
    return Type();
  if (newAttr == e)
    return ty;
  return IdxType::get(ctx, newAttr);
}

static Type substituteInPred(MLIRContext *ctx, sym::Store &store, PredType ty,
                             StringRef targetName,
                             sym::ExprHandle replacement) {
  auto p = ty.getPred();
  if (!p || !referencesSymbol(p.getValue(), targetName))
    return ty;
  auto subs = substituteOnePred(store, p.getValue(), targetName, replacement);
  if (failed(subs))
    return Type();
  return PredType::get(ctx, PredAttr::get(ctx, *subs));
}

static Type substituteInSlice(MLIRContext *ctx, sym::Store &store, SliceType ty,
                              StringRef targetName,
                              sym::ExprHandle replacement) {
  Type lo =
      substituteInType(ctx, store, ty.getLowerType(), targetName, replacement);
  Type hi =
      substituteInType(ctx, store, ty.getUpperType(), targetName, replacement);
  Type step =
      substituteInType(ctx, store, ty.getStepType(), targetName, replacement);
  if (lo == ty.getLowerType() && hi == ty.getUpperType() &&
      step == ty.getStepType())
    return ty;
  return SliceType::get(ctx, lo, hi, step);
}

static Type substituteInShaped(MLIRContext *ctx, sym::Store &store,
                               SymbolicallyShapedTypeInterface ty,
                               StringRef targetName,
                               sym::ExprHandle replacement) {
  ShapeAttr newShape = substituteShapeAttr(ctx, store, ty.getSymbolicShape(),
                                           targetName, replacement);
  LayoutAttr newLayout = substituteLayoutAttr(
      ctx, store, ty.getSymbolicLayout(), targetName, replacement);
  if (!newShape)
    return Type();
  Type result = ty;
  if (newShape != ty.getSymbolicShape())
    result = ty.cloneWithSymbolicShape(newShape);
  if (newLayout != ty.getSymbolicLayout()) {
    auto asShaped = dyn_cast<SymbolicallyShapedTypeInterface>(result);
    if (!asShaped)
      return Type();
    result = asShaped.cloneWithSymbolicLayout(newLayout);
  }
  return result;
}

static Type substituteInType(MLIRContext *ctx, sym::Store &store, Type ty,
                             StringRef targetName,
                             sym::ExprHandle replacement) {
  if (!ty)
    return ty;
  if (auto idx = dyn_cast<IdxType>(ty))
    return substituteInIdx(ctx, store, idx, targetName, replacement);
  if (auto pred = dyn_cast<PredType>(ty))
    return substituteInPred(ctx, store, pred, targetName, replacement);
  if (auto slice = dyn_cast<SliceType>(ty))
    return substituteInSlice(ctx, store, slice, targetName, replacement);
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(ty))
    return substituteInShaped(ctx, store, shaped, targetName, replacement);
  return ty;
}

// Rewrite leaves in body carriers so they reference `$WI0` from the enclosing
// workitem region after the generic has dropped the lane iter.
static LogicalResult substituteInRegionBody(Region &region, sym::Store &store,
                                            StringRef targetName,
                                            sym::ExprHandle replacement) {
  MLIRContext *ctx = region.getContext();
  auto walkResult = region.walk([&](Operation *op) -> WalkResult {
    for (Value result : op->getResults()) {
      Type oldTy = result.getType();
      Type newTy = substituteInType(ctx, store, oldTy, targetName, replacement);
      if (!newTy)
        return WalkResult::interrupt();
      if (newTy != oldTy)
        result.setType(newTy);
    }
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

// Per-axis substitution with optional leading-axis drop.
static ArrayAttr substituteOneOperandOffsets(MLIRContext *ctx, ArrayAttr inner,
                                             StringRef laneIterSym,
                                             sym::ExprHandle waveExpr,
                                             sym::Store &store,
                                             bool dropLeading) {
  if (dropLeading && inner.empty())
    return ArrayAttr{};
  SmallVector<Attribute> rewritten;
  rewritten.reserve(inner.size());
  size_t start = dropLeading ? 1u : 0u;
  for (size_t i = start; i < inner.size(); ++i) {
    auto expr = dyn_cast<ExprAttr>(inner[i]);
    if (!expr)
      return ArrayAttr{};
    auto subs = substituteOne(store, expr.getValue(), laneIterSym, waveExpr);
    if (failed(subs))
      return ArrayAttr{};
    rewritten.push_back(ExprAttr::get(ctx, *subs));
  }
  return ArrayAttr::get(ctx, rewritten);
}

// Operand-major substitution over the whole `*_offsets` attribute.
static ArrayAttr
substituteAndOptionallyDropAxis(MLIRContext *ctx, ArrayAttr perOperandOffsets,
                                StringRef laneIterSym, StringRef waveSym,
                                sym::Store &store, ArrayRef<bool> dropLeading) {
  if (!perOperandOffsets)
    return ArrayAttr{};
  if (perOperandOffsets.size() != dropLeading.size())
    return ArrayAttr{};
  auto waveExpr = sym::composeExprSym(store, waveSym);
  if (failed(waveExpr))
    return ArrayAttr{};
  SmallVector<Attribute> outer;
  outer.reserve(perOperandOffsets.size());
  for (auto [k, entry] : llvm::enumerate(perOperandOffsets)) {
    auto inner = dyn_cast<ArrayAttr>(entry);
    if (!inner)
      return ArrayAttr{};
    ArrayAttr rewritten = substituteOneOperandOffsets(
        ctx, inner, laneIterSym, *waveExpr, store, dropLeading[k]);
    if (!rewritten)
      return ArrayAttr{};
    outer.push_back(rewritten);
  }
  return ArrayAttr::get(ctx, outer);
}

// In-place generic rewrite: drop lane iter axis, sub `lane -> $WI0`, retype
// outs/results to per-lane peers. Bails on: outs leading entry not a bare
// iter sym; lane sym in conflicting non-leading positions; reduction over
// the lane axis.
struct WaveOutsClassification {
  StringRef laneIterSym;
  SmallVector<bool> outsDropLeading;
  SmallVector<Type> newOutsTypes;
};

// Leading bare iter sym on operand `k`'s offset array; fatal on failure.
static FailureOr<StringRef>
readLeadingIterSymOnOperand(ArrayAttr offsets, size_t k, sym::Store &store) {
  auto perOperand = dyn_cast<ArrayAttr>(offsets[k]);
  if (!perOperand || perOperand.empty())
    return failure();
  auto leading = dyn_cast<ExprAttr>(perOperand[0]);
  if (!leading)
    return failure();
  auto iterSym = getBareIterSym(leading, store);
  if (!iterSym)
    return failure();
  return *iterSym;
}

// Sets `factor` on a wave-distributable carrier; reads layout/shape off the
// type.
static bool isWaveDistributableOuts(Value out, int64_t waveSize,
                                    sym::Store &store,
                                    std::optional<WaveFactorization> &factor) {
  factor.reset();
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(out.getType());
  if (!shaped)
    return false;
  auto layout = shaped.getSymbolicLayout();
  if (!layout)
    return false;
  factor =
      tryFactorWaveLayout(layout, shaped.getSymbolicShape(), waveSize, store);
  return factor.has_value();
}

// Failure: malformed bare iter sym or distributed type. Skip: matched=false.
static LogicalResult tryClassifyWaveOut(Value out, size_t k,
                                        ArrayAttr outsOffsets, int64_t waveSize,
                                        StringRef waveSym, sym::Store &store,
                                        WaveOutsClassification &c,
                                        bool &matched) {
  matched = false;
  std::optional<WaveFactorization> factor;
  if (!isWaveDistributableOuts(out, waveSize, store, factor))
    return success();
  auto iterSym = readLeadingIterSymOnOperand(outsOffsets, k, store);
  if (failed(iterSym))
    return failure();
  if (!c.laneIterSym.empty() && *iterSym != c.laneIterSym)
    return failure();
  auto shaped = cast<SymbolicallyShapedTypeInterface>(out.getType());
  Type newTy =
      distributedTypeMaybeBare(shaped, waveSize, waveSym, store, *factor);
  if (!newTy)
    return failure();
  c.laneIterSym = *iterSym;
  c.outsDropLeading[k] = true;
  c.newOutsTypes[k] = newTy;
  matched = true;
  return success();
}

// Identify wave-distributable outs and the shared lane iter sym; failure when
// none qualify or any qualifier fails offset resolution.
static FailureOr<WaveOutsClassification>
classifyWaveOuts(ArrayRef<Value> outs, ArrayAttr outsOffsets, int64_t waveSize,
                 StringRef waveSym, sym::Store &store) {
  WaveOutsClassification c;
  c.outsDropLeading.assign(outs.size(), false);
  c.newOutsTypes.resize(outs.size());
  for (auto [k, out] : llvm::enumerate(outs))
    c.newOutsTypes[k] = out.getType();
  bool anyWaveOuts = false;
  for (auto [k, out] : llvm::enumerate(outs)) {
    bool matched = false;
    if (failed(tryClassifyWaveOut(out, k, outsOffsets, waveSize, waveSym, store,
                                  c, matched)))
      return failure();
    anyWaveOuts |= matched;
  }
  if (!anyWaveOuts)
    return failure();
  return c;
}

// Wave-distributable ins drop their leading axis in lockstep; soft-skip
// non-matches.
static SmallVector<bool>
classifyWaveIns(ArrayRef<Value> ins, ArrayAttr insOffsets, int64_t waveSize,
                StringRef waveSym, StringRef laneIterSym, sym::Store &store) {
  SmallVector<bool> insDropLeading(ins.size(), false);
  for (auto [k, in] : llvm::enumerate(ins)) {
    auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(in.getType());
    if (!shaped)
      continue;
    auto layout = shaped.getSymbolicLayout();
    if (!layout)
      continue;
    auto factor =
        tryFactorWaveLayout(layout, shaped.getSymbolicShape(), waveSize, store);
    if (!factor)
      continue;
    auto perOperand = dyn_cast<ArrayAttr>(insOffsets[k]);
    if (!perOperand || perOperand.empty())
      continue;
    auto leading = dyn_cast<ExprAttr>(perOperand[0]);
    if (!leading)
      continue;
    auto iterSym = getBareIterSym(leading, store);
    if (!iterSym || *iterSym != laneIterSym)
      continue;
    insDropLeading[k] = true;
  }
  return insDropLeading;
}

// Lane iter must be `Parallel`; wave-reduction across the lane axis isn't
// modelled.
static FailureOr<size_t> findAndValidateLaneIterPos(ArrayAttr iterSyms,
                                                    ArrayAttr iterKinds,
                                                    StringRef laneIterSym) {
  for (auto [i, s] : llvm::enumerate(iterSyms)) {
    if (cast<StringAttr>(s).getValue() != laneIterSym)
      continue;
    if (cast<IterKindAttr>(iterKinds[i]).getValue() != IterKind::Parallel)
      return failure();
    return i;
  }
  return failure();
}

// Iter-list slices after the lane axis is dropped.
struct IterListsMinusLane {
  SmallVector<Attribute> syms;
  SmallVector<Value> bounds;
  SmallVector<Attribute> kinds;
};

// Per-axis iter lists with the entry at `lanePos` removed.
static IterListsMinusLane buildIterListsWithoutLane(ArrayAttr iterSyms,
                                                    ValueRange iterBounds,
                                                    ArrayAttr iterKinds,
                                                    size_t lanePos) {
  IterListsMinusLane out;
  out.syms.reserve(iterSyms.size() - 1);
  out.bounds.reserve(iterBounds.size() - 1);
  out.kinds.reserve(iterKinds.size() - 1);
  for (auto [i, s] : llvm::enumerate(iterSyms)) {
    if (i == lanePos)
      continue;
    out.syms.push_back(s);
    out.bounds.push_back(iterBounds[i]);
    out.kinds.push_back(iterKinds[i]);
  }
  return out;
}

// Commit classification: iter lists, offsets, outs/result types, body sub.
static LogicalResult applyGenericRewriteMutations(
    HCGenericOp op, MLIRContext *ctx, const IterListsMinusLane &iters,
    ArrayAttr newInsOffsets, ArrayAttr newOutsOffsets, ArrayRef<Value> outs,
    ArrayRef<Type> newOutsTypes, StringRef laneIterSym, StringRef waveSym,
    sym::Store &store) {
  op.setIterSymsAttr(ArrayAttr::get(ctx, iters.syms));
  op.getIterBoundsMutable().assign(iters.bounds);
  op.setIterKindsAttr(ArrayAttr::get(ctx, iters.kinds));
  op.setInsOffsetsAttr(newInsOffsets);
  op.setOutsOffsetsAttr(newOutsOffsets);
  for (auto [k, outRef] : llvm::enumerate(outs)) {
    Value out = outRef;
    if (out.getType() != newOutsTypes[k])
      out.setType(newOutsTypes[k]);
  }
  for (auto [k, res] : llvm::enumerate(op.getResults()))
    res.setType(newOutsTypes[k]);
  // Body carriers still bind `laneIterSym`; rewrite leaves to `$WI0`.
  auto waveExpr = sym::composeExprSym(store, waveSym);
  if (failed(waveExpr))
    return failure();
  return substituteInRegionBody(op.getRegion(), store, laneIterSym, *waveExpr);
}

static LogicalResult rewriteGenericOp(HCGenericOp op, int64_t waveSize,
                                      StringRef waveSym, sym::Store &store) {
  MLIRContext *ctx = op.getContext();
  ArrayAttr insOffsets = op.getInsOffsets();
  ArrayAttr outsOffsets = op.getOutsOffsets();
  // ValueRange views into `op`'s operand storage go stale the moment
  // we mutate the operand list (the iter_bounds shrink below shifts
  // the ins / outs segments earlier). Snapshot the SSA values into
  // owned storage so subsequent `setType` calls land on the original
  // outs operands, not the operand slot that used to hold them.
  SmallVector<Value> ins(op.getIns().begin(), op.getIns().end());
  SmallVector<Value> outs(op.getOuts().begin(), op.getOuts().end());

  auto outsClass =
      classifyWaveOuts(outs, outsOffsets, waveSize, waveSym, store);
  if (failed(outsClass))
    return failure();
  SmallVector<bool> insDropLeading = classifyWaveIns(
      ins, insOffsets, waveSize, waveSym, outsClass->laneIterSym, store);

  auto lanePos = findAndValidateLaneIterPos(op.getIterSyms(), op.getIterKinds(),
                                            outsClass->laneIterSym);
  if (failed(lanePos))
    return failure();

  IterListsMinusLane iters = buildIterListsWithoutLane(
      op.getIterSyms(), op.getIterBounds(), op.getIterKinds(), *lanePos);

  ArrayAttr newInsOffsets = substituteAndOptionallyDropAxis(
      ctx, insOffsets, outsClass->laneIterSym, waveSym, store, insDropLeading);
  ArrayAttr newOutsOffsets = substituteAndOptionallyDropAxis(
      ctx, outsOffsets, outsClass->laneIterSym, waveSym, store,
      outsClass->outsDropLeading);
  if (!newInsOffsets || !newOutsOffsets)
    return failure();

  return applyGenericRewriteMutations(
      op, ctx, iters, newInsOffsets, newOutsOffsets, outs,
      outsClass->newOutsTypes, outsClass->laneIterSym, waveSym, store);
}

// Collapse a `hc.buffer_view` whose root is now per-lane (the
// producer dropped the lane axis) and whose first index operand is
// the lane scalar (`!hc.idx<"$WI0">`-typed). The view's remaining
// indices, after the lane index is consumed, must select every
// remaining axis with a full `!hc.slice` (i.e. they do not further
// constrain the view). Under those conditions the view is the
// identity on the root and gets RAUWed to the root.
//
// We use a structural rather than semantic check: each remaining
// index has a `!hc.slice` type with no lower / upper / step (i.e. the
// open-ended slice the canonicalization passes prefer). Tightening
// the recogniser later (e.g. accepting other full-range forms) is
// additive.
// True iff `firstTy` is a pinned `!hc.idx<expr>` whose expression
// references `waveSym` — i.e. an index materialized from the
// workitem region's lane scalar.
static bool isLaneScalarIdx(IdxType firstTy, StringRef waveSym) {
  if (!firstTy)
    return false;
  ExprAttr firstExpr = firstTy.getExpr();
  if (!firstExpr)
    return false;
  bool found = false;
  sym::walkSymbolNames(firstExpr.getValue(), [&](StringRef name) {
    if (name == waveSym)
      found = true;
  });
  return found;
}

// True iff every remaining index is an open-ended full `!hc.slice`
// (no lower / upper / step pinned). Tightening the recogniser later
// (other full-range forms) is additive.
static bool allOpenEndedSlices(ValueRange indices) {
  for (Value idx : indices) {
    auto sliceTy = dyn_cast<SliceType>(idx.getType());
    if (!sliceTy)
      return false;
    if (sliceTy.getLowerType() || sliceTy.getUpperType() ||
        sliceTy.getStepType())
      return false;
  }
  return true;
}

static LogicalResult rewriteBufferView(HCBufferViewOp op, StringRef waveSym) {
  ValueRange indices = op.getIndices();
  if (indices.empty())
    return failure();
  if (!isLaneScalarIdx(dyn_cast<IdxType>(indices[0].getType()), waveSym))
    return failure();
  if (!allOpenEndedSlices(indices.drop_front()))
    return failure();

  // Root must now be per-lane (i.e. the dim count matches the
  // per-axis subscript count minus 1).
  Value root = op.getBuffer();
  auto rootShaped = dyn_cast<SymbolicallyShapedTypeInterface>(root.getType());
  if (!rootShaped)
    return failure();
  ShapeAttr rootShape = rootShaped.getSymbolicShape();
  if (!rootShape || rootShape.getDims().size() != indices.size() - 1)
    return failure();

  // Element-type / shape parity check: the producer rewrite dropped
  // the leading axis and the lane sym out of the layout, leaving a
  // root whose remaining shape matches the view's result; the layout
  // slot may differ (root is typically layout-less after the
  // identity-fold; the view's result still carries the lane-pinned
  // layout from inference time). Drop the view and forward the root —
  // consumers pick up the new type via SSA tracking, and the verifier
  // on `hc.generic` is permissive on the layout slot.
  auto resShaped =
      dyn_cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
  if (!resShaped)
    return failure();
  if (resShaped.getSymbolicShape() != rootShape ||
      resShaped.getSymbolicElementType() != rootShaped.getSymbolicElementType())
    return failure();
  op.getResult().replaceAllUsesWith(root);
  op.erase();
  return success();
}

// Collapse a `hc.strip_layout` whose input matches its result (the
// producer rewrite already dropped the layout). The forward is the
// same shape `hc-lower-strip-layout` performs as its defensive
// no-op, surfaced here so the cascade doesn't leave dead strips in
// the IR for downstream passes to clean up.
static LogicalResult rewriteStripLayout(HCStripLayoutOp op) {
  if (op.getValue().getType() != op.getResult().getType())
    return failure();
  op.getResult().replaceAllUsesWith(op.getValue());
  op.erase();
  return success();
}

// Distill the wave size for the workitem region from its block-arg
// type. The block argument is typed `!hc.workitem<subgroup_size =
// #hc.expr<N>>`; pulling N off the type is structurally cheaper than
// walking back to the kernel callable's launch context and the two
// sources agree by construction (the launch-context check is on the
// substrate todo list as a backstop).
static std::optional<int64_t> waveSizeFromRegion(HCWorkitemRegionOp r) {
  if (r.getBody().empty() || r.getBody().front().getNumArguments() == 0)
    return std::nullopt;
  auto witm =
      dyn_cast<WorkitemType>(r.getBody().front().getArgument(0).getType());
  if (!witm)
    return std::nullopt;
  ExprAttr sg = witm.getSubgroupSize();
  if (!sg)
    return std::nullopt;
  return sym::getIntegerLiteralValue(sg.getValue());
}

// Drive one workitem region: collect candidate alloc / generic ops in
// the region body, rewrite them in producer-first order, then
// collapse the buffer_view / strip_layout consumers that became
// identity forwards. Each rewrite is in-place; nothing escapes the
// region body.
static void processRegion(HCWorkitemRegionOp r, int64_t waveSize,
                          StringRef waveSym, sym::Store &store) {
  SmallVector<Operation *> allocs;
  SmallVector<HCGenericOp> generics;
  r.getBody().walk([&](Operation *op) {
    if (isShapedAllocOp(op))
      allocs.push_back(op);
    else if (auto gen = dyn_cast<HCGenericOp>(op))
      generics.push_back(gen);
  });

  // Producer rewrites are generic-driven: each `hc.generic` whose
  // result is wave-distributable rewrites its iter / offset / outs
  // type, and the type change cascades onto the outs operand's
  // SSA-defining alloc op. The alloc cleanup follows: we read back
  // the alloc op's `layout` attr against its (now-rewritten) result
  // type and rebuild the shape operand to match.
  for (HCGenericOp gen : generics)
    (void)rewriteGenericOp(gen, waveSize, waveSym, store);
  for (Operation *op : allocs)
    (void)harmoniseAllocOp(op);

  // Consumer cleanup: views first (so the strip's input matches its
  // result type after the view forward), then strips.
  SmallVector<HCBufferViewOp> views;
  r.getBody().walk([&](HCBufferViewOp v) { views.push_back(v); });
  for (HCBufferViewOp v : views)
    (void)rewriteBufferView(v, waveSym);

  SmallVector<HCStripLayoutOp> strips;
  r.getBody().walk([&](HCStripLayoutOp s) { strips.push_back(s); });
  for (HCStripLayoutOp s : strips)
    (void)rewriteStripLayout(s);
}

struct HCDistributeWaveLayoutsPass
    : public hc::impl::HCDistributeWaveLayoutsBase<
          HCDistributeWaveLayoutsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();

    // Wave-distribute runs per `hc.workitem_region` body. The
    // workitem axis sym name is `$WI0` by the LaunchGeoMethod::LocalId
    // prefix convention; the substrate hard-codes the same name on
    // every other producer / consumer that touches `$WI*`, so the
    // string lives at one site.
    constexpr StringLiteral waveSym = "$WI0";

    SmallVector<HCWorkitemRegionOp> regions;
    root->walk([&](HCWorkitemRegionOp r) { regions.push_back(r); });
    for (HCWorkitemRegionOp r : regions) {
      auto waveSize = waveSizeFromRegion(r);
      if (!waveSize)
        continue;
      processRegion(r, *waveSize, waveSym, store);
    }
  }
};

} // namespace
