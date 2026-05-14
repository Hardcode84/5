// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-distribute-wave-layouts`: rewrite wave-cooperative
// layout-bearing carriers into per-lane peers before the rest of the
// substrate composes per-axis offsets through them. See the pass
// description in `include/hc/Transforms/Passes.td` and the substrate
// rationale in `doc/layouts.md` — the short version is that a
// user-spelled layout like `vector<["32", "8"], <offset = 32*fi +
// lane>>` over a 32-lane subgroup distributes one slot per lane per
// fragment index, and the rewrite makes that fact visible to flatten /
// suffix-drop / post-flatten retyper without forcing them to invent
// it.
//
// Recognition is offset-driven: for a `#hc.layout` whose first
// index sym `lane` shows up in `offset` as the affine factor `lane *
// <stride>`, substituting `lane := $WI0` (the workitem axis-0 sym, by
// the substrate's `LaunchGeoMethod::LocalId` prefix convention) gives
// the per-lane offset, and substituting `lane := 0` gives the residue
// (the iter-only term flatten composes against). The shape's leading
// dim must equal the launch context's `subgroup_size` so the lane
// extent matches the carrier extent.
//
// Rewrite scope is intentionally narrow: only carriers that live
// inside an `hc.workitem_region` body get distributed (outside the
// region the per-lane peer has nowhere to live). The four ops the
// rewrite touches — `hc.vzeros` / `hc.vones` / `hc.vfull` and their
// tensor-flavoured siblings, `hc.generic`, `hc.buffer_view`,
// `hc.strip_layout` — are the ones the WMMA accumulator chain
// exercises end-to-end; broader carriers fall through unchanged and
// the downstream `hc-lower-generic` diagnostic still fires on
// anything the rewrite missed.

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

// Factorization of a wave-distributable layout's `offset` along the
// first index sym (the lane axis). `residue` is the offset after
// substituting `lane := 0`; `laneStride` is the affine coefficient of
// `lane` (constant integer); `perLaneStorage` is `storage_size /
// wave_size`. Caller has already validated that `offset` is affine in
// `lane` with constant stride and that `storage_size` is a constant
// multiple of `wave_size`.
struct WaveFactorization {
  sym::ExprHandle residueOffset;
  int64_t laneStride;
  sym::ExprHandle perLaneStorage;
  StringRef laneIndexSymName;
};

// Substitute `target` with `replacement` in `expr` via the symbol
// store's `ixs_subs_multi`. Returns a fresh handle on success. Mirrors
// the substitution shape `composeAccessOffsetExpr` uses; extracted
// here so the pass body doesn't repeat the session / target / value
// dance.
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

// Same as `substituteOne` but the replacement is a literal integer.
static FailureOr<sym::ExprHandle> substituteInt(sym::Store &store,
                                                sym::ExprHandle expr,
                                                StringRef targetName,
                                                int64_t value) {
  auto replacement = sym::composeExprInt(store, value);
  if (failed(replacement))
    return failure();
  return substituteOne(store, expr, targetName, *replacement);
}

// Predicate-flavoured analogue of `substituteOne`: walks
// `ixs_subs_multi` to rewrite `targetName` → `replacement` inside the
// predicate node. Returned handle is canonical hash-consed; type
// uniquing on `!hc.pred<...>` propagates the substitution to every
// SSA user the same way an `ExprAttr` swap does on `!hc.idx<...>`.
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

// True iff `layout` and `shape` agree on rank: both must be present,
// non-empty, and index_syms / shape_syms / dims share the same arity.
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

// Validate the layout / shape headers and pull the lane index sym
// name off the leading entry. Returns nullopt if the layout / shape
// don't agree on rank, the leading index sym isn't a StringAttr, or
// the leading shape dim doesn't match the wave extent.
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

// Probe the lane axis of `offset` for an affine factorization: the
// residue at `lane := 0` must not still mention `lane`, and `offset[lane :=
// 1] - residue` must be a positive constant stride. Caller has
// already validated the layout / shape headers.
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

// Validate that `storage_size` is a positive constant divisible by
// `waveSize`; produce `storage_size / waveSize` as the per-lane peer's
// storage extent.
//
// The simplest sufficient condition the v0 recogniser accepts is
// `storage_size == lane_stride * wave_size * <per_lane_extent>` for
// some positive per-lane extent — i.e. the lane axis tiles
// `lane_stride * wave_size` slots, leaving the residual to encode
// the per-lane extent. We don't enforce the exact tiling here; the
// storage_size of the per-lane peer is just `storage_size /
// wave_size`.
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

// Try to factor `layout.offset` along `layout.index_syms[0]`.
// Returns nullopt when the offset isn't affine in the lane sym with a
// constant stride, or the layout / shape parities don't support a
// clean lane distribution.
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

// Compute the per-lane peer of `shaped`: drop the first shape dim and
// the layout slot. The wave-cooperative carrier's layout encodes the
// (lane, residual_indices) → wave-wide slot mapping, which is exactly
// what disappears once we project onto a single lane — each lane's
// fragment is just a vector of `prod(residual_dims)` slots indexed
// row-major by the residual iters. Downstream producers (`hc.generic`
// outs / `hc.vload` etc.) have their offset arrays rewritten in
// lockstep so the per-lane composed offset on the layout-less peer is
// the residual iter sym alone.
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

// True when `op` is one of the nullary / fill alloc ops the rewrite
// handles. Each of these owns its result-type layout slot via the
// op-side `layout` attr and a shape SSA tuple; the rewrite updates
// both in lockstep.
static bool isShapedAllocOp(Operation *op) {
  return isa<HCVZerosOp, HCVOnesOp, HCVFullOp, HCZerosOp, HCOnesOp, HCFullOp>(
      op);
}

// Build a `tuple<idx<...>, ...>` SSA tuple at `loc` from the given
// per-axis dim exprs. Each dim materialises as an empty-binding
// `hc.idx_apply` carrying `!hc.idx<dim>`.
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

// Set the `shape` operand and `layout` attribute on a nullary / fill
// alloc op. Each of the six alloc ops has the same accessor shape but
// different concrete types, so the dispatch is by op kind. Returns
// failure on an unknown op kind.
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

// Extract the `shape` SSA tuple operand off any of the six
// nullary / fill alloc ops the rewrite touches. Returns null for
// anything else.
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

// Pull pinned `!hc.idx<dim>` expressions off a `tuple<!hc.idx<...>,
// ...>` type. Returns nullopt when any element isn't a pinned idx
// (the rewriter has no signal to compare against the result type's
// shape).
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

// Read the alloc op's `shape` SSA tuple's element types and return
// the per-axis dim exprs they pin.
static std::optional<SmallVector<Attribute>>
shapeFromAllocOperand(Operation *op) {
  Value shape = getAllocShapeOperand(op);
  if (!shape)
    return std::nullopt;
  return pinnedDimsFromIdxTuple(shape.getType());
}

// Update a wave-distributable nullary / fill alloc to match the type
// the producer-side rewrite established on its result. The result
// type's shape may have shrunk (lane axis dropped) and its layout may
// have been cleared by the generic-driven cascade; this helper
// detects the shape divergence between the alloc's `shape` SSA tuple
// (still pinning the pre-rewrite per-axis dims) and the alloc's
// (now-cascaded) result type, and rebuilds both the shape operand
// and the layout attr in lockstep. No-op when the shape operand
// already matches the result-type shape (the producer rewrite didn't
// touch this alloc).
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

  // Rebuild the shape tuple from the result type's per-axis dims and
  // clear the layout attr (the per-lane peer is layout-less; the
  // generic-driven cascade has retyped the result accordingly).
  OpBuilder builder(op);
  Value newShape =
      buildShapeTuple(builder, op->getLoc(), op->getContext(), resultDims);
  if (!newShape)
    return failure();
  return patchAllocShape(op, newShape, shaped.getSymbolicLayout());
}

// Helper: extract the iter sym name (StringRef) bound by an offset
// array entry that's a bare lane index (i.e. the operand's per-axis
// offset is literally one of the generic's iter syms). Returns
// nullopt when the entry isn't a bare sym name expression.
static std::optional<StringRef> getBareIterSym(ExprAttr offsetEntry,
                                               sym::Store &store) {
  if (!offsetEntry)
    return std::nullopt;
  // A bare iter sym renders as a single ixs sym leaf; walk and accept
  // only when there's exactly one leaf and the expression reduces to
  // that leaf alone. We test that by comparing the expression handle
  // against the constructed sym handle for the recovered name.
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

// Type-level substitution: rewrite every embedded sym leaf
// `targetName` in `ty` to point at `replacement`. The walker recurses
// into the structural carriers the dialect's symbolic types expose
// (`!hc.idx`, `!hc.pred`, `!hc.slice`, the five shaped types' shape
// + layout payloads). Anything outside those carriers — builtin
// scalars, opaque MLIR types, etc. — returns unchanged. Returning
// `Type()` signals a substitution failure the caller should treat
// as a bail.
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

// Per-type-kind handlers for `substituteInType`. Each returns the
// rewritten type, `ty` when nothing changed, or `Type{}` (i.e. null)
// on a substitution failure the caller propagates upstream.
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

// Substitute the lane iter sym for the wave sym inside every value
// the rewritten generic produces in its body region. The body's
// `hc.idx_apply` / `hc.pred_apply` ops carry the iter sym as an
// ambient binding in their result types (`!hc.idx<E[i_0]>` /
// `!hc.pred<P[i_0]>`); the surrounding generic used to bind `i_0`
// for them, but the rewrite has just dropped it. Replacing the leaf
// in every carrier in topological order leaves the body referencing
// `$WI0` from the enclosing `hc.workitem_region` instead, which the
// existing lowering paths handle natively.
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

// Substitute `laneIterSym -> waveExpr` in one operand's per-axis
// offset array, optionally dropping the leading axis. Returns null
// on a malformed entry, a failed substitution, or a drop-from-empty.
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

// Apply the lane-iter substitution to an entire operand-major offsets
// attribute (`outs_offsets` / `ins_offsets` on `hc.generic`). The
// outer array maps to operands; per-operand arrays are per-axis
// offset entries.
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

// Rewrite a `hc.generic` whose outs include wave-distributable
// carriers. The rewrite is in-place: the lane iter axis (identified
// by the leading entry of any wave-distributable outs's offset array)
// is dropped from `iter_syms` / `iter_bounds` / `iter_kinds`, every
// remaining offset expression substitutes the lane iter sym for
// `$WI0`, every wave-distributable operand's offset array drops its
// leading axis, and the value-typed results / outs that carry the
// wave-distributable type are retyped to their per-lane peers.
//
// Bails (returning failure) when:
//   * the outs's offset array's leading entry isn't a bare iter sym
//     (we'd be unable to identify the lane iter axis structurally),
//   * the lane iter sym shows up in multiple operands' non-leading
//     positions with conflicting subexpression shapes (we'd need a
//     cross-axis substitution the v0 doesn't model),
//   * the generic carries reduction iters that name the lane iter sym
//     (a wave reduction across the dropped axis would lose its
//     accumulator semantics under the rewrite).
// Bundles the classification state computed by `classifyWaveOuts`
// before any IR mutation: the shared lane iter sym, per-out drop
// flags, and the rewritten per-out types.
struct WaveOutsClassification {
  StringRef laneIterSym;
  SmallVector<bool> outsDropLeading;
  SmallVector<Type> newOutsTypes;
};

// Pull the leading bare iter sym off operand `k`'s per-axis offset
// array; failure means a malformed offsets entry the caller treats
// as fatal.
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

// True iff `out` carries a wave-distributable layout; if so, sets
// `*factor` to the factorization (caller owns the lifetime). Reads
// the layout + shape off the type interface.
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

// Inspect one outs operand `out` at index `k` and update
// `classification` if it's a wave-distributable carrier. Returns
// failure for outs whose offset array's leading entry can't be
// resolved to a bare iter sym (we couldn't identify the lane axis
// structurally) or whose distributed type can't be computed; signals
// "non-wave outs, skip" via success() with `matched` set false.
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

// First pass: identify every wave-distributable outs operand and the
// lane iter sym they share. Returns failure when at least one outs is
// wave-distributable but resolution against the offsets attribute
// fails, or when none are wave-distributable (nothing to rewrite).
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

// Second pass: identify wave-distributable INS so their leading
// offset axis drops in lockstep with the outs side. The ins carrier
// is typically the same wave-cooperative value the outs writes (e.g.
// a `hc.buffer_view` forward the rewrite collapses), but the
// rewrite handles both forms via the same offset-leading-iter-sym
// detector. Soft-skip semantics — anything that doesn't look like a
// wave-distributable ins is left alone; we don't reject the rewrite.
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

// Locate the lane iter sym's index inside `iterSyms` and validate
// it's `Parallel`. A reduction over the lane axis would be a wave
// reduction the v0 doesn't model.
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

// Bundles the iter-list slices after dropping the lane axis.
struct IterListsMinusLane {
  SmallVector<Attribute> syms;
  SmallVector<Value> bounds;
  SmallVector<Attribute> kinds;
};

// Rebuild the per-axis iter lists with the lane axis at `lanePos`
// removed.
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

// Apply all the producer-side mutations once classification has
// committed: new iter lists, new offsets, retyped outs operands,
// retyped results, and the in-body lane-sym substitution.
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
  // The body's `hc.idx_apply` / `hc.pred_apply` ops still carry
  // `laneIterSym` in their result-type expressions (the iter sym was
  // an ambient binding from the surrounding generic, which we just
  // dropped). Rewrite the body in lockstep so every leaf references
  // the wave sym instead — `$WI0` is bound by the enclosing
  // `hc.workitem_region` and reaches every nested op via the same
  // ambient-binding path.
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
