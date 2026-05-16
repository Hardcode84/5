// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-load-store-to-generic`: rewrite the per-tile memory
// ops `hc.load` / `hc.vload` / `hc.store` into the body-driven
// `hc.generic` surface that owns the post-flatten compute pipeline.
// See the pass description in `include/hc/Transforms/Passes.td` and
// the design in `doc/layouts.md`.

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
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOADSTORETOGENERIC
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Materialize one bound dim as `!hc.idx<dim>` SSA via an
// `hc.idx_apply` with no listed symbols (free shape names stay
// ambient). Mirrors the helper in `hc-shaped-compute-to-generic` —
// the rewriter has full structural knowledge of the iteration space,
// so the bounds-inference pass is a no-op if it ever runs after.
static Value materializeIdxBound(OpBuilder &builder, Location loc,
                                 ExprAttr dim) {
  auto idxTy = IdxType::get(builder.getContext(), dim);
  return HCIdxApplyOp::create(builder, loc, idxTy, ValueRange{},
                              builder.getStrArrayAttr({}));
}

// Build a `tuple<idx<...>, ...>` SSA tuple from the per-axis bounds —
// every shaped allocator (`hc.zeros` / `hc.vzeros`) takes its result
// shape through one of these.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// Pull each dim off a shaped operand. Bails on non-shaped types
// (`!hc.undef`) and on shapes that carry non-`#hc.expr` dim entries —
// either case means the rewriter can't materialize bounds and the op
// stays for later inference / downstream diagnostics.
static FailureOr<SmallVector<ExprAttr>> getOperandShape(Type t) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
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

// Per-axis decomposition of an access op's index operand. `base` is the
// lower bound of the tile walk on that axis (a scalar `!hc.idx<expr>`
// contributes `base = expr`; a slice contributes `base = lower` or `0`
// when the slice's lower is absent). `step` is the per-iter stride; a
// scalar idx is always step 1, a slice contributes `step = step_expr`
// or `1` when absent. Both are stored as `ExprAttr` so callers can
// build the per-axis offset structurally without re-parsing.
struct AxisIndex {
  ExprAttr base;
  ExprAttr step;
};

// Extract the bound expression off a pinned `!hc.idx<expr>`. Anything
// else — raw `index`, untyped `!hc.idx`, non-idx — fails: there's no
// symbolic name to bind into a `lower + step*iter` offset.
static FailureOr<ExprAttr> extractPinnedIdxExpr(Type t) {
  auto idx = llvm::dyn_cast<IdxType>(t);
  if (!idx || !idx.getExpr())
    return failure();
  return idx.getExpr();
}

// `(base, step)` for a slice index operand. Accepts default-step
// slices (`step = 1`) and pinned `!hc.idx<expr>` lower / step. Anything
// else fails the whole rewrite — silently lowering would emit an
// offset the launch-body would walk without the stride contribution.
static FailureOr<AxisIndex> extractAxisIndexFromSlice(MLIRContext *ctx,
                                                      sym::Store &store,
                                                      SliceType slice,
                                                      ExprAttr stepOne) {
  ExprAttr base;
  if (Type lowerTy = slice.getLowerType()) {
    auto lowerExpr = extractPinnedIdxExpr(lowerTy);
    if (failed(lowerExpr))
      return failure();
    base = *lowerExpr;
  } else {
    auto zero = sym::composeExprInt(store, 0);
    if (failed(zero))
      return failure();
    base = ExprAttr::get(ctx, *zero);
  }
  ExprAttr step = stepOne;
  if (Type stepTy = slice.getStepType()) {
    auto stepExpr = extractPinnedIdxExpr(stepTy);
    if (failed(stepExpr))
      return failure();
    step = *stepExpr;
  }
  return AxisIndex{base, step};
}

// Extract `(base, step)` for one access op index operand. Mirrors
// `extractAccessIndexExpr` in `hc-flatten-with-layouts` on the slice
// branch.
static FailureOr<AxisIndex>
extractAxisIndex(MLIRContext *ctx, sym::Store &store, Type indexType) {
  auto litOne = sym::composeExprInt(store, 1);
  if (failed(litOne))
    return failure();
  ExprAttr stepOne = ExprAttr::get(ctx, *litOne);
  if (auto idx = llvm::dyn_cast<IdxType>(indexType)) {
    auto expr = extractPinnedIdxExpr(idx);
    if (failed(expr))
      return failure();
    return AxisIndex{*expr, stepOne};
  }
  if (auto slice = llvm::dyn_cast<SliceType>(indexType))
    return extractAxisIndexFromSlice(ctx, store, slice, stepOne);
  return failure();
}

// Compose `base + step * iterSym` for the per-axis offset on the
// memory-side operand. Folds the trivial `step == 1` to `base +
// iterSym` so the printed offset stays the form scalar-idx callers
// already produce — without the fold, otherwise-identical loads land
// on two distinct hash-consed sums and obscure the diff. ixsimpl
// hash-conses, so building via the store keeps the printed offset
// canonical and shares storage with other identical sums elsewhere in
// the IR.
static FailureOr<ExprAttr> composeBasePlusStepIter(MLIRContext *ctx,
                                                   sym::Store &store,
                                                   AxisIndex axis,
                                                   StringAttr iterSym) {
  auto symHandle = sym::composeExprSym(store, iterSym.getValue());
  if (failed(symHandle))
    return failure();
  sym::ExprHandle term = *symHandle;
  std::optional<int64_t> stepLit =
      sym::getIntegerLiteralValue(axis.step.getValue());
  if (!stepLit || *stepLit != 1) {
    auto mul = sym::composeExprBinary(store, axis.step.getValue(),
                                      sym::ExprBinaryOp::Mul, term);
    if (failed(mul))
      return failure();
    term = *mul;
  }
  auto sum = sym::composeExprBinary(store, axis.base.getValue(),
                                    sym::ExprBinaryOp::Add, term);
  if (failed(sum))
    return failure();
  return ExprAttr::get(ctx, *sum);
}

// Build an `ArrayAttr<#hc.expr>` from a list of bare iter sym names —
// identity offsets on the value-side operand. Mirrors the helper
// in `hc-shaped-compute-to-generic`.
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

// Build the per-axis offset attribute on the memory-side operand:
// `[base_0 + step_0*i_0, base_1 + step_1*i_1, ...]`. Empty `axes`
// (whole-tensor access, e.g. `hc.store %dst[]`) collapses to identity
// over the iter syms — same shape, no addressing addend. Sizes must
// match post-pre-checks; this helper just assembles the array.
static FailureOr<ArrayAttr>
composeMemoryOffsetArray(MLIRContext *ctx, sym::Store &store,
                         ArrayRef<AxisIndex> axes,
                         ArrayRef<StringAttr> iterSyms) {
  SmallVector<Attribute> exprs;
  exprs.reserve(iterSyms.size());
  if (axes.empty()) {
    for (StringAttr name : iterSyms) {
      auto handle = sym::composeExprSym(store, name.getValue());
      if (failed(handle))
        return failure();
      exprs.push_back(ExprAttr::get(ctx, *handle));
    }
  } else {
    for (auto [axis, name] : llvm::zip_equal(axes, iterSyms)) {
      auto sum = composeBasePlusStepIter(ctx, store, axis, name);
      if (failed(sum))
        return failure();
      exprs.push_back(*sum);
    }
  }
  return ArrayAttr::get(ctx, exprs);
}

// Synthesise a fresh value-typed init of the given shape. The body
// of an all-parallel `hc.generic` never reads its outs carry, so
// any zero-cost initialisation is fine — `hc.zeros` for bare tensors,
// `hc.vzeros` for bare vectors. Result type drives the choice.
// Semantic carriers are rejected by the contract gate in
// `hc-decompose-shaped-values` and never reach this pass.
static Value emitValueInit(OpBuilder &builder, Location loc, Type resultTy,
                           Value shape) {
  if (isa<BareVectorType>(resultTy))
    return HCVZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                              /*layout=*/LayoutAttr{});
  assert(isa<BareTensorType>(resultTy) &&
         "load/store-to-generic result must be a bare shaped carrier");
  return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                           /*layout=*/LayoutAttr{});
}

// Common shape:
//   * iter syms `i_0`, ..., `i_{r-1}` — distinct from the source-level
//     symbol names that index operands carry. Conflicts with a body
//     name `i_0` are theoretically possible but very unlikely in
//     practice; the verifier catches it and a proper collision-free
//     namer is its own follow-up (matches the same trade-off the
//     reduce rewriter takes).
//   * all parallel iters; one bound per axis materialised from the
//     tile shape.
//   * value-side operand carries identity offsets, memory-side
//     carries `[base_k + i_k]`.
struct CommonRewriteData {
  SmallVector<StringAttr> iterSyms;
  SmallVector<Value> iterBounds;
  ArrayAttr iterSymsAttr;
  ArrayAttr iterKindsAttr;
};

static CommonRewriteData buildCommon(OpBuilder &builder, Location loc,
                                     ArrayRef<ExprAttr> tileShape) {
  MLIRContext *ctx = builder.getContext();
  CommonRewriteData out;
  out.iterSyms.reserve(tileShape.size());
  out.iterBounds.reserve(tileShape.size());
  SmallVector<Attribute> symAttrs;
  symAttrs.reserve(tileShape.size());
  SmallVector<Attribute> kindAttrs;
  kindAttrs.reserve(tileShape.size());
  for (auto [k, dim] : llvm::enumerate(tileShape)) {
    auto sym = StringAttr::get(ctx, ("i_" + Twine(k)).str());
    out.iterSyms.push_back(sym);
    symAttrs.push_back(sym);
    out.iterBounds.push_back(materializeIdxBound(builder, loc, dim));
    kindAttrs.push_back(IterKindAttr::get(ctx, IterKind::Parallel));
  }
  out.iterSymsAttr = ArrayAttr::get(ctx, symAttrs);
  out.iterKindsAttr = ArrayAttr::get(ctx, kindAttrs);
  return out;
}

// Pull the per-axis `(base, step)` off an op's index operands. v0
// handles pinned `!hc.idx<expr>` scalar indices and slices whose
// lower/step (when present) are pinned `!hc.idx<expr>`. Anything else
// fails the whole rewrite — the launch-body lowering still owns those
// shapes.
static FailureOr<SmallVector<AxisIndex>>
collectAxisIndices(MLIRContext *ctx, sym::Store &store, ValueRange indices) {
  SmallVector<AxisIndex> axes;
  axes.reserve(indices.size());
  for (Value idx : indices) {
    auto axis = extractAxisIndex(ctx, store, idx.getType());
    if (failed(axis))
      return failure();
    axes.push_back(*axis);
  }
  return axes;
}

// Element / pointee type the body block-arg carries for a given
// operand. Mirrors `genericOperandElement` in `lib/IR/HCOps.cpp` for
// the shaped + ptr cases the rewriter actually emits — kept local
// because hardcoding the shape here is fine while we only emit
// shaped + ptr/buffer operands.
static Type bodyArgElementType(Type t) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t))
    return shaped.getSymbolicElementType();
  return {};
}

// ----- hc.load / hc.vload -----------------------------------------------

// Compose ins_offsets[0] for a load whose source rank differs from the
// result tile rank. The result type's layout maps `index_syms` → a
// single flat offset into source storage. Substituting iter syms
// (i_0, i_1, ...) for `index_syms` and the tile's per-axis bound
// expressions for `shape_syms` produces an expression in iter syms
// that is exactly the source's 0-axis index per iter point. Returns a
// single-element `ArrayAttr` wrapping that expression; bails when the
// source isn't rank-1 (no general decomposition of a scalar layout
// offset onto multi-dim source axes) or the layout / shape parities
// don't line up. Caller has already confirmed the result type carries
// the layout.
static FailureOr<ArrayAttr> composeBroadcastSourceOffset(
    MLIRContext *ctx, LayoutAttr layout, ArrayRef<ExprAttr> tileShape,
    ArrayRef<StringAttr> iterSyms, ArrayRef<ExprAttr> srcShape) {
  if (srcShape.size() != 1)
    return failure();
  if (!layout)
    return failure();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallVector<ExprAttr> iterExprs;
  iterExprs.reserve(iterSyms.size());
  for (StringAttr name : iterSyms) {
    auto handle = sym::composeExprSym(store, name.getValue());
    if (failed(handle))
      return failure();
    iterExprs.push_back(ExprAttr::get(ctx, *handle));
  }
  SmallVector<Attribute> dims(tileShape.begin(), tileShape.end());
  auto shape = ShapeAttr::get(ctx, dims);
  auto offset = composeAccessOffsetExpr(ctx, layout, shape, iterExprs);
  if (failed(offset))
    return failure();
  return ArrayAttr::get(ctx, {Attribute(*offset)});
}

// Divide `extent` by the slice's `step` when the step is a pinned
// `!hc.idx<expr>` that isn't the integer literal 1. Unit steps leave
// `extent` untouched — the hash-consed `/1` would print as a
// redundant ride-along otherwise. Absent step (null `stepTy`) also
// leaves `extent` alone since "no step" means unit by convention.
static FailureOr<sym::ExprHandle>
divideExtentByStep(sym::Store &store, sym::ExprHandle extent, Type stepTy) {
  if (!stepTy)
    return extent;
  auto stepExpr = extractPinnedIdxExpr(stepTy);
  if (failed(stepExpr))
    return failure();
  std::optional<int64_t> stepLit =
      sym::getIntegerLiteralValue(stepExpr->getValue());
  if (stepLit && *stepLit == 1)
    return extent;
  return sym::composeExprBinary(store, extent, sym::ExprBinaryOp::Div,
                                stepExpr->getValue());
}

// Per-axis extent expression of one slice when the slice's `lower`
// and `upper` are both pinned `!hc.idx<expr>`. Composes `upper -
// lower` (and divides by `step` when the step is a non-unit pinned
// literal). Used by the layout-driven gather decomposition to know
// how many cells the slice covers along each axis.
static FailureOr<ExprAttr>
extractSliceExtent(MLIRContext *ctx, sym::Store &store, SliceType slice) {
  Type lowerTy = slice.getLowerType();
  Type upperTy = slice.getUpperType();
  if (!lowerTy || !upperTy)
    return failure();
  auto lower = extractPinnedIdxExpr(lowerTy);
  if (failed(lower))
    return failure();
  auto upper = extractPinnedIdxExpr(upperTy);
  if (failed(upper))
    return failure();
  auto diff = sym::composeExprBinary(store, upper->getValue(),
                                     sym::ExprBinaryOp::Sub, lower->getValue());
  if (failed(diff))
    return failure();
  auto extent = divideExtentByStep(store, *diff, slice.getStepType());
  if (failed(extent))
    return failure();
  return ExprAttr::get(ctx, *extent);
}

// Per-axis extents for an entire `indices` operand list. Returns
// failure if any index isn't a slice with both pinned `lower` and
// `upper` (the layout-driven gather decomposition needs both bounds
// to compute the slice's flat capacity). Caller falls back to the
// per-axis identity offsets on failure.
static FailureOr<SmallVector<ExprAttr>>
collectSliceExtents(MLIRContext *ctx, sym::Store &store, ValueRange indices) {
  if (indices.empty())
    return failure();
  SmallVector<ExprAttr> extents;
  extents.reserve(indices.size());
  for (Value idx : indices) {
    auto slice = dyn_cast<SliceType>(idx.getType());
    if (!slice)
      return failure();
    auto extent = extractSliceExtent(ctx, store, slice);
    if (failed(extent))
      return failure();
    extents.push_back(*extent);
  }
  return extents;
}

// Product of `extents`. Used to compare the slice's flat capacity
// against the layout's `storage_size` before the gather
// decomposition: hash-consing in the dialect store gives pointer
// equality on the canonical handle when the two products
// structurally agree.
static FailureOr<sym::ExprHandle>
composeProductOfExtents(sym::Store &store, ArrayRef<ExprAttr> extents) {
  if (extents.empty())
    return sym::composeExprInt(store, 1);
  sym::ExprHandle prod = extents[0].getValue();
  for (size_t k = 1; k < extents.size(); ++k) {
    auto next = sym::composeExprBinary(store, prod, sym::ExprBinaryOp::Mul,
                                       extents[k].getValue());
    if (failed(next))
      return failure();
    prod = *next;
  }
  return prod;
}

// Validated preflight state for the load rewriter: source shape, the
// per-axis indices, and whether the access is a non-injective rank-1
// broadcast (source rank < tile rank, with the iter axes collapsing
// onto storage via the result layout).
struct LoadPreflight {
  SmallVector<AxisIndex> axes;
  SmallVector<ExprAttr> srcShape;
  LayoutAttr resultLayout;
  bool broadcastFromRank1;
  // Per-axis slice extents when every index is a pinned slice. Empty
  // when extracting extents failed for any axis or when the access
  // doesn't use slices at every position. The layout-driven gather
  // decomposition consumes this; the per-axis identity fallback
  // doesn't need it.
  SmallVector<ExprAttr> sliceExtents;
  // Layout-bearing buffer source path: when the access op's source is
  // defined by `hc.as_layout` with both a layout attribute and a
  // `shape=` operand, peel it. `effectiveSource` is the underlying
  // buffer (the as_layout's operand), `srcShape` reports that
  // underlying's shape, and `sourceLayout` / `sourceDeclaredShape`
  // drive the per-position gather decomposition. Empty
  // `effectiveSource` means no peel happened and the original source
  // routes through the layout-on-result paths above.
  Value effectiveSource;
  LayoutAttr sourceLayout;
  ShapeAttr sourceDeclaredShape;
};

// `hc.as_layout` with a `shape=` operand is the symmetric layout-
// driven load/store ergonomic on a pointer-rooted source: the result
// type is `!hc.buffer<elem, [D0, ...], LAY>` carrying both the
// declared shape and the layout that maps each access position to
// the underlying buffer's row-major flatten. The access op itself
// stays addressing-shape-agnostic — pinned + slice indices mix on
// the as_layout result; the layout's `offset` formula reconstructs
// the flat target.
//
// Peeling here lets the rewriter route the access through a gather
// decomposition against the *underlying* buffer's shape, which is
// what the per-axis offsets the post-flatten pipeline expects. The
// alternative — leaving the `hc.as_layout` in place — would have
// `hc-flatten-with-layouts` compose the source-side layout into the
// access offsets, but that path also assumes the access op's
// per-axis offsets are pre-layout coords and would double-apply LAY.
struct LayoutBearingSource {
  Value underlying;
  LayoutAttr layout;
  ShapeAttr declaredShape;
};

static std::optional<LayoutBearingSource>
peelLayoutBearingBufferSource(Value source) {
  auto op = source.getDefiningOp<HCAsLayoutOp>();
  if (!op)
    return std::nullopt;
  LayoutAttr layout = op.getLayoutAttr();
  if (!layout)
    return std::nullopt;
  Value shapeOperand = op.getShape();
  if (!shapeOperand)
    return std::nullopt;
  ShapeAttr declared = getStaticShapeFromTupleType(shapeOperand.getType());
  if (!declared)
    return std::nullopt;
  return LayoutBearingSource{op.getValue(), layout, declared};
}

// Element-type match check between source and result: both must be
// shaped, with matching element types. Failure means the rewriter
// can't safely emit a typed body block and the op stays for later
// inference / diagnostics.
static LogicalResult checkLoadElementTypesMatch(Type srcTy, Type resultTy) {
  Type srcElem = bodyArgElementType(srcTy);
  Type resElem = bodyArgElementType(resultTy);
  if (!srcElem || !resElem || srcElem != resElem)
    return failure();
  return success();
}

// Classify the access shape: same-rank (offsets per-axis) vs.
// rank-1 broadcast (layout-driven). Returns the broadcast flag plus
// the layout attribute (null when the result type isn't a shaped /
// has no layout).
//
// Non-injective broadcast path: source rank < tile rank means the
// user-supplied layout encodes how iter axes collapse onto source
// storage (e.g. `offset = j` for `vload(src=rank1, shape=(M,K,L))`
// emits each row K times). Only rank-1 source has a well-defined
// decomposition (a scalar offset doesn't split across N>1 axes).
static LogicalResult classifyLoadAccess(Type resultTy, ValueRange indices,
                                        ArrayRef<ExprAttr> srcShape,
                                        ArrayRef<ExprAttr> tileShape,
                                        bool &broadcastFromRank1,
                                        LayoutAttr &resultLayout) {
  auto resultShaped = dyn_cast<SymbolicallyShapedTypeInterface>(resultTy);
  resultLayout = resultShaped ? resultShaped.getSymbolicLayout() : LayoutAttr{};
  broadcastFromRank1 = indices.empty() && srcShape.size() != tileShape.size();
  if (broadcastFromRank1 && (srcShape.size() != 1 || !resultLayout))
    return failure();
  return success();
}

// Compose the layout's `offset` expression with iter syms
// substituted for `index_syms` and `tileShape` for `shape_syms`.
// The result is the per-iter-point flat position into the slice's
// row-major flatten — the value the per-axis decomposition splits
// back into source coordinates.
static FailureOr<sym::ExprHandle>
composeLayoutFlatOffset(MLIRContext *ctx, sym::Store &store, LayoutAttr layout,
                        ArrayRef<ExprAttr> tileShape,
                        ArrayRef<StringAttr> iterSyms) {
  SmallVector<ExprAttr> iterExprs;
  iterExprs.reserve(iterSyms.size());
  for (StringAttr name : iterSyms) {
    auto handle = sym::composeExprSym(store, name.getValue());
    if (failed(handle))
      return failure();
    iterExprs.push_back(ExprAttr::get(ctx, *handle));
  }
  SmallVector<Attribute> dims(tileShape.begin(), tileShape.end());
  auto tileShapeAttr = ShapeAttr::get(ctx, dims);
  auto flatOr = composeAccessOffsetExpr(ctx, layout, tileShapeAttr, iterExprs);
  if (failed(flatOr))
    return failure();
  return flatOr->getValue();
}

// Row-major inner products of `extents`: result[k] = product of
// extents[k+1..N-1], with result[N-1] = 1. Used by the layout-driven
// gather to split a flat row-major position back into per-axis
// coordinates.
static FailureOr<SmallVector<sym::ExprHandle>>
composeInnerProducts(sym::Store &store, ArrayRef<ExprAttr> extents) {
  size_t rank = extents.size();
  SmallVector<sym::ExprHandle> innerProds(rank);
  auto one = sym::composeExprInt(store, 1);
  if (failed(one))
    return failure();
  innerProds[rank - 1] = *one;
  for (size_t k = rank - 1; k > 0; --k) {
    auto next = sym::composeExprBinary(store, extents[k].getValue(),
                                       sym::ExprBinaryOp::Mul, innerProds[k]);
    if (failed(next))
      return failure();
    innerProds[k - 1] = *next;
  }
  return innerProds;
}

// One axis of the layout-driven decomposition: `base + Mod(floor(flat
// / inner_prod), extent)`. The last axis (`isLast=true`) skips the
// `/inner_prod` since `inner_prod_{N-1} = 1` would print as a
// redundant `/1`. The floor wrapper is mandatory on the non-last
// axes — `composeExprBinary(Div)` is exact-rational; without the
// floor we'd carry `1/16*X` through every later use and downstream
// ixsimpl never recovers an integer index.
static FailureOr<ExprAttr>
composeDecomposedAxis(MLIRContext *ctx, sym::Store &store, sym::ExprHandle flat,
                      sym::ExprHandle innerProd, ExprAttr extent, ExprAttr base,
                      bool isLast) {
  sym::ExprHandle divFlat = flat;
  if (!isLast) {
    auto d =
        sym::composeExprBinary(store, flat, sym::ExprBinaryOp::Div, innerProd);
    if (failed(d))
      return failure();
    auto floored = sym::composeExprFloor(store, *d);
    if (failed(floored))
      return failure();
    divFlat = *floored;
  }
  auto m = sym::composeExprBinary(store, divFlat, sym::ExprBinaryOp::Mod,
                                  extent.getValue());
  if (failed(m))
    return failure();
  auto s = sym::composeExprBinary(store, base.getValue(),
                                  sym::ExprBinaryOp::Add, *m);
  if (failed(s))
    return failure();
  return ExprAttr::get(ctx, *s);
}

// Compose source-side per-axis offsets via the result type's layout
// when the source is a multi-dim slice whose intent shape product
// matches the layout's `storage_size`. The result type's layout maps
// `index_syms` -> a single flat tile-storage offset; substituting
// iter syms (`i_0`, `i_1`, ...) for `index_syms` and `tileShape` for
// `shape_syms` produces an expression in iter syms that gives the
// per-iter-point flat position into the slice's row-major flatten.
// That flat position decomposes back into source-axis coordinates:
//
//   axis_k = base_k + (flat / inner_prod_k) % extent_k
//
// where `inner_prod_k = product(extent[k+1..N-1])` and
// `inner_prod_{N-1} = 1`. The per-axis `% extent_k` keeps the
// decomposition well-defined when ixsimpl can't prove the upper
// bound; the simplifier folds it when bounds are statically tight.
//
// Bails when:
//   * any iter sym fails to compose (rank/parity mismatch with the
//     layout's `index_syms`),
//   * the layout's `storage_size` doesn't structurally equal the
//     slice extent product (with a wider layout, `% extent` would
//     alias OOB lanes onto valid cells; with a narrower layout some
//     positions stop short, both of which need explicit masking the
//     generic surface doesn't carry on this slot).
//
// Caller has already confirmed every index is a pinned slice and
// `pf.sliceExtents` rank-matches `pf.axes`.
static FailureOr<ArrayAttr> composeLayoutGatherSourceOffsets(
    MLIRContext *ctx, sym::Store &store, LayoutAttr layout,
    ArrayRef<ExprAttr> tileShape, ArrayRef<StringAttr> iterSyms,
    ArrayRef<AxisIndex> axes, ArrayRef<ExprAttr> sliceExtents) {
  size_t rank = axes.size();
  if (rank == 0 || sliceExtents.size() != rank)
    return failure();

  auto productExtents = composeProductOfExtents(store, sliceExtents);
  if (failed(productExtents))
    return failure();
  if (!(*productExtents == layout.getStorageSize().getValue()))
    return failure();

  auto flat = composeLayoutFlatOffset(ctx, store, layout, tileShape, iterSyms);
  if (failed(flat))
    return failure();
  auto innerProds = composeInnerProducts(store, sliceExtents);
  if (failed(innerProds))
    return failure();

  SmallVector<Attribute> axisOffsets;
  axisOffsets.reserve(rank);
  for (size_t k = 0; k < rank; ++k) {
    auto axis = composeDecomposedAxis(ctx, store, *flat, (*innerProds)[k],
                                      sliceExtents[k], axes[k].base,
                                      /*isLast=*/k + 1 == rank);
    if (failed(axis))
      return failure();
    axisOffsets.push_back(*axis);
  }
  return ArrayAttr::get(ctx, axisOffsets);
}

// Build the per-position binding expression list for an access into a
// layout-bearing buffer. Pinned `!hc.idx<expr>` positions bind their
// idx's expression directly; slice positions bind `lo + step *
// iter_sym`, consuming `iterSyms` in order (one per slice axis,
// matching the result tile rank). The access is expected to carry
// exactly `iterSyms.size()` slice positions — the preflight pinned-
// vs-slice classification has already enforced that.
static FailureOr<SmallVector<ExprAttr>>
composeLayoutBearingAccessBindings(MLIRContext *ctx, sym::Store &store,
                                   ValueRange indices, ArrayRef<AxisIndex> axes,
                                   ArrayRef<StringAttr> iterSyms) {
  SmallVector<ExprAttr> bindings;
  bindings.reserve(indices.size());
  size_t sliceSeen = 0;
  for (auto [i, idx] : llvm::enumerate(indices)) {
    Type ty = idx.getType();
    if (llvm::isa<IdxType>(ty)) {
      bindings.push_back(axes[i].base);
      continue;
    }
    if (!llvm::isa<SliceType>(ty))
      return failure();
    if (sliceSeen >= iterSyms.size())
      return failure();
    auto binding =
        composeBasePlusStepIter(ctx, store, axes[i], iterSyms[sliceSeen]);
    if (failed(binding))
      return failure();
    bindings.push_back(*binding);
    ++sliceSeen;
  }
  if (sliceSeen != iterSyms.size())
    return failure();
  return bindings;
}

// Compose the source-side per-axis offsets for an access whose source
// is a layout-bearing buffer (the as_layout result peeled to its
// underlying). The layout's `offset` is composed with the declared
// shape (from the as_layout's `shape=` operand) and per-position
// bindings; the resulting flat offset decomposes against the
// underlying's shape via the same row-major split that the result-
// layout gather uses, with per-axis base 0 (the buffer-view chain
// downstream of the peeled source carries any tile origin).
//
// The structural invariant binding this to the verifier is:
//
//   storage_size(LAY) == product(underlying.shape)
//
// The `hc.as_layout` verifier already enforces this on the op (the
// layout's storage_size matches the operand's flat capacity), so the
// rewrite can rely on the row-major decomposition closing without a
// secondary `% extent` guard.
//
// Caller has already validated:
//   * iter syms count == result tile rank == slice axis count
//   * layout.index_syms count == access indices count
//   * layout.shape_syms count == declared shape rank
static FailureOr<ArrayAttr> composeLayoutBearingBufferOffsets(
    MLIRContext *ctx, sym::Store &store, LayoutAttr layout,
    ShapeAttr declaredShape, ValueRange indices, ArrayRef<AxisIndex> axes,
    ArrayRef<StringAttr> iterSyms, ArrayRef<ExprAttr> underlyingShape) {
  auto bindingsOr =
      composeLayoutBearingAccessBindings(ctx, store, indices, axes, iterSyms);
  if (failed(bindingsOr))
    return failure();
  auto flatOr =
      composeAccessOffsetExpr(ctx, layout, declaredShape, *bindingsOr);
  if (failed(flatOr))
    return failure();
  sym::ExprHandle flat = flatOr->getValue();

  size_t rank = underlyingShape.size();
  if (rank == 0)
    return failure();
  auto innerProds = composeInnerProducts(store, underlyingShape);
  if (failed(innerProds))
    return failure();
  auto zero = sym::composeExprInt(store, 0);
  if (failed(zero))
    return failure();
  ExprAttr base = ExprAttr::get(ctx, *zero);

  SmallVector<Attribute> axisOffsets;
  axisOffsets.reserve(rank);
  for (size_t k = 0; k < rank; ++k) {
    auto axis = composeDecomposedAxis(ctx, store, flat, (*innerProds)[k],
                                      underlyingShape[k], base,
                                      /*isLast=*/k + 1 == rank);
    if (failed(axis))
      return failure();
    axisOffsets.push_back(*axis);
  }
  return ArrayAttr::get(ctx, axisOffsets);
}

// Layout-bearing buffer source preflight. Captures the peeled
// underlying as `effectiveSource`, validates rank parity between
// access indices and the layout's `index_syms`, between the declared
// shape and `shape_syms`, and between slice positions and the result
// tile rank.
static FailureOr<LoadPreflight> preflightLoadLikeLayoutBearing(
    MLIRContext *ctx, sym::Store &store, const LayoutBearingSource &peel,
    Type resultTy, ValueRange indices, ArrayRef<ExprAttr> tileShape) {
  if (indices.size() != peel.layout.getIndexSyms().size())
    return failure();
  if (peel.declaredShape.getDims().size() != peel.layout.getShapeSyms().size())
    return failure();
  auto axesOr = collectAxisIndices(ctx, store, indices);
  if (failed(axesOr))
    return failure();
  size_t sliceCount = 0;
  for (Value idx : indices)
    if (llvm::isa<SliceType>(idx.getType()))
      ++sliceCount;
  if (sliceCount != tileShape.size())
    return failure();
  if (failed(checkLoadElementTypesMatch(peel.underlying.getType(), resultTy)))
    return failure();
  auto srcShape = getOperandShape(peel.underlying.getType());
  if (failed(srcShape))
    return failure();
  LoadPreflight pf;
  pf.axes = std::move(*axesOr);
  pf.srcShape = std::move(*srcShape);
  pf.effectiveSource = peel.underlying;
  pf.sourceLayout = peel.layout;
  pf.sourceDeclaredShape = peel.declaredShape;
  return pf;
}

// Plain (non-layout-bearing) load preflight. Same shape as the
// pre-as_layout path: rank-parity gate, per-axis index harvest,
// element-type compatibility, source-shape extraction, then the
// result-layout / broadcast classification that picks the gather
// decomposition vs identity-offset path downstream. Result-layout
// slice extents are harvested up front so the gather composer can
// translate the layout's flat offset back into per-axis source
// coords.
// Pad partial indices with default full-slice axes (`base=0,
// step=1`) so the per-axis offset composer sees one axis per tile
// dim. Only pads when the source's rank matches the tile rank — the
// rank-1 broadcast path keeps the empty-axes form for the layout-
// driven gather to pick up. A no-op for empty or already-rank-equal
// axis lists.
static LogicalResult
padTrailingFullSliceAxes(MLIRContext *ctx, sym::Store &store, size_t targetRank,
                         size_t srcRank, SmallVectorImpl<AxisIndex> &axes) {
  if (axes.empty() || axes.size() >= targetRank || srcRank != targetRank)
    return success();
  auto zero = sym::composeExprInt(store, 0);
  auto one = sym::composeExprInt(store, 1);
  if (failed(zero) || failed(one))
    return failure();
  ExprAttr zeroAttr = ExprAttr::get(ctx, *zero);
  ExprAttr oneAttr = ExprAttr::get(ctx, *one);
  while (axes.size() < targetRank)
    axes.push_back(AxisIndex{zeroAttr, oneAttr});
  return success();
}

// Slice-extents are only meaningful on the layout-driven gather path
// (`resultLayout != null && !broadcastFromRank1`); other paths return
// an empty list. Extraction failure on the gather path is fine —
// `composeLoadInsOffsets` falls back to the per-axis identity offsets
// when the layout-driven gather doesn't fit the structural shape.
static SmallVector<ExprAttr> harvestLayoutSliceExtents(MLIRContext *ctx,
                                                       sym::Store &store,
                                                       LayoutAttr resultLayout,
                                                       bool broadcastFromRank1,
                                                       ValueRange indices) {
  if (!resultLayout || broadcastFromRank1)
    return {};
  auto extents = collectSliceExtents(ctx, store, indices);
  if (failed(extents))
    return {};
  return std::move(*extents);
}

static FailureOr<LoadPreflight>
preflightLoadLikePlain(MLIRContext *ctx, sym::Store &store, Value source,
                       Type resultTy, ValueRange indices,
                       ArrayRef<ExprAttr> tileShape) {
  // Empty index list is a legal shape (`hc.load %t[], shape ...`): the
  // access addresses the operand at the tile origin, which is just the
  // iter syms with no addressing addend. Non-empty lists may match
  // the tile rank exactly (the canonical fully-indexed form) or fall
  // short of it (NumPy `X[gid[0]:]` against a rank-2 `X` — trailing
  // axes are implicit-full slices). Over-indexing is inconsistent IR.
  if (!indices.empty() && indices.size() > tileShape.size())
    return failure();
  auto axesOr = collectAxisIndices(ctx, store, indices);
  if (failed(axesOr))
    return failure();

  if (failed(checkLoadElementTypesMatch(source.getType(), resultTy)))
    return failure();

  auto srcShape = getOperandShape(source.getType());
  if (failed(srcShape))
    return failure();

  if (failed(padTrailingFullSliceAxes(ctx, store, tileShape.size(),
                                      srcShape->size(), *axesOr)))
    return failure();

  bool broadcastFromRank1 = false;
  LayoutAttr resultLayout;
  if (failed(classifyLoadAccess(resultTy, indices, *srcShape, tileShape,
                                broadcastFromRank1, resultLayout)))
    return failure();

  SmallVector<ExprAttr> sliceExtents = harvestLayoutSliceExtents(
      ctx, store, resultLayout, broadcastFromRank1, indices);

  return LoadPreflight{std::move(*axesOr), std::move(*srcShape), resultLayout,
                       broadcastFromRank1, std::move(sliceExtents)};
}

// Top-level load preflight. Layout-bearing buffer source
// (`hc.as_layout` with `shape=` operand) gets its own path because
// the access's index list rank-matches the layout's `index_syms`,
// not the underlying buffer's shape — running it through the plain
// path's rank-parity gate would reject the mixed pinned + slice
// ergonomic form outright.
static FailureOr<LoadPreflight>
preflightLoadLike(MLIRContext *ctx, sym::Store &store, Value source,
                  Type resultTy, ValueRange indices,
                  ArrayRef<ExprAttr> tileShape) {
  if (auto peel = peelLayoutBearingBufferSource(source))
    return preflightLoadLikeLayoutBearing(ctx, store, *peel, resultTy, indices,
                                          tileShape);
  return preflightLoadLikePlain(ctx, store, source, resultTy, indices,
                                tileShape);
}

// Pick the right ins_offsets[0] for a load:
//   * Layout-bearing buffer source (source is `hc.as_layout` with
//     `shape=` operand): per-position bindings (pinned-idx expr for
//     pinned, `lo + step*iter` for slice) compose through the
//     source's layout, then decompose against the peeled underlying's
//     shape.
//   * Rank-1 broadcast (source rank < tile rank, layout-driven
//     collapse onto rank-1 storage): single flat offset.
//   * Layout-driven gather (result has a non-default layout, all
//     indices are pinned slices with extents flattening to the
//     layout's `storage_size`): per-axis row-major decomposition of
//     the layout's flat offset.
//   * Otherwise: per-axis `base + step*iter` identity offsets.
static FailureOr<ArrayAttr>
composeLoadInsOffsets(MLIRContext *ctx, sym::Store &store,
                      const LoadPreflight &pf, ValueRange indices,
                      ArrayRef<ExprAttr> tileShape,
                      ArrayRef<StringAttr> iterSyms) {
  if (pf.sourceLayout)
    return composeLayoutBearingBufferOffsets(ctx, store, pf.sourceLayout,
                                             pf.sourceDeclaredShape, indices,
                                             pf.axes, iterSyms, pf.srcShape);
  if (pf.broadcastFromRank1)
    return composeBroadcastSourceOffset(ctx, pf.resultLayout, tileShape,
                                        iterSyms, pf.srcShape);
  if (pf.resultLayout && !pf.sliceExtents.empty()) {
    auto layoutDriven =
        composeLayoutGatherSourceOffsets(ctx, store, pf.resultLayout, tileShape,
                                         iterSyms, pf.axes, pf.sliceExtents);
    if (succeeded(layoutDriven))
      return *layoutDriven;
    // Fall through: the identity offsets are correct when the layout
    // is structurally identity over the tile shape; when the layout
    // is genuinely non-identity and we couldn't decompose, the
    // identity fallback is wrong but matches the pre-fix behaviour
    // and downstream verifiers diagnose the storage-size mismatch.
  }
  return composeMemoryOffsetArray(ctx, store, pf.axes, iterSyms);
}

// Emit a fresh block carrying one src arg and one dst arg, terminated
// with `hc.yield %src` — the trivial "copy element through" body the
// load rewrites all share.
static void populateLoadBody(HCGenericOp generic, Type srcElem, Type resElem,
                             Location loc) {
  Block *body = new Block();
  BlockArgument bv = body->addArgument(srcElem, loc);
  body->addArgument(resElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  HCYieldOp::create(bodyBuilder, loc, ValueRange{bv});
}

// Common load rewriter for both `hc.load` and `hc.vload`. The two ops
// have identical operand layouts (source, indices, shape) and the
// only difference is which value-init op the result type wants.
template <typename OpT>
static LogicalResult rewriteLoadLike(OpT op, sym::Store &store) {
  Type resultTy = op.getResult().getType();
  auto tileShape = getOperandShape(resultTy);
  if (failed(tileShape))
    return failure();

  // Source operand is `getBuffer` on `hc.load` and `getSource` on
  // `hc.vload`; the type is what tells us the body element type.
  Value source;
  if constexpr (std::is_same_v<OpT, HCLoadOp>)
    source = op.getBuffer();
  else
    source = op.getSource();

  MLIRContext *ctx = op.getContext();
  auto pf = preflightLoadLike(ctx, store, source, resultTy, op.getIndices(),
                              *tileShape);
  if (failed(pf))
    return failure();
  // Layout-bearing buffer source path peels `hc.as_layout` and routes
  // through the underlying; everything downstream (body element type,
  // generic op operand) keys off the effective source.
  Value effSource = pf->effectiveSource ? pf->effectiveSource : source;

  Location loc = op.getLoc();
  OpBuilder builder(op);
  CommonRewriteData common = buildCommon(builder, loc, *tileShape);
  Value shapeTuple = buildShapeTuple(builder, loc, common.iterBounds);
  Value initOut = emitValueInit(builder, loc, resultTy, shapeTuple);

  auto inOff = composeLoadInsOffsets(ctx, store, *pf, op.getIndices(),
                                     *tileShape, common.iterSyms);
  if (failed(inOff))
    return failure();
  ArrayAttr outOff = offsetArrayFromIterSyms(ctx, store, common.iterSyms);
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {*inOff});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  SmallVector<Value> insArr{effSource};
  SmallVector<Value> outsArr{initOut};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{resultTy}, common.iterSymsAttr,
      ValueRange(common.iterBounds), common.iterKindsAttr, ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Type srcElem = bodyArgElementType(effSource.getType());
  Type resElem = bodyArgElementType(resultTy);
  populateLoadBody(generic, srcElem, resElem, loc);

  op->replaceAllUsesWith(generic.getResults());
  op->erase();
  return success();
}

// ----- hc.load_mask -----------------------------------------------------

// Mask companion for `hc.load` / `hc.vload`: produces a bare predicate
// carrier whose lane `i_k` says whether the source's per-axis slice
// subscript `(lo_k + step_k * i_k)` stays in-bounds against the source's
// k-th dim `D_k`. Pre-flatten the source still carries its multi-dim
// shape, so `D_k` is the source operand's k-th symbolic dim — a sym
// leaf for kernel-arg buffers (`"A"`, `"B"`, ...) or an integer literal
// for static bare tensors.
//
// Rewrite shape:
//
//   %m = hc.generic
//       iter (parallel i_0 = %S_0, parallel i_1 = %S_1, ...)
//       outs (%init at [#hc.expr<"i_0">, #hc.expr<"i_1">, ...]
//             : !hc.bare_(tensor|vector)<!hc.pred, ...>)
//       -> (!hc.bare_(tensor|vector)<!hc.pred, ...>) {
//   ^bb0(%iv: !hc.pred):
//     %p_pinned = hc.pred_apply ()
//                 : () -> !hc.pred<"(lo_0 + step_0*i_0 < D_0)
//                                && (lo_1 + step_1*i_1 < D_1) && ...">
//     %p = builtin.unrealized_conversion_cast %p_pinned : ... to !hc.pred
//     hc.yield %p : !hc.pred
//   }
//
// The pinned-pred → unpinned-pred UCC is the same bridge
// `hc.yield_predicated`'s consumers use for body-computed masks; the
// downstream `bindIterSymsInClone` (in `hc-lower-generic`'s value-outs
// path) binds each iter sym to its per-lane compile-time integer on
// the planted apply, so the launch-body resolver sees a direct
// dataflow edge instead of falling through ambient context. The
// hash-consed structural conjunction shares storage with any other
// identical bound predicate elsewhere in the IR.
//
// Bails (op stays for legacy lowering / diagnostics):
//   * Non-shaped source / non-shaped result types (`!hc.undef`, raw
//     pointers, ...) — no shape to source bounds from.
//   * Source axis carrying `#hc.dyn` instead of `#hc.expr` — host-owned
//     size, no in-IR sym to bound against.
//   * Index operand mismatch (rank, non-canonical slice with `!hc.undef`
//     lower / step parts, scalar idx with no expression) — same shapes
//     the data-side `rewriteLoadLike` punts on.
//   * Rank-0 mask (no slice axes) — pathological shape that should
//     have been folded earlier; if it ever lands, `hc-full-mask` is
//     the right primitive.
// Per-axis (lo, step, src-dim) carriers, slice axes only — scalar
// idx axes drop out of the result rank by construction so they
// contribute no iter dimension and no bounds term.
struct SliceAxisInfo {
  ExprAttr lo;
  ExprAttr step;
  ExprAttr srcDim;
};

// Collect the slice-only axes from `hc.load_mask`'s parallel
// (axis, isSlice) arrays. Caller is responsible for shape-padding
// partial indices into the rank-matched form (synthetic full-slice
// axes have `isSlice=true`). Returns failure when the resulting
// slice-axis count doesn't equal the result tile rank, or when
// there are no slice axes at all (rank-0 mask is a pathological
// shape — `hc-full-mask` is the right primitive).
static FailureOr<SmallVector<SliceAxisInfo>>
collectMaskSliceAxes(sym::Store &store, ArrayRef<AxisIndex> axes,
                     ArrayRef<bool> isSliceAxis, ArrayRef<ExprAttr> srcShape,
                     ArrayRef<ExprAttr> tileShape) {
  if (axes.size() != srcShape.size() || axes.size() != isSliceAxis.size())
    return failure();
  SmallVector<SliceAxisInfo> sliceAxes;
  for (auto [k, axis] : llvm::enumerate(axes)) {
    if (!isSliceAxis[k])
      continue;
    sliceAxes.push_back({axis.base, axis.step, srcShape[k]});
  }
  if (sliceAxes.size() != tileShape.size() || sliceAxes.empty())
    return failure();
  return sliceAxes;
}

// Compose the structural per-axis comparison `lo + step * i_k < D_k`
// for one slice axis. Step==1 is folded so the printed cmp matches the
// scalar-idx caller form.
static FailureOr<sym::PredHandle>
composeSliceAxisBound(sym::Store &store, StringAttr iterSym,
                      const SliceAxisInfo &info) {
  auto iterHandle = sym::composeExprSym(store, iterSym.getValue());
  if (failed(iterHandle))
    return failure();
  sym::ExprHandle term = *iterHandle;
  std::optional<int64_t> stepLit =
      sym::getIntegerLiteralValue(info.step.getValue());
  if (!stepLit || *stepLit != 1) {
    auto mul = sym::composeExprBinary(store, info.step.getValue(),
                                      sym::ExprBinaryOp::Mul, term);
    if (failed(mul))
      return failure();
    term = *mul;
  }
  auto sum = sym::composeExprBinary(store, info.lo.getValue(),
                                    sym::ExprBinaryOp::Add, term);
  if (failed(sum))
    return failure();
  return sym::composePredCmp(store, *sum, sym::PredCmpOp::Lt,
                             info.srcDim.getValue());
}

// Build the conjunction of per-axis bounds across every slice axis.
// Hash-consing shares identical bounds expressions with any other
// producer, so two load_masks reading the same buffer with the same
// slice geometry emit one pred_apply each pointing at the same
// canonical node. Empty `sliceAxes` returns failure rather than the
// rank-0 conjunction — `hc-full-mask` is the right primitive for
// that pathological shape, and an empty fold here would leave the
// outer `composeMaskConjunctionForResult` deref-on-empty-optional.
static FailureOr<sym::PredHandle>
composeMaskConjunction(sym::Store &store, ArrayRef<SliceAxisInfo> sliceAxes,
                       ArrayRef<StringAttr> iterSyms) {
  if (sliceAxes.empty())
    return failure();
  std::optional<sym::PredHandle> conjunction;
  for (auto [k, info] : llvm::enumerate(sliceAxes)) {
    auto cmp = composeSliceAxisBound(store, iterSyms[k], info);
    if (failed(cmp))
      return failure();
    if (!conjunction) {
      conjunction = *cmp;
      continue;
    }
    auto andP = sym::composePredAnd(store, *conjunction, *cmp);
    if (failed(andP))
      return failure();
    conjunction = *andP;
  }
  return *conjunction;
}

// Emit the predicate body of a `hc.load_mask` rewrite: a `hc.pred_apply`
// pinned with `conjunction`, bridged through a `unrealized_conversion_cast`
// to the unpinned `!hc.pred` element type, yielded.
static void populateMaskBody(HCGenericOp generic, MLIRContext *ctx,
                             Location loc, Type predElem,
                             sym::PredHandle conjunction) {
  Block *body = new Block();
  body->addArgument(predElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());

  PredAttr predAttr = PredAttr::get(ctx, conjunction);
  Type pinnedTy = PredType::get(ctx, predAttr);
  Value predPinned = HCPredApplyOp::create(bodyBuilder, loc, pinnedTy,
                                           /*operands=*/ValueRange{},
                                           bodyBuilder.getStrArrayAttr({}))
                         .getResult();
  Value predUnpinned =
      UnrealizedConversionCastOp::create(bodyBuilder, loc, predElem, predPinned)
          .getResult(0);
  HCYieldOp::create(bodyBuilder, loc, ValueRange{predUnpinned});
}

// Turn a list of per-axis source offsets into the in-bounds
// conjunction `offset_k < srcDim_k AND ...`. Shared between the
// result-layout gather mask and the source-layout (layout-bearing
// buffer) mask: both decompose the layout's flat offset into source
// coords first, then ask the same per-axis bound.
static FailureOr<sym::PredHandle>
maskConjunctionFromOffsets(sym::Store &store, ArrayAttr offsets,
                           ArrayRef<ExprAttr> srcShape) {
  if (offsets.size() != srcShape.size())
    return failure();
  std::optional<sym::PredHandle> conjunction;
  for (auto [offsetAttr, srcDim] : llvm::zip_equal(offsets, srcShape)) {
    auto offsetExpr = dyn_cast<ExprAttr>(offsetAttr);
    if (!offsetExpr)
      return failure();
    auto cmp = sym::composePredCmp(store, offsetExpr.getValue(),
                                   sym::PredCmpOp::Lt, srcDim.getValue());
    if (failed(cmp))
      return failure();
    if (!conjunction) {
      conjunction = *cmp;
      continue;
    }
    auto andP = sym::composePredAnd(store, *conjunction, *cmp);
    if (failed(andP))
      return failure();
    conjunction = *andP;
  }
  if (!conjunction)
    return failure();
  return *conjunction;
}

// Compose the layout-driven mask conjunction: for every axis the
// per-element bound is `decomposed_offset_k < srcDim_k`, where the
// decomposed offset is the same row-major-flatten of the layout's
// `composeAccessOffsetExpr` that the data-side gather uses. This
// keeps the predicate's "in-bounds" notion structurally aligned with
// the data access — without it, the mask says "in-bounds" for tile
// positions whose layout-driven source coordinates actually run off
// the slice (e.g. lanes 16..31 in a 16x16 WMMA tile under
// `WAVE_ACC_FRAG_LAYOUT`).
//
// Caller has already confirmed slice extents flatten to the layout's
// `storage_size` and built the per-axis `AxisIndex` carriers.
static FailureOr<sym::PredHandle> composeLayoutGatherMaskConjunction(
    MLIRContext *ctx, sym::Store &store, LayoutAttr layout,
    ArrayRef<ExprAttr> tileShape, ArrayRef<StringAttr> iterSyms,
    ArrayRef<AxisIndex> axes, ArrayRef<ExprAttr> sliceExtents,
    ArrayRef<ExprAttr> srcShape) {
  auto offsetsOr = composeLayoutGatherSourceOffsets(
      ctx, store, layout, tileShape, iterSyms, axes, sliceExtents);
  if (failed(offsetsOr))
    return failure();
  return maskConjunctionFromOffsets(store, *offsetsOr, srcShape);
}

// Mask companion of the layout-bearing buffer load path. Same
// decomposition (`LAY.offset` composed with per-position bindings,
// then row-major split against the peeled underlying's shape) drives
// both the data offsets and the mask's in-bounds bound; without the
// per-axis bound the predicate would flag in-bounds lanes whose
// decomposed source coords actually run past the underlying tile.
static FailureOr<sym::PredHandle> composeLayoutBearingMaskConjunction(
    MLIRContext *ctx, sym::Store &store, LayoutAttr layout,
    ShapeAttr declaredShape, ValueRange indices, ArrayRef<AxisIndex> axes,
    ArrayRef<StringAttr> iterSyms, ArrayRef<ExprAttr> underlyingShape) {
  auto offsetsOr = composeLayoutBearingBufferOffsets(
      ctx, store, layout, declaredShape, indices, axes, iterSyms,
      underlyingShape);
  if (failed(offsetsOr))
    return failure();
  return maskConjunctionFromOffsets(store, *offsetsOr, underlyingShape);
}

// Pick the right mask conjunction for an `hc.load_mask`. Three
// shapes, all driven by the same in-bounds-per-source-axis principle
// the matching data-side load uses, so masked-in lanes line up with
// the data path's actual addressing:
//   * Layout-bearing buffer source (`hc.as_layout` source with
//     `shape=` operand) — the per-position bindings compose through
//     the source layout and decompose against the peeled underlying.
//   * Result-layout gather (layout on the mask result type, all
//     pinned slices) — same decomposition through the result layout
//     and slice extents.
//   * Plain per-axis bound — `lo + step*iter < srcDim`.
static FailureOr<sym::PredHandle> composeMaskConjunctionForResult(
    MLIRContext *ctx, sym::Store &store, Value source, Type resultTy,
    ValueRange indices, ArrayRef<ExprAttr> srcShape,
    ArrayRef<ExprAttr> tileShape, ArrayRef<StringAttr> iterSyms,
    ArrayRef<AxisIndex> axes, ArrayRef<SliceAxisInfo> sliceAxes) {
  if (auto peel = peelLayoutBearingBufferSource(source)) {
    auto conjunction = composeLayoutBearingMaskConjunction(
        ctx, store, peel->layout, peel->declaredShape, indices, axes, iterSyms,
        srcShape);
    if (succeeded(conjunction))
      return *conjunction;
  }
  auto resultShaped = dyn_cast<SymbolicallyShapedTypeInterface>(resultTy);
  LayoutAttr resultLayout =
      resultShaped ? resultShaped.getSymbolicLayout() : LayoutAttr{};
  if (resultLayout) {
    auto extents = collectSliceExtents(ctx, store, indices);
    if (succeeded(extents) && axes.size() == extents->size()) {
      auto conjunction = composeLayoutGatherMaskConjunction(
          ctx, store, resultLayout, tileShape, iterSyms, axes, *extents,
          srcShape);
      if (succeeded(conjunction))
        return *conjunction;
    }
  }
  return composeMaskConjunction(store, sliceAxes, iterSyms);
}

// Bail-out checks for a `hc.load_mask` rewrite. Two flavours of source:
//   * Layout-bearing buffer source (`hc.as_layout` with `shape=`) —
//     `srcShape` is the peeled underlying's shape; indices rank-
//     match the layout's `index_syms`, not the underlying source.
//   * Plain shaped source — `srcShape` is the source's shape and
//     indices must rank-match.
struct MaskPreflight {
  SmallVector<ExprAttr> srcShape;
  SmallVector<AxisIndex> axes;
  // Slice-only axes per `collectMaskSliceAxes`. Empty when the access
  // doesn't fit the rank-matched assumption — only legal for the
  // layout-bearing path, which uses `axes` and the captured layout
  // instead.
  SmallVector<SliceAxisInfo> sliceAxes;
};

// Resolve the mask source's effective shape and whether it's
// layout-bearing. `srcShape` is the peeled underlying's shape for
// layout-bearing sources (`hc.as_layout` with `shape=`) and the
// source's own shape otherwise. Over-indexing the plain shape
// remains inconsistent IR.
static FailureOr<bool>
resolveMaskSourceShape(Value source, ValueRange indices,
                       SmallVectorImpl<ExprAttr> &srcShape) {
  if (auto peel = peelLayoutBearingBufferSource(source)) {
    auto shapeOr = getOperandShape(peel->underlying.getType());
    if (failed(shapeOr))
      return failure();
    srcShape = std::move(*shapeOr);
    return true;
  }
  auto shapeOr = getOperandShape(source.getType());
  if (failed(shapeOr))
    return failure();
  srcShape = std::move(*shapeOr);
  if (indices.size() > srcShape.size())
    return failure();
  return false;
}

// Build the parallel (axis, isSlice) arrays for `hc.load_mask`'s
// indices, padding trailing axes with synthetic full-slice entries
// when the index list is rank-short. Partial-index form
// (`X[gid[0]:]` against rank-2 X): the trailing axes are implicit-
// full slices, contributing a structurally trivial `0 + 1*iter <
// srcDim` bound. Layout-bearing path skips padding — the layout's
// own `index_syms` drive that flow.
static LogicalResult buildMaskAxes(MLIRContext *ctx, sym::Store &store,
                                   ValueRange indices, size_t srcRank,
                                   bool layoutBearing,
                                   SmallVectorImpl<AxisIndex> &axes,
                                   SmallVectorImpl<bool> &isSliceAxis) {
  auto axesOr = collectAxisIndices(ctx, store, indices);
  if (failed(axesOr))
    return failure();
  axes = std::move(*axesOr);
  isSliceAxis.reserve(srcRank);
  for (Value idx : indices)
    isSliceAxis.push_back(isa<SliceType>(idx.getType()));
  if (layoutBearing)
    return success();
  if (failed(padTrailingFullSliceAxes(ctx, store, srcRank, srcRank, axes)))
    return failure();
  while (isSliceAxis.size() < axes.size())
    isSliceAxis.push_back(true);
  return success();
}

static FailureOr<MaskPreflight>
preflightLoadMask(MLIRContext *ctx, sym::Store &store, Value source,
                  ValueRange indices, ArrayRef<ExprAttr> tileShape) {
  MaskPreflight pf;
  auto layoutBearing = resolveMaskSourceShape(source, indices, pf.srcShape);
  if (failed(layoutBearing))
    return failure();
  SmallVector<bool> isSliceAxis;
  if (failed(buildMaskAxes(ctx, store, indices, pf.srcShape.size(),
                           *layoutBearing, pf.axes, isSliceAxis)))
    return failure();
  // `collectMaskSliceAxes` is only well-defined on the rank-matched
  // pure-slice form; let it fail silently when the layout-bearing
  // path needs the per-position bindings instead.
  if (pf.axes.size() == pf.srcShape.size()) {
    auto sliceAxesOr = collectMaskSliceAxes(store, pf.axes, isSliceAxis,
                                            pf.srcShape, tileShape);
    if (succeeded(sliceAxesOr))
      pf.sliceAxes = std::move(*sliceAxesOr);
  }
  return pf;
}

static LogicalResult rewriteLoadMask(HCLoadMaskOp op, sym::Store &store) {
  Type resultTy = op.getMask().getType();
  auto tileShape = getOperandShape(resultTy);
  if (failed(tileShape))
    return failure();

  Value source = op.getSource();
  ValueRange indices = op.getIndices();
  MLIRContext *ctx = op.getContext();

  auto pf = preflightLoadMask(ctx, store, source, indices, *tileShape);
  if (failed(pf))
    return failure();

  // Iter sym names ahead of `buildCommon` — the conjunction composer
  // needs them, and `buildCommon` mints SSA `idx_apply` / `tuple`
  // ops we don't want orphaned if the conjunction picker bails. The
  // names depend only on `tileShape.size()` (canonical `i_0`,
  // `i_1`, ...), so generating them out-of-band stays in lockstep
  // with the names `buildCommon` would mint.
  SmallVector<StringAttr> iterSymNames;
  iterSymNames.reserve(tileShape->size());
  for (size_t k = 0; k < tileShape->size(); ++k)
    iterSymNames.push_back(StringAttr::get(ctx, ("i_" + Twine(k)).str()));

  // Conjunction picker routes through the source-layout / result-
  // layout / per-axis paths; failure here means none of the three
  // could fit the access shape and the op stays for the legacy
  // launch-body handler to pick up.
  auto conjunction = composeMaskConjunctionForResult(
      ctx, store, source, resultTy, indices, pf->srcShape, *tileShape,
      iterSymNames, pf->axes, pf->sliceAxes);
  if (failed(conjunction))
    return failure();

  Location loc = op.getLoc();
  OpBuilder builder(op);
  CommonRewriteData common = buildCommon(builder, loc, *tileShape);
  Value shapeTuple = buildShapeTuple(builder, loc, common.iterBounds);
  Value initOut = emitValueInit(builder, loc, resultTy, shapeTuple);

  ArrayAttr outOff = offsetArrayFromIterSyms(ctx, store, common.iterSyms);
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{resultTy}, common.iterSymsAttr,
      ValueRange(common.iterBounds), common.iterKindsAttr,
      /*ins=*/ValueRange{}, /*outs=*/ValueRange{initOut},
      /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Type predElem = getUnpinnedPredType(ctx);
  populateMaskBody(generic, ctx, loc, predElem, *conjunction);

  op->replaceAllUsesWith(generic.getResults());
  op->erase();
  return success();
}

// ----- hc.store ---------------------------------------------------------

// Validated preflight state for the store rewriter.
struct StorePreflight {
  SmallVector<ExprAttr> tileShape;
  SmallVector<AxisIndex> axes;
  Type srcElem;
  Type dstElem;
  Type maskElem; // null when there's no mask
  // Layout-bearing buffer dest path: when `op.getDest()` is defined
  // by `hc.as_layout` with both a layout attribute and a `shape=`
  // operand, peel it. `effectiveDest` is the underlying buffer,
  // `dstShape` is its shape, and `destLayout` / `destDeclaredShape`
  // drive the per-position scatter decomposition. Empty
  // `effectiveDest` means no peel happened.
  Value effectiveDest;
  SmallVector<ExprAttr> dstShape;
  LayoutAttr destLayout;
  ShapeAttr destDeclaredShape;
};

// Element-type match between src and dst, plus mask element type
// extraction. Returns failure on mismatch or on a mask whose tile rank
// doesn't match `tileShape`.
static LogicalResult collectStoreElementTypes(HCStoreOp op, Type dstType,
                                              ArrayRef<ExprAttr> tileShape,
                                              Type &srcElem, Type &dstElem,
                                              Type &maskElem) {
  srcElem = bodyArgElementType(op.getSource().getType());
  dstElem = bodyArgElementType(dstType);
  if (!srcElem || !dstElem || srcElem != dstElem)
    return failure();
  if (Value mask = op.getMask()) {
    auto maskShape = getOperandShape(mask.getType());
    if (failed(maskShape) || maskShape->size() != tileShape.size())
      return failure();
    maskElem = bodyArgElementType(mask.getType());
    if (!maskElem)
      return failure();
  }
  return success();
}

// Layout-bearing buffer dest preflight: mirrors
// `preflightLoadLikeLayoutBearing` on the scatter side. The peeled underlying
// becomes the generic op's `outs` operand; the captured layout / declared shape
// drive the per-axis offset decomposition.
static FailureOr<StorePreflight>
preflightStoreLayoutBearing(MLIRContext *ctx, sym::Store &store, HCStoreOp op,
                            const LayoutBearingSource &peel,
                            ArrayRef<ExprAttr> tileShape) {
  if (!isa<BufferType>(peel.underlying.getType()))
    return failure();
  ValueRange indices = op.getIndices();
  if (indices.size() != peel.layout.getIndexSyms().size())
    return failure();
  if (peel.declaredShape.getDims().size() != peel.layout.getShapeSyms().size())
    return failure();
  size_t sliceCount = 0;
  for (Value idx : indices)
    if (llvm::isa<SliceType>(idx.getType()))
      ++sliceCount;
  if (sliceCount != tileShape.size())
    return failure();
  auto axes = collectAxisIndices(ctx, store, indices);
  if (failed(axes))
    return failure();
  auto dstShape = getOperandShape(peel.underlying.getType());
  if (failed(dstShape))
    return failure();
  Type srcElem, dstElem, maskElem;
  if (failed(collectStoreElementTypes(op, peel.underlying.getType(), tileShape,
                                      srcElem, dstElem, maskElem)))
    return failure();
  StorePreflight pf;
  pf.tileShape.assign(tileShape.begin(), tileShape.end());
  pf.axes = std::move(*axes);
  pf.srcElem = srcElem;
  pf.dstElem = dstElem;
  pf.maskElem = maskElem;
  pf.effectiveDest = peel.underlying;
  pf.dstShape = std::move(*dstShape);
  pf.destLayout = peel.layout;
  pf.destDeclaredShape = peel.declaredShape;
  return pf;
}

// Validates the store's operand shapes and element types. Tensor-dst
// is rejected (separate slice); src and dst element types must match;
// when present, mask must match `src`'s tile shape (the
// `hc.store` verifier already enforces this, but the body block needs
// the element type extracted up front).
static FailureOr<StorePreflight>
preflightStore(MLIRContext *ctx, sym::Store &store, HCStoreOp op) {
  auto tileShape = getOperandShape(op.getSource().getType());
  if (failed(tileShape))
    return failure();

  // Layout-bearing buffer dest: dest is `hc.as_layout` with `shape=`
  // operand whose underlying is the actual storage. Peel before the
  // tensor-dst / rank-parity gates below so the mixed pinned+slice
  // access form on the scatter side can route through the layout-
  // driven decomposition (the load-side mirror).
  if (auto peel = peelLayoutBearingBufferSource(op.getDest()))
    return preflightStoreLayoutBearing(ctx, store, op, *peel, *tileShape);

  // Tensor / bare_tensor dst is workgroup-shared LDS storage; the IR
  // models it as a value-typed operand even though the runtime
  // semantic is in-place mutation. A clean rewrite would need to
  // produce a new SSA tensor result and propagate it through every
  // subsequent use of `%dst`, which crosses op boundaries we can't
  // resolve in a local rewrite. Buffer dst falls through cleanly
  // because polymorphic outs already model ptr/buffer in-place writes
  // without an SSA result.
  if (!isa<BufferType>(op.getDest().getType()))
    return failure();

  ValueRange indices = op.getIndices();
  if (!indices.empty() && indices.size() != tileShape->size())
    return failure();
  auto axes = collectAxisIndices(ctx, store, indices);
  if (failed(axes))
    return failure();

  Type srcElem, dstElem, maskElem;
  if (failed(collectStoreElementTypes(op, op.getDest().getType(), *tileShape,
                                      srcElem, dstElem, maskElem)))
    return failure();

  StorePreflight pf;
  pf.tileShape = std::move(*tileShape);
  pf.axes = std::move(*axes);
  pf.srcElem = srcElem;
  pf.dstElem = dstElem;
  pf.maskElem = maskElem;
  return pf;
}

// Build the body block of the store generic: src arg, optional mask
// arg, dst arg, terminated with `hc.yield_predicated` (masked) or
// plain `hc.yield`. The masked terminator routes through
// `hc.ptr_store_pred` downstream so masked-out lanes leave the
// existing dst contents in place.
static void populateStoreBody(HCGenericOp generic, const StorePreflight &pf,
                              Location loc, bool hasMask) {
  Block *body = new Block();
  BlockArgument sv = body->addArgument(pf.srcElem, loc);
  BlockArgument mv;
  if (hasMask)
    mv = body->addArgument(pf.maskElem, loc);
  body->addArgument(pf.dstElem, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  if (hasMask)
    HCYieldPredicatedOp::create(bodyBuilder, loc, ValueRange{sv},
                                ValueRange{mv});
  else
    HCYieldOp::create(bodyBuilder, loc, ValueRange{sv});
}

// Pick the right outs_offsets[0] for a store. Mirrors
// `composeLoadInsOffsets` on the scatter side: layout-bearing buffer
// dest routes through `composeLayoutBearingBufferOffsets`; the
// plain path keeps the per-axis identity offsets.
static FailureOr<ArrayAttr>
composeStoreOutsOffsets(MLIRContext *ctx, sym::Store &store,
                        const StorePreflight &pf, ValueRange indices,
                        ArrayRef<StringAttr> iterSyms) {
  if (pf.destLayout)
    return composeLayoutBearingBufferOffsets(ctx, store, pf.destLayout,
                                             pf.destDeclaredShape, indices,
                                             pf.axes, iterSyms, pf.dstShape);
  return composeMemoryOffsetArray(ctx, store, pf.axes, iterSyms);
}

static LogicalResult rewriteStore(HCStoreOp op, sym::Store &store) {
  MLIRContext *ctx = op.getContext();
  auto pf = preflightStore(ctx, store, op);
  if (failed(pf))
    return failure();

  Value src = op.getSource();
  Value dst = pf->effectiveDest ? pf->effectiveDest : op.getDest();
  Value mask = op.getMask();

  Location loc = op.getLoc();
  OpBuilder builder(op);
  CommonRewriteData common = buildCommon(builder, loc, pf->tileShape);

  ArrayAttr inOff = offsetArrayFromIterSyms(ctx, store, common.iterSyms);
  auto outOffArr = composeStoreOutsOffsets(ctx, store, *pf, op.getIndices(),
                                           common.iterSyms);
  if (failed(outOffArr))
    return failure();
  // Masked path: mask rides as an extra ins slot with identity offsets
  // (same shape as src, both tile-local).
  SmallVector<Attribute> insOffArr{inOff};
  SmallVector<Value> insArr{src};
  if (mask) {
    insOffArr.push_back(inOff);
    insArr.push_back(mask);
  }
  ArrayAttr insOffsets = ArrayAttr::get(ctx, insOffArr);
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {*outOffArr});

  SmallVector<Value> outsArr{dst};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{}, common.iterSymsAttr,
      ValueRange(common.iterBounds), common.iterKindsAttr, ValueRange(insArr),
      ValueRange(outsArr), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  populateStoreBody(generic, *pf, loc, /*hasMask=*/(bool)mask);

  op->erase();
  return success();
}

struct HCLoadStoreToGenericPass
    : public hc::impl::HCLoadStoreToGenericBase<HCLoadStoreToGenericPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
    // Collect first, mutate after — `op->erase()` inside the walk
    // would invalidate the iterator the walk is driving.
    SmallVector<HCLoadOp> loads;
    SmallVector<HCVLoadOp> vloads;
    SmallVector<HCStoreOp> stores;
    SmallVector<HCLoadMaskOp> loadMasks;
    root->walk([&](Operation *op) {
      if (auto l = dyn_cast<HCLoadOp>(op))
        loads.push_back(l);
      else if (auto v = dyn_cast<HCVLoadOp>(op))
        vloads.push_back(v);
      else if (auto s = dyn_cast<HCStoreOp>(op))
        stores.push_back(s);
      else if (auto m = dyn_cast<HCLoadMaskOp>(op))
        loadMasks.push_back(m);
    });
    for (HCLoadOp l : loads)
      (void)rewriteLoadLike(l, store);
    for (HCVLoadOp v : vloads)
      (void)rewriteLoadLike(v, store);
    for (HCStoreOp s : stores)
      (void)rewriteStore(s, store);
    for (HCLoadMaskOp m : loadMasks)
      (void)rewriteLoadMask(m, store);
  }
};

} // namespace
