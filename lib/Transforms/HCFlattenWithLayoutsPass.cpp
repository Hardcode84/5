// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-flatten-with-layouts`. Type-only flatten: every
// `SymbolicallyShapedTypeInterface` value loses its `#hc.layout` slot
// AND collapses its shape to a single entry. Tensors / vectors get a
// concrete `storage_size_expr` from the layout (or the dim product
// when no layout is attached). Buffers collapse to `[?]` (`#hc.dyn`
// sentinel) because the host owns the allocation.
//
// The op-level structural invariants stay nD: `hc.generic` keeps its
// per-axis offset arrays at the original logical rank for downstream
// fusion / vectorization, and the rewriter does not touch them.
//
// Per-access ops (`hc.load`, `hc.vload`, `hc.store`, `hc.load_mask`)
// also get rewritten in this pass: their multi-index lists collapse
// to a single 1D base-offset SSA value computed from the operand's
// layout. The composition runs *during* conversion so the layout is
// still on the (pre-conversion) operand type when we read it. Common
// cases lower to one `hc.materialize_bound_expr` typed
// `!hc.idx<offset_expr>`; the per-element traversal of the loaded
// tile (which depends on the storage being contiguous from that base)
// is the consuming pass's problem. `hc.buffer_view` and `hc.vec` stay
// untouched here — buffer_view is a sub-view producer with different
// semantics than a single base offset, and vec has no indices.
//
// `hc.as_layout` is dropped unconditionally — both endpoints route
// through the converter, so the relabel becomes cosmetic by the time
// the rewriter sees it.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCFLATTENWITHLAYOUTS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Compute the 1D `storage_size_expr` for `originalShape` given an
// optional `layout`. With a layout, that is `layout.storage_size`
// after substituting `layout.shape_syms` with the original shape's
// per-axis expressions. Without a layout, the type sits on the
// implicit identity-row-major contract and the storage size is the
// product of the original dimensions. Rank-0 falls out as the empty
// product `1`.
//
// Caller's responsibility: the shape's entries must all be
// `ExprAttr` (no `DynSizeAttr`). Buffers — the only flatten input
// that legitimately carries a `?` post-collapse — handle that case
// directly in the converter instead of routing through here, so
// any `DynSize` here would mean a bug upstream and the cast asserts.
//
// All work goes through hash-consed ixsimpl handles via the
// dialect-owned store: no rendering, no parsing, no string traffic.
// `ixs_subs_multi` wants raw `ixs_node *` arrays for both targets
// and replacements, so we collect those via `composeExprSym` /
// `getNode()` and let the substitution canonicalize through the same
// store any other producer of these expressions would.
static FailureOr<ExprAttr> computeStorageSizeExpr(MLIRContext *ctx,
                                                  LayoutAttr layout,
                                                  ShapeAttr originalShape) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

  if (!layout) {
    ArrayRef<Attribute> dims = originalShape.getDims();
    if (dims.empty()) {
      auto one = sym::composeExprInt(store, 1);
      if (failed(one))
        return failure();
      return ExprAttr::get(ctx, *one);
    }
    sym::ExprHandle product = llvm::cast<ExprAttr>(dims[0]).getValue();
    for (Attribute dim : dims.drop_front()) {
      auto next = sym::composeExprBinary(store, product, sym::ExprBinaryOp::Mul,
                                         llvm::cast<ExprAttr>(dim).getValue());
      if (failed(next))
        return failure();
      product = *next;
    }
    return ExprAttr::get(ctx, product);
  }

  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> dims = originalShape.getDims();
  if (shapeSyms.size() != dims.size())
    return failure();

  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(shapeSyms.size());
  replacements.reserve(shapeSyms.size());
  for (auto [sym, dim] : llvm::zip_equal(shapeSyms, dims)) {
    auto symHandle =
        sym::composeExprSym(store, llvm::cast<StringAttr>(sym).getValue());
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(
        const_cast<ixs_node *>(llvm::cast<ExprAttr>(dim).getValue().raw()));
  }

  sym::Session session(store);
  ixs_node *bound = ixs_subs_multi(
      session.raw(),
      const_cast<ixs_node *>(layout.getStorageSize().getValue().raw()),
      static_cast<uint32_t>(targets.size()), targets.data(),
      replacements.data());
  if (!bound)
    return failure();
  return ExprAttr::get(ctx, sym::ExprHandle(bound));
}

// Pull the symbolic expression that names the SSA index value at an
// access site. Three legal sources:
//   - `!hc.idx<expr>`               -> `expr` (the typical case;
//     index ops produced by the kernel scope are pinned),
//   - `!hc.slice<lower=!hc.idx<expr>, ...>` -> the lower-bound expr
//     (slice's first element address is the access base; the tile
//     walk steps from there),
//   - `!hc.slice` with no `lower`   -> integer `0` (Python-style
//     full-slice; base offset on this axis is 0).
// Anything else (raw `index`, untyped `!hc.idx`, slice without a
// pinned lower) -> failure: the rewrite can't compose a base offset
// without a symbolic name to bind to the layout's index_sym, and
// silently leaving the multi-index list alone would mismatch the
// op's now-1D operand type.
static FailureOr<ExprAttr>
extractAccessIndexExpr(MLIRContext *ctx, sym::Store &store, Type indexType) {
  if (auto idx = llvm::dyn_cast<IdxType>(indexType)) {
    if (ExprAttr expr = idx.getExpr())
      return expr;
    return failure();
  }
  if (auto slice = llvm::dyn_cast<SliceType>(indexType)) {
    Type lowerTy = slice.getLowerType();
    if (!lowerTy) {
      auto zero = sym::composeExprInt(store, 0);
      if (failed(zero))
        return failure();
      return ExprAttr::get(ctx, *zero);
    }
    if (auto lowerIdx = llvm::dyn_cast<IdxType>(lowerTy))
      if (ExprAttr expr = lowerIdx.getExpr())
        return expr;
    return failure();
  }
  return failure();
}

// Build the identity row-major offset for an access into a layout-less
// shaped operand: `i_0 * (d_1 * ... * d_{n-1}) + i_1 * (d_2 * ... *
// d_{n-1}) + ... + i_{n-1}`. Right-to-left fold gives a single ixsimpl
// pass on the way out, which canonicalizes the result for free.
// Rank-0 returns `0`. Caller checks rank parity.
static FailureOr<sym::ExprHandle>
identityRowMajorOffset(sym::Store &store, ArrayRef<ExprAttr> indexExprs,
                       ArrayRef<Attribute> dims) {
  auto zero = sym::composeExprInt(store, 0);
  if (failed(zero))
    return failure();
  if (indexExprs.empty())
    return *zero;
  sym::ExprHandle accum = *zero;
  for (size_t i = 0; i < indexExprs.size(); ++i) {
    sym::ExprHandle term = indexExprs[i].getValue();
    for (size_t j = i + 1; j < dims.size(); ++j) {
      auto dimExpr = llvm::dyn_cast<ExprAttr>(dims[j]);
      if (!dimExpr)
        return failure();
      auto next = sym::composeExprBinary(store, term, sym::ExprBinaryOp::Mul,
                                         dimExpr.getValue());
      if (failed(next))
        return failure();
      term = *next;
    }
    auto added =
        sym::composeExprBinary(store, accum, sym::ExprBinaryOp::Add, term);
    if (failed(added))
      return failure();
    accum = *added;
  }
  return accum;
}

// Compose the access base offset for a multi-index access into a shaped
// operand. With a layout, substitute `shape_syms` positionally with the
// operand's shape entries and `index_syms` positionally with the access
// site's index expressions, then evaluate `layout.offset`. Without a
// layout, fall back to identity row-major over the operand's shape (the
// canonical contract for layout-less shaped types). Rank parity between
// the operand shape and the index list is the caller's responsibility —
// the verifier on the access op already enforces it pre-rewrite, but
// the helper still bails on mismatch instead of producing nonsense.
static FailureOr<ExprAttr>
composeAccessOffsetExpr(MLIRContext *ctx, LayoutAttr layout,
                        ShapeAttr originalShape,
                        ArrayRef<ExprAttr> indexExprs) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  ArrayRef<Attribute> dims = originalShape.getDims();
  if (dims.size() != indexExprs.size())
    return failure();

  if (!layout) {
    auto offset = identityRowMajorOffset(store, indexExprs, dims);
    if (failed(offset))
      return failure();
    return ExprAttr::get(ctx, *offset);
  }

  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = layout.getIndexSyms();
  if (shapeSyms.size() != dims.size() || indexSyms.size() != indexExprs.size())
    return failure();

  // Single substitution pass over `shape_syms ++ index_syms`. Params
  // intentionally don't expand here — they survive as free symbols in
  // the resulting offset, the same way `computeStorageSizeExpr` lets
  // them survive in the post-flatten storage size. Whoever resolves
  // those symbols downstream (launch context for stride params,
  // materialize_bound_expr lowering for closed-form ones) does it
  // uniformly across both surfaces.
  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(shapeSyms.size() + indexSyms.size());
  replacements.reserve(shapeSyms.size() + indexSyms.size());
  auto pushPair = [&](StringRef name,
                      sym::ExprHandle replacement) -> LogicalResult {
    auto symHandle = sym::composeExprSym(store, name);
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(const_cast<ixs_node *>(replacement.raw()));
    return success();
  };
  for (auto [sym, dim] : llvm::zip_equal(shapeSyms, dims)) {
    auto dimExpr = llvm::dyn_cast<ExprAttr>(dim);
    if (!dimExpr)
      return failure();
    if (failed(pushPair(llvm::cast<StringAttr>(sym).getValue(),
                        dimExpr.getValue())))
      return failure();
  }
  for (auto [sym, idx] : llvm::zip_equal(indexSyms, indexExprs)) {
    if (failed(
            pushPair(llvm::cast<StringAttr>(sym).getValue(), idx.getValue())))
      return failure();
  }

  sym::Session session(store);
  ixs_node *bound = ixs_subs_multi(
      session.raw(),
      const_cast<ixs_node *>(layout.getOffset().getValue().raw()),
      static_cast<uint32_t>(targets.size()), targets.data(),
      replacements.data());
  if (!bound)
    return failure();
  return ExprAttr::get(ctx, sym::ExprHandle(bound));
}

// Materialize the composed offset as an SSA value typed
// `!hc.idx<offset_expr>` via `hc.materialize_bound_expr`. The
// downstream `hc-materialize-bound-exprs` pass resolves the type's
// pinned expression against the in-scope symbol bindings (kernel
// shape symbols, stride params, etc.) when it lowers the
// materialize op into concrete SSA arithmetic.
static Value materializeOffsetSSA(ConversionPatternRewriter &rewriter,
                                  Location loc, ExprAttr offsetExpr) {
  auto idxType = IdxType::get(rewriter.getContext(), offsetExpr);
  return HCMaterializeBoundExprOp::create(rewriter, loc, idxType).getResult();
}

// Per-access-op helper: extracts each index operand's symbolic expr,
// composes the base offset against the operand's pre-flatten layout
// + shape, materializes the offset as a single SSA value. Returns
// failure (and leaves the IR untouched) if any index can't yield a
// symbolic expression, or if the operand isn't a shaped HC type, or
// if rank parity breaks. Failure here means the access op stays on
// the multi-index surface; `RetypeAnyHCOp` then handles type-only
// retyping and the lowering downstream still has to deal with the
// uncomposed access.
static FailureOr<Value>
composeAccessBaseOffset(ConversionPatternRewriter &rewriter, Operation *op,
                        Value preFlattenOperand, OperandRange indices) {
  auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(
      preFlattenOperand.getType());
  if (!shaped)
    return failure();
  ShapeAttr originalShape = shaped.getSymbolicShape();
  if (!originalShape)
    return failure();
  // Buffers ride on a `[?]` post-flatten shape and an absent layout
  // here would mean we'd fall back to identity row-major over the
  // wrong dims. Today every buffer carries the default strided
  // layout, so a missing layout on a buffer is a frontend bug we
  // surface as a rewrite failure instead of silently emitting `0`.
  LayoutAttr layout = shaped.getSymbolicLayout();
  if (!layout && llvm::isa<BufferType>(preFlattenOperand.getType()))
    return failure();
  for (Attribute dim : originalShape.getDims())
    if (!llvm::isa<ExprAttr>(dim))
      return failure();

  MLIRContext *ctx = op->getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallVector<ExprAttr> indexExprs;
  indexExprs.reserve(indices.size());
  for (Value idx : indices) {
    auto expr = extractAccessIndexExpr(ctx, store, idx.getType());
    if (failed(expr))
      return failure();
    indexExprs.push_back(*expr);
  }
  auto offsetExpr =
      composeAccessOffsetExpr(ctx, layout, originalShape, indexExprs);
  if (failed(offsetExpr))
    return failure();
  return materializeOffsetSSA(rewriter, op->getLoc(), *offsetExpr);
}

// `TypeConverter` that turns every shaped value into its 1D
// layout-less form. The `SymbolicallyShapedTypeInterface` dispatch
// stays generic across the four shells we collapse (tensor / vector /
// bare_tensor / bare_vector) — adding a sixth implementor would join
// without touching this code. Buffers are deliberately exempt: their
// default strided layout's `storage_size` is the literal `0`
// placeholder the frontend emits (the host owns the allocation), so
// 1D-collapsing them would fold every buffer shape to `[0]`. They
// keep their original nD shape and lose only the layout slot for
// now; a separate slice owns the buffer-side ABI.
//
// Tuples and function types recurse via the converter's own
// `convertTypes`. The catch-all identity is registered first so the
// shaped / tuple / function overrides take precedence on dispatch
// (last-registered-wins).
//
// `unrealized_conversion_cast` materializations are the same safety
// net `hc-canonicalize-layouts` uses: the pre- and post-flatten
// forms are observationally identical at the value level but
// distinct MLIR types, so an op whose operand declaration pins one
// form would reject the other without an explicit bridge. Surviving
// casts on a boundary nothing converted fold via
// `reconcile-unrealized-casts` downstream.
class FlattenLayoutConverter : public TypeConverter {
public:
  FlattenLayoutConverter() {
    addConversion([](Type t) -> Type { return t; });

    addConversion([](Type t) -> std::optional<Type> {
      auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
      if (!shaped)
        return std::nullopt;
      LayoutAttr layout = shaped.getSymbolicLayout();
      ShapeAttr originalShape = shaped.getSymbolicShape();
      if (!originalShape)
        return std::nullopt;

      MLIRContext *ctx = t.getContext();

      // Buffers collapse to `[?]`: the host owns the allocation and
      // the default strided layout's `storage_size = 0` placeholder
      // is informational, so a `#hc.dyn` sentinel is the honest
      // 1D form. Any consumer that needs a concrete extent reaches
      // for the host descriptor instead of trying to derive it from
      // the in-IR symbol set.
      if (isa<BufferType>(t)) {
        ShapeAttr collapsed = ShapeAttr::get(ctx, {DynSizeAttr::get(ctx)});
        if (collapsed == originalShape && !layout)
          return Type(shaped);
        Type withShape = shaped.cloneWithSymbolicShape(collapsed);
        auto reshaped = cast<SymbolicallyShapedTypeInterface>(withShape);
        return reshaped.cloneWithSymbolicLayout(LayoutAttr{});
      }

      bool alreadyFlat = !layout && originalShape.getDims().size() == 1 &&
                         llvm::isa<ExprAttr>(originalShape.getDims().front());
      if (alreadyFlat)
        return Type(shaped);

      FailureOr<ExprAttr> storageSize =
          computeStorageSizeExpr(ctx, layout, originalShape);
      if (failed(storageSize))
        return std::nullopt;
      ShapeAttr collapsed = ShapeAttr::get(ctx, {*storageSize});
      Type withShape = shaped.cloneWithSymbolicShape(collapsed);
      auto reshaped = cast<SymbolicallyShapedTypeInterface>(withShape);
      return reshaped.cloneWithSymbolicLayout(LayoutAttr{});
    });

    addConversion([this](TupleType tuple) -> std::optional<Type> {
      SmallVector<Type> elements;
      if (failed(convertTypes(tuple.getTypes(), elements)))
        return std::nullopt;
      return TupleType::get(tuple.getContext(), elements);
    });

    addConversion([this](FunctionType fn) -> std::optional<Type> {
      SmallVector<Type> ins;
      SmallVector<Type> outs;
      if (failed(convertTypes(fn.getInputs(), ins)))
        return std::nullopt;
      if (failed(convertTypes(fn.getResults(), outs)))
        return std::nullopt;
      return FunctionType::get(fn.getContext(), ins, outs);
    });

    auto cast = [](OpBuilder &builder, Type resultType, ValueRange inputs,
                   Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    addSourceMaterialization(cast);
    addTargetMaterialization(cast);
  }
};

// `hc.as_layout` was the layout-relabel surface op. Once both endpoints
// route through `FlattenLayoutConverter`, the op is purely cosmetic —
// the source value already has the converted type that the result is
// asking for. Drop the op and replace its result with the converted
// operand. The conversion driver runs to fixpoint, so a chain of
// `as_layout` ops collapses without us walking it ourselves.
struct DropAsLayout : public OpConversionPattern<HCAsLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HCAsLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

// Compose multi-index access ops down to a single 1D base offset. Pre-
// flatten the operand's type still carries the layout (and the original
// nD shape); we read both off `op.get<Operand>().getType()`, build the
// composed offset, and rebuild the op with the converted (1D) operand
// type and a one-element index list. The shape SSA tuple — when the op
// has one — passes through unchanged: it still describes the loaded
// tile's logical multi-dim shape, which the downstream lowering walks
// from the new base.
//
// Each pattern bumps benefit above the generic `RetypeAnyHCOp` so the
// driver picks it first; on failure (unbound index, untyped operand,
// rank mismatch) the rewrite signals failure and `RetypeAnyHCOp` does
// the type-only retyping fallback — the access op stays on the nD
// index list and the consuming pass owns the uncomposed access.
//
// Per-op pattern bodies are nearly identical except for the operand
// shape (which fields carry the buffer / source / dest, the indices,
// and the optional shape / source / mask passthroughs); a small CRTP
// base would shrink the boilerplate but obscures which ODS attribute
// each op carries. Four short rewrites are easier to read.

template <typename OpT>
struct ComposeAccessOffsetBase : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  ComposeAccessOffsetBase(const TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<OpT>(converter, ctx, /*benefit=*/2) {}
};

struct ComposeLoadOffsets : public ComposeAccessOffsetBase<HCLoadOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;

  LogicalResult
  matchAndRewrite(HCLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIndices().size() == 1)
      return failure();
    auto base =
        composeAccessBaseOffset(rewriter, op, op.getBuffer(), op.getIndices());
    if (failed(base))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    rewriter.replaceOpWithNewOp<HCLoadOp>(op, resultType, adaptor.getBuffer(),
                                          ValueRange{*base},
                                          adaptor.getShape());
    return success();
  }
};

struct ComposeVLoadOffsets : public ComposeAccessOffsetBase<HCVLoadOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;

  LogicalResult
  matchAndRewrite(HCVLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIndices().size() == 1)
      return failure();
    auto base =
        composeAccessBaseOffset(rewriter, op, op.getSource(), op.getIndices());
    if (failed(base))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    rewriter.replaceOpWithNewOp<HCVLoadOp>(op, resultType, adaptor.getSource(),
                                           ValueRange{*base},
                                           adaptor.getShape());
    return success();
  }
};

struct ComposeLoadMaskOffsets : public ComposeAccessOffsetBase<HCLoadMaskOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;

  LogicalResult
  matchAndRewrite(HCLoadMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIndices().size() == 1)
      return failure();
    auto base =
        composeAccessBaseOffset(rewriter, op, op.getSource(), op.getIndices());
    if (failed(base))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType)
      return failure();
    rewriter.replaceOpWithNewOp<HCLoadMaskOp>(
        op, resultType, adaptor.getSource(), ValueRange{*base},
        adaptor.getShape());
    return success();
  }
};

struct ComposeStoreOffsets : public ComposeAccessOffsetBase<HCStoreOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;

  LogicalResult
  matchAndRewrite(HCStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIndices().size() == 1)
      return failure();
    auto base =
        composeAccessBaseOffset(rewriter, op, op.getDest(), op.getIndices());
    if (failed(base))
      return failure();
    rewriter.replaceOpWithNewOp<HCStoreOp>(
        op, adaptor.getDest(), ValueRange{*base}, adaptor.getSource(),
        adaptor.getMask());
    return success();
  }
};

// Generic op-rebuild pattern for HC dialect ops. The function/SCF
// populators retype signatures and structural ops, but ops in the
// middle of the IR (`hc.generic`, `hc.load`, `hc.cast`, ...) need to
// be rebuilt with converted operand/result types so the post-flatten
// IR carries 1D types end-to-end without `unrealized_conversion_cast`
// stranded in the middle. This pattern is "type-only" by design —
// attributes (including the per-axis `#hc.expr` offset arrays on
// `hc.generic`) and regions are carried over unchanged. Body block
// args of `hc.generic` and friends carry scalar element types that
// don't change under the flatten, so no signature conversion is
// needed inside the regions.
//
// Scoped to ops in the `hc` dialect to stay out of the way of
// upstream patterns (func / scf / call) and avoid silent wins over
// patterns the rest of the codebase relies on. Returning `failure()`
// on a no-op match lets the driver short-circuit to the next pattern
// without us paying for a clone.
struct RetypeAnyHCOp : public ConversionPattern {
  RetypeAnyHCOp(const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getDialect() != op->getContext()->getLoadedDialect<HCDialect>())
      return failure();
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return failure();
    bool changed = false;
    for (auto [oldT, newT] :
         llvm::zip_equal(op->getResultTypes(), resultTypes)) {
      if (oldT != newT) {
        changed = true;
        break;
      }
    }
    if (!changed) {
      for (auto [oldVal, newVal] :
           llvm::zip_equal(op->getOperands(), operands)) {
        if (oldVal.getType() != newVal.getType()) {
          changed = true;
          break;
        }
      }
    }
    if (!changed)
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (size_t i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();
    Operation *newOp = rewriter.create(state);
    for (auto [oldRegion, newRegion] :
         llvm::zip_equal(op->getRegions(), newOp->getRegions()))
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct HCFlattenWithLayoutsPass final
    : public hc::impl::HCFlattenWithLayoutsBase<HCFlattenWithLayoutsPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    FlattenLayoutConverter converter;

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    // Per-access-op patterns are listed first by intent — the
    // conversion driver still picks via benefit (2 vs the generic
    // retype's 1), but having them grouped reads as the design.
    patterns
        .add<ComposeLoadOffsets, ComposeVLoadOffsets, ComposeLoadMaskOffsets,
             ComposeStoreOffsets, DropAsLayout, RetypeAnyHCOp>(converter, ctx);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto fn = dyn_cast<FunctionOpInterface>(op))
        if (auto fnType = dyn_cast<FunctionType>(fn.getFunctionType()))
          return converter.isSignatureLegal(fnType);
      if (isa<func::ReturnOp, func::CallOp>(op))
        return converter.isLegal(op);
      // `hc.as_layout` is always illegal: the rewriter above unconditionally
      // drops it. Without that gate the driver would consider the op legal
      // when both endpoints already have the same converted type, leaving
      // the cosmetic relabel in place.
      if (isa<HCAsLayoutOp>(op))
        return false;
      // HC dialect ops are legal iff every operand and every result type
      // is already in its converted form. The `RetypeAnyHCOp` pattern
      // takes care of the rebuild when one side still carries the
      // pre-flatten shape.
      if (op->getDialect() == ctx->getLoadedDialect<HCDialect>())
        return converter.isLegal(op);
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
