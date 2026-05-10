// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-flatten-with-layouts`. Type-only flatten: every
// `SymbolicallyShapedTypeInterface` value (except buffers) loses its
// `#hc.layout` slot AND collapses its shape to a single
// `storage_size_expr` entry. Op surfaces stay untouched — the per-axis
// offsets on `hc.generic` and the multi-index lists on `hc.load` /
// `hc.store` / `hc.vload` / `hc.buffer_view` remain nD because
// downstream fusion / vectorization wants the per-axis structure.
// Composing the layout offset into those access expressions is a
// separate slice (the operand types just got 1D, but the ops still
// describe the original logical access; whoever turns those into
// concrete addresses owns the layout->offset materialization).
//
// Buffer types are skipped: their default strided layout has
// `storage_size = 0` (informational, the host owns the allocation),
// so 1D-collapse would fold every shape to `[0]`. Buffer flatten
// gets its own slice once the buffer-side ABI lands.
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

      // Buffers keep their nD shape (the buffer-side ABI slice owns
      // 1D-collapse — the default strided layout's `storage_size = 0`
      // placeholder would otherwise fold every buffer to `[0]`).
      // The layout slot still drops to honor the
      // *no-`#hc.layout`-survives* invariant.
      if (isa<BufferType>(t)) {
        if (!layout)
          return Type(shaped);
        return shaped.cloneWithSymbolicLayout(LayoutAttr{});
      }

      bool alreadyFlat = !layout && originalShape.getDims().size() == 1;
      if (alreadyFlat)
        return Type(shaped);

      MLIRContext *ctx = t.getContext();
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

    patterns.add<DropAsLayout, RetypeAnyHCOp>(converter, ctx);

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
