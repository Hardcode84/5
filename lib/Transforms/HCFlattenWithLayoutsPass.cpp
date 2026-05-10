// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-flatten-with-layouts`. This pass only establishes
// the *no-`#hc.layout`-survives* invariant on shaped types: the shape
// itself stays multi-rank. The 1D collapse + per-access offset
// materialization land later, on top of the pointer / elementwise
// memory carrier — they have nowhere meaningful to go until those ops
// exist.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
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

namespace mlir::hc {
#define GEN_PASS_DEF_HCFLATTENWITHLAYOUTS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// `TypeConverter` that drops every `#hc.layout` slot off shaped types
// reachable through `SymbolicallyShapedTypeInterface`, leaving the
// shape itself alone. Tuple and function types recurse so a layout
// buried under a `tuple<...>` operand or a kernel-signature result
// gets the same fold without per-shell hand-rolling. Mirrors the
// catch-all-then-override registration order
// `HCCanonicalizeLayoutsPass` settled on; the more specific
// `SymbolicallyShapedTypeInterface` dispatch takes precedence over
// the identity by virtue of being registered last (last-registered-
// wins is MLIR's documented dispatch order).
//
// `unrealized_conversion_cast` materializations are the same safety
// net the canonicalize pass uses: the layout-bearing and
// layout-stripped forms are observationally identical at the value
// level, but they are *distinct* MLIR types, so an op whose operand
// declaration pins one form would reject the other without an
// explicit bridge. Surviving casts on a boundary that no consumer
// triggered fold via the standard `reconcile-unrealized-casts` pass
// downstream.
class StripLayoutConverter : public TypeConverter {
public:
  StripLayoutConverter() {
    addConversion([](Type t) -> Type { return t; });

    addConversion([](Type t) -> std::optional<Type> {
      auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
      if (!shaped)
        return std::nullopt;
      // Already layout-less — return the type unchanged so the
      // converter reports the slot as legal and we don't waste
      // a clone-and-replace on no-op rewrites.
      if (!shaped.getSymbolicLayout())
        return Type(shaped);
      return shaped.cloneWithSymbolicLayout(LayoutAttr{});
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
// route through `StripLayoutConverter`, the op is purely cosmetic —
// the source value already has the layout-stripped type that the
// result is asking for. Drop the op and replace its result with the
// converted operand. The conversion driver runs to fixpoint, so a
// chain of `as_layout` ops collapses without us walking it ourselves.
struct DropAsLayout : public OpConversionPattern<HCAsLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HCAsLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

struct HCFlattenWithLayoutsPass final
    : public hc::impl::HCFlattenWithLayoutsBase<HCFlattenWithLayoutsPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    StripLayoutConverter converter;

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    patterns.add<DropAsLayout>(converter, ctx);

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
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
