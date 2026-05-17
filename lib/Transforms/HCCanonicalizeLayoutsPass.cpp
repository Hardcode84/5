// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-canonicalize-layouts`: strip explicit identity layouts
// from shaped Values and collapse `hc.as_layout` chains. See `doc/layouts.md`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCCANONICALIZELAYOUTS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Compose-built so hash-consed handles compare by pointer.
struct CanonicalIdentity {
  sym::ExprHandle offset;
  sym::ExprHandle storage;
};

// Names → ixsimpl symbol handles, position-indexed.
static FailureOr<SmallVector<sym::ExprHandle>>
liftSymbolHandles(sym::Store &store, ArrayRef<Attribute> names) {
  SmallVector<sym::ExprHandle> handles;
  handles.reserve(names.size());
  for (Attribute name : names) {
    auto handle =
        sym::composeExprSym(store, llvm::cast<StringAttr>(name).getValue());
    if (failed(handle))
      return failure();
    handles.push_back(*handle);
  }
  return handles;
}

// `tailProducts[k] = prod_{j>k} shape[j]`; right-to-left so partials are
// canonical.
static FailureOr<SmallVector<sym::ExprHandle>>
buildTailProducts(sym::Store &store, ArrayRef<sym::ExprHandle> shapeHandles) {
  size_t n = shapeHandles.size();
  SmallVector<sym::ExprHandle> tailProducts(n);
  auto one = sym::composeExprInt(store, 1);
  if (failed(one))
    return failure();
  tailProducts[n - 1] = *one;
  for (size_t k = n - 1; k > 0; --k) {
    auto next = sym::composeExprBinary(store, shapeHandles[k],
                                       sym::ExprBinaryOp::Mul, tailProducts[k]);
    if (failed(next))
      return failure();
    tailProducts[k - 1] = *next;
  }
  return tailProducts;
}

// `offset = sum_k (i_k * tailProducts[k])`; last term skips the `*1` for
// canonical form.
static FailureOr<sym::ExprHandle>
buildOffsetSum(sym::Store &store, ArrayRef<sym::ExprHandle> indexHandles,
               ArrayRef<sym::ExprHandle> tailProducts) {
  size_t n = indexHandles.size();
  sym::ExprHandle acc;
  for (size_t k = 0; k < n; ++k) {
    sym::ExprHandle term = indexHandles[k];
    if (k + 1 != n) {
      auto product = sym::composeExprBinary(
          store, indexHandles[k], sym::ExprBinaryOp::Mul, tailProducts[k]);
      if (failed(product))
        return failure();
      term = *product;
    }
    if (k == 0) {
      acc = term;
      continue;
    }
    auto sum = sym::composeExprBinary(store, acc, sym::ExprBinaryOp::Add, term);
    if (failed(sum))
      return failure();
    acc = *sum;
  }
  return acc;
}

static FailureOr<CanonicalIdentity>
buildCanonicalIdentity(sym::Store &store, ArrayRef<Attribute> shapeSyms,
                       ArrayRef<Attribute> indexSyms) {
  size_t n = shapeSyms.size();
  if (n == 0) {
    auto zero = sym::composeExprInt(store, 0);
    auto one = sym::composeExprInt(store, 1);
    if (failed(zero) || failed(one))
      return failure();
    return CanonicalIdentity{*zero, *one};
  }

  FailureOr<SmallVector<sym::ExprHandle>> shapeHandles =
      liftSymbolHandles(store, shapeSyms);
  if (failed(shapeHandles))
    return failure();
  FailureOr<SmallVector<sym::ExprHandle>> indexHandles =
      liftSymbolHandles(store, indexSyms);
  if (failed(indexHandles))
    return failure();

  FailureOr<SmallVector<sym::ExprHandle>> tailProducts =
      buildTailProducts(store, *shapeHandles);
  if (failed(tailProducts))
    return failure();

  FailureOr<sym::ExprHandle> offset =
      buildOffsetSum(store, *indexHandles, *tailProducts);
  if (failed(offset))
    return failure();

  // `storage = shape[0] * tailProducts[0]`.
  auto storage = sym::composeExprBinary(
      store, (*shapeHandles)[0], sym::ExprBinaryOp::Mul, (*tailProducts)[0]);
  if (failed(storage))
    return failure();

  return CanonicalIdentity{*offset, *storage};
}

// Explicit identity only; absent layout is identity by definition.
static bool isIdentityLayout(LayoutAttr layout, ShapeAttr shape) {
  if (!layout)
    return false;
  if (!shape)
    return false;
  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = layout.getIndexSyms();
  if (shape.getDims().size() != shapeSyms.size())
    return false;
  // Identity contract: one index per axis, rightmost-fastest.
  if (shapeSyms.size() != indexSyms.size())
    return false;
  if (!layout.getParams().empty())
    return false;

  auto &store =
      layout.getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
  FailureOr<CanonicalIdentity> canonical =
      buildCanonicalIdentity(store, shapeSyms, indexSyms);
  if (failed(canonical))
    return false;
  // Hash-consed: pointer equality on handles = structural equivalence.
  return canonical->offset == layout.getOffset().getValue() &&
         canonical->storage == layout.getStorageSize().getValue();
}

// Strip explicit identity off shaped types; recurse through tuples / funcs.
// Layout-bearing and layout-stripped are distinct types, so UCC source/target
// materializations bridge them; survivors fold via
// `reconcile-unrealized-casts`.
class StripIdentityLayoutConverter : public TypeConverter {
public:
  StripIdentityLayoutConverter() {
    // Catch-all first; last-registered wins.
    addConversion([](Type t) -> Type { return t; });

    addConversion([](Type t) -> std::optional<Type> {
      auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
      if (!shaped)
        return std::nullopt;
      LayoutAttr layout = shaped.getSymbolicLayout();
      if (!isIdentityLayout(layout, shaped.getSymbolicShape()))
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

// `as_layout(as_layout(%v, L1), L2)` → `as_layout(%v, L2)`; outer `shape=`
// survives.
struct CollapseAsLayoutChain : public OpConversionPattern<HCAsLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HCAsLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inner = adaptor.getValue().getDefiningOp<HCAsLayoutOp>();
    if (!inner)
      return rewriter.notifyMatchFailure(op, "operand is not an as_layout");
    rewriter.replaceOpWithNewOp<HCAsLayoutOp>(
        op, op.getType(), inner.getValue(), op.getShape(), op.getLayoutAttr());
    return success();
  }
};

struct HCCanonicalizeLayoutsPass final
    : public hc::impl::HCCanonicalizeLayoutsBase<HCCanonicalizeLayoutsPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    StripIdentityLayoutConverter converter;

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    // Conversion-aware populators; bare `setType` skips materialization
    // tracking.
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);

    // Populator installs patterns and configures legality.
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    patterns.add<CollapseAsLayoutChain>(converter, ctx);

    // Legal iff signature converts to itself.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto fn = dyn_cast<FunctionOpInterface>(op))
        if (auto fnType = dyn_cast<FunctionType>(fn.getFunctionType()))
          return converter.isSignatureLegal(fnType);
      if (isa<func::ReturnOp, func::CallOp>(op))
        return converter.isLegal(op);
      if (auto al = dyn_cast<HCAsLayoutOp>(op))
        return !al.getValue().getDefiningOp<HCAsLayoutOp>();
      // Anything needing the explicit form gets its own pattern.
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    // Partial conversion leaves dead chain links; sweep.
    SmallVector<HCAsLayoutOp> dead;
    getOperation()->walk([&](HCAsLayoutOp op) {
      if (op.use_empty())
        dead.push_back(op);
    });
    for (HCAsLayoutOp op : dead)
      op.erase();
  }
};

} // namespace
