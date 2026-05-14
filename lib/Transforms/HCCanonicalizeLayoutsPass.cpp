// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-canonicalize-layouts`: strip explicit identity layouts
// from shaped Values and collapse `hc.as_layout` chains. See the pass
// description in `include/hc/Transforms/Passes.td` and the design in
// `doc/layouts.md`.

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

// Build canonical identity (offset, storage_size) ixsimpl handles in
// terms of a layout's OWN shape_syms / index_syms — no parser, no string
// concatenation, just leaf-and-compose calls so we go through the same
// hash-consed nodes any other producer of these expressions would. The
// "tail product" stride for index k is prod_{j > k} shape[j], encoded
// right-to-left as a fold so partial products are themselves canonical
// nodes (matters when the producer of `layout`'s offset built it the
// same way — pointer equality after hash-consing is what makes the
// identity check work).
struct CanonicalIdentity {
  sym::ExprHandle offset;
  sym::ExprHandle storage;
};

static FailureOr<CanonicalIdentity>
buildCanonicalIdentity(sym::Store &store, ArrayRef<Attribute> shapeSyms,
                       ArrayRef<Attribute> indexSyms) {
  auto liftSym = [&](Attribute name) -> FailureOr<sym::ExprHandle> {
    return sym::composeExprSym(store, llvm::cast<StringAttr>(name).getValue());
  };

  size_t n = shapeSyms.size();
  if (n == 0) {
    auto zero = sym::composeExprInt(store, 0);
    auto one = sym::composeExprInt(store, 1);
    if (failed(zero) || failed(one))
      return failure();
    return CanonicalIdentity{*zero, *one};
  }

  // Cache shape-syms once: liftSym creates a fresh ixs_node per call,
  // and we'd otherwise materialize each shape symbol O(n) times across
  // the offset / storage builds.
  SmallVector<sym::ExprHandle> shapeHandles;
  shapeHandles.reserve(n);
  for (Attribute name : shapeSyms) {
    auto handle = liftSym(name);
    if (failed(handle))
      return failure();
    shapeHandles.push_back(*handle);
  }

  // tailProducts[k] = prod_{j > k} shape[j]; rightmost stride is 1.
  // Build right-to-left so each partial result is itself canonical.
  SmallVector<sym::ExprHandle> tailProducts(n);
  {
    auto one = sym::composeExprInt(store, 1);
    if (failed(one))
      return failure();
    tailProducts[n - 1] = *one;
    for (size_t k = n - 1; k > 0; --k) {
      auto next = sym::composeExprBinary(
          store, shapeHandles[k], sym::ExprBinaryOp::Mul, tailProducts[k]);
      if (failed(next))
        return failure();
      tailProducts[k - 1] = *next;
    }
  }

  // offset = sum_k (i_k * tailProducts[k])
  sym::ExprHandle offsetAcc;
  bool offsetInit = false;
  for (size_t k = 0; k < n; ++k) {
    auto i_k = liftSym(indexSyms[k]);
    if (failed(i_k))
      return failure();
    sym::ExprHandle term;
    if (k + 1 == n) {
      // Last dim's stride is 1 by construction; skip the trivial multiply
      // so the canonical form matches what ixsimpl produces from a
      // standard textual `... + i_{n-1}` expression.
      term = *i_k;
    } else {
      auto product = sym::composeExprBinary(store, *i_k, sym::ExprBinaryOp::Mul,
                                            tailProducts[k]);
      if (failed(product))
        return failure();
      term = *product;
    }
    if (!offsetInit) {
      offsetAcc = term;
      offsetInit = true;
    } else {
      auto sum = sym::composeExprBinary(store, offsetAcc,
                                        sym::ExprBinaryOp::Add, term);
      if (failed(sum))
        return failure();
      offsetAcc = *sum;
    }
  }

  // storage = prod_k shape[k]; tailProducts[0] already holds prod_{j>0},
  // so multiply by shape[0] once.
  auto storageHandle = sym::composeExprBinary(
      store, shapeHandles[0], sym::ExprBinaryOp::Mul, tailProducts[0]);
  if (failed(storageHandle))
    return failure();

  return CanonicalIdentity{offsetAcc, *storageHandle};
}

// True iff `layout` is the identity-layout contract for `shape`.
// Absent layout (`!layout`) is the v0 identity by definition; this helper
// only fires for the explicit case so the pass can decide whether to
// strip the slot.
static bool isIdentityLayout(LayoutAttr layout, ShapeAttr shape) {
  if (!layout)
    return false;
  if (!shape)
    return false;
  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = layout.getIndexSyms();
  if (shape.getDims().size() != shapeSyms.size())
    return false;
  // `LayoutAttr` enforces `shape_syms.size() == index_syms.size()`; the
  // identity contract is "rightmost-fastest over the dims" with one
  // index per axis.
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
  // ixsimpl handles are hash-consed against the shared store; pointer
  // equality on the canonical handles matches structural equivalence,
  // and the LayoutAttr constructor already normalized the stored ones
  // through the same store on parse.
  return canonical->offset == layout.getOffset().getValue() &&
         canonical->storage == layout.getStorageSize().getValue();
}

// TypeConverter that strips explicit identity layouts off shaped HC
// types (via `SymbolicallyShapedTypeInterface`) and recurses into the
// container types we encounter in real IR (tuples, function types).
// All other types pass through unchanged. The interface dispatch keeps
// this agnostic to which of the five shaped shells we see — adding a
// sixth implementor of the interface would automatically join this
// fold without touching the pass.
//
// Source/target materializations build `unrealized_conversion_cast`
// ops as a safety net: we can't return the input unchanged because the
// layout-bearing and layout-stripped forms are *distinct* MLIR types
// even if they're observationally identical, and a use whose operand
// declaration pins the explicit form would otherwise verify-fail.
// Mirrors the choice in `HCDecomposeShapedValuesPass`. Surviving UCC
// ops (e.g. on a boundary where no consumer triggered the conversion)
// fold via the standard `reconcile-unrealized-casts` pass downstream.
class StripIdentityLayoutConverter : public TypeConverter {
public:
  StripIdentityLayoutConverter() {
    // Catch-all identity. Registered first so that more specific
    // overrides (registered below) take precedence per MLIR's
    // last-registered-wins dispatch order.
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

    // Recurse into containers via the converter's own `convertTypes`
    // so nested shaped types get the same fold without us having to
    // hand-roll a recursion. `this` capture is stable: addConversion
    // stores the lambda by value on the converter itself.
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

// Collapses `hc.as_layout(hc.as_layout(%v, L1), L2)` to
// `hc.as_layout(%v, L2)` by reaching past the inner op's source. The
// conversion driver runs to fixpoint, so chains of any length collapse
// without us hand-rolling a worklist; the inner op is dropped by DCE
// once nothing references it.
struct CollapseAsLayoutChain : public OpConversionPattern<HCAsLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HCAsLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inner = adaptor.getValue().getDefiningOp<HCAsLayoutOp>();
    if (!inner)
      return rewriter.notifyMatchFailure(op, "operand is not an as_layout");
    rewriter.replaceOpWithNewOp<HCAsLayoutOp>(
        op, op.getType(), inner.getValue(), op.getLayoutAttr());
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

    // Func op signatures + return / call body. Upstream populators
    // know how to clone-and-replace these correctly so we don't have
    // to hand-roll the type-rewrite plumbing — and crucially they go
    // through the conversion-aware rewrite paths instead of bare
    // `setType`, which the driver doesn't track for materialization.
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);

    // SCF structural ops (`scf.for`, `scf.if`, `scf.while`,
    // `scf.index_switch`, plus their yield / condition terminators).
    // The populator both adds the patterns AND configures dynamic
    // legality for those ops on the same target — see
    // `mlir/Dialect/SCF/Transforms/StructuralTypeConversions.cpp`.
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    patterns.add<CollapseAsLayoutChain>(converter, ctx);

    // Func-like ops legal iff their signature converts to itself; the
    // body legality is enforced by the per-op conversion patterns
    // above.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto fn = dyn_cast<FunctionOpInterface>(op))
        if (auto fnType = dyn_cast<FunctionType>(fn.getFunctionType()))
          return converter.isSignatureLegal(fnType);
      if (isa<func::ReturnOp, func::CallOp>(op))
        return converter.isLegal(op);
      if (auto al = dyn_cast<HCAsLayoutOp>(op))
        return !al.getValue().getDefiningOp<HCAsLayoutOp>();
      // Anything else: declared legal regardless of operand types.
      // Producers of layout-bearing values get retyped by the patterns
      // above; the consumer observes the new type and stays happy
      // because the layout-stripped form is interchangeable. If a
      // future pipeline grows an op that *requires* the explicit form
      // here, give it its own conversion pattern alongside the SCF
      // ones — don't smear that knowledge across this default.
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    // `applyPartialConversion` rewrites in place but doesn't DCE pure
    // ops that lost their uses to a chain collapse. Sweep dead
    // `hc.as_layout` ops once at the end so a `triple_as_layout` chain
    // doesn't leave two redundant intermediates behind.
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
