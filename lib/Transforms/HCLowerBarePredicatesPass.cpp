// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-bare-predicates`, the first HC-to-upstream lowering
// slice for explicit validity masks.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERBAREPREDICATES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static Type convertElementType(Type type) {
  if (isa<PredType>(type))
    return IntegerType::get(type.getContext(), 1);
  if (type.isIntOrIndexOrFloat())
    return type;
  return {};
}

static FailureOr<SmallVector<int64_t>> staticIntegerShape(ShapeAttr shape,
                                                          Operation *diagOp) {
  SmallVector<int64_t> dims;
  dims.reserve(shape.getDims().size());
  for (auto [index, attr] : llvm::enumerate(shape.getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<int64_t> value =
        expr ? sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()))
             : std::nullopt;
    if (!value || *value < 0) {
      if (diagOp)
        return diagOp->emitOpError("expected static non-negative integer "
                                   "dimension at #")
               << index << ", got " << attr;
      return failure();
    }
    dims.push_back(*value);
  }
  return dims;
}

static Type convertBareVectorType(BareVectorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(shaped.getSymbolicShape(), nullptr);
  if (failed(dims))
    return {};
  if (dims->empty())
    return element;
  return mlir::VectorType::get(*dims, element);
}

static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                             Location loc) {
  if (inputs.size() != 1)
    return {};
  return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
      .getResult(0);
}

class HCBarePredicateTypeConverter : public TypeConverter {
public:
  HCBarePredicateTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });
    addConversion([](PredType type) -> Type {
      return IntegerType::get(type.getContext(), 1);
    });
    addConversion([](BareVectorType type) -> std::optional<Type> {
      if (Type converted = convertBareVectorType(type))
        return converted;
      return Type(type);
    });
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

static bool signatureIsLegal(FunctionType fnType,
                             const TypeConverter &converter) {
  return converter.isSignatureLegal(fnType);
}

struct ConvertFuncSignatureOp : public OpConversionPattern<func::FuncOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType fnType = op.getFunctionType();
    TypeConverter::SignatureConversion blockConversion(fnType.getNumInputs());
    if (failed(typeConverter->convertSignatureArgs(fnType.getInputs(),
                                                   blockConversion)))
      return failure();

    SmallVector<Type> inputs;
    SmallVector<Type> results;
    if (failed(typeConverter->convertTypes(fnType.getInputs(), inputs)) ||
        failed(typeConverter->convertTypes(fnType.getResults(), results)))
      return failure();

    FunctionType convertedFnType =
        FunctionType::get(rewriter.getContext(), inputs, results);
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(convertedFnType);
      if (!op.isExternal())
        rewriter.applySignatureConversion(&op.getBody().front(),
                                          blockConversion, typeConverter);
    });
    return success();
  }
};

struct ConvertFuncReturnOp : public OpConversionPattern<func::ReturnOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::ReturnOp::create(rewriter, op.getLoc(), adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertFullMaskOp : public OpConversionPattern<HCFullMaskOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCFullMaskOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getMask().getType());
    if (!converted || converted == op.getMask().getType())
      return failure();

    TypedAttr value;
    if (auto shaped = dyn_cast<ShapedType>(converted))
      value = DenseElementsAttr::get(shaped, rewriter.getBoolAttr(true));
    else if (converted.isInteger(1))
      value = rewriter.getBoolAttr(true);
    else
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, converted, value);
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<HCSelectOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResult = typeConverter->convertType(op.getResult().getType());
    if (!convertedResult || convertedResult == op.getResult().getType())
      return failure();

    Value falseValue = adaptor.getFalseValue();
    if (isa<mlir::VectorType>(convertedResult))
      falseValue = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                               convertedResult, falseValue);

    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(), falseValue);
    return success();
  }
};

static void populateBarePredicateLoweringPatterns(TypeConverter &converter,
                                                  MLIRContext *ctx,
                                                  RewritePatternSet &patterns) {
  patterns.add<ConvertFuncSignatureOp, ConvertFuncReturnOp, ConvertFullMaskOp,
               ConvertSelectOp>(converter, ctx);
}

static ConversionTarget
makeBarePredicateLoweringTarget(MLIRContext *ctx,
                                const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target.addLegalOp<UnrealizedConversionCastOp, arith::ConstantOp,
                    arith::SelectOp, vector::BroadcastOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return signatureIsLegal(op.getFunctionType(), converter) &&
           converter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<HCFullMaskOp, HCSelectOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  return target;
}

struct HCLowerBarePredicatesPass
    : public hc::impl::HCLowerBarePredicatesBase<HCLowerBarePredicatesPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    HCBarePredicateTypeConverter converter;

    RewritePatternSet patterns(ctx);
    populateBarePredicateLoweringPatterns(converter, ctx, patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    ConversionTarget target = makeBarePredicateLoweringTarget(ctx, converter);
    if (failed(applyPartialConversion(getOperation(), target, frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace

// `createHCLowerBarePredicatesPass()` is emitted by tablegen.
