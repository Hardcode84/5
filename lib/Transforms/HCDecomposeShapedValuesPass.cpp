// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-decompose-shaped-values`, the HC-to-HC boundary that makes
// implicit tensor/vector validity explicit before upstream dialect conversion.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCDECOMPOSESHAPEDVALUES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static bool isSemanticShaped(Type type) {
  return isa<mlir::hc::TensorType, mlir::hc::VectorType>(type);
}

static Type bareDataType(Type type) {
  if (auto tensor = dyn_cast<mlir::hc::TensorType>(type))
    return BareTensorType::get(type.getContext(), tensor.getElementType(),
                               tensor.getShape());
  if (auto vector = dyn_cast<mlir::hc::VectorType>(type))
    return BareVectorType::get(type.getContext(), vector.getElementType(),
                               vector.getShape());
  return {};
}

static Type bareMaskType(Type type) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return {};
  Type pred = getUnpinnedPredType(type.getContext());
  if (isa<mlir::hc::TensorType, BareTensorType>(type))
    return BareTensorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape());
  if (isa<mlir::hc::VectorType, BareVectorType>(type))
    return BareVectorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape());
  return {};
}

static bool functionTypeIsLegal(TypeAttr attr, const TypeConverter &converter) {
  if (!attr)
    return true;
  auto fnType = dyn_cast<FunctionType>(attr.getValue());
  return !fnType || converter.isSignatureLegal(fnType);
}

static bool callableSignatureIsLegal(Operation *op,
                                     const TypeConverter &converter) {
  if (auto func = dyn_cast<FunctionOpInterface>(op)) {
    auto fnType = dyn_cast<FunctionType>(func.getFunctionType());
    return !fnType || converter.isSignatureLegal(fnType);
  }
  if (auto kernel = dyn_cast<HCKernelOp>(op))
    return functionTypeIsLegal(kernel.getFunctionTypeAttr(), converter);
  if (auto func = dyn_cast<HCFuncOp>(op))
    return functionTypeIsLegal(func.getFunctionTypeAttr(), converter);
  if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op))
    return functionTypeIsLegal(intrinsic.getFunctionTypeAttr(), converter);
  return true;
}

static bool regionsAreLegal(Operation *op, const TypeConverter &converter) {
  return llvm::all_of(op->getRegions(), [&](Region &region) {
    return converter.isLegal(&region);
  });
}

static FailureOr<Value> expectOne(ValueRange values, Operation *op,
                                  StringRef what) {
  if (values.size() == 1)
    return values.front();
  return op->emitOpError("expected one converted value for ")
         << what << ", got " << values.size();
}

static FailureOr<std::pair<Value, Value>>
expectSplit(ValueRange values, Operation *op, StringRef what) {
  if (values.size() == 2)
    return std::make_pair(values[0], values[1]);
  return op->emitOpError("expected decomposed data and mask for ")
         << what << ", got " << values.size() << " value(s)";
}

static LogicalResult collectOneToOneOperands(ArrayRef<ValueRange> operands,
                                             Operation *op, StringRef what,
                                             SmallVectorImpl<Value> &values) {
  values.reserve(values.size() + operands.size());
  for (ValueRange operand : operands) {
    FailureOr<Value> value = expectOne(operand, op, what);
    if (failed(value))
      return failure();
    values.push_back(*value);
  }
  return success();
}

static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> flattened;
  for (ValueRange range : values)
    llvm::append_range(flattened, range);
  return flattened;
}

static LogicalResult convertFunctionType(FunctionType fnType,
                                         const TypeConverter &converter,
                                         SmallVectorImpl<Type> &inputs,
                                         SmallVectorImpl<Type> &results) {
  if (failed(converter.convertTypes(fnType.getInputs(), inputs)))
    return failure();
  return converter.convertTypes(fnType.getResults(), results);
}

static void replaceSingleResultWithSplit(ConversionPatternRewriter &rewriter,
                                         Operation *op, Value data,
                                         Value mask) {
  SmallVector<SmallVector<Value>> replacements;
  replacements.push_back({data, mask});
  rewriter.replaceOpWithMultiple(op, std::move(replacements));
}

static Value materializeSourceCast(OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) {
  if (!isSemanticShaped(type) || inputs.size() != 2)
    return {};
  if (inputs[0].getType() != bareDataType(type) ||
      inputs[1].getType() != bareMaskType(type))
    return {};
  return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
      .getResult(0);
}

static SmallVector<Value> materializeTargetCast(OpBuilder &builder,
                                                TypeRange types,
                                                ValueRange inputs, Location loc,
                                                Type originalType) {
  if (!isSemanticShaped(originalType) || types.size() != 2 ||
      inputs.size() != 1)
    return {};
  if (types[0] != bareDataType(originalType) ||
      types[1] != bareMaskType(originalType))
    return {};
  return UnrealizedConversionCastOp::create(builder, loc, types, inputs)
      .getResults();
}

class HCShapedTypeConverter : public TypeConverter {
public:
  HCShapedTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });
    addConversion(
        [](mlir::hc::TensorType type,
           SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          results.push_back(bareDataType(type));
          results.push_back(bareMaskType(type));
          return success();
        });
    addConversion(
        [](mlir::hc::VectorType type,
           SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          results.push_back(bareDataType(type));
          results.push_back(bareMaskType(type));
          return success();
        });
    addSourceMaterialization(materializeSourceCast);
    addTargetMaterialization(materializeTargetCast);
  }
};

template <typename OpT>
struct ConvertCallableSignatureOp : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using Base::Base;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OneToNOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    TypeAttr fnTypeAttr = op.getFunctionTypeAttr();
    if (!fnTypeAttr)
      return failure();

    FunctionType fnType = cast<FunctionType>(fnTypeAttr.getValue());
    TypeConverter::SignatureConversion blockConversion(fnType.getNumInputs());
    if (failed(this->typeConverter->convertSignatureArgs(fnType.getInputs(),
                                                         blockConversion)))
      return failure();

    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    if (failed(convertFunctionType(fnType, *this->typeConverter,
                                   convertedInputs, convertedResults)))
      return failure();

    rewriter.applySignatureConversion(&op.getBody().front(), blockConversion,
                                      this->typeConverter);
    FunctionType convertedFnType = FunctionType::get(
        rewriter.getContext(), convertedInputs, convertedResults);
    rewriter.modifyOpInPlace(
        op, [&] { op.setFunctionTypeAttr(TypeAttr::get(convertedFnType)); });
    return success();
  }
};

struct ConvertCallOp : public OpConversionPattern<HCCallOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCCallOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> convertedArgs = flattenValues(adaptor.getArgs());
    SmallVector<Type> convertedResults;
    SmallVector<unsigned> resultWidths;
    for (Type resultType : op.getResultTypes()) {
      unsigned start = convertedResults.size();
      if (failed(typeConverter->convertTypes(resultType, convertedResults)))
        return failure();
      resultWidths.push_back(convertedResults.size() - start);
    }

    auto newCall = HCCallOp::create(rewriter, op.getLoc(), convertedResults,
                                    op.getCalleeAttr(), convertedArgs);
    newCall->setAttrs(op->getAttrs());

    SmallVector<ValueRange> replacements;
    unsigned offset = 0;
    for (unsigned width : resultWidths) {
      replacements.push_back(newCall->getResults().slice(offset, width));
      offset += width;
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

struct ConvertStoreOp : public OpConversionPattern<HCStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalSourceType = op.getSource().getType();
    if (!isSemanticShaped(originalSourceType))
      return failure();

    FailureOr<std::pair<Value, Value>> source =
        expectSplit(adaptor.getSource(), op, "store source");
    if (failed(source))
      return failure();

    SmallVector<Value> indices;
    if (failed(collectOneToOneOperands(adaptor.getIndices(), op, "store index",
                                       indices)))
      return failure();

    if (adaptor.getDest().size() == 1) {
      HCStoreOp::create(rewriter, op.getLoc(), adaptor.getDest().front(),
                        indices, source->first, source->second);
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<std::pair<Value, Value>> dest =
        expectSplit(adaptor.getDest(), op, "store destination");
    if (failed(dest))
      return failure();
    // Tensor-backed stores update the decomposed payload and validity channels
    // independently; a later lowering pass decides how those channels alias.
    HCStoreOp::create(rewriter, op.getLoc(), dest->first, indices,
                      source->first, Value{});
    HCStoreOp::create(rewriter, op.getLoc(), dest->second, indices,
                      source->second, Value{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertReturnOp : public OpConversionPattern<HCReturnOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCReturnOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    HCReturnOp::create(rewriter, op.getLoc(),
                       flattenValues(adaptor.getValues()));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertLoadOp : public OpConversionPattern<HCLoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<Value> buffer = expectOne(adaptor.getBuffer(), op, "load buffer");
    FailureOr<Value> shape = expectOne(adaptor.getShape(), op, "load shape");
    if (failed(buffer) || failed(shape))
      return failure();
    SmallVector<Value> indices;
    if (failed(collectOneToOneOperands(adaptor.getIndices(), op, "load index",
                                       indices)))
      return failure();

    auto data =
        HCLoadOp::create(rewriter, op.getLoc(), bareDataType(originalType),
                         *buffer, indices, *shape);
    auto mask =
        HCLoadMaskOp::create(rewriter, op.getLoc(), bareMaskType(originalType),
                             *buffer, indices, *shape);
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getMask());
    return success();
  }
};

struct ConvertVLoadOp : public OpConversionPattern<HCVLoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCVLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<Value> shape = expectOne(adaptor.getShape(), op, "vload shape");
    if (failed(shape))
      return failure();
    SmallVector<Value> indices;
    if (failed(collectOneToOneOperands(adaptor.getIndices(), op, "vload index",
                                       indices)))
      return failure();

    Value dataSource;
    Value maskSource;
    if (adaptor.getSource().size() == 1) {
      dataSource = adaptor.getSource().front();
    } else {
      FailureOr<std::pair<Value, Value>> source =
          expectSplit(adaptor.getSource(), op, "vload source");
      if (failed(source))
        return failure();
      dataSource = source->first;
      maskSource = source->second;
    }
    auto data =
        HCVLoadOp::create(rewriter, op.getLoc(), bareDataType(originalType),
                          dataSource, indices, *shape);
    Value maskValue;
    if (maskSource) {
      maskValue =
          HCVLoadOp::create(rewriter, op.getLoc(), bareMaskType(originalType),
                            maskSource, indices, *shape)
              .getResult();
    } else {
      maskValue = HCLoadMaskOp::create(rewriter, op.getLoc(),
                                       bareMaskType(originalType), dataSource,
                                       indices, *shape)
                      .getMask();
    }
    replaceSingleResultWithSplit(rewriter, op, data.getResult(), maskValue);
    return success();
  }
};

template <typename OpT>
struct ConvertNullaryAllocOp : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using Base::Base;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<Value> shape =
        expectOne(adaptor.getShape(), op, "allocator shape");
    if (failed(shape))
      return failure();
    auto data = OpT::create(rewriter, op.getLoc(), bareDataType(originalType),
                            *shape, op.getDtypeAttr());
    auto mask =
        HCFullMaskOp::create(rewriter, op.getLoc(), bareMaskType(originalType));
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getMask());
    return success();
  }
};

template <typename OpT>
struct ConvertFillAllocOp : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using Base::Base;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<Value> fill = expectOne(adaptor.getValue(), op, "allocator fill");
    FailureOr<Value> shape =
        expectOne(adaptor.getShape(), op, "allocator shape");
    if (failed(fill) || failed(shape))
      return failure();
    auto data = OpT::create(rewriter, op.getLoc(), bareDataType(originalType),
                            *fill, *shape, op.getDtypeAttr());
    auto mask =
        HCFullMaskOp::create(rewriter, op.getLoc(), bareMaskType(originalType));
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getMask());
    return success();
  }
};

struct ConvertBufferViewOp : public OpConversionPattern<HCBufferViewOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBufferViewOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    Value dataSource;
    Value maskSource;
    // Buffer roots are not decomposed by this pass, so a single adapted operand
    // means the view starts from fully valid buffer storage.
    if (adaptor.getBuffer().size() == 1 &&
        isa<BufferType>(op.getBuffer().getType())) {
      dataSource = adaptor.getBuffer().front();
    } else {
      FailureOr<std::pair<Value, Value>> source =
          expectSplit(adaptor.getBuffer(), op, "buffer_view source");
      if (failed(source))
        return failure();
      dataSource = source->first;
      maskSource = source->second;
    }
    SmallVector<Value> indices;
    if (failed(collectOneToOneOperands(adaptor.getIndices(), op,
                                       "buffer_view index", indices)))
      return failure();

    auto data = HCBufferViewOp::create(
        rewriter, op.getLoc(), bareDataType(originalType), dataSource, indices);
    Value maskValue;
    if (maskSource) {
      maskValue = HCBufferViewOp::create(rewriter, op.getLoc(),
                                         bareMaskType(originalType), maskSource,
                                         indices)
                      .getResult();
    } else {
      maskValue = HCFullMaskOp::create(rewriter, op.getLoc(),
                                       bareMaskType(originalType))
                      .getMask();
    }
    replaceSingleResultWithSplit(rewriter, op, data.getResult(), maskValue);
    return success();
  }
};

struct ConvertGetItemOp : public OpConversionPattern<HCGetItemOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCGetItemOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    Value dataSource;
    Value maskSource;
    // Buffer roots are not decomposed by this pass, so a single adapted operand
    // means the item starts from fully valid buffer storage.
    if (adaptor.getBase().size() == 1 &&
        isa<BufferType>(op.getBase().getType())) {
      dataSource = adaptor.getBase().front();
    } else {
      FailureOr<std::pair<Value, Value>> source =
          expectSplit(adaptor.getBase(), op, "getitem base");
      if (failed(source))
        return failure();
      dataSource = source->first;
      maskSource = source->second;
    }
    SmallVector<Value> indices;
    if (failed(collectOneToOneOperands(adaptor.getIndices(), op,
                                       "getitem index", indices)))
      return failure();

    auto data = HCGetItemOp::create(
        rewriter, op.getLoc(), bareDataType(originalType), dataSource, indices);
    Value maskValue;
    if (maskSource) {
      maskValue =
          HCGetItemOp::create(rewriter, op.getLoc(), bareMaskType(originalType),
                              maskSource, indices)
              .getResult();
    } else {
      maskValue = HCFullMaskOp::create(rewriter, op.getLoc(),
                                       bareMaskType(originalType))
                      .getMask();
    }
    replaceSingleResultWithSplit(rewriter, op, data.getResult(), maskValue);
    return success();
  }
};

struct ConvertVecOp : public OpConversionPattern<HCVecOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCVecOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<std::pair<Value, Value>> source =
        expectSplit(adaptor.getValue(), op, "vec source");
    if (failed(source))
      return failure();
    auto data = HCVecOp::create(rewriter, op.getLoc(),
                                bareDataType(originalType), source->first);
    auto mask = HCVecOp::create(rewriter, op.getLoc(),
                                bareMaskType(originalType), source->second);
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getResult());
    return success();
  }
};

struct ConvertWithInactiveOp : public OpConversionPattern<HCWithInactiveOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCWithInactiveOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<std::pair<Value, Value>> source =
        expectSplit(adaptor.getValue(), op, "with_inactive source");
    FailureOr<Value> inactive =
        expectOne(adaptor.getInactive(), op, "with_inactive fill");
    if (failed(source) || failed(inactive))
      return failure();
    auto data =
        HCSelectOp::create(rewriter, op.getLoc(), bareDataType(originalType),
                           source->second, source->first, *inactive);
    auto mask =
        HCFullMaskOp::create(rewriter, op.getLoc(), bareMaskType(originalType));
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getMask());
    return success();
  }
};

static void populateShapedDecompositionPatterns(TypeConverter &converter,
                                                MLIRContext *ctx,
                                                RewritePatternSet &patterns) {
  patterns.add<ConvertCallableSignatureOp<HCKernelOp>,
               ConvertCallableSignatureOp<HCFuncOp>, ConvertCallOp,
               ConvertStoreOp, ConvertReturnOp>(converter, ctx);
  patterns.add<ConvertLoadOp, ConvertVLoadOp, ConvertBufferViewOp,
               ConvertGetItemOp, ConvertVecOp, ConvertWithInactiveOp>(converter,
                                                                      ctx);
  patterns
      .add<ConvertNullaryAllocOp<HCVZerosOp>, ConvertNullaryAllocOp<HCVOnesOp>,
           ConvertNullaryAllocOp<HCZerosOp>, ConvertNullaryAllocOp<HCOnesOp>,
           ConvertNullaryAllocOp<HCEmptyOp>, ConvertFillAllocOp<HCVFullOp>,
           ConvertFillAllocOp<HCFullOp>>(converter, ctx);
}

static ConversionTarget
makeStrictShapedDecompositionTarget(MLIRContext *ctx,
                                    const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addDynamicallyLegalOp<HCKernelOp, HCFuncOp>([&](Operation *op) {
    return regionsAreLegal(op, converter) &&
           callableSignatureIsLegal(op, converter);
  });
  target.addDynamicallyLegalOp<HCCallOp, HCStoreOp, HCReturnOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return converter.isLegal(op) && regionsAreLegal(op, converter) &&
           callableSignatureIsLegal(op, converter);
  });
  return target;
}

static ConversionTarget
makePartialShapedDecompositionTarget(MLIRContext *ctx,
                                     const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<HCKernelOp, HCFuncOp>([&](Operation *op) {
    return regionsAreLegal(op, converter) &&
           callableSignatureIsLegal(op, converter);
  });
  target.addDynamicallyLegalOp<
      HCCallOp, HCStoreOp, HCReturnOp, HCLoadOp, HCVLoadOp, HCBufferViewOp,
      HCGetItemOp, HCVecOp, HCWithInactiveOp, HCVZerosOp, HCVOnesOp, HCZerosOp,
      HCOnesOp, HCEmptyOp, HCVFullOp, HCFullOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  return target;
}

struct HCDecomposeShapedValuesPass
    : public hc::impl::HCDecomposeShapedValuesBase<
          HCDecomposeShapedValuesPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    HCShapedTypeConverter converter;

    RewritePatternSet patterns(ctx);
    populateShapedDecompositionPatterns(converter, ctx, patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    ConversionTarget target =
        strictMode ? makeStrictShapedDecompositionTarget(ctx, converter)
                   : makePartialShapedDecompositionTarget(ctx, converter);
    LogicalResult result =
        strictMode
            ? applyFullConversion(getOperation(), target, frozenPatterns)
            : applyPartialConversion(getOperation(), target, frozenPatterns);
    if (failed(result))
      signalPassFailure();
  }
};

} // namespace

// `createHCDecomposeShapedValuesPass()` is emitted by tablegen.
