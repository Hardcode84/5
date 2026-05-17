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

// Carry layout onto both halves; dropping it collapses non-injective access
// (broadcasts, WMMA fragments where storage_size < product(shape)) to identity.
static Type bareDataType(Type type) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  LayoutAttr layout = shaped ? shaped.getSymbolicLayout() : LayoutAttr{};
  if (auto tensor = dyn_cast<mlir::hc::TensorType>(type))
    return BareTensorType::get(type.getContext(), tensor.getElementType(),
                               tensor.getShape(), layout);
  if (auto vector = dyn_cast<mlir::hc::VectorType>(type))
    return BareVectorType::get(type.getContext(), vector.getElementType(),
                               vector.getShape(), layout);
  return {};
}

static Type bareMaskType(Type type) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return {};
  Type pred = getUnpinnedPredType(type.getContext());
  LayoutAttr layout = shaped.getSymbolicLayout();
  if (isa<mlir::hc::TensorType, BareTensorType>(type))
    return BareTensorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape(), layout);
  if (isa<mlir::hc::VectorType, BareVectorType>(type))
    return BareVectorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape(), layout);
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

static LogicalResult
convertResultTypes(TypeRange resultTypes, const TypeConverter &converter,
                   SmallVectorImpl<Type> &convertedResults,
                   SmallVectorImpl<unsigned> &resultWidths) {
  for (Type resultType : resultTypes) {
    unsigned start = convertedResults.size();
    if (failed(converter.convertTypes(resultType, convertedResults)))
      return failure();
    resultWidths.push_back(convertedResults.size() - start);
  }
  return success();
}

static SmallVector<TypeRange> resultTypeSlices(TypeRange convertedResults,
                                               ArrayRef<unsigned> widths) {
  SmallVector<TypeRange> slices;
  slices.reserve(widths.size());
  unsigned offset = 0;
  for (unsigned width : widths) {
    slices.push_back(convertedResults.slice(offset, width));
    offset += width;
  }
  return slices;
}

static FailureOr<SmallVector<Value>>
adaptValuesToTypes(ValueRange values, TypeRange targetTypes, Operation *op,
                   ConversionPatternRewriter &rewriter, StringRef what) {
  if (values.size() == targetTypes.size())
    return SmallVector<Value>(values);
  if (values.size() == 1) {
    auto cast = UnrealizedConversionCastOp::create(rewriter, op->getLoc(),
                                                   targetTypes, values.front());
    return SmallVector<Value>(cast.getResults());
  }
  return op->emitOpError("expected ")
         << targetTypes.size() << " converted value(s) for " << what << ", got "
         << values.size();
}

static LogicalResult buildForRangeBodySignatureConversion(
    HCForRangeOp op, TypeRange convertedResultTypes,
    ArrayRef<unsigned> resultWidths, const TypeConverter &converter,
    TypeConverter::SignatureConversion &bodyConversion) {
  Block &body = op.getBody().front();
  if (body.getNumArguments() != 1 + resultWidths.size())
    return failure();

  SmallVector<Type> ivTypes;
  if (failed(converter.convertType(body.getArgument(0).getType(), ivTypes)))
    return failure();
  bodyConversion.addInputs(0, ivTypes);

  SmallVector<TypeRange> resultSlices =
      resultTypeSlices(convertedResultTypes, resultWidths);
  for (auto [index, types] : llvm::enumerate(resultSlices)) {
    SmallVector<Type> copiedTypes(types.begin(), types.end());
    bodyConversion.addInputs(index + 1, copiedTypes);
  }
  return success();
}

static LogicalResult
adaptForRangeIterInits(HCForRangeOp op, ArrayRef<ValueRange> adaptedIterInits,
                       TypeRange convertedResultTypes,
                       ArrayRef<unsigned> resultWidths,
                       ConversionPatternRewriter &rewriter,
                       SmallVectorImpl<Value> &convertedIterInits) {
  if (adaptedIterInits.size() != resultWidths.size())
    return op.emitOpError("expected ")
           << resultWidths.size() << " iter init group(s), got "
           << adaptedIterInits.size();

  SmallVector<TypeRange> resultSlices =
      resultTypeSlices(convertedResultTypes, resultWidths);
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip_equal(adaptedIterInits, resultSlices))) {
    auto [values, types] = pair;
    FailureOr<SmallVector<Value>> adapted = adaptValuesToTypes(
        values, types, op, rewriter,
        Twine("for_range iter init #").concat(Twine(index)).str());
    if (failed(adapted))
      return failure();
    llvm::append_range(convertedIterInits, *adapted);
  }
  return success();
}

static void replaceOpWithResultSlices(ConversionPatternRewriter &rewriter,
                                      Operation *op,
                                      ResultRange convertedResults,
                                      ArrayRef<unsigned> resultWidths) {
  SmallVector<ValueRange> replacements;
  unsigned offset = 0;
  for (unsigned width : resultWidths) {
    replacements.push_back(convertedResults.slice(offset, width));
    offset += width;
  }
  rewriter.replaceOpWithMultiple(op, replacements);
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

// Slot 0/1 → `.data`/`.mask`; higher slots → `.2`, `.3`, ...
static void appendSplitParameterNames(MLIRContext *ctx, StringRef name,
                                      size_t typeCount,
                                      SmallVectorImpl<Attribute> &out) {
  StringRef suffixes[] = {"data", "mask"};
  for (auto [index, suffix] : llvm::enumerate(suffixes)) {
    if (index >= typeCount)
      break;
    SmallString<32> splitName(name);
    splitName += ".";
    splitName += suffix;
    out.push_back(StringAttr::get(ctx, splitName));
  }
  for (size_t index = 2; index < typeCount; ++index) {
    SmallString<32> splitName(name);
    splitName += ".";
    splitName += Twine(index).str();
    out.push_back(StringAttr::get(ctx, splitName));
  }
}

// Const kwargs don't consume an input slot; failure surfaces as `{}` upstream.
static LogicalResult
convertOneParameter(MLIRContext *ctx, StringAttr parameter,
                    const llvm::SmallDenseSet<StringRef> &constKwargs,
                    FunctionType originalFnType, const TypeConverter &converter,
                    unsigned &inputIndex,
                    SmallVectorImpl<Attribute> &converted) {
  StringRef name = parameter.getValue();
  if (constKwargs.contains(name)) {
    converted.push_back(parameter);
    return success();
  }
  if (inputIndex >= originalFnType.getNumInputs())
    return failure();
  SmallVector<Type> convertedTypes;
  if (failed(converter.convertType(originalFnType.getInput(inputIndex++),
                                   convertedTypes)))
    return failure();
  if (convertedTypes.size() == 1) {
    converted.push_back(parameter);
    return success();
  }
  appendSplitParameterNames(ctx, name, convertedTypes.size(), converted);
  return success();
}

static ArrayAttr convertIntrinsicParameters(HCIntrinsicOp op,
                                            FunctionType originalFnType,
                                            const TypeConverter &converter) {
  ArrayAttr parameters = op.getParametersAttr();
  if (!parameters)
    return {};

  llvm::SmallDenseSet<StringRef> constKwargs;
  if (ArrayAttr attrs = op.getConstKwargsAttr())
    for (StringAttr attr : attrs.getAsRange<StringAttr>())
      constKwargs.insert(attr.getValue());

  MLIRContext *ctx = op.getContext();
  SmallVector<Attribute> converted;
  unsigned inputIndex = 0;
  for (StringAttr parameter : parameters.getAsRange<StringAttr>())
    if (failed(convertOneParameter(ctx, parameter, constKwargs, originalFnType,
                                   converter, inputIndex, converted)))
      return {};

  if (inputIndex != originalFnType.getNumInputs())
    return {};
  return ArrayAttr::get(ctx, converted);
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
    ArrayAttr convertedParameters;
    if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op.getOperation())) {
      convertedParameters =
          convertIntrinsicParameters(intrinsic, fnType, *this->typeConverter);
      if (intrinsic.getParametersAttr() && !convertedParameters)
        return op.emitOpError("failed to convert intrinsic parameter metadata");
    }
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionTypeAttr(TypeAttr::get(convertedFnType));
      if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op.getOperation()))
        if (convertedParameters)
          intrinsic.setParametersAttr(convertedParameters);
    });
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
    if (failed(convertResultTypes(op.getResultTypes(), *typeConverter,
                                  convertedResults, resultWidths)))
      return failure();

    auto newCall = HCCallOp::create(rewriter, op.getLoc(), convertedResults,
                                    op.getCalleeAttr(), convertedArgs);
    newCall->setAttrs(op->getAttrs());

    replaceOpWithResultSlices(rewriter, op, newCall->getResults(),
                              resultWidths);
    return success();
  }
};

struct ConvertCallIntrinsicOp : public OpConversionPattern<HCCallIntrinsicOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCCallIntrinsicOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> convertedArgs = flattenValues(adaptor.getArgs());
    SmallVector<Type> convertedResults;
    SmallVector<unsigned> resultWidths;
    if (failed(convertResultTypes(op.getResultTypes(), *typeConverter,
                                  convertedResults, resultWidths)))
      return failure();

    auto newCall =
        HCCallIntrinsicOp::create(rewriter, op.getLoc(), convertedResults,
                                  op.getCalleeAttr(), convertedArgs);
    newCall->setAttrs(op->getAttrs());

    replaceOpWithResultSlices(rewriter, op, newCall->getResults(),
                              resultWidths);
    return success();
  }
};

struct ConvertForRangeOp : public OpConversionPattern<HCForRangeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCForRangeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> lower =
        expectOne(adaptor.getLower(), op, "for_range lower");
    FailureOr<Value> upper =
        expectOne(adaptor.getUpper(), op, "for_range upper");
    FailureOr<Value> step = expectOne(adaptor.getStep(), op, "for_range step");
    if (failed(lower) || failed(upper) || failed(step))
      return failure();

    SmallVector<Type> convertedResults;
    SmallVector<unsigned> resultWidths;
    if (failed(convertResultTypes(op.getResultTypes(), *typeConverter,
                                  convertedResults, resultWidths)))
      return failure();
    SmallVector<Value> convertedIterInits;
    if (failed(adaptForRangeIterInits(op, adaptor.getIterInits(),
                                      convertedResults, resultWidths, rewriter,
                                      convertedIterInits)))
      return failure();

    TypeConverter::SignatureConversion bodyConversion(
        op.getBody().front().getNumArguments());
    if (failed(buildForRangeBodySignatureConversion(
            op, convertedResults, resultWidths, *typeConverter,
            bodyConversion)))
      return failure();

    auto newLoop =
        HCForRangeOp::create(rewriter, op.getLoc(), convertedResults, *lower,
                             *upper, *step, convertedIterInits);
    newLoop->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getBody(), newLoop.getBody(),
                                newLoop.getBody().begin());
    rewriter.applySignatureConversion(&newLoop.getBody().front(),
                                      bodyConversion, typeConverter);

    replaceOpWithResultSlices(rewriter, op, newLoop->getResults(),
                              resultWidths);
    return success();
  }
};

struct ConvertIfOp : public OpConversionPattern<HCIfOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIfOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> cond = expectOne(adaptor.getCond(), op, "if condition");
    if (failed(cond))
      return failure();

    SmallVector<Type> convertedResults;
    SmallVector<unsigned> resultWidths;
    if (failed(convertResultTypes(op.getResultTypes(), *typeConverter,
                                  convertedResults, resultWidths)))
      return failure();

    auto newIf = HCIfOp::create(rewriter, op.getLoc(), convertedResults, *cond);
    newIf->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().begin());

    replaceOpWithResultSlices(rewriter, op, newIf->getResults(), resultWidths);
    return success();
  }
};

template <typename OpT>
struct ConvertCollectiveRegionOp : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using Base::Base;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OneToNOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convertedResults;
    SmallVector<unsigned> resultWidths;
    if (failed(convertResultTypes(op->getResultTypes(), *this->typeConverter,
                                  convertedResults, resultWidths)))
      return failure();

    TypeConverter::SignatureConversion bodyConversion(
        op.getBody().front().getNumArguments());
    if (failed(this->typeConverter->convertSignatureArgs(
            op.getBody().front().getArgumentTypes(), bodyConversion)))
      return failure();

    auto newRegion = OpT::create(rewriter, op.getLoc(), convertedResults,
                                 op.getCapturesAttr());
    newRegion->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getBody(), newRegion.getBody(),
                                newRegion.getBody().begin());
    rewriter.applySignatureConversion(&newRegion.getBody().front(),
                                      bodyConversion, this->typeConverter);

    replaceOpWithResultSlices(rewriter, op, newRegion->getResults(),
                              resultWidths);
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
    // Payload and validity channels updated independently; aliasing decided
    // later.
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

struct ConvertYieldOp : public OpConversionPattern<HCYieldOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCYieldOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<HCForRangeOp, HCIfOp, HCWorkitemRegionOp, HCSubgroupRegionOp>(
            op->getParentOp()))
      return failure();
    HCYieldOp::create(rewriter, op.getLoc(),
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
                         *buffer, indices, *shape, /*layout=*/LayoutAttr{});
    // Empty indices: no per-axis carriers to plant predicates from; full_mask
    // seed.
    Value maskValue = indices.empty()
                          ? HCFullMaskOp::create(rewriter, op.getLoc(),
                                                 bareMaskType(originalType))
                                .getMask()
                          : HCLoadMaskOp::create(rewriter, op.getLoc(),
                                                 bareMaskType(originalType),
                                                 *buffer, indices, *shape)
                                .getMask();
    replaceSingleResultWithSplit(rewriter, op, data.getResult(), maskValue);
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
                          dataSource, indices, *shape, /*layout=*/LayoutAttr{});
    Value maskValue;
    if (maskSource) {
      maskValue =
          HCVLoadOp::create(rewriter, op.getLoc(), bareMaskType(originalType),
                            maskSource, indices, *shape,
                            /*layout=*/LayoutAttr{})
              .getResult();
    } else if (indices.empty()) {
      // Broadcast vload: no per-axis carriers → load_mask can't plant.
      maskValue = HCFullMaskOp::create(rewriter, op.getLoc(),
                                       bareMaskType(originalType))
                      .getMask();
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
                            *shape, op.getDtypeAttr(), /*layout=*/LayoutAttr{});
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
                            *fill, *shape, op.getDtypeAttr(),
                            /*layout=*/LayoutAttr{});
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
    // Buffer roots stay undecomposed; one adapted operand → fully valid
    // storage.
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

    // Carry `unit_axes` through; result rank = residual indices + unit axes.
    DenseI64ArrayAttr unitAxes = op.getUnitAxesAttr();
    auto data = HCBufferViewOp::create(rewriter, op.getLoc(),
                                       bareDataType(originalType), dataSource,
                                       indices, unitAxes);
    Value maskValue;
    if (maskSource) {
      maskValue = HCBufferViewOp::create(rewriter, op.getLoc(),
                                         bareMaskType(originalType), maskSource,
                                         indices, unitAxes)
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
    // Buffer roots stay undecomposed; one adapted operand → fully valid
    // storage.
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
    auto data =
        HCVecOp::create(rewriter, op.getLoc(), bareDataType(originalType),
                        source->first, /*layout=*/LayoutAttr{});
    auto mask =
        HCVecOp::create(rewriter, op.getLoc(), bareMaskType(originalType),
                        source->second, /*layout=*/LayoutAttr{});
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getResult());
    return success();
  }
};

// One strip per channel; layout dropped on data and mask independently.
struct ConvertStripLayoutOp : public OpConversionPattern<HCStripLayoutOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCStripLayoutOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalType = op.getResult().getType();
    if (!isSemanticShaped(originalType))
      return failure();

    FailureOr<std::pair<Value, Value>> source =
        expectSplit(adaptor.getValue(), op, "strip_layout source");
    if (failed(source))
      return failure();
    auto data = HCStripLayoutOp::create(
        rewriter, op.getLoc(), bareDataType(originalType), source->first);
    auto mask = HCStripLayoutOp::create(
        rewriter, op.getLoc(), bareMaskType(originalType), source->second);
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

// Data: clone op on data values. Validity: `hc.and` of operand masks,
// typed at the op's broadcast result so both halves share shape downstream.
template <typename OpT>
struct ConvertElementwiseBinaryShapedOp : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using Base::Base;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalResultType = op.getResult().getType();
    if (!isSemanticShaped(originalResultType))
      return failure();

    FailureOr<std::pair<Value, Value>> lhs =
        expectSplit(adaptor.getLhs(), op, "elementwise binary lhs");
    FailureOr<std::pair<Value, Value>> rhs =
        expectSplit(adaptor.getRhs(), op, "elementwise binary rhs");
    if (failed(lhs) || failed(rhs))
      return failure();

    auto data =
        OpT::create(rewriter, op.getLoc(), bareDataType(originalResultType),
                    lhs->first, rhs->first);
    auto mask =
        HCAndOp::create(rewriter, op.getLoc(), bareMaskType(originalResultType),
                        lhs->second, rhs->second);
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getResult());
    return success();
  }
};

// Validity passes through; unary arith doesn't gate lanes further.
template <typename OpT>
struct ConvertElementwiseUnaryShapedOp : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using Base::Base;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalResultType = op.getResult().getType();
    if (!isSemanticShaped(originalResultType))
      return failure();

    FailureOr<std::pair<Value, Value>> source =
        expectSplit(adaptor.getValue(), op, "elementwise unary value");
    if (failed(source))
      return failure();

    auto data = OpT::create(rewriter, op.getLoc(),
                            bareDataType(originalResultType), source->first);
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 source->second);
    return success();
  }
};

// Unary-shape; `target` attr threaded through. Mask passes through.
struct ConvertAsTypeShapedOp : public OpConversionPattern<HCAsTypeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCAsTypeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalResultType = op.getResult().getType();
    if (!isSemanticShaped(originalResultType))
      return failure();

    FailureOr<std::pair<Value, Value>> source =
        expectSplit(adaptor.getValue(), op, "astype value");
    if (failed(source))
      return failure();

    auto data = HCAsTypeOp::create(rewriter, op.getLoc(),
                                   bareDataType(originalResultType),
                                   source->first, op.getTargetAttr());
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 source->second);
    return success();
  }
};

// Single-arg: mask passthrough. Multi-arg: AND of operand masks.
struct ConvertBuiltinCallShapedOp
    : public OpConversionPattern<HCBuiltinCallOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBuiltinCallOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalResultType = op.getResult().getType();
    if (!isSemanticShaped(originalResultType))
      return failure();

    SmallVector<Value> dataArgs;
    SmallVector<Value> maskArgs;
    dataArgs.reserve(adaptor.getArgs().size());
    maskArgs.reserve(adaptor.getArgs().size());
    for (auto [index, group] : llvm::enumerate(adaptor.getArgs())) {
      FailureOr<std::pair<Value, Value>> split = expectSplit(
          group, op, Twine("builtin_call arg #").concat(Twine(index)).str());
      if (failed(split))
        return failure();
      dataArgs.push_back(split->first);
      maskArgs.push_back(split->second);
    }

    auto data = HCBuiltinCallOp::create(rewriter, op.getLoc(),
                                        bareDataType(originalResultType),
                                        op.getNameAttr(), dataArgs);

    Value maskValue = maskArgs.front();
    for (Value m : ArrayRef<Value>(maskArgs).drop_front()) {
      maskValue =
          HCAndOp::create(rewriter, op.getLoc(),
                          bareMaskType(originalResultType), maskValue, m)
              .getResult();
    }

    replaceSingleResultWithSplit(rewriter, op, data.getResult(), maskValue);
    return success();
  }
};

// Data: clone reduction. Mask: full_mask of reduced shape (invalid lanes
// already neutral via upstream init/load_mask; per-tile validity is
// store-side).
struct ConvertReduceShapedOp : public OpConversionPattern<HCReduceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCReduceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalResultType = op.getResult().getType();
    if (!isSemanticShaped(originalResultType))
      return failure();

    FailureOr<std::pair<Value, Value>> value =
        expectSplit(adaptor.getValue(), op, "reduce value");
    if (failed(value))
      return failure();

    auto data = HCReduceOp::create(
        rewriter, op.getLoc(), bareDataType(originalResultType), value->first,
        op.getKindAttr(), op.getAxisAttr(), op.getKeepdimsAttr());
    auto mask = HCFullMaskOp::create(rewriter, op.getLoc(),
                                     bareMaskType(originalResultType));
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getMask());
    return success();
  }
};

// Mask: full_mask over result. Invalid K lanes already carry additive identity
// via upstream load_mask / fill; per-tile validity is store-side.
struct ConvertMatmulShapedOp : public OpConversionPattern<HCMatmulOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCMatmulOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type originalResultType = op.getResult().getType();
    if (!isSemanticShaped(originalResultType))
      return failure();

    FailureOr<std::pair<Value, Value>> lhs =
        expectSplit(adaptor.getLhs(), op, "matmul lhs");
    FailureOr<std::pair<Value, Value>> rhs =
        expectSplit(adaptor.getRhs(), op, "matmul rhs");
    if (failed(lhs) || failed(rhs))
      return failure();

    auto data = HCMatmulOp::create(rewriter, op.getLoc(),
                                   bareDataType(originalResultType), lhs->first,
                                   rhs->first);
    auto mask = HCFullMaskOp::create(rewriter, op.getLoc(),
                                     bareMaskType(originalResultType));
    replaceSingleResultWithSplit(rewriter, op, data.getResult(),
                                 mask.getMask());
    return success();
  }
};

static void populateShapedDecompositionPatterns(TypeConverter &converter,
                                                MLIRContext *ctx,
                                                RewritePatternSet &patterns) {
  patterns.add<ConvertCallableSignatureOp<HCKernelOp>,
               ConvertCallableSignatureOp<HCFuncOp>,
               ConvertCallableSignatureOp<HCIntrinsicOp>, ConvertCallOp,
               ConvertCallIntrinsicOp, ConvertForRangeOp, ConvertIfOp,
               ConvertCollectiveRegionOp<HCWorkitemRegionOp>,
               ConvertCollectiveRegionOp<HCSubgroupRegionOp>, ConvertStoreOp,
               ConvertReturnOp, ConvertYieldOp>(converter, ctx);
  patterns
      .add<ConvertLoadOp, ConvertVLoadOp, ConvertBufferViewOp, ConvertGetItemOp,
           ConvertVecOp, ConvertStripLayoutOp, ConvertWithInactiveOp>(converter,
                                                                      ctx);
  patterns
      .add<ConvertNullaryAllocOp<HCVZerosOp>, ConvertNullaryAllocOp<HCVOnesOp>,
           ConvertNullaryAllocOp<HCZerosOp>, ConvertNullaryAllocOp<HCOnesOp>,
           ConvertNullaryAllocOp<HCEmptyOp>, ConvertFillAllocOp<HCVFullOp>,
           ConvertFillAllocOp<HCFullOp>>(converter, ctx);
  patterns.add<ConvertElementwiseBinaryShapedOp<HCAddOp>,
               ConvertElementwiseBinaryShapedOp<HCSubOp>,
               ConvertElementwiseBinaryShapedOp<HCMulOp>,
               ConvertElementwiseBinaryShapedOp<HCDivOp>,
               ConvertElementwiseBinaryShapedOp<HCModOp>,
               ConvertElementwiseUnaryShapedOp<HCNegOp>,
               ConvertElementwiseUnaryShapedOp<HCNotOp>, ConvertAsTypeShapedOp,
               ConvertBuiltinCallShapedOp, ConvertReduceShapedOp,
               ConvertMatmulShapedOp>(converter, ctx);
}

static ConversionTarget
makeShapedDecompositionTarget(MLIRContext *ctx,
                              const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addDynamicallyLegalOp<HCKernelOp, HCFuncOp, HCIntrinsicOp>(
      [&](Operation *op) {
        return regionsAreLegal(op, converter) &&
               callableSignatureIsLegal(op, converter);
      });
  target.addDynamicallyLegalOp<HCCallOp, HCCallIntrinsicOp, HCStoreOp,
                               HCReturnOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<HCAddOp, HCSubOp, HCMulOp, HCDivOp, HCModOp,
                               HCNegOp, HCNotOp, HCAsTypeOp, HCBuiltinCallOp,
                               HCReduceOp, HCMatmulOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<HCForRangeOp, HCIfOp, HCWorkitemRegionOp,
                               HCSubgroupRegionOp>([&](Operation *op) {
    return converter.isLegal(op) && regionsAreLegal(op, converter);
  });
  target.addDynamicallyLegalOp<HCYieldOp>([&](Operation *op) {
    if (!isa<HCForRangeOp, HCIfOp, HCWorkitemRegionOp, HCSubgroupRegionOp>(
            op->getParentOp()))
      return true;
    return converter.isLegal(op);
  });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return converter.isLegal(op) && regionsAreLegal(op, converter) &&
           callableSignatureIsLegal(op, converter);
  });
  return target;
}

// Belt-and-suspenders sweep: catch any semantic carrier the target's
// legality lists missed; diagnostic names the offender at the source.
static bool isSemanticShapedType(Type type) {
  return isa<mlir::hc::TensorType, mlir::hc::VectorType>(type);
}

static LogicalResult assertNoSemanticShapedSurvives(Operation *rootOp) {
  WalkResult walk = rootOp->walk([&](Operation *op) {
    auto bail = [&](Type type, StringRef role) -> WalkResult {
      op->emitOpError("semantic shaped type ")
          << type << " survived hc-decompose-shaped-values on " << role
          << "; every !hc.tensor / !hc.vector must split into bare "
             "(data, mask) pairs before the pass returns";
      return WalkResult::interrupt();
    };
    for (Value v : op->getOperands())
      if (isSemanticShapedType(v.getType()))
        return bail(v.getType(), "operand");
    for (Type t : op->getResultTypes())
      if (isSemanticShapedType(t))
        return bail(t, "result");
    for (Region &region : op->getRegions())
      for (Block &block : region.getBlocks())
        for (BlockArgument arg : block.getArguments())
          if (isSemanticShapedType(arg.getType()))
            return bail(arg.getType(), "block argument");
    return WalkResult::advance();
  });
  return success(!walk.wasInterrupted());
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

    ConversionTarget target = makeShapedDecompositionTarget(ctx, converter);
    if (failed(applyFullConversion(getOperation(), target, frozenPatterns)))
      return signalPassFailure();
    if (failed(assertNoSemanticShapedSurvives(getOperation())))
      return signalPassFailure();
  }
};

} // namespace
