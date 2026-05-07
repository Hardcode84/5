// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-launch-body`, the launch-body lowering slice that runs
// after HC kernels have been wrapped in `gpu.launch`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

#include <type_traits>

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLAUNCHBODY
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static ixs_node *rawNode(ExprAttr expr) {
  return const_cast<ixs_node *>(expr.getNode());
}

static ixs_node *rawNode(PredAttr pred) {
  return const_cast<ixs_node *>(pred.getNode());
}

static std::optional<StringRef> exactSymbolName(ExprAttr expr) {
  if (!expr)
    return std::nullopt;
  ixs_node *node = rawNode(expr);
  if (ixs_node_tag(node) != IXS_SYM)
    return std::nullopt;
  return StringRef(ixs_node_sym_name(node));
}

static std::optional<StringRef> exactSymbolName(Type type) {
  auto idx = dyn_cast<IdxType>(type);
  return idx ? exactSymbolName(idx.getExpr()) : std::nullopt;
}

static bool isOneNode(ixs_node *node) {
  return ixs_node_tag(node) == IXS_INT && ixs_node_int_val(node) == 1;
}

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

static FailureOr<SmallVector<int64_t>>
staticIntegerShape(SymbolicallyShapedTypeInterface shaped) {
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(shaped.getSymbolicShape(), nullptr);
  if (failed(dims))
    return failure();
  return *dims;
}

static Type convertBareTensorType(BareTensorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(shaped);
  if (failed(dims))
    return {};
  Attribute memorySpace = gpu::AddressSpaceAttr::get(
      type.getContext(), gpu::AddressSpace::Workgroup);
  return MemRefType::get(*dims, element, MemRefLayoutAttrInterface{},
                         memorySpace);
}

static Type convertBareVectorType(BareVectorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(shaped);
  if (failed(dims))
    return {};
  if (dims->empty())
    return element;
  for (int64_t dim : *dims)
    if (dim <= 0)
      return {};
  return mlir::VectorType::get(*dims, element);
}

static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                             Location loc) {
  if (inputs.size() != 1)
    return {};
  return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
      .getResult(0);
}

static Value castIfNeeded(OpBuilder &builder, Location loc, Value value,
                          Type type) {
  if (value.getType() == type)
    return value;
  return UnrealizedConversionCastOp::create(builder, loc, type, value)
      .getResult(0);
}

class HCLaunchBodyTypeConverter : public TypeConverter {
public:
  HCLaunchBodyTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });
    addConversion(
        [](IdxType type) -> Type { return IndexType::get(type.getContext()); });
    addConversion([](PredType type) -> Type {
      return IntegerType::get(type.getContext(), 1);
    });
    addConversion([](BareTensorType type) -> std::optional<Type> {
      if (Type converted = convertBareTensorType(type))
        return converted;
      return Type(type);
    });
    addConversion([](BareVectorType type) -> std::optional<Type> {
      if (Type converted = convertBareVectorType(type))
        return converted;
      return Type(type);
    });
    addConversion([&](TupleType type) -> std::optional<Type> {
      SmallVector<Type> elements;
      if (failed(convertTypes(type.getTypes(), elements)))
        return std::nullopt;
      return TupleType::get(type.getContext(), elements);
    });
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

struct BoundValues {
  llvm::StringMap<Value> symbols;

  void bind(StringRef name, Value value) {
    if (!name.empty())
      symbols.try_emplace(name, value);
  }

  Value lookup(StringRef name) const {
    auto it = symbols.find(name);
    return it == symbols.end() ? Value{} : it->second;
  }
};

static Value dim(OpBuilder &builder, Location loc, Value memref, int64_t axis) {
  return memref::DimOp::create(builder, loc, memref, axis).getResult();
}

static void bindShapeSymbols(OpBuilder &builder, Location loc, BufferType type,
                             Value memref, BoundValues &boundValues) {
  for (auto [axis, attr] : llvm::enumerate(type.getShape().getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<StringRef> symbol = exactSymbolName(expr);
    if (!symbol)
      continue;
    boundValues.bind(*symbol, dim(builder, loc, memref, axis));
  }
}

static void bindLaunchDim3(StringRef prefix, gpu::KernelDim3 values,
                           BoundValues &boundValues) {
  Value dims[] = {values.x, values.y, values.z};
  for (auto [axis, value] : llvm::enumerate(dims)) {
    SmallString<16> name(prefix);
    name += Twine(axis).str();
    boundValues.bind(name, value);
  }
}

static Value indexCast(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), value)
      .getResult(0);
}

static BoundValues collectBoundValues(Operation *anchor,
                                      ConversionPatternRewriter &rewriter) {
  BoundValues boundValues;
  auto launch = anchor->getParentOfType<gpu::LaunchOp>();
  if (!launch)
    return boundValues;

  bindLaunchDim3("$WG", launch.getBlockIds(), boundValues);
  bindLaunchDim3("$WI", launch.getThreadIds(), boundValues);
  bindLaunchDim3("$WGS", launch.getBlockSizeOperandValues(), boundValues);

  launch.walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1)
      return;
    Value input = cast.getInputs().front();
    Type outputType = cast.getOutputs().front().getType();

    if (auto buffer = dyn_cast<BufferType>(outputType)) {
      if (isa<MemRefType>(input.getType()))
        bindShapeSymbols(rewriter, anchor->getLoc(), buffer, input,
                         boundValues);
      return;
    }

    if (std::optional<StringRef> symbol = exactSymbolName(outputType))
      boundValues.bind(*symbol, indexCast(rewriter, anchor->getLoc(), input));
  });

  return boundValues;
}

class ExprLowerer {
public:
  ExprLowerer(OpBuilder &builder, Location loc, const BoundValues &boundValues)
      : builder(builder), loc(loc), boundValues(boundValues) {}

  FailureOr<Value> lower(ExprAttr expr) {
    if (!expr)
      return failure();
    return lowerNode(rawNode(expr));
  }

  FailureOr<Value> lower(PredAttr pred) {
    if (!pred)
      return failure();
    return lowerPredNode(rawNode(pred));
  }

  Value constant(int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value).getResult();
  }

private:
  FailureOr<Value> lowerNode(ixs_node *node) {
    switch (ixs_node_tag(node)) {
    case IXS_INT:
      return constant(ixs_node_int_val(node));
    case IXS_RAT:
      return lowerRational(node);
    case IXS_SYM:
      return lowerSymbol(node);
    case IXS_ADD:
      return lowerAdd(node);
    case IXS_MUL:
      return lowerMul(node);
    case IXS_CEIL:
      return lowerCeil(node);
    case IXS_FLOOR:
      return lowerNode(ixs_node_unary_arg(node));
    case IXS_MOD:
      return lowerBinary<arith::RemUIOp>(node);
    default:
      return failure();
    }
  }

  FailureOr<Value> lowerPredNode(ixs_node *node) {
    switch (ixs_node_tag(node)) {
    case IXS_TRUE:
      return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true))
          .getResult();
    case IXS_FALSE:
      return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(false))
          .getResult();
    case IXS_CMP:
      return lowerCmp(node);
    case IXS_AND:
      return lowerLogic<arith::AndIOp>(node);
    case IXS_OR:
      return lowerLogic<arith::OrIOp>(node);
    case IXS_NOT: {
      FailureOr<Value> value = lowerPredNode(ixs_node_unary_arg(node));
      if (failed(value))
        return failure();
      Value trueValue =
          arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
      return arith::XOrIOp::create(builder, loc, *value, trueValue).getResult();
    }
    default:
      return failure();
    }
  }

  FailureOr<Value> lowerRational(ixs_node *node) {
    int64_t numerator = ixs_node_rat_num(node);
    int64_t denominator = ixs_node_rat_den(node);
    if (denominator == 1)
      return constant(numerator);
    return failure();
  }

  FailureOr<Value> lowerSymbol(ixs_node *node) {
    StringRef name(ixs_node_sym_name(node));
    if (Value value = boundValues.lookup(name))
      return value;
    return failure();
  }

  FailureOr<Value> lowerAdd(ixs_node *node) {
    FailureOr<Value> result = lowerNode(ixs_node_add_coeff(node));
    if (failed(result))
      return failure();

    for (uint32_t index = 0, end = ixs_node_add_nterms(node); index != end;
         ++index) {
      FailureOr<Value> term = lowerNode(ixs_node_add_term(node, index));
      if (failed(term))
        return failure();
      FailureOr<Value> scaled =
          scaleTerm(ixs_node_add_term_coeff(node, index), *term);
      if (failed(scaled))
        return failure();
      *result = arith::AddIOp::create(builder, loc, *result, *scaled);
    }
    return *result;
  }

  FailureOr<Value> scaleTerm(ixs_node *coeff, Value term) {
    if (isOneNode(coeff))
      return term;
    if (ixs_node_tag(coeff) == IXS_INT) {
      Value coeffValue = constant(ixs_node_int_val(coeff));
      return arith::MulIOp::create(builder, loc, coeffValue, term).getResult();
    }
    if (ixs_node_tag(coeff) != IXS_RAT)
      return failure();

    int64_t numerator = ixs_node_rat_num(coeff);
    int64_t denominator = ixs_node_rat_den(coeff);
    if (denominator <= 0)
      return failure();
    Value scaled = term;
    if (numerator != 1)
      scaled = arith::MulIOp::create(builder, loc, constant(numerator), term);
    if (denominator != 1)
      scaled =
          arith::DivUIOp::create(builder, loc, scaled, constant(denominator));
    return scaled;
  }

  FailureOr<Value> lowerMul(ixs_node *node) {
    FailureOr<std::pair<Value, Value>> fraction = lowerAsFraction(node);
    if (failed(fraction))
      return failure();
    if (auto denominator =
            fraction->second.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<IntegerAttr>(denominator.getValue()))
        if (attr.getInt() == 1)
          return fraction->first;
    }
    return arith::DivUIOp::create(builder, loc, fraction->first,
                                  fraction->second)
        .getResult();
  }

  FailureOr<Value> lowerCeil(ixs_node *node) {
    FailureOr<std::pair<Value, Value>> fraction =
        lowerAsFraction(ixs_node_unary_arg(node));
    if (failed(fraction))
      return failure();
    return arith::CeilDivUIOp::create(builder, loc, fraction->first,
                                      fraction->second)
        .getResult();
  }

  FailureOr<std::pair<Value, Value>> lowerAsFraction(ixs_node *node) {
    if (ixs_node_tag(node) != IXS_MUL) {
      FailureOr<Value> value = lowerNode(node);
      if (failed(value))
        return failure();
      return std::pair<Value, Value>{*value, constant(1)};
    }

    ixs_node *coeff = ixs_node_mul_coeff(node);
    int64_t numerator = 1;
    int64_t denominator = 1;
    if (ixs_node_tag(coeff) == IXS_INT) {
      numerator = ixs_node_int_val(coeff);
    } else if (ixs_node_tag(coeff) == IXS_RAT) {
      numerator = ixs_node_rat_num(coeff);
      denominator = ixs_node_rat_den(coeff);
    } else {
      return failure();
    }
    if (denominator <= 0)
      return failure();

    Value numeratorValue = constant(numerator);
    for (uint32_t index = 0, end = ixs_node_mul_nfactors(node); index != end;
         ++index) {
      int32_t exponent = ixs_node_mul_factor_exp(node, index);
      if (exponent < 0)
        return failure();
      FailureOr<Value> factor =
          lowerNode(ixs_node_mul_factor_base(node, index));
      if (failed(factor))
        return failure();
      for (int32_t power = 0; power < exponent; ++power)
        numeratorValue =
            arith::MulIOp::create(builder, loc, numeratorValue, *factor);
    }
    return std::pair<Value, Value>{numeratorValue, constant(denominator)};
  }

  template <typename OpT> FailureOr<Value> lowerBinary(ixs_node *node) {
    FailureOr<Value> lhs = lowerNode(ixs_node_binary_lhs(node));
    FailureOr<Value> rhs = lowerNode(ixs_node_binary_rhs(node));
    if (failed(lhs) || failed(rhs))
      return failure();
    return OpT::create(builder, loc, *lhs, *rhs).getResult();
  }

  FailureOr<Value> lowerCmp(ixs_node *node) {
    FailureOr<Value> lhs = lowerNode(ixs_node_binary_lhs(node));
    FailureOr<Value> rhs = lowerNode(ixs_node_binary_rhs(node));
    if (failed(lhs) || failed(rhs))
      return failure();
    arith::CmpIPredicate pred = arith::CmpIPredicate::eq;
    switch (ixs_node_cmp_op(node)) {
    case IXS_CMP_LT:
      pred = arith::CmpIPredicate::slt;
      break;
    case IXS_CMP_LE:
      pred = arith::CmpIPredicate::sle;
      break;
    case IXS_CMP_GT:
      pred = arith::CmpIPredicate::sgt;
      break;
    case IXS_CMP_GE:
      pred = arith::CmpIPredicate::sge;
      break;
    case IXS_CMP_EQ:
      pred = arith::CmpIPredicate::eq;
      break;
    case IXS_CMP_NE:
      pred = arith::CmpIPredicate::ne;
      break;
    }
    return arith::CmpIOp::create(builder, loc, pred, *lhs, *rhs).getResult();
  }

  template <typename OpT> FailureOr<Value> lowerLogic(ixs_node *node) {
    uint32_t count = ixs_node_logic_nargs(node);
    if (count == 0)
      return failure();
    FailureOr<Value> result = lowerPredNode(ixs_node_logic_arg(node, 0));
    if (failed(result))
      return failure();
    for (uint32_t index = 1; index != count; ++index) {
      FailureOr<Value> arg = lowerPredNode(ixs_node_logic_arg(node, index));
      if (failed(arg))
        return failure();
      *result = OpT::create(builder, loc, *result, *arg).getResult();
    }
    return *result;
  }

  OpBuilder &builder;
  Location loc;
  const BoundValues &boundValues;
};

static Value sourceMemRef(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (cast && cast.getInputs().size() == 1 && cast.getOutputs().size() == 1) {
    Value input = cast.getInputs().front();
    if (isa<MemRefType>(input.getType()))
      return input;
  }
  return isa<MemRefType>(source.getType()) ? source : Value();
}

static Type convertIntrinsicBoundaryType(Type type,
                                         const TypeConverter &converter) {
  if (isa<BareTensorType, BareVectorType>(type))
    return type;
  auto tuple = dyn_cast<TupleType>(type);
  if (!tuple)
    return converter.convertType(type);

  SmallVector<Type> elements;
  elements.reserve(tuple.size());
  for (Type element : tuple.getTypes()) {
    Type converted = convertIntrinsicBoundaryType(element, converter);
    if (!converted)
      return {};
    elements.push_back(converted);
  }
  return TupleType::get(type.getContext(), elements);
}

static LogicalResult convertIntrinsicFunctionSignature(
    FunctionType fnType, const TypeConverter &converter,
    SmallVectorImpl<Type> &inputs, SmallVectorImpl<Type> &results) {
  for (Type input : fnType.getInputs()) {
    Type converted = convertIntrinsicBoundaryType(input, converter);
    if (!converted)
      return failure();
    inputs.push_back(converted);
  }
  for (Type result : fnType.getResults()) {
    Type converted = convertIntrinsicBoundaryType(result, converter);
    if (!converted)
      return failure();
    results.push_back(converted);
  }
  return success();
}

static LogicalResult convertIntrinsicBodySignature(
    FunctionType fnType, TypeConverter::SignatureConversion &bodyConversion,
    const TypeConverter &converter) {
  for (auto [index, input] : llvm::enumerate(fnType.getInputs())) {
    Type converted = convertIntrinsicBoundaryType(input, converter);
    if (!converted)
      return failure();
    bodyConversion.addInputs(index, converted);
  }
  return success();
}

struct ConvertIntrinsicSignatureOp : public OpConversionPattern<HCIntrinsicOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIntrinsicOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<FunctionType> fnType = op.getFunctionType();
    if (!fnType)
      return failure();

    SmallVector<Type> inputs;
    SmallVector<Type> results;
    if (failed(convertIntrinsicFunctionSignature(*fnType, *typeConverter,
                                                 inputs, results)))
      return failure();

    auto converted = FunctionType::get(rewriter.getContext(), inputs, results);
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(converted);
      if (!op.getBody().empty()) {
        TypeConverter::SignatureConversion bodyConversion(
            fnType->getNumInputs());
        (void)convertIntrinsicBodySignature(*fnType, bodyConversion,
                                            *typeConverter);
        rewriter.applySignatureConversion(&op.getBody().front(), bodyConversion,
                                          typeConverter);
      }
    });
    return success();
  }
};

struct ConvertMaterializeBoundExprOp
    : public OpConversionPattern<HCMaterializeBoundExprOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCMaterializeBoundExprOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    BoundValues boundValues = collectBoundValues(op, rewriter);
    ExprLowerer lowerer(rewriter, op.getLoc(), boundValues);

    if (auto idx = dyn_cast<IdxType>(op.getResult().getType())) {
      FailureOr<Value> lowered = lowerer.lower(idx.getExpr());
      if (failed(lowered))
        return op.emitOpError("failed to lower bound index expression");
      rewriter.replaceOp(op, *lowered);
      return success();
    }
    if (auto pred = dyn_cast<PredType>(op.getResult().getType())) {
      FailureOr<Value> lowered = lowerer.lower(pred.getPred());
      if (failed(lowered))
        return op.emitOpError("failed to lower bound predicate expression");
      rewriter.replaceOp(op, *lowered);
      return success();
    }
    return failure();
  }
};

static FailureOr<TypedAttr> splatAttr(OpBuilder &builder, Type type,
                                      int64_t value) {
  Type elementType = type;
  if (auto shaped = dyn_cast<ShapedType>(type))
    elementType = shaped.getElementType();

  Attribute scalar;
  if (elementType.isInteger(1))
    scalar = builder.getBoolAttr(value != 0);
  else if (elementType.isIndex())
    scalar = builder.getIndexAttr(value);
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    scalar = builder.getIntegerAttr(intType, value);
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    scalar = builder.getFloatAttr(floatType, value);
  else
    return failure();

  if (auto shaped = dyn_cast<ShapedType>(type))
    return cast<TypedAttr>(DenseElementsAttr::get(shaped, scalar));
  return cast<TypedAttr>(scalar);
}

static Value constantSplat(OpBuilder &builder, Location loc, Type type,
                           int64_t value) {
  FailureOr<TypedAttr> attr = splatAttr(builder, type, value);
  if (failed(attr))
    return {};
  return arith::ConstantOp::create(builder, loc, type, *attr).getResult();
}

struct ConvertConstOp : public OpConversionPattern<HCConstOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCConstOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();

    Attribute value = op.getValue();
    if (converted.isIndex()) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return op.emitOpError("expected integer literal for index constant");
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, integer.getInt());
      return success();
    }

    if (auto typed = dyn_cast<TypedAttr>(value)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, converted, typed);
      return success();
    }
    return failure();
  }
};

template <typename OpT, typename ArithOpT>
struct ConvertIntBinaryOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted || !converted.isIntOrIndex())
      return failure();
    rewriter.template replaceOpWithNewOp<ArithOpT>(
        op, converted, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SliceAxis {
  Value offset;
  bool isSlice = false;
};

static Value zeroIndex(OpBuilder &builder, Location loc) {
  return arith::ConstantIndexOp::create(builder, loc, 0).getResult();
}

static LogicalResult collectSliceAxis(Operation *op, Value index,
                                      OpBuilder &builder, SliceAxis &axis) {
  auto slice = index.getDefiningOp<HCSliceExprOp>();
  if (!slice)
    return failure();
  if (Value step = slice.getStep()) {
    APInt stepValue;
    if (!matchPattern(step, m_ConstantInt(&stepValue)) ||
        stepValue.getSExtValue() != 1)
      return op->emitOpError("only unit-stride slices lower to vector ops");
  }
  axis.offset =
      slice.getLower() ? slice.getLower() : zeroIndex(builder, op->getLoc());
  axis.isSlice = true;
  return success();
}

static FailureOr<SmallVector<SliceAxis>>
collectAxes(Operation *op, ValueRange indices, OpBuilder &builder) {
  SmallVector<SliceAxis> axes;
  axes.reserve(indices.size());
  for (Value index : indices) {
    SliceAxis axis;
    if (isa<SliceType>(index.getType())) {
      if (failed(collectSliceAxis(op, index, builder, axis)))
        return failure();
    } else {
      if (!index.getType().isIndex())
        return op->emitOpError("expected index or slice subscript after "
                               "launch-body scalar lowering");
      axis.offset = index;
    }
    axes.push_back(axis);
  }
  return axes;
}

static AffineMap transferPermutationMap(MLIRContext *ctx, int64_t sourceRank,
                                        ArrayRef<SliceAxis> axes) {
  SmallVector<AffineExpr> results;
  results.reserve(axes.size());
  for (auto [axis, info] : llvm::enumerate(axes))
    if (info.isSlice)
      results.push_back(getAffineDimExpr(axis, ctx));
  return AffineMap::get(sourceRank, 0, results, ctx);
}

static SmallVector<Value> zeroOffsets(OpBuilder &builder, Location loc,
                                      int64_t rank) {
  SmallVector<Value> offsets;
  offsets.reserve(rank);
  for (int64_t axis = 0; axis != rank; ++axis)
    offsets.push_back(zeroIndex(builder, loc));
  return offsets;
}

static mlir::VectorType vectorTypeForMemRef(MemRefType memrefType) {
  if (!memrefType.hasStaticShape())
    return {};
  return mlir::VectorType::get(memrefType.getShape(),
                               memrefType.getElementType());
}

static Value allocateWorkgroupMemRef(OpBuilder &builder, Location loc,
                                     MemRefType type) {
  return memref::AllocaOp::create(builder, loc, type).getResult();
}

static LogicalResult writeVectorToMemRef(OpBuilder &builder, Location loc,
                                         Value vector, Value memref) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType)
    return failure();
  vector::TransferWriteOp::create(
      builder, loc, vector, memref,
      zeroOffsets(builder, loc, memrefType.getRank()));
  return success();
}

static FailureOr<Value> readMemRefAsVector(OpBuilder &builder, Location loc,
                                           Value memref,
                                           mlir::VectorType vectorType) {
  memref = sourceMemRef(memref);
  if (!memref)
    return failure();
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType || memrefType.getRank() != vectorType.getRank())
    return failure();
  Value padding = constantSplat(builder, loc, vectorType.getElementType(), 0);
  if (!padding)
    return failure();
  return vector::TransferReadOp::create(
             builder, loc, vectorType, memref,
             zeroOffsets(builder, loc, memrefType.getRank()),
             std::optional<Value>(padding))
      .getResult();
}

template <typename OpT>
struct ConvertLoadLikeOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    auto resultMemRefType = dyn_cast_if_present<MemRefType>(converted);
    auto resultVectorType = dyn_cast_if_present<mlir::VectorType>(converted);
    mlir::VectorType transferVectorType =
        resultMemRefType ? vectorTypeForMemRef(resultMemRefType)
                         : resultVectorType;
    if (!transferVectorType)
      return failure();

    Value source = [&]() -> Value {
      if constexpr (std::is_same_v<OpT, HCLoadOp>)
        return adaptor.getBuffer();
      else
        return adaptor.getSource();
    }();
    Value memref = sourceMemRef(source);
    if (!memref)
      return op.emitOpError(
          "expected load source to be a memref or memref ABI cast");
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || memrefType.getRank() !=
                           static_cast<int64_t>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected ranked memref with one subscript per axis");

    FailureOr<SmallVector<SliceAxis>> axes =
        collectAxes(op.getOperation(), adaptor.getIndices(), rewriter);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != transferVectorType.getRank())
      return op.emitOpError("load result rank must match slice subscript rank");

    SmallVector<Value> offsets;
    offsets.reserve(axes->size());
    for (const SliceAxis &axis : *axes)
      offsets.push_back(axis.offset);

    Value padding = constantSplat(rewriter, op.getLoc(),
                                  transferVectorType.getElementType(), 0);
    if (!padding)
      return failure();

    Value loaded = vector::TransferReadOp::create(
                       rewriter, op.getLoc(), transferVectorType, memref,
                       offsets, std::optional<Value>(padding),
                       transferPermutationMap(rewriter.getContext(),
                                              memrefType.getRank(), *axes))
                       .getResult();
    if (resultVectorType) {
      rewriter.replaceOp(op, loaded);
      return success();
    }

    Value shared =
        allocateWorkgroupMemRef(rewriter, op.getLoc(), resultMemRefType);
    if (failed(writeVectorToMemRef(rewriter, op.getLoc(), loaded, shared)))
      return failure();
    rewriter.replaceOp(op, shared);
    return success();
  }
};

struct ConvertLoadMaskOp : public OpConversionPattern<HCLoadMaskOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCLoadMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getMask().getType());
    auto resultMemRefType = dyn_cast_if_present<MemRefType>(converted);
    auto resultVectorType = dyn_cast_if_present<mlir::VectorType>(converted);
    mlir::VectorType maskType = resultMemRefType
                                    ? vectorTypeForMemRef(resultMemRefType)
                                    : resultVectorType;
    if (!maskType || !maskType.getElementType().isInteger(1))
      return failure();

    Value memref = sourceMemRef(adaptor.getSource());
    if (!memref)
      return op.emitOpError(
          "expected mask source to be a memref or memref ABI cast");
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || memrefType.getRank() !=
                           static_cast<int64_t>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected ranked memref with one subscript per axis");

    FailureOr<SmallVector<SliceAxis>> axes =
        collectAxes(op.getOperation(), adaptor.getIndices(), rewriter);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != maskType.getRank())
      return op.emitOpError("mask result rank must match slice subscript rank");

    SmallVector<Value> maskSizes;
    for (auto [axis, info] : llvm::enumerate(*axes)) {
      if (!info.isSlice)
        continue;
      Value extent = dim(rewriter, op.getLoc(), memref, axis);
      maskSizes.push_back(
          arith::SubIOp::create(rewriter, op.getLoc(), extent, info.offset));
    }

    Value mask = vector::CreateMaskOp::create(rewriter, op.getLoc(), maskType,
                                              maskSizes);
    if (resultVectorType) {
      rewriter.replaceOp(op, mask);
      return success();
    }

    Value shared =
        allocateWorkgroupMemRef(rewriter, op.getLoc(), resultMemRefType);
    if (failed(writeVectorToMemRef(rewriter, op.getLoc(), mask, shared)))
      return failure();
    rewriter.replaceOp(op, shared);
    return success();
  }
};

struct ConvertFullMaskOp : public OpConversionPattern<HCFullMaskOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCFullMaskOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getMask().getType());
    if (!converted)
      return failure();
    if (auto memrefType = dyn_cast<MemRefType>(converted)) {
      mlir::VectorType vectorType = vectorTypeForMemRef(memrefType);
      if (!vectorType)
        return failure();
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, 1);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      Value shared = allocateWorkgroupMemRef(rewriter, op.getLoc(), memrefType);
      if (failed(writeVectorToMemRef(rewriter, op.getLoc(), vector, shared)))
        return failure();
      rewriter.replaceOp(op, shared);
      return success();
    }
    FailureOr<TypedAttr> attr = splatAttr(rewriter, converted, 1);
    if (failed(attr))
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, converted, *attr);
    return success();
  }
};

template <typename OpT, int64_t FillValue>
struct ConvertNullaryShapedConstantOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (auto memrefType = dyn_cast<MemRefType>(converted)) {
      mlir::VectorType vectorType = vectorTypeForMemRef(memrefType);
      if (!vectorType)
        return failure();
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, FillValue);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      Value shared = allocateWorkgroupMemRef(rewriter, op.getLoc(), memrefType);
      if (failed(writeVectorToMemRef(rewriter, op.getLoc(), vector, shared)))
        return failure();
      rewriter.replaceOp(op, shared);
      return success();
    }
    FailureOr<TypedAttr> attr = splatAttr(rewriter, converted, FillValue);
    if (failed(attr))
      return failure();
    rewriter.template replaceOpWithNewOp<arith::ConstantOp>(op, converted,
                                                            *attr);
    return success();
  }
};

template <typename OpT>
struct ConvertFillShapedConstantOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    auto vectorType = dyn_cast_if_present<mlir::VectorType>(converted);
    auto memrefType = dyn_cast_if_present<MemRefType>(converted);
    if (!vectorType && memrefType)
      vectorType = vectorTypeForMemRef(memrefType);
    if (!vectorType)
      return failure();
    Value vector = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                               vectorType, adaptor.getValue());
    if (!memrefType) {
      rewriter.replaceOp(op, vector);
      return success();
    }
    Value shared = allocateWorkgroupMemRef(rewriter, op.getLoc(), memrefType);
    if (failed(writeVectorToMemRef(rewriter, op.getLoc(), vector, shared)))
      return failure();
    rewriter.replaceOp(op, shared);
    return success();
  }
};

struct ConvertEmptyOp : public OpConversionPattern<HCEmptyOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCEmptyOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = dyn_cast_if_present<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!memrefType)
      return failure();
    rewriter.replaceOp(
        op, allocateWorkgroupMemRef(rewriter, op.getLoc(), memrefType));
    return success();
  }
};

struct ConvertVecOp : public OpConversionPattern<HCVecOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCVecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    auto vectorType = dyn_cast_if_present<mlir::VectorType>(converted);
    if (isa<MemRefType>(adaptor.getValue().getType())) {
      if (!vectorType)
        return failure();
      FailureOr<Value> vector = readMemRefAsVector(
          rewriter, op.getLoc(), adaptor.getValue(), vectorType);
      if (failed(vector))
        return failure();
      rewriter.replaceOp(op, *vector);
      return success();
    }
    if (!converted || converted != adaptor.getValue().getType())
      return failure();
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<HCSelectOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();

    if (auto memrefType = dyn_cast<MemRefType>(converted)) {
      mlir::VectorType vectorType = vectorTypeForMemRef(memrefType);
      if (!vectorType)
        return failure();

      mlir::VectorType maskType =
          mlir::VectorType::get(vectorType.getShape(), rewriter.getI1Type());
      FailureOr<Value> condition = readMemRefAsVector(
          rewriter, op.getLoc(), adaptor.getCondition(), maskType);
      FailureOr<Value> trueValue = readMemRefAsVector(
          rewriter, op.getLoc(), adaptor.getTrueValue(), vectorType);
      if (failed(condition) || failed(trueValue))
        return op.emitOpError(
            "expected tensor select operands to lower to memrefs");

      Value falseValue =
          vector::BroadcastOp::create(rewriter, op.getLoc(), vectorType,
                                      adaptor.getFalseValue())
              .getResult();
      Value selected =
          arith::SelectOp::create(rewriter, op.getLoc(), vectorType, *condition,
                                  *trueValue, falseValue)
              .getResult();
      Value shared = allocateWorkgroupMemRef(rewriter, op.getLoc(), memrefType);
      if (failed(writeVectorToMemRef(rewriter, op.getLoc(), selected, shared)))
        return failure();
      rewriter.replaceOp(op, shared);
      return success();
    }

    Value falseValue = adaptor.getFalseValue();
    if (isa<mlir::VectorType>(converted))
      falseValue = vector::BroadcastOp::create(rewriter, op.getLoc(), converted,
                                               falseValue)
                       .getResult();

    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, converted, adaptor.getCondition(), adaptor.getTrueValue(),
        falseValue);
    return success();
  }
};

struct ConvertCallIntrinsicOp : public OpConversionPattern<HCCallIntrinsicOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCCallIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    operands.reserve(op.getArgs().size());
    for (auto [original, converted] :
         llvm::zip_equal(op.getArgs(), adaptor.getArgs())) {
      Type boundary =
          convertIntrinsicBoundaryType(original.getType(), *typeConverter);
      if (!boundary)
        return failure();
      operands.push_back(
          castIfNeeded(rewriter, op.getLoc(), converted, boundary));
    }

    SmallVector<Type> results;
    results.reserve(op.getResultTypes().size());
    for (Type result : op.getResultTypes()) {
      Type boundary = convertIntrinsicBoundaryType(result, *typeConverter);
      if (!boundary)
        return failure();
      results.push_back(boundary);
    }

    OperationState state(op.getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(results);
    state.addAttributes(op->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

struct ConvertBufferViewOp : public OpConversionPattern<HCBufferViewOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBufferViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (auto sourceMemRefType =
            dyn_cast<MemRefType>(adaptor.getBuffer().getType())) {
      auto resultMemRefType = dyn_cast_if_present<MemRefType>(converted);
      if (!resultMemRefType)
        return failure();
      FailureOr<SmallVector<SliceAxis>> axes =
          collectAxes(op.getOperation(), adaptor.getIndices(), rewriter);
      if (failed(axes))
        return failure();
      if (static_cast<int64_t>(axes->size()) != sourceMemRefType.getRank())
        return op.emitOpError("expected one view subscript per memref axis");
      if (!resultMemRefType.hasStaticShape())
        return failure();
      if (llvm::count_if(*axes, [](const SliceAxis &axis) {
            return axis.isSlice;
          }) != resultMemRefType.getRank())
        return op.emitOpError(
            "view result rank must match slice subscript rank");

      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides;
      offsets.reserve(axes->size());
      sizes.reserve(axes->size());
      strides.reserve(axes->size());
      SmallVector<int64_t> resultShape;
      resultShape.reserve(resultMemRefType.getRank());
      int64_t resultAxis = 0;
      for (const SliceAxis &info : *axes) {
        offsets.push_back(info.offset);
        strides.push_back(rewriter.getIndexAttr(1));
        if (info.isSlice) {
          int64_t size = resultMemRefType.getDimSize(resultAxis++);
          resultShape.push_back(size);
          sizes.push_back(rewriter.getIndexAttr(size));
        } else {
          sizes.push_back(rewriter.getIndexAttr(1));
        }
      }

      MemRefType subviewType = memref::SubViewOp::inferRankReducedResultType(
          resultShape, sourceMemRefType, offsets, sizes, strides);
      rewriter.replaceOpWithNewOp<memref::SubViewOp>(
          op, subviewType, adaptor.getBuffer(), offsets, sizes, strides);
      return success();
    }

    auto sourceType = dyn_cast<mlir::VectorType>(adaptor.getBuffer().getType());
    if (!sourceType || !converted)
      return failure();

    FailureOr<SmallVector<SliceAxis>> axes =
        collectAxes(op.getOperation(), adaptor.getIndices(), rewriter);
    if (failed(axes))
      return failure();
    if (static_cast<int64_t>(axes->size()) < sourceType.getRank())
      return op.emitOpError(
          "expected at least one view subscript per vector axis");

    ArrayRef<SliceAxis> localAxes(*axes);
    localAxes = localAxes.take_front(sourceType.getRank());
    SmallVector<int64_t> permutation;
    SmallVector<OpFoldResult> positions;
    permutation.reserve(localAxes.size());
    for (auto [axis, info] : llvm::enumerate(localAxes)) {
      if (info.isSlice)
        continue;
      permutation.push_back(axis);
      positions.push_back(info.offset);
    }
    for (auto [axis, info] : llvm::enumerate(localAxes)) {
      if (info.isSlice)
        permutation.push_back(axis);
    }

    Value source = adaptor.getBuffer();
    if (!llvm::equal(permutation, llvm::seq<int64_t>(0, localAxes.size())))
      source = vector::TransposeOp::create(rewriter, op.getLoc(), source,
                                           permutation)
                   .getResult();

    Value result = source;
    if (!positions.empty())
      result =
          vector::ExtractOp::create(rewriter, op.getLoc(), source, positions)
              .getResult();

    if (result.getType() != converted) {
      if (!isa<mlir::VectorType>(result.getType()) ||
          !isa<mlir::VectorType>(converted))
        return failure();
      result =
          vector::ShapeCastOp::create(rewriter, op.getLoc(), converted, result)
              .getResult();
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertDivOp : public OpConversionPattern<HCDivOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted || !converted.isIntOrIndex())
      return failure();
    rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, converted, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

struct ConvertNegOp : public OpConversionPattern<HCNegOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCNegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted || !converted.isIntOrIndex())
      return failure();
    Value zero =
        converted.isIndex()
            ? arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0)
                  .getResult()
            : arith::ConstantOp::create(rewriter, op.getLoc(), converted,
                                        rewriter.getZeroAttr(converted));
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, converted, zero,
                                               adaptor.getValue());
    return success();
  }
};

template <typename OpT> static constexpr arith::CmpIPredicate cmpPredicate() {
  if constexpr (std::is_same_v<OpT, HCCmpLtOp>)
    return arith::CmpIPredicate::slt;
  if constexpr (std::is_same_v<OpT, HCCmpLeOp>)
    return arith::CmpIPredicate::sle;
  if constexpr (std::is_same_v<OpT, HCCmpGtOp>)
    return arith::CmpIPredicate::sgt;
  if constexpr (std::is_same_v<OpT, HCCmpGeOp>)
    return arith::CmpIPredicate::sge;
  if constexpr (std::is_same_v<OpT, HCCmpEqOp>)
    return arith::CmpIPredicate::eq;
  if constexpr (std::is_same_v<OpT, HCCmpNeOp>)
    return arith::CmpIPredicate::ne;
}

template <typename OpT> struct ConvertCmpOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted || !converted.isInteger(1))
      return failure();
    rewriter.template replaceOpWithNewOp<arith::CmpIOp>(
        op, cmpPredicate<OpT>(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertCastOp : public OpConversionPattern<HCCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (adaptor.getSource().getType() == converted) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, converted, adaptor.getSource());
    return success();
  }
};

struct ConvertBufferDimOp : public OpConversionPattern<HCBufferDimOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBufferDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getDim().getType());
    if (!converted || !converted.isIndex())
      return failure();
    Value memref = sourceMemRef(adaptor.getBuffer());
    if (!memref)
      return op.emitOpError(
          "expected buffer to be a memref or memref ABI cast");
    rewriter.replaceOp(op, dim(rewriter, op.getLoc(), memref, op.getAxis()));
    return success();
  }
};

template <typename OpT>
struct AdaptRegionlessOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> results;
    if (failed(
            this->typeConverter->convertTypes(op->getResultTypes(), results)))
      return failure();
    OperationState state(op.getLoc(), op->getName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(results);
    state.addAttributes(op->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

struct ConvertForRangeOp : public OpConversionPattern<HCForRangeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCForRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLower().getType().isIndex() ||
        !adaptor.getUpper().getType().isIndex() ||
        !adaptor.getStep().getType().isIndex())
      return op.emitOpError("requires index lower, upper, and step");

    auto loop = scf::ForOp::create(rewriter, op.getLoc(), adaptor.getLower(),
                                   adaptor.getUpper(), adaptor.getStep(),
                                   adaptor.getIterInits());

    Block &src = op.getBody().front();
    Block &dst = loop.getRegion().front();
    Operation *dstTerminator = dst.empty() ? nullptr : &dst.back();
    if (dstTerminator)
      rewriter.setInsertionPoint(dstTerminator);
    else
      rewriter.setInsertionPointToEnd(&dst);

    IRMapping mapping;
    for (auto [oldArg, newArg] :
         llvm::zip_equal(src.getArguments(), dst.getArguments()))
      mapping.map(oldArg, newArg);

    auto yield = dyn_cast<HCYieldOp>(src.back());
    if (!yield)
      return op.emitOpError("body must end with `hc.yield`");
    for (Operation &nested : src) {
      if (&nested == yield.getOperation())
        break;
      rewriter.clone(nested, mapping);
    }

    SmallVector<Value> yielded;
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      Value mapped = mapping.lookupOrDefault(value);
      yielded.push_back(castIfNeeded(rewriter, op.getLoc(), mapped,
                                     loop.getResultTypes()[index]));
    }
    if (dstTerminator)
      rewriter.replaceOpWithNewOp<scf::YieldOp>(dstTerminator, yielded);
    else
      scf::YieldOp::create(rewriter, op.getLoc(), yielded);
    rewriter.replaceOp(op, loop.getResults());
    return success();
  }
};

struct ConvertIfOp : public OpConversionPattern<HCIfOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> results;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), results)))
      return failure();
    auto ifOp =
        scf::IfOp::create(rewriter, op.getLoc(), results, adaptor.getCond(),
                          /*withElseRegion=*/!op.getElseRegion().empty());
    auto cloneRegion = [&](Region &srcRegion,
                           Region &dstRegion) -> LogicalResult {
      Block &src = srcRegion.front();
      Block &dst = dstRegion.front();
      Operation *dstTerminator = dst.empty() ? nullptr : &dst.back();
      if (dstTerminator)
        rewriter.setInsertionPoint(dstTerminator);
      else
        rewriter.setInsertionPointToEnd(&dst);
      IRMapping mapping;
      auto yield = dyn_cast<HCYieldOp>(src.back());
      if (!yield)
        return failure();
      for (Operation &nested : src) {
        if (&nested == yield.getOperation())
          break;
        rewriter.clone(nested, mapping);
      }
      SmallVector<Value> yielded;
      for (auto [index, value] : llvm::enumerate(yield.getValues())) {
        Value mapped = mapping.lookupOrDefault(value);
        yielded.push_back(
            castIfNeeded(rewriter, op.getLoc(), mapped, results[index]));
      }
      if (dstTerminator)
        rewriter.replaceOpWithNewOp<scf::YieldOp>(dstTerminator, yielded);
      else
        scf::YieldOp::create(rewriter, op.getLoc(), yielded);
      return success();
    };
    if (failed(cloneRegion(op.getThenRegion(), ifOp.getThenRegion())))
      return op.emitOpError("then region must end with `hc.yield`");
    if (!op.getElseRegion().empty() &&
        failed(cloneRegion(op.getElseRegion(), ifOp.getElseRegion())))
      return op.emitOpError("else region must end with `hc.yield`");
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

static void populateLaunchBodyLoweringPatterns(TypeConverter &converter,
                                               MLIRContext *ctx,
                                               RewritePatternSet &patterns) {
  patterns.add<ConvertMaterializeBoundExprOp, ConvertConstOp,
               ConvertIntBinaryOp<HCAddOp, arith::AddIOp>,
               ConvertIntBinaryOp<HCSubOp, arith::SubIOp>,
               ConvertIntBinaryOp<HCMulOp, arith::MulIOp>, ConvertDivOp,
               ConvertIntBinaryOp<HCModOp, arith::RemUIOp>, ConvertNegOp,
               ConvertCmpOp<HCCmpLtOp>, ConvertCmpOp<HCCmpLeOp>,
               ConvertCmpOp<HCCmpGtOp>, ConvertCmpOp<HCCmpGeOp>,
               ConvertCmpOp<HCCmpEqOp>, ConvertCmpOp<HCCmpNeOp>, ConvertCastOp,
               ConvertBufferDimOp, ConvertIntrinsicSignatureOp,
               ConvertLoadLikeOp<HCLoadOp>, ConvertLoadLikeOp<HCVLoadOp>,
               ConvertLoadMaskOp, ConvertFullMaskOp,
               ConvertNullaryShapedConstantOp<HCVZerosOp, 0>,
               ConvertNullaryShapedConstantOp<HCVOnesOp, 1>,
               ConvertNullaryShapedConstantOp<HCZerosOp, 0>,
               ConvertNullaryShapedConstantOp<HCOnesOp, 1>,
               ConvertFillShapedConstantOp<HCVFullOp>,
               ConvertFillShapedConstantOp<HCFullOp>, ConvertEmptyOp,
               ConvertVecOp, ConvertSelectOp, ConvertCallIntrinsicOp,
               ConvertBufferViewOp, ConvertForRangeOp, ConvertIfOp>(converter,
                                                                    ctx);

  patterns.add<AdaptRegionlessOp<HCTupleOp>, AdaptRegionlessOp<HCSliceExprOp>,
               AdaptRegionlessOp<HCGetItemOp>>(converter, ctx);
}

static bool regionsAreLegal(Operation *op, const TypeConverter &converter) {
  return llvm::all_of(op->getRegions(), [&](Region &region) {
    return converter.isLegal(&region);
  });
}

static ConversionTarget
makeLaunchBodyLoweringTarget(MLIRContext *ctx, const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                         gpu::GPUDialect, memref::MemRefDialect,
                         scf::SCFDialect, vector::VectorDialect>();
  target.addLegalOp<HCStoreOp, HCUndefValueOp, UnrealizedConversionCastOp>();
  target.addIllegalOp<HCMaterializeBoundExprOp, HCConstOp, HCAddOp, HCSubOp,
                      HCMulOp, HCDivOp, HCModOp, HCNegOp, HCCmpLtOp, HCCmpLeOp,
                      HCCmpGtOp, HCCmpGeOp, HCCmpEqOp, HCCmpNeOp, HCCastOp,
                      HCBufferDimOp, HCLoadOp, HCVLoadOp, HCLoadMaskOp,
                      HCBufferViewOp, HCVecOp, HCVZerosOp, HCVOnesOp, HCVFullOp,
                      HCFullMaskOp, HCZerosOp, HCOnesOp, HCFullOp, HCEmptyOp,
                      HCSelectOp, HCForRangeOp, HCIfOp, HCYieldOp>();
  target.addDynamicallyLegalOp<HCIntrinsicOp>([&](HCIntrinsicOp op) {
    std::optional<FunctionType> fnType = op.getFunctionType();
    if (!fnType)
      return true;
    SmallVector<Type> inputs;
    SmallVector<Type> results;
    if (failed(convertIntrinsicFunctionSignature(*fnType, converter, inputs,
                                                 results)))
      return false;
    return llvm::equal(fnType->getInputs(), inputs) &&
           llvm::equal(fnType->getResults(), results);
  });
  target.addDynamicallyLegalOp<HCCallIntrinsicOp>([&](HCCallIntrinsicOp op) {
    for (Value arg : op.getArgs()) {
      Type boundary = convertIntrinsicBoundaryType(arg.getType(), converter);
      if (!boundary || arg.getType() != boundary)
        return false;
    }
    for (Type result : op.getResultTypes()) {
      Type boundary = convertIntrinsicBoundaryType(result, converter);
      if (!boundary || result != boundary)
        return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<HCTupleOp, HCSliceExprOp, HCGetItemOp>(
      [&](Operation *op) {
        return converter.isLegal(op) && regionsAreLegal(op, converter);
      });
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) { return !isa<HCDialect>(op->getDialect()); });
  return target;
}

struct HCLowerLaunchBodyPass
    : public hc::impl::HCLowerLaunchBodyBase<HCLowerLaunchBodyPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    HCLaunchBodyTypeConverter converter;
    RewritePatternSet patterns(ctx);
    populateLaunchBodyLoweringPatterns(converter, ctx, patterns);

    ConversionTarget target = makeLaunchBodyLoweringTarget(ctx, converter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// `createHCLowerLaunchBodyPass()` is emitted by tablegen.
