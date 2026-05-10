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

// Build a `BoundValues` for one `hc.idx_apply` / `hc.pred_apply` op.
// Explicit operand bindings are authoritative (direct map insert
// rather than `bind()`, which uses `try_emplace`); the ambient walk
// fills in everything else for free symbols left unlisted (e.g.
// launch geometry like `$WG0`).
static BoundValues collectApplyBindings(Operation *op,
                                        ConversionPatternRewriter &rewriter,
                                        ArrayAttr symbols,
                                        ValueRange convertedOperands) {
  BoundValues boundValues = collectBoundValues(op, rewriter);
  for (auto [attr, operand] : llvm::zip(symbols, convertedOperands)) {
    StringRef name = cast<StringAttr>(attr).getValue();
    if (name.empty())
      continue;
    boundValues.symbols[name] = indexCast(rewriter, op->getLoc(), operand);
  }
  return boundValues;
}

struct ConvertIdxApplyOp : public OpConversionPattern<HCIdxApplyOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIdxApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto idx = dyn_cast<IdxType>(op.getResult().getType());
    if (!idx || !idx.getExpr())
      return op.emitOpError("expected `!hc.idx<expr>` result type");

    BoundValues boundValues = collectApplyBindings(
        op, rewriter, op.getSymbolsAttr(), adaptor.getOperands());
    ExprLowerer lowerer(rewriter, op.getLoc(), boundValues);
    FailureOr<Value> lowered = lowerer.lower(idx.getExpr());
    if (failed(lowered))
      return op.emitOpError("failed to lower idx_apply expression");
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

struct ConvertPredApplyOp : public OpConversionPattern<HCPredApplyOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPredApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred = dyn_cast<PredType>(op.getResult().getType());
    if (!pred || !pred.getPred())
      return op.emitOpError("expected `!hc.pred<pred>` result type");

    BoundValues boundValues = collectApplyBindings(
        op, rewriter, op.getSymbolsAttr(), adaptor.getOperands());
    ExprLowerer lowerer(rewriter, op.getLoc(), boundValues);
    FailureOr<Value> lowered = lowerer.lower(pred.getPred());
    if (failed(lowered))
      return op.emitOpError("failed to lower pred_apply predicate");
    rewriter.replaceOp(op, *lowered);
    return success();
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
  Value stride;
  bool isSlice = false;
};

static Value zeroIndex(OpBuilder &builder, Location loc) {
  return arith::ConstantIndexOp::create(builder, loc, 0).getResult();
}

static Value oneIndex(OpBuilder &builder, Location loc) {
  return arith::ConstantIndexOp::create(builder, loc, 1).getResult();
}

static LogicalResult collectSliceAxis(Operation *op, Value index,
                                      OpBuilder &builder, SliceAxis &axis,
                                      bool requireUnitStride = true) {
  auto slice = index.getDefiningOp<HCSliceExprOp>();
  if (!slice)
    return failure();
  if (Value step = slice.getStep()) {
    APInt stepValue;
    if (requireUnitStride && (!matchPattern(step, m_ConstantInt(&stepValue)) ||
                              stepValue.getSExtValue() != 1))
      return op->emitOpError("only unit-stride slices lower to vector ops");
    axis.stride = step;
  } else {
    axis.stride = oneIndex(builder, op->getLoc());
  }
  axis.offset =
      slice.getLower() ? slice.getLower() : zeroIndex(builder, op->getLoc());
  axis.isSlice = true;
  return success();
}

static FailureOr<SmallVector<SliceAxis>>
collectAxes(Operation *op, ValueRange indices, OpBuilder &builder,
            bool requireUnitStride = true) {
  SmallVector<SliceAxis> axes;
  axes.reserve(indices.size());
  for (Value index : indices) {
    SliceAxis axis;
    if (isa<SliceType>(index.getType())) {
      if (failed(collectSliceAxis(op, index, builder, axis, requireUnitStride)))
        return failure();
    } else {
      if (!index.getType().isIndex())
        return op->emitOpError("expected index or slice subscript after "
                               "launch-body scalar lowering");
      axis.offset = index;
      axis.stride = oneIndex(builder, op->getLoc());
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

  // i1 needs per-element stores: a `vector<NxI1>` lowers to a packed-bit
  // store (LLVM packs the i1 lanes into the vector's bit width — N bits =
  // ceil(N/8) bytes), but `memref<NxI1>` accessed at element granularity
  // gets one i1 per *byte* (i1 has store size 1). The downstream
  // per-element scalar loads emitted by `transfer_to_scf full_unroll =
  // true` for the WMMA fragment paths reach past byte 0/1 into bytes the
  // packed write never touched, then read uninitialized LDS for every
  // lane whose `lane%16 >= 2`. Decompose the vector store into N scalar
  // i1 stores so both ends of the wire agree on byte-per-element layout.
  // For wider types (f16/f32) the packed-vector and per-element stores
  // touch the same bytes, so the extra unroll just bloats IR — keep the
  // single transfer_write there.
  if (memrefType.getElementType().isInteger(1)) {
    auto vectorType = dyn_cast<mlir::VectorType>(vector.getType());
    if (!vectorType || vectorType.getRank() != memrefType.getRank() ||
        vectorType.getShape() != memrefType.getShape())
      return failure();
    SmallVector<int64_t> shape(memrefType.getShape().begin(),
                               memrefType.getShape().end());
    int64_t total = 1;
    for (int64_t d : shape)
      total *= d;
    SmallVector<Value> dimValues;
    dimValues.reserve(shape.size());
    for (int64_t d : shape)
      dimValues.push_back(
          arith::ConstantIndexOp::create(builder, loc, d).getResult());
    for (int64_t lin = 0; lin < total; ++lin) {
      // Walk linear index back into per-axis coords (last axis fastest).
      SmallVector<Value> coords(shape.size());
      SmallVector<int64_t> coordInts(shape.size());
      int64_t remaining = lin;
      for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
           --axis) {
        coordInts[axis] = remaining % shape[axis];
        remaining /= shape[axis];
        coords[axis] =
            arith::ConstantIndexOp::create(builder, loc, coordInts[axis])
                .getResult();
      }
      Value element = vector::ExtractOp::create(builder, loc, vector, coordInts)
                          .getResult();
      memref::StoreOp::create(builder, loc, element, memref, coords);
    }
    return success();
  }

  vector::TransferWriteOp::create(
      builder, loc, vector, memref,
      zeroOffsets(builder, loc, memrefType.getRank()));
  return success();
}

// Linearize the enclosing launch's 3-D thread id and block size into a single
// `(tid, wgSize)` pair so the cooperative-copy loop only has to reason about
// one dimension. The dim3 layout is the standard
// `lin = (tz * by + ty) * bx + tx`. For the common `block_y = block_z = 1`
// shape the y/z multiplications fold to identity and the resulting IR
// collapses to plain `tx` / `bx`.
static FailureOr<std::pair<Value, Value>>
linearizedThreadAndSize(OpBuilder &builder, Location loc, Operation *anchor) {
  auto launch = anchor->getParentOfType<gpu::LaunchOp>();
  if (!launch)
    return failure();
  gpu::KernelDim3 tids = launch.getThreadIds();
  // `getBlockSizeOperandValues` reaches the values defined above the launch
  // (the operands that pin the block shape). Inside the body they're still
  // dominating SSA values, and using the outer form keeps the cooperative
  // loop's IR close to the values the materializeBoundExpr lowering already
  // surfaces under the `$WGS*` symbols.
  gpu::KernelDim3 sizes = launch.getBlockSizeOperandValues();
  Value tzBy = arith::MulIOp::create(builder, loc, tids.z, sizes.y);
  Value tzByPlusTy = arith::AddIOp::create(builder, loc, tzBy, tids.y);
  Value rowSpan = arith::MulIOp::create(builder, loc, tzByPlusTy, sizes.x);
  Value linearTid = arith::AddIOp::create(builder, loc, rowSpan, tids.x);
  Value bxBy = arith::MulIOp::create(builder, loc, sizes.x, sizes.y);
  Value wgSize = arith::MulIOp::create(builder, loc, bxBy, sizes.z);
  return std::pair<Value, Value>{linearTid, wgSize};
}

// Cooperative copy from a slice of a device-memory memref into a workgroup-AS
// LDS memref. Each thread of the enclosing wave is responsible for a strided
// subset of the LDS tile's elements (`lane, lane + wgSize, lane + 2*wgSize,
// ...`); a closing `gpu.barrier` makes the fully populated LDS visible to
// every thread before it's read back as per-lane fragments.
//
// Why one element per thread per chunk (rather than vectorized 2/4/8-wide
// loads): the prior lowering materialized the entire tile as a per-lane
// `vector<MxNxT>` value, blew register pressure into the hundreds of SGPR
// spills, and -- on real gfx11 hardware -- corrupted the WMMA inputs for
// most lanes. Scalar per-element loads keep the per-lane register footprint
// constant regardless of tile size; the compiler still coalesces the
// uniform-stride global loads into wide accesses.
//
// OOB elements pad with zero (matching the prior `transfer_read` semantics).
// The bounds check is per-element rather than at the tile level so partial
// tiles at the edge of `M`/`N`/`K` get correct zero padding without
// over-reading the source.
static LogicalResult emitCooperativeCopy(OpBuilder &builder, Location loc,
                                         Operation *anchor, Value sourceMemRef,
                                         ArrayRef<SliceAxis> axes, Value lds,
                                         MemRefType ldsType) {
  FailureOr<std::pair<Value, Value>> tidAndSize =
      linearizedThreadAndSize(builder, loc, anchor);
  if (failed(tidAndSize))
    return failure();
  Value linearTid = tidAndSize->first;
  Value wgSize = tidAndSize->second;

  ArrayRef<int64_t> ldsShape = ldsType.getShape();
  int64_t totalElements = 1;
  for (int64_t d : ldsShape)
    totalElements *= d;

  Value totalVal =
      arith::ConstantIndexOp::create(builder, loc, totalElements).getResult();
  Value c0 = zeroIndex(builder, loc);
  Value c1 = oneIndex(builder, loc);

  // Per-lane chunk count. `ceildiv` so the trailing partial chunk still runs
  // (its in-range `scf.if` then gates the actual work for the lanes that
  // would otherwise step past `totalVal`).
  Value chunks =
      arith::CeilDivUIOp::create(builder, loc, totalVal, wgSize).getResult();
  Type elementType = ldsType.getElementType();
  Value padding = constantSplat(builder, loc, elementType, 0);
  if (!padding)
    return failure();

  scf::ForOp loop =
      scf::ForOp::create(builder, loc, c0, chunks, c1, ValueRange{});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());

    Value chunkIdx = loop.getInductionVar();
    Value chunkOffset =
        arith::MulIOp::create(builder, loc, chunkIdx, wgSize).getResult();
    Value lin =
        arith::AddIOp::create(builder, loc, chunkOffset, linearTid).getResult();
    Value inRange = arith::CmpIOp::create(
                        builder, loc, arith::CmpIPredicate::ult, lin, totalVal)
                        .getResult();

    auto rangeIf = scf::IfOp::create(builder, loc, TypeRange{}, inRange,
                                     /*withElseRegion=*/false);
    OpBuilder::InsertionGuard rangeGuard(builder);
    builder.setInsertionPointToStart(&rangeIf.getThenRegion().front());

    // Unlinearize `lin` into per-axis coordinates of the LDS tile. Walk the
    // axes back-to-front so the innermost axis (fastest-varying) absorbs the
    // remainder first; this matches the canonical row-major flatten order.
    SmallVector<Value> coords(ldsShape.size());
    Value remaining = lin;
    for (int64_t axis = static_cast<int64_t>(ldsShape.size()) - 1; axis >= 0;
         --axis) {
      Value dim = arith::ConstantIndexOp::create(builder, loc, ldsShape[axis])
                      .getResult();
      coords[axis] =
          arith::RemUIOp::create(builder, loc, remaining, dim).getResult();
      if (axis > 0)
        remaining =
            arith::DivUIOp::create(builder, loc, remaining, dim).getResult();
    }

    // Translate LDS coords to source indices. Non-slice axes contribute their
    // fixed offset; slice axes scale the LDS coord by the slice's stride and
    // add the slice's base offset, matching the `axes`-driven addressing in
    // the per-lane vector path below.
    SmallVector<Value> srcIndices;
    srcIndices.reserve(axes.size());
    int64_t sliceCoordIdx = 0;
    for (const SliceAxis &info : axes) {
      if (!info.isSlice) {
        srcIndices.push_back(info.offset);
        continue;
      }
      Value coord = coords[sliceCoordIdx++];
      Value scaled =
          arith::MulIOp::create(builder, loc, coord, info.stride).getResult();
      srcIndices.push_back(
          arith::AddIOp::create(builder, loc, info.offset, scaled).getResult());
    }

    // Per-element bounds check on the source: every slice-axis index must be
    // less than the source dim. Non-slice axes carry a fixed in-bounds offset
    // by construction (the slice dialect rejects scalar subscripts past the
    // end statically) so they don't need a runtime check.
    Value inBounds =
        arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true))
            .getResult();
    for (auto [axisIdx, info] : llvm::enumerate(axes)) {
      if (!info.isSlice)
        continue;
      Value extent = dim(builder, loc, sourceMemRef, axisIdx);
      Value check =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                srcIndices[axisIdx], extent)
              .getResult();
      inBounds =
          arith::AndIOp::create(builder, loc, inBounds, check).getResult();
    }

    auto loadIf = scf::IfOp::create(builder, loc, TypeRange{elementType},
                                    inBounds, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard thenGuard(builder);
      builder.setInsertionPointToStart(&loadIf.getThenRegion().front());
      Value loaded =
          memref::LoadOp::create(builder, loc, sourceMemRef, srcIndices)
              .getResult();
      scf::YieldOp::create(builder, loc, loaded);
    }
    {
      OpBuilder::InsertionGuard elseGuard(builder);
      builder.setInsertionPointToStart(&loadIf.getElseRegion().front());
      scf::YieldOp::create(builder, loc, padding);
    }

    memref::StoreOp::create(builder, loc, loadIf.getResult(0), lds, coords);
  }

  // Make the cooperative writes visible to every thread before any per-lane
  // reader sees the LDS tile. Without this, multi-wave workgroups race; for
  // single-wave workgroups it's redundant but cheap and the canonicalizer
  // doesn't (and shouldn't) drop the safety net.
  gpu::BarrierOp::create(builder, loc);
  return success();
}

static FailureOr<Value> materializeShapedResult(OpBuilder &builder,
                                                Location loc,
                                                Type convertedType,
                                                Value vector) {
  if (auto vectorType = dyn_cast_if_present<mlir::VectorType>(convertedType)) {
    if (vector.getType() != vectorType)
      return failure();
    return vector;
  }

  auto memrefType = dyn_cast_if_present<MemRefType>(convertedType);
  if (!memrefType)
    return failure();
  mlir::VectorType vectorType = vectorTypeForMemRef(memrefType);
  if (!vectorType || vector.getType() != vectorType)
    return failure();

  Value shared = allocateWorkgroupMemRef(builder, loc, memrefType);
  if (failed(writeVectorToMemRef(builder, loc, vector, shared)))
    return failure();
  return shared;
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

  // Mirror `writeVectorToMemRef`'s i1 special case on the read side. LLVM
  // packs `vector<NxI1>` into ceil(N/8) bytes (one bit per lane), but the
  // matching memref uses one *byte* per element. A bare `vector.transfer_read
  // : vector<NxI1>` therefore reads ceil(N/8) bytes from row 0 and reinterprets
  // them as N bits — element 0 lands at the LSB of byte 0, element 1 at bit 1
  // of byte 0 (which the byte-per-element store left zero), element 8 at bit 0
  // of byte 1, and so on. Decompose into N scalar `memref.load`s + inserts so
  // the read agrees with the write's byte-per-element layout. f16/f32 don't
  // need this — vector and memref already touch the same bytes.
  if (memrefType.getElementType().isInteger(1) &&
      vectorType.getElementType().isInteger(1) &&
      vectorType.getShape() == memrefType.getShape()) {
    SmallVector<int64_t> shape(memrefType.getShape().begin(),
                               memrefType.getShape().end());
    int64_t total = 1;
    for (int64_t d : shape)
      total *= d;
    Value result = arith::ConstantOp::create(builder, loc, vectorType,
                                             builder.getZeroAttr(vectorType))
                       .getResult();
    for (int64_t lin = 0; lin < total; ++lin) {
      SmallVector<Value> coords(shape.size());
      SmallVector<int64_t> coordInts(shape.size());
      int64_t remaining = lin;
      for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
           --axis) {
        coordInts[axis] = remaining % shape[axis];
        remaining /= shape[axis];
        coords[axis] =
            arith::ConstantIndexOp::create(builder, loc, coordInts[axis])
                .getResult();
      }
      Value element =
          memref::LoadOp::create(builder, loc, memref, coords).getResult();
      result =
          vector::InsertOp::create(builder, loc, element, result, coordInts)
              .getResult();
    }
    return result;
  }

  Value padding = constantSplat(builder, loc, vectorType.getElementType(), 0);
  if (!padding)
    return failure();
  return vector::TransferReadOp::create(
             builder, loc, vectorType, memref,
             zeroOffsets(builder, loc, memrefType.getRank()),
             std::optional<Value>(padding))
      .getResult();
}

static FailureOr<Value> shapedValueAsVector(OpBuilder &builder, Location loc,
                                            Value value, Type convertedType) {
  if (auto vectorType = dyn_cast<mlir::VectorType>(convertedType)) {
    if (value.getType() == vectorType)
      return value;
    return failure();
  }
  auto memrefType = dyn_cast<MemRefType>(convertedType);
  if (!memrefType)
    return failure();
  mlir::VectorType vectorType = vectorTypeForMemRef(memrefType);
  if (!vectorType)
    return failure();
  return readMemRefAsVector(builder, loc, value, vectorType);
}

static Value extractVectorElement(OpBuilder &builder, Location loc,
                                  Value vector, ArrayRef<int64_t> coordinates) {
  if (coordinates.empty())
    return vector;
  SmallVector<OpFoldResult> positions;
  positions.reserve(coordinates.size());
  for (int64_t coordinate : coordinates)
    positions.push_back(builder.getI64IntegerAttr(coordinate));
  return vector::ExtractOp::create(builder, loc, vector, positions).getResult();
}

static SmallVector<SmallVector<int64_t>>
staticVectorCoordinates(ArrayRef<int64_t> shape) {
  int64_t elementCount = 1;
  for (int64_t dim : shape)
    elementCount *= dim;

  SmallVector<SmallVector<int64_t>> coordinates;
  coordinates.reserve(elementCount);
  for (int64_t linear = 0; linear != elementCount; ++linear) {
    int64_t remaining = linear;
    SmallVector<int64_t> coordinate(shape.size(), 0);
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
         --axis) {
      coordinate[axis] = remaining % shape[axis];
      remaining /= shape[axis];
    }
    coordinates.push_back(std::move(coordinate));
  }
  return coordinates;
}

static Value scaleIndexOffset(OpBuilder &builder, Location loc, Value stride,
                              int64_t coordinate) {
  if (coordinate == 0)
    return arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  if (coordinate == 1)
    return stride;
  Value coordinateValue =
      arith::ConstantIndexOp::create(builder, loc, coordinate);
  return arith::MulIOp::create(builder, loc, stride, coordinateValue)
      .getResult();
}

static SmallVector<Value> storeIndicesForCoordinate(OpBuilder &builder,
                                                    Location loc,
                                                    ArrayRef<SliceAxis> axes,
                                                    ArrayRef<int64_t> coords) {
  SmallVector<Value> indices;
  indices.reserve(axes.size());
  int64_t sliceAxis = 0;
  for (const SliceAxis &axis : axes) {
    if (!axis.isSlice) {
      indices.push_back(axis.offset);
      continue;
    }
    Value scaled =
        scaleIndexOffset(builder, loc, axis.stride, coords[sliceAxis++]);
    indices.push_back(
        arith::AddIOp::create(builder, loc, axis.offset, scaled).getResult());
  }
  return indices;
}

static bool isUnitStride(Value stride) {
  APInt step;
  return matchPattern(stride, m_ConstantInt(&step)) && step.getSExtValue() == 1;
}

static bool hasNonUnitStrideSlice(ArrayRef<SliceAxis> axes) {
  return llvm::any_of(axes, [](const SliceAxis &axis) {
    return axis.isSlice && !isUnitStride(axis.stride);
  });
}

// Fold a constant index `Value` into an `IntegerAttr` so subview-result-type
// inference picks up a static stride/offset; non-constants stay as the SSA
// value the caller supplied.
static OpFoldResult asIndexFold(OpBuilder &builder, Value v) {
  APInt constant;
  if (matchPattern(v, m_ConstantInt(&constant)))
    return builder.getIndexAttr(constant.getSExtValue());
  return v;
}

// Encode the slice's per-axis offsets and strides into a `memref.subview` so
// the resulting memref's affine layout reflects the strided access pattern;
// the caller can then issue zero-offset transfers against the subview.
// Constant offsets and strides are folded into static layout components so
// the subview type stays as concrete as the slice expression allows.
static Value makeStridedSubview(OpBuilder &builder, Location loc, Value memref,
                                MemRefType memrefType,
                                mlir::VectorType resultVectorType,
                                ArrayRef<SliceAxis> axes) {
  SmallVector<OpFoldResult> subOffsets;
  SmallVector<OpFoldResult> subSizes;
  SmallVector<OpFoldResult> subStrides;
  subOffsets.reserve(axes.size());
  subSizes.reserve(axes.size());
  subStrides.reserve(axes.size());
  int64_t sliceIdx = 0;
  for (const SliceAxis &info : axes) {
    subOffsets.push_back(asIndexFold(builder, info.offset));
    if (info.isSlice) {
      int64_t size = resultVectorType.getDimSize(sliceIdx++);
      subSizes.push_back(builder.getIndexAttr(size));
      subStrides.push_back(asIndexFold(builder, info.stride));
    } else {
      subSizes.push_back(builder.getIndexAttr(1));
      subStrides.push_back(builder.getIndexAttr(1));
    }
  }
  MemRefType subviewType = memref::SubViewOp::inferResultType(
      memrefType, subOffsets, subSizes, subStrides);
  return memref::SubViewOp::create(builder, loc, subviewType, memref,
                                   subOffsets, subSizes, subStrides)
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
        collectAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                    /*requireUnitStride=*/false);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != transferVectorType.getRank())
      return op.emitOpError("load result rank must match slice subscript rank");

    // LDS-staged result: use a cooperative per-lane copy so each thread of
    // the wave only handles its share of the tile elements. The previous
    // path materialized the whole tile as a per-lane vector, then had every
    // lane redundantly write it to the same LDS bytes -- correct on paper
    // but it drove SGPR spills into the hundreds and corrupted WMMA inputs
    // on real gfx11 hardware.
    if (resultMemRefType) {
      Value lds =
          allocateWorkgroupMemRef(rewriter, op.getLoc(), resultMemRefType);
      if (failed(emitCooperativeCopy(rewriter, op.getLoc(), op.getOperation(),
                                     memref, *axes, lds, resultMemRefType)))
        return failure();
      rewriter.replaceOp(op, lds);
      return success();
    }

    // Per-lane vector result: every thread independently materializes its
    // own fragment via `vector.transfer_read`. Strided slices fold the
    // per-axis stride into a `memref.subview` so the step ends up encoded in
    // the subview's affine layout; the read then walks zero offsets against
    // the subview. Unit-stride slices skip the subview to keep the simpler
    // IR shape existing tests pin down.
    Value padding = constantSplat(rewriter, op.getLoc(),
                                  transferVectorType.getElementType(), 0);
    if (!padding)
      return failure();

    Value transferSource = memref;
    SmallVector<Value> transferOffsets;
    AffineMap permutationMap = transferPermutationMap(
        rewriter.getContext(), memrefType.getRank(), *axes);
    if (hasNonUnitStrideSlice(*axes)) {
      transferSource = makeStridedSubview(
          rewriter, op.getLoc(), memref, memrefType, transferVectorType, *axes);
      auto subviewType = cast<MemRefType>(transferSource.getType());
      transferOffsets =
          zeroOffsets(rewriter, op.getLoc(), subviewType.getRank());
      permutationMap = transferPermutationMap(rewriter.getContext(),
                                              subviewType.getRank(), *axes);
    } else {
      transferOffsets.reserve(axes->size());
      for (const SliceAxis &axis : *axes)
        transferOffsets.push_back(axis.offset);
    }

    // i1 vectors must use per-element scalar loads. LLVM packs `vector<NxI1>`
    // into ceil(N/8) bytes, so a single `vector.transfer_read` of i1 lowers
    // to a 2-byte load that picks element 0 from bit 0 of byte 0 and element
    // 8 from bit 0 of byte 1 — everything else aliases the byte-per-element
    // memref's zero padding. Mirror `writeVectorToMemRef`'s i1 path: emit N
    // `memref.load`s + `vector.insert`s so both ends agree on the
    // byte-per-element layout. Only intercept the unit-stride path (the
    // strided one already routes through a `memref.subview`, where the same
    // logic still applies — handle it the same way).
    Value loaded;
    if (transferVectorType.getElementType().isInteger(1)) {
      ArrayRef<int64_t> shape = transferVectorType.getShape();
      int64_t total = 1;
      for (int64_t d : shape)
        total *= d;
      Value zero =
          arith::ConstantOp::create(rewriter, op.getLoc(), transferVectorType,
                                    rewriter.getZeroAttr(transferVectorType))
              .getResult();
      Value vec = zero;
      for (int64_t lin = 0; lin < total; ++lin) {
        SmallVector<int64_t> resultCoords(shape.size());
        int64_t remaining = lin;
        for (int64_t a = static_cast<int64_t>(shape.size()) - 1; a >= 0; --a) {
          resultCoords[a] = remaining % shape[a];
          remaining /= shape[a];
        }
        SmallVector<Value> sourceCoords(transferOffsets.begin(),
                                        transferOffsets.end());
        for (size_t i = 0; i < permutationMap.getNumResults(); ++i) {
          auto expr = permutationMap.getResult(i);
          if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
            unsigned srcAxis = dim.getPosition();
            Value off = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                       resultCoords[i])
                            .getResult();
            sourceCoords[srcAxis] =
                arith::AddIOp::create(rewriter, op.getLoc(),
                                      sourceCoords[srcAxis], off)
                    .getResult();
          }
        }
        Value elem = memref::LoadOp::create(rewriter, op.getLoc(),
                                            transferSource, sourceCoords)
                         .getResult();
        vec = vector::InsertOp::create(rewriter, op.getLoc(), elem, vec,
                                       resultCoords)
                  .getResult();
      }
      loaded = vec;
    } else {
      loaded =
          vector::TransferReadOp::create(
              rewriter, op.getLoc(), transferVectorType, transferSource,
              transferOffsets, std::optional<Value>(padding), permutationMap)
              .getResult();
    }
    FailureOr<Value> result =
        materializeShapedResult(rewriter, op.getLoc(), converted, loaded);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
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
        collectAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                    /*requireUnitStride=*/false);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != maskType.getRank())
      return op.emitOpError("mask result rank must match slice subscript rank");

    // The mask size is the count of strided positions that stay in-bounds.
    // Unit-stride collapses to `extent - offset`; for wider strides the count
    // becomes `ceildiv(extent - offset, stride)` so e.g. a stride-2 slice into
    // an 8-row tail of a 24-row buffer reports 4 valid lanes, not 8.
    // `vector.create_mask` signed-clamps the result to `[0, N]`, so a negative
    // `extent - offset` (offset past the end) lands on a zero-clamped,
    // all-false mask without an explicit guard here.
    SmallVector<Value> maskSizes;
    for (auto [axis, info] : llvm::enumerate(*axes)) {
      if (!info.isSlice)
        continue;
      Value extent = dim(rewriter, op.getLoc(), memref, axis);
      Value remaining =
          arith::SubIOp::create(rewriter, op.getLoc(), extent, info.offset);
      Value size = remaining;
      if (!isUnitStride(info.stride)) {
        Value strideMinusOne =
            arith::SubIOp::create(rewriter, op.getLoc(), info.stride,
                                  oneIndex(rewriter, op.getLoc()));
        Value adjusted = arith::AddIOp::create(rewriter, op.getLoc(), remaining,
                                               strideMinusOne);
        size = arith::DivSIOp::create(rewriter, op.getLoc(), adjusted,
                                      info.stride);
      }
      maskSizes.push_back(size);
    }

    Value mask = vector::CreateMaskOp::create(rewriter, op.getLoc(), maskType,
                                              maskSizes);
    FailureOr<Value> result =
        materializeShapedResult(rewriter, op.getLoc(), converted, mask);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
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
    auto vectorType = dyn_cast<mlir::VectorType>(converted);
    if (auto memrefType = dyn_cast<MemRefType>(converted))
      vectorType = vectorTypeForMemRef(memrefType);
    if (vectorType) {
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, 1);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      FailureOr<Value> result =
          materializeShapedResult(rewriter, op.getLoc(), converted, vector);
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
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
    auto vectorType = dyn_cast<mlir::VectorType>(converted);
    if (auto memrefType = dyn_cast<MemRefType>(converted))
      vectorType = vectorTypeForMemRef(memrefType);
    if (vectorType) {
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, FillValue);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      FailureOr<Value> result =
          materializeShapedResult(rewriter, op.getLoc(), converted, vector);
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
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
    FailureOr<Value> result =
        materializeShapedResult(rewriter, op.getLoc(), converted, vector);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
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
      FailureOr<Value> result =
          materializeShapedResult(rewriter, op.getLoc(), converted, selected);
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
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

struct ConvertStoreOp : public OpConversionPattern<HCStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memref = sourceMemRef(adaptor.getDest());
    if (!memref)
      return op.emitOpError(
          "expected store destination to be a memref or memref ABI cast");
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || memrefType.getRank() !=
                           static_cast<int64_t>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected ranked memref with one subscript per axis");

    FailureOr<SmallVector<SliceAxis>> axes =
        collectAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                    /*requireUnitStride=*/false);
    if (failed(axes))
      return failure();

    Type convertedSource = typeConverter->convertType(op.getSource().getType());
    FailureOr<Value> source = shapedValueAsVector(
        rewriter, op.getLoc(), adaptor.getSource(), convertedSource);
    if (failed(source))
      return op.emitOpError("expected store source to lower to a vector");
    auto sourceType = dyn_cast<mlir::VectorType>(source->getType());
    if (!sourceType)
      return op.emitOpError("expected store source to be a vector");

    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != sourceType.getRank())
      return op.emitOpError(
          "store source rank must match slice subscript rank");

    Value mask;
    if (Value originalMask = op.getMask()) {
      Type convertedMask = typeConverter->convertType(originalMask.getType());
      FailureOr<Value> maskVector = shapedValueAsVector(
          rewriter, op.getLoc(), adaptor.getMask(), convertedMask);
      if (failed(maskVector))
        return op.emitOpError("expected store mask to lower to a vector");
      auto maskType = dyn_cast<mlir::VectorType>(maskVector->getType());
      auto expectedMaskType =
          mlir::VectorType::get(sourceType.getShape(), rewriter.getI1Type());
      if (maskType != expectedMaskType)
        return op.emitOpError("store mask type ")
               << maskVector->getType() << " must match " << expectedMaskType;
      mask = *maskVector;
    }

    for (ArrayRef<int64_t> coordinate :
         staticVectorCoordinates(sourceType.getShape())) {
      Value element =
          extractVectorElement(rewriter, op.getLoc(), *source, coordinate);
      SmallVector<Value> indices =
          storeIndicesForCoordinate(rewriter, op.getLoc(), *axes, coordinate);
      if (!mask) {
        memref::StoreOp::create(rewriter, op.getLoc(), element, memref,
                                indices);
        continue;
      }

      Value guard =
          extractVectorElement(rewriter, op.getLoc(), mask, coordinate);
      auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), TypeRange{}, guard,
                                    /*withElseRegion=*/false);
      Block &thenBlock = ifOp.getThenRegion().front();
      Operation *terminator = thenBlock.empty() ? nullptr : &thenBlock.back();
      if (terminator)
        rewriter.setInsertionPoint(terminator);
      else
        rewriter.setInsertionPointToEnd(&thenBlock);
      memref::StoreOp::create(rewriter, op.getLoc(), element, memref, indices);
      if (!terminator)
        scf::YieldOp::create(rewriter, op.getLoc());
      rewriter.setInsertionPointAfter(ifOp);
    }

    rewriter.eraseOp(op);
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
  patterns.add<
      ConvertMaterializeBoundExprOp, ConvertIdxApplyOp, ConvertPredApplyOp,
      ConvertConstOp, ConvertIntBinaryOp<HCAddOp, arith::AddIOp>,
      ConvertIntBinaryOp<HCSubOp, arith::SubIOp>,
      ConvertIntBinaryOp<HCMulOp, arith::MulIOp>, ConvertDivOp,
      ConvertIntBinaryOp<HCModOp, arith::RemUIOp>, ConvertNegOp,
      ConvertCmpOp<HCCmpLtOp>, ConvertCmpOp<HCCmpLeOp>, ConvertCmpOp<HCCmpGtOp>,
      ConvertCmpOp<HCCmpGeOp>, ConvertCmpOp<HCCmpEqOp>, ConvertCmpOp<HCCmpNeOp>,
      ConvertCastOp, ConvertBufferDimOp, ConvertIntrinsicSignatureOp,
      ConvertLoadLikeOp<HCLoadOp>, ConvertLoadLikeOp<HCVLoadOp>,
      ConvertLoadMaskOp, ConvertFullMaskOp,
      ConvertNullaryShapedConstantOp<HCVZerosOp, 0>,
      ConvertNullaryShapedConstantOp<HCVOnesOp, 1>,
      ConvertNullaryShapedConstantOp<HCZerosOp, 0>,
      ConvertNullaryShapedConstantOp<HCOnesOp, 1>,
      ConvertFillShapedConstantOp<HCVFullOp>,
      ConvertFillShapedConstantOp<HCFullOp>, ConvertEmptyOp, ConvertVecOp,
      ConvertSelectOp, ConvertStoreOp, ConvertCallIntrinsicOp,
      ConvertBufferViewOp, ConvertForRangeOp, ConvertIfOp>(converter, ctx);

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
  target.addLegalOp<HCUndefValueOp, UnrealizedConversionCastOp>();
  target.addIllegalOp<
      HCMaterializeBoundExprOp, HCIdxApplyOp, HCPredApplyOp, HCConstOp, HCAddOp,
      HCSubOp, HCMulOp, HCDivOp, HCModOp, HCNegOp, HCCmpLtOp, HCCmpLeOp,
      HCCmpGtOp, HCCmpGeOp, HCCmpEqOp, HCCmpNeOp, HCCastOp, HCBufferDimOp,
      HCLoadOp, HCVLoadOp, HCLoadMaskOp, HCBufferViewOp, HCVecOp, HCVZerosOp,
      HCVOnesOp, HCVFullOp, HCFullMaskOp, HCZerosOp, HCOnesOp, HCFullOp,
      HCEmptyOp, HCSelectOp, HCStoreOp, HCForRangeOp, HCIfOp, HCYieldOp>();
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
