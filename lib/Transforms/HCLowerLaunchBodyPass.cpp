// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-launch-body`, the launch-body lowering pass that runs
// after HC kernels have been wrapped in `gpu.launch`.
//
// Workgroup-AS storage and kernel-arg buffers are both off memref. Workgroup
// tiles materialize as `hc.alloc` + `hc.ptr_*` against `!hc.ptr<workgroup, T>`;
// kernel-arg buffers arrive as a `(!hc.ptr<global, T>, dim*, stride*)` UCC
// fragment that this pass walks to recover dims/strides and to emit
// `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`. The
// downstream `hc-lower-to-llvm` finishes the LLVM-dialect lowering.
// `doc/layouts.md` "hc.ptr and memory ops" holds the contract.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

// Bare tensors collapse to `!hc.ptr<workgroup, T>` — the launch-body owns
// the workgroup tile only as opaque-flat storage; the rank and layout were
// frontend semantics, and once memory is in hand the lowering treats every
// LDS allocation as a flat element-typed buffer. Static dims are required
// because `hc.alloc` workgroup needs a constant element count downstream.
static Type convertBareTensorType(BareTensorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  if (failed(staticIntegerShape(shaped)))
    return {};
  return PtrType::get(type.getContext(), AddrSpace::Workgroup, element);
}

// Total element count for the workgroup tile — product of static dims.
static FailureOr<int64_t> bareTensorElementCount(BareTensorType type) {
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(type));
  if (failed(dims))
    return failure();
  int64_t total = 1;
  for (int64_t d : *dims) {
    if (d < 0)
      return failure();
    total *= d;
  }
  return total;
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

// Kernel-arg fragment: the UCC at the launch boundary expands a single
// `!hc.buffer<T, [dims]>` into 1 + 2N values — a global pointer, then per-axis
// dim values, then per-axis element-stride values, in that order. The host
// wrapper builds it (`hc-lower-kernels-to-gpu-launch`); launch-body sees the
// UCC, walks back through it, and turns the per-element load/store into
// `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`.
struct KernelArgSource {
  Value ptr;
  SmallVector<Value> dims;
  SmallVector<Value> strides;

  unsigned rank() const { return dims.size(); }
};

static std::optional<KernelArgSource> resolveKernelArg(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getOutputs().size() != 1)
    return std::nullopt;
  if (cast.getInputs().size() < 1)
    return std::nullopt;
  if ((cast.getInputs().size() - 1) % 2 != 0)
    return std::nullopt;

  Value ptr = cast.getInputs().front();
  auto ptrType = dyn_cast<PtrType>(ptr.getType());
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Global)
    return std::nullopt;

  unsigned rank = (cast.getInputs().size() - 1) / 2;
  KernelArgSource info;
  info.ptr = ptr;
  info.dims.reserve(rank);
  info.strides.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value dim = cast.getInputs()[1 + axis];
    if (!dim.getType().isIndex())
      return std::nullopt;
    info.dims.push_back(dim);
  }
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value stride = cast.getInputs()[1 + rank + axis];
    if (!stride.getType().isIndex())
      return std::nullopt;
    info.strides.push_back(stride);
  }
  return info;
}

// Compose the linear element offset for a multi-axis access against a
// kernel-arg buffer: `sum_axis(indices[axis] * strides[axis])`. The result
// is in element units (matching torch/numpy `tensor.stride(i)` reporting),
// not bytes — `hc.ptr_offset` does its own element-size scaling on the
// way to LLVM.
static Value linearizeKernelArgOffset(OpBuilder &builder, Location loc,
                                      const KernelArgSource &source,
                                      ValueRange indices) {
  assert(indices.size() == source.rank() &&
         "kernel-arg offset rank must match source rank");
  Value offset;
  for (auto [index, stride] : llvm::zip_equal(indices, source.strides)) {
    Value term = arith::MulIOp::create(builder, loc, index, stride).getResult();
    offset = offset
                 ? arith::AddIOp::create(builder, loc, offset, term).getResult()
                 : term;
  }
  if (!offset)
    offset = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  return offset;
}

// Bind each free shape symbol from `type` to its matching dim value pulled
// out of the kernel-arg UCC fragment. Replaces the prior `memref.dim`
// chain — the dims now ride as explicit UCC inputs (one per axis).
static void bindShapeSymbols(BufferType type, const KernelArgSource &source,
                             BoundValues &boundValues) {
  for (auto [axis, attr] : llvm::enumerate(type.getShape().getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<StringRef> symbol = exactSymbolName(expr);
    if (!symbol)
      continue;
    if (axis >= source.rank())
      continue;
    boundValues.bind(*symbol, source.dims[axis]);
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
    if (cast.getOutputs().size() != 1)
      return;
    Type outputType = cast.getOutputs().front().getType();

    if (auto buffer = dyn_cast<BufferType>(outputType)) {
      if (auto info = resolveKernelArg(cast.getResult(0)))
        bindShapeSymbols(buffer, *info, boundValues);
      return;
    }

    if (cast.getInputs().size() != 1)
      return;
    Value input = cast.getInputs().front();
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

// Post-flatten access-op shape: one scalar `index` subscript carrying the
// already-linearised element offset, against a 1-rank kernel-arg ABI whose
// only stride is constant 1, and a 1-rank iter shape from the result vector
// (load/vload), source vector (store), or mask vector (load_mask). The
// pre-flatten per-axis structure has been folded into the composed offset
// by `hc-flatten-with-layouts`; the remaining lane walk over the flat tile
// is a unit-stride bump over `[composed_offset, composed_offset + N)`.
//
// Synthesizing a single full-slice axis (`offset = composed`, `stride = 1`,
// `isSlice = true`) lets the existing per-lane machinery in
// `kernelArgLaneIndices` / `storeIndicesForCoordinate` /
// `linearizeKernelArgOffset` fall through unchanged: the kernel-arg's
// unit stride collapses through the lin formula and each lane's offset
// reduces to `composed + lane`. Non-contiguous post-flatten tiles
// (strided slice survivors) intentionally fail this detect and route
// through the slice-aware path — they need the rank-N kernel-arg ABI
// to recompute per-axis offsets and are tracked separately.
static std::optional<SmallVector<SliceAxis>>
synthesizePostFlattenAxes(OpBuilder &builder, Location loc,
                          const KernelArgSource &kernelArg, ValueRange indices,
                          ArrayRef<int64_t> iterShape) {
  if (kernelArg.rank() != 1)
    return std::nullopt;
  if (indices.size() != 1)
    return std::nullopt;
  if (iterShape.size() != 1)
    return std::nullopt;
  if (!indices[0].getType().isIndex())
    return std::nullopt;
  APInt stride;
  if (!matchPattern(kernelArg.strides[0], m_ConstantInt(&stride)) ||
      stride.getSExtValue() != 1)
    return std::nullopt;
  SliceAxis axis;
  axis.offset = indices[0];
  axis.stride = oneIndex(builder, loc);
  axis.isSlice = true;
  return SmallVector<SliceAxis>{axis};
}

// Allocate a flat `!hc.ptr<workgroup, T>` sized for `count` elements. The
// downstream `hc-lower-to-llvm` rewrites the alloc into a sibling
// addrspace(3) global; rank-erasure happens at this boundary because the
// AMDGPU LDS surface is one-dimensional anyway.
static Value allocateWorkgroupPtr(OpBuilder &builder, Location loc,
                                  PtrType ptrType, int64_t count) {
  Value countValue =
      arith::ConstantIndexOp::create(builder, loc, count).getResult();
  return HCAllocOp::create(builder, loc, ptrType, countValue).getResult();
}

// Walk linear `lin` into per-axis coords for `shape` (last axis fastest).
// Returns the constant-index `Value` per axis.
static SmallVector<Value> unlinearizeCoords(OpBuilder &builder, Location loc,
                                            int64_t lin,
                                            ArrayRef<int64_t> shape) {
  SmallVector<Value> coords(shape.size());
  int64_t remaining = lin;
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
       --axis) {
    coords[axis] =
        arith::ConstantIndexOp::create(builder, loc, remaining % shape[axis])
            .getResult();
    remaining /= shape[axis];
  }
  return coords;
}

// Per-element store from a multi-dim `vector<...xT>` value into a flat
// `!hc.ptr<workgroup, T>` buffer. Emits one `hc.ptr_offset` + `hc.ptr_store`
// per lane in row-major order. We unconditionally use the per-element form
// (rather than a single vector-typed `hc.ptr_store`) so the matching reads
// can also be per-element scalars and the i1 byte-vs-bit discrepancy LLVM
// has between scalar and vector i1 stores never bites.
static LogicalResult writeVectorToWorkgroupPtr(OpBuilder &builder, Location loc,
                                               Value vector, Value ptr,
                                               ArrayRef<int64_t> shape) {
  auto vectorType = dyn_cast<mlir::VectorType>(vector.getType());
  if (!vectorType || vectorType.getShape() != shape)
    return failure();
  auto ptrType = dyn_cast<PtrType>(ptr.getType());
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();

  int64_t total = 1;
  for (int64_t d : shape)
    total *= d;
  for (int64_t lin = 0; lin != total; ++lin) {
    SmallVector<Value> coords = unlinearizeCoords(builder, loc, lin, shape);
    SmallVector<int64_t> coordInts(shape.size());
    int64_t remaining = lin;
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
         --axis) {
      coordInts[axis] = remaining % shape[axis];
      remaining /= shape[axis];
    }
    Value element =
        vector::ExtractOp::create(builder, loc, vector, coordInts).getResult();
    Value linVal =
        arith::ConstantIndexOp::create(builder, loc, lin).getResult();
    Value addr =
        HCPtrOffsetOp::create(builder, loc, ptrType, ptr, linVal).getResult();
    HCPtrStoreOp::create(builder, loc, element, addr);
  }
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

// Cooperative copy from a slice of a kernel-arg `!hc.ptr<global, T>` into a
// workgroup-AS LDS pointer. Each thread of the enclosing wave is responsible
// for a strided
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
//
// LDS writes route through `hc.ptr_offset` + `hc.ptr_store` against the
// flat workgroup buffer. The linear index already in hand from the
// per-thread chunk loop *is* the flat offset; the per-axis coords only
// participate in computing the source offset back into the kernel-arg
// global pointer.
static LogicalResult
emitCooperativeCopy(OpBuilder &builder, Location loc, Operation *anchor,
                    const KernelArgSource &source, ArrayRef<SliceAxis> axes,
                    Value lds, ArrayRef<int64_t> ldsShape, Type elementType) {
  FailureOr<std::pair<Value, Value>> tidAndSize =
      linearizedThreadAndSize(builder, loc, anchor);
  if (failed(tidAndSize))
    return failure();
  Value linearTid = tidAndSize->first;
  Value wgSize = tidAndSize->second;

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
  Value padding = constantSplat(builder, loc, elementType, 0);
  if (!padding)
    return failure();

  auto ptrType = cast<PtrType>(lds.getType());

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
      Value extent = source.dims[axisIdx];
      Value check =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                srcIndices[axisIdx], extent)
              .getResult();
      inBounds =
          arith::AndIOp::create(builder, loc, inBounds, check).getResult();
    }

    // Predicated load against the kernel-arg ptr: in-bounds gives the
    // tile element, OOB lanes pad with zero (matching the prior
    // `transfer_read` semantics). The branchless `hc.ptr_load_pred` keeps
    // the inner if-then-else flat for downstream LLVM SLP recombination.
    auto sourcePtrType = cast<PtrType>(source.ptr.getType());
    Value flatSrc = linearizeKernelArgOffset(builder, loc, source, srcIndices);
    Value srcAddr =
        HCPtrOffsetOp::create(builder, loc, sourcePtrType, source.ptr, flatSrc)
            .getResult();
    auto loadIf = scf::IfOp::create(builder, loc, TypeRange{elementType},
                                    inBounds, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard thenGuard(builder);
      builder.setInsertionPointToStart(&loadIf.getThenRegion().front());
      Value loaded =
          HCPtrLoadOp::create(builder, loc, elementType, srcAddr).getResult();
      scf::YieldOp::create(builder, loc, loaded);
    }
    {
      OpBuilder::InsertionGuard elseGuard(builder);
      builder.setInsertionPointToStart(&loadIf.getElseRegion().front());
      scf::YieldOp::create(builder, loc, padding);
    }

    Value addr =
        HCPtrOffsetOp::create(builder, loc, ptrType, lds, lin).getResult();
    HCPtrStoreOp::create(builder, loc, loadIf.getResult(0), addr);
  }

  // Make the cooperative writes visible to every thread before any per-lane
  // reader sees the LDS tile. Without this, multi-wave workgroups race; for
  // single-wave workgroups it's redundant but cheap and the canonicalizer
  // doesn't (and shouldn't) drop the safety net.
  gpu::BarrierOp::create(builder, loc);
  return success();
}

// Walk back through an `unrealized_conversion_cast` to recover a workgroup-AS
// `!hc.ptr` value. Counterpart to `resolveKernelArg` for the launch-body's
// LDS path; kernel-arg buffers come in as ptr+dims+strides UCC bundles
// for now. Returns null if `value` doesn't ultimately derive from a
// workgroup ptr.
static Value sourcePtr(Value value) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (cast && cast.getInputs().size() == 1 && cast.getOutputs().size() == 1) {
    Value input = cast.getInputs().front();
    if (auto ptr = dyn_cast<PtrType>(input.getType()))
      if (ptr.getAddrSpace() == AddrSpace::Workgroup)
        return input;
  }
  if (auto ptr = dyn_cast<PtrType>(value.getType()))
    if (ptr.getAddrSpace() == AddrSpace::Workgroup)
      return value;
  return {};
}

// Captures the underlying `!hc.ptr<workgroup, T>` source plus an indexing
// pattern that maps result-vector lanes back to source-tile flat offsets.
// Materialized eagerly from a (possibly view-laden) bare-tensor SSA value;
// see `resolvePtrViewSource`. The trivial pattern (no `hc.buffer_view` in
// the chain) carries one full-slice `SliceAxis` per source dim — gives the
// per-element loop a single shape to iterate without a separate "no view"
// path.
struct PtrViewSource {
  Value sourcePtr;
  PtrType ptrType;
  SmallVector<int64_t> sourceShape;
  SmallVector<SliceAxis> axes;
};

// Walk back through a single `hc.buffer_view` to find the underlying
// workgroup ptr and the index pattern. The buffer_view ops sit in the
// pre-conversion IR — `getRemappedValue` walks `replaceOp` records to
// produce the post-conversion ptr for the source tile, which by this point
// has been lowered (or is being lowered in this same partial-conversion
// pass). Two-deep view chains (`view(view(...))`) are vanishingly rare in
// the surface; if they ever show up we'd unroll the chain here.
//
// For non-view sources (`hc.alloc`, cooperative-copy result, fresh LDS from
// a `select`), we synthesize a trivial axis pattern (every axis a full
// slice over the source's static dim). That keeps the loader uniform — the
// strided per-element loop with trivial axes degenerates to the same flat
// `0..N` iteration the no-view path would emit.
static FailureOr<PtrViewSource>
resolvePtrViewSource(Value original, ConversionPatternRewriter &rewriter) {
  if (auto bv = original.getDefiningOp<HCBufferViewOp>()) {
    Value sourceTile = bv.getBuffer();
    auto bareSrc = dyn_cast<BareTensorType>(sourceTile.getType());
    if (!bareSrc)
      return failure();
    FailureOr<SmallVector<int64_t>> sourceShape =
        staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bareSrc));
    if (failed(sourceShape))
      return failure();
    Value remappedSrc = rewriter.getRemappedValue(sourceTile);
    if (!remappedSrc)
      return failure();
    Value srcPtr = sourcePtr(remappedSrc);
    if (!srcPtr)
      return failure();
    auto srcPtrType = dyn_cast<PtrType>(srcPtr.getType());
    if (!srcPtrType || srcPtrType.getAddrSpace() != AddrSpace::Workgroup)
      return failure();
    // Walk through remapped index operands so the per-axis offset/stride
    // values come out as `index`-typed SSA the per-element loop can plug
    // into `arith.muli`/`arith.addi` directly. The pre-conversion form
    // here is `!hc.idx<...>` / `!hc.slice<...>` from the buffer_view's
    // original operands, so we have to remap each index value (the slice
    // op's `lower`/`upper`/`step` are the slice op's own operands and get
    // remapped on its lowering, which has already happened by the time
    // any ptr-view consumer runs).
    SmallVector<Value> remappedIndices;
    remappedIndices.reserve(bv.getIndices().size());
    for (Value idx : bv.getIndices()) {
      Value remapped = rewriter.getRemappedValue(idx);
      if (!remapped)
        return failure();
      remappedIndices.push_back(remapped);
    }
    FailureOr<SmallVector<SliceAxis>> axes =
        collectAxes(bv.getOperation(), remappedIndices, rewriter,
                    /*requireUnitStride=*/true);
    if (failed(axes))
      return failure();
    if (axes->size() != sourceShape->size())
      return bv.emitOpError("expected one view subscript per source axis");
    return PtrViewSource{srcPtr, srcPtrType, *sourceShape, std::move(*axes)};
  }

  Value remapped = rewriter.getRemappedValue(original);
  if (!remapped)
    return failure();
  Value srcPtr = sourcePtr(remapped);
  if (!srcPtr)
    return failure();
  auto srcPtrType = dyn_cast<PtrType>(srcPtr.getType());
  if (!srcPtrType || srcPtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  auto bare = dyn_cast<BareTensorType>(original.getType());
  if (!bare)
    return failure();
  FailureOr<SmallVector<int64_t>> shape =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bare));
  if (failed(shape))
    return failure();

  SmallVector<SliceAxis> trivialAxes;
  trivialAxes.reserve(shape->size());
  Value zero = zeroIndex(rewriter, original.getLoc());
  Value one = oneIndex(rewriter, original.getLoc());
  for (size_t i = 0; i < shape->size(); ++i) {
    SliceAxis ax;
    ax.offset = zero;
    ax.stride = one;
    ax.isSlice = true;
    trivialAxes.push_back(ax);
  }
  return PtrViewSource{srcPtr, srcPtrType, *shape, std::move(trivialAxes)};
}

// Per-element load of a multi-dim `vector<...xT>` from a workgroup ptr,
// honoring any `hc.buffer_view` index pattern in the source chain. Lane V
// reads from the source's flat offset `base + sum_k(view_idx[k] *
// source_stride[slice_axis_k])` — base is the contribution of scalar source
// axes, the slice contribution scales the lane coord by the source's
// row-major stride at that source axis. For trivial (full-slice) patterns,
// this collapses to the contiguous `0..N-1` linear walk a no-view source
// wants. For strided patterns (column-of-2D-tile in WMMA), it threads each
// lane through a separate `hc.ptr_offset` + `hc.ptr_load`.
//
// The per-element shape preserves the byte-per-element layout the
// cooperative store and other consumers use, sidestepping the i1
// packed/unpacked discrepancy LLVM has between scalar and vector i1 stores.
static FailureOr<Value>
loadVectorFromPtrView(ConversionPatternRewriter &rewriter, Location loc,
                      Value original, mlir::VectorType vectorType,
                      ArrayRef<int64_t> viewShape) {
  FailureOr<PtrViewSource> src = resolvePtrViewSource(original, rewriter);
  if (failed(src))
    return failure();
  if (vectorType.getShape() != viewShape)
    return failure();

  SmallVector<int64_t> sourceStrides(src->sourceShape.size(), 1);
  for (int64_t axis = static_cast<int64_t>(src->sourceShape.size()) - 2;
       axis >= 0; --axis)
    sourceStrides[axis] = sourceStrides[axis + 1] * src->sourceShape[axis + 1];

  SmallVector<int64_t> sliceAxisPositions;
  for (int64_t i = 0; i < static_cast<int64_t>(src->axes.size()); ++i)
    if (src->axes[i].isSlice)
      sliceAxisPositions.push_back(i);
  if (sliceAxisPositions.size() != viewShape.size())
    return failure();

  Value baseOffset = zeroIndex(rewriter, loc);
  for (size_t i = 0; i < src->axes.size(); ++i) {
    if (src->axes[i].isSlice)
      continue;
    Value strideVal =
        arith::ConstantIndexOp::create(rewriter, loc, sourceStrides[i])
            .getResult();
    Value contrib =
        arith::MulIOp::create(rewriter, loc, src->axes[i].offset, strideVal)
            .getResult();
    baseOffset =
        arith::AddIOp::create(rewriter, loc, baseOffset, contrib).getResult();
  }

  int64_t total = 1;
  for (int64_t d : viewShape)
    total *= d;
  Type elementType = vectorType.getElementType();
  Value result = arith::ConstantOp::create(rewriter, loc, vectorType,
                                           rewriter.getZeroAttr(vectorType))
                     .getResult();
  for (int64_t lin = 0; lin != total; ++lin) {
    SmallVector<int64_t> coordInts(viewShape.size());
    int64_t remaining = lin;
    for (int64_t axis = static_cast<int64_t>(viewShape.size()) - 1; axis >= 0;
         --axis) {
      coordInts[axis] = remaining % viewShape[axis];
      remaining /= viewShape[axis];
    }
    Value flat = baseOffset;
    for (size_t vi = 0; vi < sliceAxisPositions.size(); ++vi) {
      int64_t srcAxisPos = sliceAxisPositions[vi];
      const SliceAxis &ax = src->axes[srcAxisPos];
      Value coordVal =
          arith::ConstantIndexOp::create(rewriter, loc, coordInts[vi])
              .getResult();
      Value scaledStride =
          arith::MulIOp::create(rewriter, loc, coordVal, ax.stride).getResult();
      Value indexInSource =
          arith::AddIOp::create(rewriter, loc, ax.offset, scaledStride)
              .getResult();
      Value srcStrideVal = arith::ConstantIndexOp::create(
                               rewriter, loc, sourceStrides[srcAxisPos])
                               .getResult();
      Value contrib =
          arith::MulIOp::create(rewriter, loc, indexInSource, srcStrideVal)
              .getResult();
      flat = arith::AddIOp::create(rewriter, loc, flat, contrib).getResult();
    }
    Value addr =
        HCPtrOffsetOp::create(rewriter, loc, src->ptrType, src->sourcePtr, flat)
            .getResult();
    Value element =
        HCPtrLoadOp::create(rewriter, loc, elementType, addr).getResult();
    result = vector::InsertOp::create(rewriter, loc, element, result, coordInts)
                 .getResult();
  }
  return result;
}

// Materialize a vector value as a workgroup-AS LDS tile of the given shape.
// Allocates a flat `!hc.ptr<workgroup, T>` and writes the vector lanes
// per-element. For the per-lane vector path the converted type is already
// a `vector<...xT>` and we just pass the value through.
static FailureOr<Value>
materializeShapedResult(OpBuilder &builder, Location loc, Type convertedType,
                        Value vector, ArrayRef<int64_t> shape) {
  if (auto vectorType = dyn_cast_if_present<mlir::VectorType>(convertedType)) {
    if (vector.getType() != vectorType)
      return failure();
    return vector;
  }

  auto ptrType = dyn_cast_if_present<PtrType>(convertedType);
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  int64_t total = 1;
  for (int64_t d : shape)
    total *= d;
  Value lds = allocateWorkgroupPtr(builder, loc, ptrType, total);
  if (failed(writeVectorToWorkgroupPtr(builder, loc, vector, lds, shape)))
    return failure();
  return lds;
}

// Read a converted shaped value as a multi-dim vector. The per-lane path
// is a no-op (the value is already a vector); the LDS path expands into
// per-element `hc.ptr_load`s, walking back through any `hc.buffer_view`
// chain on the original source so strided views (column-of-2D, etc.)
// produce correctly strided per-lane offsets.
static FailureOr<Value> shapedValueAsVector(ConversionPatternRewriter &rewriter,
                                            Location loc, Value original,
                                            Value remapped, Type convertedType,
                                            ArrayRef<int64_t> shape) {
  if (auto vectorType = dyn_cast<mlir::VectorType>(convertedType)) {
    if (remapped.getType() == vectorType)
      return remapped;
    return failure();
  }
  auto ptrType = dyn_cast<PtrType>(convertedType);
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  Type elementType = ptrType.getElementType();
  if (!elementType)
    return failure();
  mlir::VectorType vectorType = mlir::VectorType::get(shape, elementType);
  return loadVectorFromPtrView(rewriter, loc, original, vectorType, shape);
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

// Build the per-axis index list for one lane of the result vector. The
// caller feeds these into `hc.ptr_offset` + `hc.ptr_load` against the
// kernel-arg pointer. `axes` carries the slice's per-axis offsets and
// strides (`offset + coord * stride` for slice axes; just `offset` for
// scalar axes). The result vector's coords identify which slice axis we're
// walking; non-slice axes get the same `axes[k].offset` for every lane.
static SmallVector<Value> kernelArgLaneIndices(OpBuilder &builder, Location loc,
                                               ArrayRef<SliceAxis> axes,
                                               ArrayRef<int64_t> resultCoord) {
  SmallVector<Value> indices;
  indices.reserve(axes.size());
  int64_t sliceCursor = 0;
  for (const SliceAxis &info : axes) {
    if (!info.isSlice) {
      indices.push_back(info.offset);
      continue;
    }
    int64_t coord = resultCoord[sliceCursor++];
    Value scaled = scaleIndexOffset(builder, loc, info.stride, coord);
    indices.push_back(
        arith::AddIOp::create(builder, loc, info.offset, scaled).getResult());
  }
  return indices;
}

template <typename OpT>
struct ConvertLoadLikeOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    auto resultPtrType = dyn_cast_if_present<PtrType>(converted);
    auto resultVectorType = dyn_cast_if_present<mlir::VectorType>(converted);

    auto resultBareTensor = dyn_cast<BareTensorType>(op.getResult().getType());
    auto resultBareVector = dyn_cast<BareVectorType>(op.getResult().getType());

    // The shape we'll iterate over per-element comes from the original HC
    // result type — the converted ptr is rank-erased, the vector type
    // already carries it.
    SmallVector<int64_t> resultShape;
    if (resultBareTensor) {
      FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(
          cast<SymbolicallyShapedTypeInterface>(resultBareTensor));
      if (failed(dims))
        return failure();
      resultShape = std::move(*dims);
    } else if (resultBareVector) {
      FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(
          cast<SymbolicallyShapedTypeInterface>(resultBareVector));
      if (failed(dims))
        return failure();
      resultShape = std::move(*dims);
    } else {
      return failure();
    }

    Type elementType;
    if (resultPtrType)
      elementType = resultPtrType.getElementType();
    else if (resultVectorType)
      elementType = resultVectorType.getElementType();
    if (!elementType)
      return failure();

    Value source = [&]() -> Value {
      if constexpr (std::is_same_v<OpT, HCLoadOp>)
        return adaptor.getBuffer();
      else
        return adaptor.getSource();
    }();
    std::optional<KernelArgSource> kernelArg = resolveKernelArg(source);
    if (!kernelArg)
      return op.emitOpError(
          "expected load source to be a kernel-arg ptr ABI cast");
    if (kernelArg->rank() != static_cast<unsigned>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected kernel-arg source rank to match index rank");
    auto sourcePtrType = cast<PtrType>(kernelArg->ptr.getType());

    SmallVector<SliceAxis> axes;
    if (auto synthesized =
            synthesizePostFlattenAxes(rewriter, op.getLoc(), *kernelArg,
                                      adaptor.getIndices(), resultShape)) {
      axes = std::move(*synthesized);
    } else {
      FailureOr<SmallVector<SliceAxis>> collected =
          collectAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                      /*requireUnitStride=*/false);
      if (failed(collected))
        return failure();
      axes = std::move(*collected);
    }
    if (llvm::count_if(axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != static_cast<int64_t>(resultShape.size()))
      return op.emitOpError("load result rank must match slice subscript rank");

    // LDS-staged result: use a cooperative per-lane copy so each thread of
    // the wave only handles its share of the tile elements. The previous
    // path materialized the whole tile as a per-lane vector, then had every
    // lane redundantly write it to the same LDS bytes -- correct on paper
    // but it drove SGPR spills into the hundreds and corrupted WMMA inputs
    // on real gfx11 hardware.
    if (resultPtrType) {
      int64_t total = 1;
      for (int64_t d : resultShape)
        total *= d;
      Value lds =
          allocateWorkgroupPtr(rewriter, op.getLoc(), resultPtrType, total);
      KernelArgSource argCopy = *kernelArg;
      if (failed(emitCooperativeCopy(rewriter, op.getLoc(), op.getOperation(),
                                     argCopy, axes, lds, resultShape,
                                     elementType)))
        return failure();
      rewriter.replaceOp(op, lds);
      return success();
    }

    // Per-lane vector result: each thread materializes its own fragment via
    // per-element scalar `hc.ptr_offset` + `hc.ptr_load`s + `vector.insert`s.
    // Switching to per-element keeps a single shape for unit and non-unit
    // stride, sidesteps the i1 packed-vs-byte discrepancy that
    // `vector.transfer_read` of `vector<Nxi1>` triggered, and lets LLVM's
    // SLP recombine adjacent scalar loads when the slice is unit-stride.
    Value zero =
        arith::ConstantOp::create(rewriter, op.getLoc(), resultVectorType,
                                  rewriter.getZeroAttr(resultVectorType))
            .getResult();
    Value laneVec = zero;
    for (ArrayRef<int64_t> resultCoord : staticVectorCoordinates(resultShape)) {
      SmallVector<Value> indices =
          kernelArgLaneIndices(rewriter, op.getLoc(), axes, resultCoord);
      Value flat =
          linearizeKernelArgOffset(rewriter, op.getLoc(), *kernelArg, indices);
      Value addr = HCPtrOffsetOp::create(rewriter, op.getLoc(), sourcePtrType,
                                         kernelArg->ptr, flat)
                       .getResult();
      Value elem = HCPtrLoadOp::create(rewriter, op.getLoc(), elementType, addr)
                       .getResult();
      laneVec = vector::InsertOp::create(rewriter, op.getLoc(), elem, laneVec,
                                         resultCoord)
                    .getResult();
    }
    rewriter.replaceOp(op, laneVec);
    return success();
  }
};

// Helper: probe whether `stride` is the constant 1. Used to decide between
// `extent - offset` and `ceildiv(extent - offset, stride)` when sizing the
// mask. (We dropped the `isUnitStride`/`hasNonUnitStrideSlice` helpers
// elsewhere when the load path stopped routing through `memref.subview`,
// but the mask sizing still needs the per-axis distinction.)
static bool maskAxisIsUnitStride(const SliceAxis &axis) {
  APInt step;
  return matchPattern(axis.stride, m_ConstantInt(&step)) &&
         step.getSExtValue() == 1;
}

struct ConvertLoadMaskOp : public OpConversionPattern<HCLoadMaskOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCLoadMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getMask().getType());

    SmallVector<int64_t> maskShape;
    if (auto resultBareTensor =
            dyn_cast<BareTensorType>(op.getMask().getType())) {
      FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(
          cast<SymbolicallyShapedTypeInterface>(resultBareTensor));
      if (failed(dims))
        return failure();
      maskShape = std::move(*dims);
    } else if (auto resultBareVector =
                   dyn_cast<BareVectorType>(op.getMask().getType())) {
      FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(
          cast<SymbolicallyShapedTypeInterface>(resultBareVector));
      if (failed(dims))
        return failure();
      maskShape = std::move(*dims);
    } else {
      return failure();
    }
    mlir::VectorType maskVectorType =
        mlir::VectorType::get(maskShape, rewriter.getI1Type());

    // Compute slice extents from either a kernel-arg ptr (per-axis dims
    // ride as UCC inputs) or a workgroup-staged tile (its bare-tensor
    // shape is statically known).
    std::optional<KernelArgSource> kernelArg =
        resolveKernelArg(adaptor.getSource());
    Value workgroupPtr;
    SmallVector<int64_t> sourceStaticShape;
    if (!kernelArg) {
      workgroupPtr = sourcePtr(adaptor.getSource());
      if (!workgroupPtr)
        return op.emitOpError(
            "expected mask source to be a kernel-arg ptr or workgroup ptr");
      auto bare = dyn_cast<BareTensorType>(op.getSource().getType());
      if (!bare)
        return op.emitOpError(
            "workgroup-source mask requires the original source to be a "
            "bare tensor (the static shape supplies the extents)");
      FailureOr<SmallVector<int64_t>> dims =
          staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bare));
      if (failed(dims))
        return failure();
      sourceStaticShape = std::move(*dims);
    } else if (kernelArg->rank() !=
               static_cast<unsigned>(adaptor.getIndices().size())) {
      return op.emitOpError("expected kernel-arg ptr rank to match index rank");
    }

    SmallVector<SliceAxis> axes;
    std::optional<SmallVector<SliceAxis>> synthesized;
    if (kernelArg)
      synthesized = synthesizePostFlattenAxes(rewriter, op.getLoc(), *kernelArg,
                                              adaptor.getIndices(), maskShape);
    if (synthesized) {
      axes = std::move(*synthesized);
    } else {
      FailureOr<SmallVector<SliceAxis>> collected =
          collectAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                      /*requireUnitStride=*/false);
      if (failed(collected))
        return failure();
      axes = std::move(*collected);
    }
    if (llvm::count_if(axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != static_cast<int64_t>(maskShape.size()))
      return op.emitOpError("mask result rank must match slice subscript rank");

    // The mask size is the count of strided positions that stay in-bounds.
    // Unit-stride collapses to `extent - offset`; for wider strides the count
    // becomes `ceildiv(extent - offset, stride)` so e.g. a stride-2 slice into
    // an 8-row tail of a 24-row buffer reports 4 valid lanes, not 8.
    // `vector.create_mask` signed-clamps the result to `[0, N]`, so a negative
    // `extent - offset` (offset past the end) lands on a zero-clamped,
    // all-false mask without an explicit guard here.
    SmallVector<Value> maskSizes;
    for (auto [axis, info] : llvm::enumerate(axes)) {
      if (!info.isSlice)
        continue;
      Value extent;
      if (kernelArg) {
        extent = kernelArg->dims[axis];
      } else {
        extent = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                sourceStaticShape[axis])
                     .getResult();
      }
      Value remaining =
          arith::SubIOp::create(rewriter, op.getLoc(), extent, info.offset);
      Value size = remaining;
      if (!maskAxisIsUnitStride(info)) {
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

    Value mask = vector::CreateMaskOp::create(rewriter, op.getLoc(),
                                              maskVectorType, maskSizes);
    FailureOr<Value> result = materializeShapedResult(
        rewriter, op.getLoc(), converted, mask, maskShape);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

// Static shape from the original BareTensor or BareVector result type — used
// to drive per-element materialization for the LDS path. Returns failure
// for any type the converter wouldn't have produced a vector or workgroup
// ptr from.
static FailureOr<SmallVector<int64_t>> shapedResultShape(Type type) {
  if (auto bt = dyn_cast<BareTensorType>(type))
    return staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bt));
  if (auto bv = dyn_cast<BareVectorType>(type))
    return staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bv));
  return failure();
}

struct ConvertFullMaskOp : public OpConversionPattern<HCFullMaskOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCFullMaskOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getMask().getType());
    if (!converted)
      return failure();

    FailureOr<SmallVector<int64_t>> shape =
        shapedResultShape(op.getMask().getType());
    if (succeeded(shape)) {
      mlir::VectorType vectorType =
          mlir::VectorType::get(*shape, rewriter.getI1Type());
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, 1);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      FailureOr<Value> result = materializeShapedResult(
          rewriter, op.getLoc(), converted, vector, *shape);
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

    FailureOr<SmallVector<int64_t>> shape =
        shapedResultShape(op.getResult().getType());
    if (succeeded(shape)) {
      Type elementType =
          isa<mlir::VectorType>(converted)
              ? cast<mlir::VectorType>(converted).getElementType()
              : cast<PtrType>(converted).getElementType();
      mlir::VectorType vectorType = mlir::VectorType::get(*shape, elementType);
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, FillValue);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      FailureOr<Value> result = materializeShapedResult(
          rewriter, op.getLoc(), converted, vector, *shape);
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
    if (!converted)
      return failure();
    FailureOr<SmallVector<int64_t>> shape =
        shapedResultShape(op.getResult().getType());
    if (failed(shape))
      return failure();
    Type elementType = isa<mlir::VectorType>(converted)
                           ? cast<mlir::VectorType>(converted).getElementType()
                           : cast<PtrType>(converted).getElementType();
    mlir::VectorType vectorType = mlir::VectorType::get(*shape, elementType);
    Value vector = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                               vectorType, adaptor.getValue());
    FailureOr<Value> result = materializeShapedResult(
        rewriter, op.getLoc(), converted, vector, *shape);
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
    auto ptrType = dyn_cast_if_present<PtrType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
      return failure();
    auto bare = dyn_cast<BareTensorType>(op.getResult().getType());
    if (!bare)
      return failure();
    FailureOr<int64_t> count = bareTensorElementCount(bare);
    if (failed(count))
      return failure();
    rewriter.replaceOp(
        op, allocateWorkgroupPtr(rewriter, op.getLoc(), ptrType, *count));
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
    if (!vectorType)
      return failure();

    // bare_tensor → bare_vector: the bare_tensor source converts to a
    // workgroup ptr; load it into the per-lane vector via per-element
    // `hc.ptr_load`s, walking back through any `hc.buffer_view` chain so
    // strided views (e.g. column of a 2D tile) get correctly strided
    // per-lane offsets.
    if (auto ptrType = dyn_cast<PtrType>(adaptor.getValue().getType())) {
      if (ptrType.getAddrSpace() != AddrSpace::Workgroup)
        return failure();
      FailureOr<SmallVector<int64_t>> shape =
          shapedResultShape(op.getValue().getType());
      if (failed(shape))
        return failure();
      FailureOr<Value> vector = loadVectorFromPtrView(
          rewriter, op.getLoc(), op.getValue(), vectorType, *shape);
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

    // Bare-tensor result: condition + true value are workgroup ptrs, load
    // them as vectors, blend, store the result back to a fresh LDS tile.
    if (auto ptrType = dyn_cast<PtrType>(converted)) {
      if (ptrType.getAddrSpace() != AddrSpace::Workgroup)
        return failure();
      FailureOr<SmallVector<int64_t>> shape =
          shapedResultShape(op.getResult().getType());
      if (failed(shape))
        return failure();
      Type elementType = ptrType.getElementType();
      mlir::VectorType vectorType = mlir::VectorType::get(*shape, elementType);
      mlir::VectorType maskType =
          mlir::VectorType::get(*shape, rewriter.getI1Type());

      FailureOr<Value> condition = loadVectorFromPtrView(
          rewriter, op.getLoc(), op.getCondition(), maskType, *shape);
      FailureOr<Value> trueValue = loadVectorFromPtrView(
          rewriter, op.getLoc(), op.getTrueValue(), vectorType, *shape);
      if (failed(condition) || failed(trueValue))
        return op.emitOpError(
            "expected tensor select operands to lower to workgroup ptrs");

      Value falseValue =
          vector::BroadcastOp::create(rewriter, op.getLoc(), vectorType,
                                      adaptor.getFalseValue())
              .getResult();
      Value selected =
          arith::SelectOp::create(rewriter, op.getLoc(), vectorType, *condition,
                                  *trueValue, falseValue)
              .getResult();
      FailureOr<Value> result = materializeShapedResult(
          rewriter, op.getLoc(), converted, selected, *shape);
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
    std::optional<KernelArgSource> kernelArg =
        resolveKernelArg(adaptor.getDest());
    if (!kernelArg)
      return op.emitOpError(
          "expected store destination to be a kernel-arg ptr ABI cast");
    if (kernelArg->rank() != static_cast<unsigned>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected kernel-arg destination rank to match index rank");
    auto destPtrType = cast<PtrType>(kernelArg->ptr.getType());

    Type convertedSource = typeConverter->convertType(op.getSource().getType());
    FailureOr<SmallVector<int64_t>> sourceShape =
        shapedResultShape(op.getSource().getType());
    if (failed(sourceShape))
      return failure();
    FailureOr<Value> source =
        shapedValueAsVector(rewriter, op.getLoc(), op.getSource(),
                            adaptor.getSource(), convertedSource, *sourceShape);
    if (failed(source))
      return op.emitOpError("expected store source to lower to a vector");
    auto sourceType = dyn_cast<mlir::VectorType>(source->getType());
    if (!sourceType)
      return op.emitOpError("expected store source to be a vector");

    SmallVector<SliceAxis> axes;
    if (auto synthesized =
            synthesizePostFlattenAxes(rewriter, op.getLoc(), *kernelArg,
                                      adaptor.getIndices(), *sourceShape)) {
      axes = std::move(*synthesized);
    } else {
      FailureOr<SmallVector<SliceAxis>> collected =
          collectAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                      /*requireUnitStride=*/false);
      if (failed(collected))
        return failure();
      axes = std::move(*collected);
    }
    if (llvm::count_if(axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != sourceType.getRank())
      return op.emitOpError(
          "store source rank must match slice subscript rank");

    Value mask;
    if (Value originalMask = op.getMask()) {
      Type convertedMask = typeConverter->convertType(originalMask.getType());
      FailureOr<SmallVector<int64_t>> maskShape =
          shapedResultShape(originalMask.getType());
      if (failed(maskShape))
        return failure();
      FailureOr<Value> maskVector =
          shapedValueAsVector(rewriter, op.getLoc(), originalMask,
                              adaptor.getMask(), convertedMask, *maskShape);
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
          storeIndicesForCoordinate(rewriter, op.getLoc(), axes, coordinate);
      Value flat =
          linearizeKernelArgOffset(rewriter, op.getLoc(), *kernelArg, indices);
      Value addr = HCPtrOffsetOp::create(rewriter, op.getLoc(), destPtrType,
                                         kernelArg->ptr, flat)
                       .getResult();
      if (!mask) {
        HCPtrStoreOp::create(rewriter, op.getLoc(), element, addr);
        continue;
      }

      // Predicated store: emit `hc.ptr_store_pred` directly so the mask
      // rides as a first-class operand instead of via an `scf.if` guard.
      // Keeps masked stores legible for downstream patterns and matches
      // the symmetric `hc.ptr_load_pred` we emit for masked loads.
      Value guard =
          extractVectorElement(rewriter, op.getLoc(), mask, coordinate);
      HCPtrStorePredOp::create(rewriter, op.getLoc(), element, addr, guard);
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
    // bare_tensor → bare_tensor view: source is a workgroup ptr. The
    // strided per-element loader (`loadVectorFromPtrView`) walks back
    // through this op to reconstruct the source-axis pattern, so the
    // buffer_view itself just needs to type-resolve cleanly. We pass the
    // source ptr through unchanged; the op survives in the IR as a
    // metadata anchor for the consumer chain walk and gets DCE'd once
    // every consumer has lowered. (The result-ptr type matches the source
    // because both are workgroup-AS, same element type, rank-erased.)
    if (auto sourcePtrType = dyn_cast<PtrType>(adaptor.getBuffer().getType())) {
      if (sourcePtrType.getAddrSpace() != AddrSpace::Workgroup)
        return failure();
      auto resultPtrType = dyn_cast_if_present<PtrType>(converted);
      if (!resultPtrType ||
          resultPtrType.getAddrSpace() != AddrSpace::Workgroup)
        return failure();
      if (resultPtrType.getElementType() != sourcePtrType.getElementType())
        return failure();
      rewriter.replaceOp(op, adaptor.getBuffer());
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
    std::optional<KernelArgSource> kernelArg =
        resolveKernelArg(adaptor.getBuffer());
    if (!kernelArg)
      return op.emitOpError("expected buffer to be a kernel-arg ptr ABI cast");
    int64_t axis = op.getAxis();
    if (axis < 0 || static_cast<unsigned>(axis) >= kernelArg->rank())
      return op.emitOpError(
          "buffer_dim axis out of range for the kernel-arg source");
    rewriter.replaceOp(op, kernelArg->dims[axis]);
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
      ConvertIdxApplyOp, ConvertPredApplyOp, ConvertConstOp,
      ConvertIntBinaryOp<HCAddOp, arith::AddIOp>,
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
  target
      .addLegalDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                       scf::SCFDialect, vector::VectorDialect>();
  // HC ptr-family ops are produced by this pass (workgroup tiles) and must
  // pass through to the downstream `hc-lower-to-llvm` slot.
  target.addLegalOp<HCUndefValueOp, UnrealizedConversionCastOp, HCAllocOp,
                    HCPtrOffsetOp, HCPtrLoadOp, HCPtrStoreOp, HCPtrLoadPredOp,
                    HCPtrStorePredOp>();
  target.addIllegalOp<HCIdxApplyOp, HCPredApplyOp, HCConstOp, HCAddOp, HCSubOp,
                      HCMulOp, HCDivOp, HCModOp, HCNegOp, HCCmpLtOp, HCCmpLeOp,
                      HCCmpGtOp, HCCmpGeOp, HCCmpEqOp, HCCmpNeOp, HCCastOp,
                      HCBufferDimOp, HCLoadOp, HCVLoadOp, HCLoadMaskOp,
                      HCBufferViewOp, HCVecOp, HCVZerosOp, HCVOnesOp, HCVFullOp,
                      HCFullMaskOp, HCZerosOp, HCOnesOp, HCFullOp, HCEmptyOp,
                      HCSelectOp, HCStoreOp, HCForRangeOp, HCIfOp, HCYieldOp>();
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
