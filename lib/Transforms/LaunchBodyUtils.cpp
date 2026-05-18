// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared launch-body infrastructure. Hoisted out of
// HCLowerLaunchBodyPass.cpp's anonymous namespace so the slice
// passes that grew out of it can consume the same primitives
// directly. See LaunchBodyUtils.h for the public surface.

#include "LaunchBodyUtils.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

#include <utility>

using namespace mlir;
using namespace mlir::hc;

namespace mlir::hc {

ixs_node *rawNode(ExprAttr expr) {
  return const_cast<ixs_node *>(expr.getNode());
}

ixs_node *rawNode(PredAttr pred) {
  return const_cast<ixs_node *>(pred.getNode());
}

std::optional<StringRef> exactSymbolName(ExprAttr expr) {
  if (!expr)
    return std::nullopt;
  ixs_node *node = rawNode(expr);
  if (ixs_node_tag(node) != IXS_SYM)
    return std::nullopt;
  return StringRef(ixs_node_sym_name(node));
}

std::optional<StringRef> exactSymbolName(Type type) {
  auto idx = dyn_cast<IdxType>(type);
  return idx ? exactSymbolName(idx.getExpr()) : std::nullopt;
}

Type convertElementType(Type type) {
  if (isa<PredType>(type))
    return IntegerType::get(type.getContext(), 1);
  if (type.isIntOrIndexOrFloat())
    return type;
  return {};
}

FailureOr<SmallVector<int64_t>> staticIntegerShape(ShapeAttr shape,
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

FailureOr<SmallVector<int64_t>>
staticIntegerShape(SymbolicallyShapedTypeInterface shaped) {
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(shaped.getSymbolicShape(), nullptr);
  if (failed(dims))
    return failure();
  return *dims;
}

Type convertBareTensorType(BareTensorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  if (failed(staticIntegerShape(shaped)))
    return {};
  return PtrType::get(type.getContext(), AddrSpace::Workgroup, element);
}

FailureOr<int64_t> bareTensorElementCount(BareTensorType type) {
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

Type convertBareVectorType(BareVectorType type) {
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

Value castIfNeeded(OpBuilder &builder, Location loc, Value value, Type type) {
  if (value.getType() == type)
    return value;
  return UnrealizedConversionCastOp::create(builder, loc, type, value)
      .getResult(0);
}

// Post-flatten bare-carrier UCC: rank-N bundle -> rank-1 buffer view
// + idx aux. Validate shape, recurse into inner source.
static std::optional<KernelArgSource>
resolveKernelArgViaBareCarrier(UnrealizedConversionCastOp cast, Value source) {
  if (cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1 ||
      source != cast.getOutputs()[0])
    return std::nullopt;
  auto bufOut = dyn_cast<BufferType>(source.getType());
  if (!bufOut || bufOut.getShape().getDims().size() != 1)
    return std::nullopt;
  return resolveKernelArg(cast.getInputs()[0]);
}

// Canonical kernel-arg UCC shape check.
static std::optional<std::pair<Value, unsigned>>
validateKernelArgCastShape(UnrealizedConversionCastOp cast) {
  if (cast.getOutputs().size() != 1 || cast.getInputs().size() < 1 ||
      (cast.getInputs().size() - 1) % 2 != 0)
    return std::nullopt;
  Value ptr = cast.getInputs().front();
  auto ptrType = dyn_cast<PtrType>(ptr.getType());
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Global)
    return std::nullopt;
  unsigned rank = static_cast<unsigned>((cast.getInputs().size() - 1) / 2);
  return std::make_pair(ptr, rank);
}

static std::optional<SmallVector<Value>>
extractKernelArgIndexInputs(UnrealizedConversionCastOp cast, unsigned start,
                            unsigned rank) {
  SmallVector<Value> out;
  out.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value v = cast.getInputs()[start + axis];
    if (!v.getType().isIndex())
      return std::nullopt;
    out.push_back(v);
  }
  return out;
}

std::optional<KernelArgSource> resolveKernelArg(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast)
    return std::nullopt;

  if (std::optional<KernelArgSource> via =
          resolveKernelArgViaBareCarrier(cast, source))
    return via;

  std::optional<std::pair<Value, unsigned>> shape =
      validateKernelArgCastShape(cast);
  if (!shape)
    return std::nullopt;
  auto [ptr, rank] = *shape;
  std::optional<SmallVector<Value>> dims =
      extractKernelArgIndexInputs(cast, /*start=*/1, rank);
  if (!dims)
    return std::nullopt;
  std::optional<SmallVector<Value>> strides =
      extractKernelArgIndexInputs(cast, /*start=*/1 + rank, rank);
  if (!strides)
    return std::nullopt;

  KernelArgSource info;
  info.ptr = ptr;
  info.dims = std::move(*dims);
  info.strides = std::move(*strides);
  return info;
}

static bool isPostFlattenKernelArgSource(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1 ||
      source != cast.getOutputs()[0])
    return false;
  auto bufOut = dyn_cast<BufferType>(source.getType());
  return bufOut && bufOut.getShape().getDims().size() == 1;
}

static KernelArgSource flatKernelArgView(OpBuilder &builder, Location loc,
                                         const KernelArgSource &inner) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  KernelArgSource flat;
  flat.ptr = inner.ptr;
  flat.dims = {zero};
  flat.strides = {one};
  return flat;
}

std::optional<KernelArgSource>
resolveAccessKernelArg(OpBuilder &builder, Location loc, Value source) {
  auto info = resolveKernelArg(source);
  if (!info)
    return std::nullopt;
  if (isPostFlattenKernelArgSource(source) && info->rank() > 1)
    return flatKernelArgView(builder, loc, *info);
  return info;
}

Value linearizeKernelArgOffset(OpBuilder &builder, Location loc,
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

Value indexCast(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), value)
      .getResult(0);
}

// `$STRIDE_<axis>_<argname>` -> corresponding stride value.
static Value resolveBundleStrideSym(const KernelArgSource &source,
                                    StringRef symName) {
  if (!symName.consume_front("$STRIDE_"))
    return Value{};
  unsigned axis = 0;
  if (symName.consumeInteger(10, axis))
    return Value{};
  if (!symName.starts_with("_"))
    return Value{};
  if (axis >= source.strides.size())
    return Value{};
  return source.strides[axis];
}

static std::optional<Value> lookupBundleDim(BufferType bundleType,
                                            const KernelArgSource &info,
                                            StringRef symbol) {
  for (auto [axis, attr] : llvm::enumerate(bundleType.getShape().getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<StringRef> dimSym = exactSymbolName(expr);
    if (dimSym && *dimSym == symbol && axis < info.dims.size())
      return info.dims[axis];
  }
  return std::nullopt;
}

static Value resolveToBundleIndex(Value value) {
  auto idxType = dyn_cast<IdxType>(value.getType());
  if (!idxType)
    return Value{};
  std::optional<StringRef> symbol = exactSymbolName(idxType.getExpr());
  if (!symbol)
    return Value{};
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1)
    return Value{};
  auto bundleType = dyn_cast<BufferType>(cast.getInputs()[0].getType());
  if (!bundleType)
    return Value{};
  std::optional<KernelArgSource> info = resolveKernelArg(cast.getInputs()[0]);
  if (!info)
    return Value{};
  if (std::optional<Value> dim = lookupBundleDim(bundleType, *info, *symbol))
    return *dim;
  return resolveBundleStrideSym(*info, *symbol);
}

Value indexCastViaBundle(OpBuilder &builder, Location loc, Value value) {
  if (Value direct = resolveToBundleIndex(value))
    return direct;
  return indexCast(builder, loc, value);
}

Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                      Location loc) {
  if (inputs.size() != 1)
    return {};
  // Short-circuit `idx<sym>` -> `index` through the bundle root;
  // bare UCC won't fold via reconcileUnrealizedCasts.
  if (type.isIndex()) {
    if (Value direct = resolveToBundleIndex(inputs.front()))
      return direct;
  }
  return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
      .getResult(0);
}

HCLaunchBodyTypeConverter::HCLaunchBodyTypeConverter() {
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

// Bind free shape syms to per-axis dim values from the UCC fragment.
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

// `$WO[k] = $WG[k] * $WGS[k]`: workgroup-tile upper-left corner.
static void bindWorkOffsets(gpu::LaunchOp launch, OpBuilder &builder,
                            Location loc, BoundValues &boundValues) {
  gpu::KernelDim3 blockIds = launch.getBlockIds();
  gpu::KernelDim3 blockSizes = launch.getBlockSizeOperandValues();
  Value bi[] = {blockIds.x, blockIds.y, blockIds.z};
  Value bs[] = {blockSizes.x, blockSizes.y, blockSizes.z};
  for (auto axis : llvm::seq<size_t>(0, std::size(bi))) {
    SmallString<8> name("$WO");
    name += Twine(axis).str();
    Value product = arith::MulIOp::create(builder, loc, bi[axis], bs[axis]);
    boundValues.bind(name, product);
  }
}

static void bindPreFlattenKernelArgCast(UnrealizedConversionCastOp cast,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        BoundValues &boundValues) {
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
    boundValues.bind(*symbol, indexCast(rewriter, loc, input));
}

static void bindPostFlattenRetypeCast(UnrealizedConversionCastOp cast,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc, BoundValues &boundValues) {
  if (cast.getInputs().size() != 1)
    return;
  for (Value output : cast.getOutputs().drop_front()) {
    std::optional<StringRef> symbol = exactSymbolName(output.getType());
    if (symbol)
      boundValues.bind(*symbol, indexCastViaBundle(rewriter, loc, output));
  }
}

static void bindKernelArgCastSymbols(UnrealizedConversionCastOp cast,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, BoundValues &boundValues) {
  if (cast.getOutputs().size() == 1) {
    bindPreFlattenKernelArgCast(cast, rewriter, loc, boundValues);
    return;
  }
  bindPostFlattenRetypeCast(cast, rewriter, loc, boundValues);
}

static void bindAncestorBlockArgSymbols(Operation *anchor, gpu::LaunchOp launch,
                                        ConversionPatternRewriter &rewriter,
                                        BoundValues &boundValues) {
  for (Block *block = anchor->getBlock(); block;) {
    for (BlockArgument arg : block->getArguments()) {
      std::optional<StringRef> symbol = exactSymbolName(arg.getType());
      if (!symbol)
        continue;
      boundValues.bind(*symbol, indexCast(rewriter, anchor->getLoc(), arg));
    }
    Operation *parent = block->getParentOp();
    if (!parent || parent == launch.getOperation())
      break;
    block = parent->getBlock();
  }
}

BoundValues collectBoundValues(Operation *anchor,
                               ConversionPatternRewriter &rewriter) {
  BoundValues boundValues;
  auto launch = anchor->getParentOfType<gpu::LaunchOp>();
  if (!launch)
    return boundValues;

  bindLaunchDim3("$WG", launch.getBlockIds(), boundValues);
  bindLaunchDim3("$WI", launch.getThreadIds(), boundValues);
  bindLaunchDim3("$WGS", launch.getBlockSizeOperandValues(), boundValues);
  bindWorkOffsets(launch, rewriter, anchor->getLoc(), boundValues);

  launch.walk([&](UnrealizedConversionCastOp cast) {
    bindKernelArgCastSymbols(cast, rewriter, anchor->getLoc(), boundValues);
  });

  bindAncestorBlockArgSymbols(anchor, launch, rewriter, boundValues);

  return boundValues;
}

BoundValues collectApplyBindings(Operation *op,
                                 ConversionPatternRewriter &rewriter,
                                 ArrayAttr symbols,
                                 ValueRange convertedOperands) {
  BoundValues boundValues = collectBoundValues(op, rewriter);
  for (auto [attr, operand] : llvm::zip(symbols, convertedOperands)) {
    StringRef name = cast<StringAttr>(attr).getValue();
    if (name.empty())
      continue;
    boundValues.symbols[name] =
        indexCastViaBundle(rewriter, op->getLoc(), operand);
  }
  return boundValues;
}

static bool isOneNode(ixs_node *node) {
  return ixs_node_tag(node) == IXS_INT && ixs_node_int_val(node) == 1;
}

ExprLowerer::ExprLowerer(OpBuilder &builder, Location loc,
                         const BoundValues &boundValues)
    : builder(builder), loc(loc), boundValues(boundValues) {}

FailureOr<Value> ExprLowerer::lower(ExprAttr expr) {
  if (!expr)
    return failure();
  return lowerNode(rawNode(expr));
}

FailureOr<Value> ExprLowerer::lower(PredAttr pred) {
  if (!pred)
    return failure();
  return lowerPredNode(rawNode(pred));
}

Value ExprLowerer::constant(int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value).getResult();
}

FailureOr<Value> ExprLowerer::lowerNode(ixs_node *node) {
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

FailureOr<Value> ExprLowerer::lowerPredNode(ixs_node *node) {
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

FailureOr<Value> ExprLowerer::lowerRational(ixs_node *node) {
  int64_t numerator = ixs_node_rat_num(node);
  int64_t denominator = ixs_node_rat_den(node);
  if (denominator == 1)
    return constant(numerator);
  return failure();
}

FailureOr<Value> ExprLowerer::lowerSymbol(ixs_node *node) {
  StringRef name(ixs_node_sym_name(node));
  if (Value value = boundValues.lookup(name))
    return value;
  if (unresolvedSymbol.empty())
    unresolvedSymbol = name.str();
  return failure();
}

FailureOr<Value> ExprLowerer::lowerAdd(ixs_node *node) {
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

FailureOr<Value> ExprLowerer::scaleTerm(ixs_node *coeff, Value term) {
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

FailureOr<Value> ExprLowerer::lowerMul(ixs_node *node) {
  FailureOr<std::pair<Value, Value>> fraction = lowerAsFraction(node);
  if (failed(fraction))
    return failure();
  if (auto denominator = fraction->second.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(denominator.getValue()))
      if (attr.getInt() == 1)
        return fraction->first;
  }
  return arith::DivUIOp::create(builder, loc, fraction->first, fraction->second)
      .getResult();
}

FailureOr<Value> ExprLowerer::lowerCeil(ixs_node *node) {
  FailureOr<std::pair<Value, Value>> fraction =
      lowerAsFraction(ixs_node_unary_arg(node));
  if (failed(fraction))
    return failure();
  return arith::CeilDivUIOp::create(builder, loc, fraction->first,
                                    fraction->second)
      .getResult();
}

FailureOr<std::pair<Value, Value>>
ExprLowerer::lowerAsFraction(ixs_node *node) {
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
    FailureOr<Value> factor = lowerNode(ixs_node_mul_factor_base(node, index));
    if (failed(factor))
      return failure();
    for (int32_t power = 0; power < exponent; ++power)
      numeratorValue =
          arith::MulIOp::create(builder, loc, numeratorValue, *factor);
  }
  return std::pair<Value, Value>{numeratorValue, constant(denominator)};
}

template <typename OpT>
FailureOr<Value> ExprLowerer::lowerBinary(ixs_node *node) {
  FailureOr<Value> lhs = lowerNode(ixs_node_binary_lhs(node));
  FailureOr<Value> rhs = lowerNode(ixs_node_binary_rhs(node));
  if (failed(lhs) || failed(rhs))
    return failure();
  return OpT::create(builder, loc, *lhs, *rhs).getResult();
}

FailureOr<Value> ExprLowerer::lowerCmp(ixs_node *node) {
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

template <typename OpT>
FailureOr<Value> ExprLowerer::lowerLogic(ixs_node *node) {
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

} // namespace mlir::hc
