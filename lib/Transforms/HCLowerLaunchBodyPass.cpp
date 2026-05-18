// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-body`: runs after HC kernels are in `gpu.launch`.
// Workgroup tiles -> `hc.alloc` + `!hc.ptr<workgroup, T>`. Kernel-args
// arrive as `(ptr<global>, dim*, stride*)` UCC; walk back, emit
// `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`.
// Contract: `doc/layouts.md` "hc.ptr and memory ops".

#include "hc/Transforms/Passes.h"

#include "LaunchUtils.h"
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

// `bare_tensor` -> `!hc.ptr<workgroup, T>`; flat storage, static dims
// required (alloc needs a constant element count).
static Type convertBareTensorType(BareTensorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  if (failed(staticIntegerShape(shaped)))
    return {};
  return PtrType::get(type.getContext(), AddrSpace::Workgroup, element);
}

// Product of static dims.
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

static Value resolveToBundleIndex(Value value);

static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
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

// Launch-boundary UCC: `!hc.buffer<T, [dims]>` -> `(ptr<global>,
// dim_0..dim_{r-1}, stride_0..stride_{r-1})`.
struct KernelArgSource {
  Value ptr;
  SmallVector<Value> dims;
  SmallVector<Value> strides;

  unsigned rank() const { return dims.size(); }
};

static std::optional<KernelArgSource> resolveKernelArg(Value source);

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

// `rank` `index`-typed inputs starting at `start`; fails on type mismatch.
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

static std::optional<KernelArgSource> resolveKernelArg(Value source) {
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

// Rank-1 carrier output of the post-flatten retype UCC. Lets a
// consumer distinguish rank-N native vs single-composed-1D access.
static bool isPostFlattenKernelArgSource(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1 ||
      source != cast.getOutputs()[0])
    return false;
  auto bufOut = dyn_cast<BufferType>(source.getType());
  return bufOut && bufOut.getShape().getDims().size() == 1;
}

// Synthetic rank-1 view: same ptr, placeholder dim, unit stride.
// Placeholder folds away at canonicalize.
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

// Kernel-arg view shaped to match the access: rank-N for native,
// synthetic rank-1 for the post-flatten retype.
static std::optional<KernelArgSource>
resolveAccessKernelArg(OpBuilder &builder, Location loc, Value source) {
  auto info = resolveKernelArg(source);
  if (!info)
    return std::nullopt;
  if (isPostFlattenKernelArgSource(source) && info->rank() > 1)
    return flatKernelArgView(builder, loc, *info);
  return info;
}

// Linear element offset: `sum_axis(indices[a] * strides[a])`. In
// element units; `hc.ptr_offset` scales by element size downstream.
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

static Value indexCast(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), value)
      .getResult(0);
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

// `$STRIDE_<axis>_<argname>` -> corresponding stride value, per
// `buildDefaultStridedBufferLayout`. Null on miss.
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

// Find bundle dim sym matching `symbol`; null on miss.
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

// Short-circuit `idx<sym>->index` through the kernel-arg bundle;
// fresh UCC otherwise.
static Value indexCastViaBundle(OpBuilder &builder, Location loc, Value value) {
  if (Value direct = resolveToBundleIndex(value))
    return direct;
  return indexCast(builder, loc, value);
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
// CSE folds per-apply duplicates downstream.
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

// Pre-flatten UCC: multi-input bundle -> single buffer output. Bind
// buffer's shape syms (M, N, ...) to per-axis dim inputs.
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

// Post-flatten retype UCC: rank-N buffer -> (rank-1 buffer, idx<sym>
// aux*). Bind each idx-typed output's symbol so ambient lowering
// can resolve applies that reference it.
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

// Block args typed `!hc.idx<sym>` (e.g. `hc.for_range` IV) self-bind
// for `sym`. Only ancestor blocks of `anchor` qualify: sibling
// scopes shadow / fail dominance.
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

static BoundValues collectBoundValues(Operation *anchor,
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
    // Record first miss; caller emits the diagnostic.
    if (unresolvedSymbol.empty())
      unresolvedSymbol = name.str();
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
  std::string unresolvedSymbol;

public:
  // First unresolved sym, empty if all bound. See doc/layouts.md
  // "Free symbols in layout offsets".
  StringRef lastUnresolvedSymbol() const { return unresolvedSymbol; }
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

// Operand bindings override ambient; ambient fills free-sym gaps.
static BoundValues collectApplyBindings(Operation *op,
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
    if (failed(lowered)) {
      StringRef sym = lowerer.lastUnresolvedSymbol();
      if (!sym.empty())
        return op.emitOpError("cannot lower idx_apply: free symbol '")
               << sym << "' has no binding in the surrounding scope "
               << "(launch geometry, kernel-arg aux idx, or ancestor block "
                  "argument); supply it via the access op's aux operands or "
                  "the producing op's binding slot";
      return op.emitOpError("failed to lower idx_apply expression");
    }
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
    if (failed(lowered)) {
      StringRef sym = lowerer.lastUnresolvedSymbol();
      if (!sym.empty())
        return op.emitOpError("cannot lower pred_apply: free symbol '")
               << sym << "' has no binding in the surrounding scope";
      return op.emitOpError("failed to lower pred_apply predicate");
    }
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

// Float counterpart to `ConvertIntBinaryOp`.
template <typename OpT, typename ArithOpT>
struct ConvertFloatBinaryOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted || !isa<FloatType>(converted))
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

// Post-flatten: one composed offset, rank-1 unit-stride ABI,
// rank-1 iter. Synthesize a full-slice axis so the per-lane machinery
// sees `composed + lane`. Strided tiles bail to the slice-aware path.
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

// Flat `!hc.ptr<workgroup, T>` alloc for `count` elements. AMDGPU
// LDS is 1D; rank erases here.
static Value allocateWorkgroupPtr(OpBuilder &builder, Location loc,
                                  PtrType ptrType, int64_t count) {
  Value countValue =
      arith::ConstantIndexOp::create(builder, loc, count).getResult();
  return HCAllocOp::create(builder, loc, ptrType, countValue).getResult();
}

// Linear -> per-axis coords, last axis fastest. Constant-index SSAs.
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

// Per-element store from vector -> flat workgroup ptr, rightmost-
// fastest. Scalar form dodges LLVM's i1 byte-vs-bit store mismatch.
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

// Walk back through UCC to recover a workgroup-AS `!hc.ptr`. Null on miss.
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

// Workgroup-ptr source + lane->source-offset indexing pattern. No
// buffer_view in chain -> one full-slice axis per source dim.
struct PtrViewSource {
  Value sourcePtr;
  PtrType ptrType;
  SmallVector<int64_t> sourceShape;
  SmallVector<SliceAxis> axes;
};

// Workgroup ptr from remapped tile; AS-checked.
static FailureOr<std::pair<Value, PtrType>>
resolveWorkgroupPtr(Value remapped) {
  Value srcPtr = sourcePtr(remapped);
  if (!srcPtr)
    return failure();
  auto srcPtrType = dyn_cast<PtrType>(srcPtr.getType());
  if (!srcPtrType || srcPtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  return std::make_pair(srcPtr, srcPtrType);
}

// Buffer-view source: remap each subscript, materialize axes.
static FailureOr<PtrViewSource>
resolveBufferViewSource(HCBufferViewOp bv,
                        ConversionPatternRewriter &rewriter) {
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
  FailureOr<std::pair<Value, PtrType>> ptr = resolveWorkgroupPtr(remappedSrc);
  if (failed(ptr))
    return failure();
  // Slice operands lowered before any ptr-view consumer runs.
  SmallVector<Value> remappedIndices;
  remappedIndices.reserve(bv.getIndices().size());
  for (Value idx : bv.getIndices()) {
    Value remapped = rewriter.getRemappedValue(idx);
    if (!remapped)
      return failure();
    remappedIndices.push_back(remapped);
  }
  // Non-unit strides ok: per-element loop scales `stride*iter +
  // offset` per lane.
  FailureOr<SmallVector<SliceAxis>> axes =
      collectAxes(bv.getOperation(), remappedIndices, rewriter,
                  /*requireUnitStride=*/false);
  if (failed(axes))
    return failure();
  if (axes->size() != sourceShape->size())
    return bv.emitOpError("expected one view subscript per source axis");
  return PtrViewSource{ptr->first, ptr->second, *sourceShape, std::move(*axes)};
}

// Trivial axes: full slice `(offset=0, stride=1)` per dim; loop
// degenerates to flat `0..N`.
static SmallVector<SliceAxis>
buildTrivialSliceAxes(ArrayRef<int64_t> shape,
                      ConversionPatternRewriter &rewriter, Location loc) {
  SmallVector<SliceAxis> trivialAxes;
  trivialAxes.reserve(shape.size());
  Value zero = zeroIndex(rewriter, loc);
  Value one = oneIndex(rewriter, loc);
  for (size_t i = 0; i < shape.size(); ++i) {
    SliceAxis ax;
    ax.offset = zero;
    ax.stride = one;
    ax.isSlice = true;
    trivialAxes.push_back(ax);
  }
  return trivialAxes;
}

// Walk one `hc.buffer_view` to its workgroup ptr + index pattern.
// Non-view sources (`hc.alloc`, fresh LDS) get a trivial axis
// pattern so the loader is uniform.
static FailureOr<PtrViewSource>
resolvePtrViewSource(Value original, ConversionPatternRewriter &rewriter) {
  if (auto bv = original.getDefiningOp<HCBufferViewOp>())
    return resolveBufferViewSource(bv, rewriter);

  Value remapped = rewriter.getRemappedValue(original);
  if (!remapped)
    return failure();
  FailureOr<std::pair<Value, PtrType>> ptr = resolveWorkgroupPtr(remapped);
  if (failed(ptr))
    return failure();
  auto bare = dyn_cast<BareTensorType>(original.getType());
  if (!bare)
    return failure();
  FailureOr<SmallVector<int64_t>> shape =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bare));
  if (failed(shape))
    return failure();
  SmallVector<SliceAxis> trivialAxes =
      buildTrivialSliceAxes(*shape, rewriter, original.getLoc());
  return PtrViewSource{ptr->first, ptr->second, *shape, std::move(trivialAxes)};
}

// Per-element load from workgroup ptr through any `hc.buffer_view`
// chain. Lane V offset: `base + sum_k(view_idx[k] *
// source_stride[slice_axis_k])`. Trivial pattern collapses to
// contiguous; strided threads per-lane `hc.ptr_offset`+`hc.ptr_load`.
// Per-element form sidesteps LLVM's scalar/vector i1 discrepancy.

// Row-major strides: last axis = 1, each prev = next_stride * next_dim.
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0; --axis)
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  return strides;
}

// Positions of slice axes in source order.
static SmallVector<int64_t>
collectSliceAxisPositions(ArrayRef<SliceAxis> axes) {
  SmallVector<int64_t> positions;
  for (int64_t i = 0; i < static_cast<int64_t>(axes.size()); ++i)
    if (axes[i].isSlice)
      positions.push_back(i);
  return positions;
}

// Scalar (non-slice) axes' base: `sum_k(offset_k * source_stride_k)`.
static Value computeScalarAxesBaseOffset(OpBuilder &rewriter, Location loc,
                                         ArrayRef<SliceAxis> axes,
                                         ArrayRef<int64_t> sourceStrides) {
  Value base = zeroIndex(rewriter, loc);
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i].isSlice)
      continue;
    Value strideVal =
        arith::ConstantIndexOp::create(rewriter, loc, sourceStrides[i])
            .getResult();
    Value contrib =
        arith::MulIOp::create(rewriter, loc, axes[i].offset, strideVal)
            .getResult();
    base = arith::AddIOp::create(rewriter, loc, base, contrib).getResult();
  }
  return base;
}

// Lane idx -> per-axis view coords, row-major.
static SmallVector<int64_t> unflattenLaneIndex(int64_t lin,
                                               ArrayRef<int64_t> viewShape) {
  SmallVector<int64_t> coords(viewShape.size());
  int64_t remaining = lin;
  for (int64_t axis = static_cast<int64_t>(viewShape.size()) - 1; axis >= 0;
       --axis) {
    coords[axis] = remaining % viewShape[axis];
    remaining /= viewShape[axis];
  }
  return coords;
}

// Add per-lane slice-axis contributions:
// `base += (offset + coord*stride) * source_stride`.
static Value applyLaneSliceOffset(OpBuilder &rewriter, Location loc, Value base,
                                  ArrayRef<int64_t> coordInts,
                                  ArrayRef<int64_t> sliceAxisPositions,
                                  ArrayRef<SliceAxis> axes,
                                  ArrayRef<int64_t> sourceStrides) {
  Value flat = base;
  for (size_t vi = 0; vi < sliceAxisPositions.size(); ++vi) {
    int64_t srcAxisPos = sliceAxisPositions[vi];
    const SliceAxis &ax = axes[srcAxisPos];
    Value coordVal =
        arith::ConstantIndexOp::create(rewriter, loc, coordInts[vi])
            .getResult();
    Value scaledStride =
        arith::MulIOp::create(rewriter, loc, coordVal, ax.stride).getResult();
    Value indexInSource =
        arith::AddIOp::create(rewriter, loc, ax.offset, scaledStride)
            .getResult();
    Value srcStrideVal =
        arith::ConstantIndexOp::create(rewriter, loc, sourceStrides[srcAxisPos])
            .getResult();
    Value contrib =
        arith::MulIOp::create(rewriter, loc, indexInSource, srcStrideVal)
            .getResult();
    flat = arith::AddIOp::create(rewriter, loc, flat, contrib).getResult();
  }
  return flat;
}

static FailureOr<Value>
loadVectorFromPtrView(ConversionPatternRewriter &rewriter, Location loc,
                      Value original, mlir::VectorType vectorType,
                      ArrayRef<int64_t> viewShape) {
  FailureOr<PtrViewSource> src = resolvePtrViewSource(original, rewriter);
  if (failed(src))
    return failure();
  if (vectorType.getShape() != viewShape)
    return failure();

  SmallVector<int64_t> sourceStrides = computeRowMajorStrides(src->sourceShape);
  SmallVector<int64_t> sliceAxisPositions =
      collectSliceAxisPositions(src->axes);
  if (sliceAxisPositions.size() != viewShape.size())
    return failure();

  Value baseOffset =
      computeScalarAxesBaseOffset(rewriter, loc, src->axes, sourceStrides);

  int64_t total = 1;
  for (int64_t d : viewShape)
    total *= d;
  Type elementType = vectorType.getElementType();
  Value result = arith::ConstantOp::create(rewriter, loc, vectorType,
                                           rewriter.getZeroAttr(vectorType))
                     .getResult();
  for (int64_t lin = 0; lin != total; ++lin) {
    SmallVector<int64_t> coordInts = unflattenLaneIndex(lin, viewShape);
    Value flat =
        applyLaneSliceOffset(rewriter, loc, baseOffset, coordInts,
                             sliceAxisPositions, src->axes, sourceStrides);
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

// Materialize vector via converter's carrier: `vector<...xT>`
// (per-workitem register, passthrough) or `!hc.ptr<workgroup, T>`
// (LDS, alloc + per-element write).
//
// Scalar splat from a constant vector for the splat fast path.
// Non-splat -> null, caller falls back to unrolled stores.
static Value extractSplatScalar(OpBuilder &builder, Location loc,
                                Value vector) {
  auto cst = vector.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return Value{};
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return Value{};
  TypedAttr scalarAttr = dense.getSplatValue<TypedAttr>();
  return arith::ConstantOp::create(builder, loc, scalarAttr).getResult();
}

// Workgroup-cooperative LDS fill as `hc.generic` with one parallel
// iter per slot. `hc-lower-generic`'s collective path dispatches one
// slot per thread. Uniform surface for `hc-insert-workgroup-barriers`.
static LogicalResult emitInitFillGeneric(OpBuilder &builder, Location loc,
                                         Value lds, PtrType ptrType,
                                         Value scalar, int64_t total) {
  MLIRContext *ctx = builder.getContext();
  Type elemTy = ptrType.getElementType();
  if (!elemTy || scalar.getType() != elemTy)
    return failure();

  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  FailureOr<sym::ExprHandle> linExpr = sym::composeExprSym(store, "lin");
  if (failed(linExpr))
    return failure();
  ArrayAttr outOff = ArrayAttr::get(ctx, {ExprAttr::get(ctx, *linExpr)});
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  StringAttr linSym = StringAttr::get(ctx, "lin");
  ArrayAttr iterSyms = ArrayAttr::get(ctx, {linSym});
  ArrayAttr iterKinds =
      ArrayAttr::get(ctx, {IterKindAttr::get(ctx, IterKind::Parallel)});

  Value totalVal =
      arith::ConstantIndexOp::create(builder, loc, total).getResult();

  auto generic = HCGenericOp::create(
      builder, loc,
      /*resultTypes=*/TypeRange{}, iterSyms, ValueRange(totalVal), iterKinds,
      /*ins=*/ValueRange{}, /*outs=*/ValueRange{lds},
      /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  body->addArgument(elemTy, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  HCYieldOp::create(bodyBuilder, loc, ValueRange{scalar});
  return success();
}

static FailureOr<Value> writeVectorToFreshLDS(OpBuilder &builder, Location loc,
                                              PtrType ptrType, Value vector,
                                              ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t d : shape)
    total *= d;
  Value lds = allocateWorkgroupPtr(builder, loc, ptrType, total);

  // Splat: route through `hc.generic` for uniform barrier-pass surface.
  if (Value scalar = extractSplatScalar(builder, loc, vector)) {
    if (failed(emitInitFillGeneric(builder, loc, lds, ptrType, scalar, total)))
      return failure();
    return lds;
  }

  // Non-splat fallback: unrolled per-element stores. New callers
  // should route through `hc.generic` to keep the barrier surface uniform.
  if (failed(writeVectorToWorkgroupPtr(builder, loc, vector, lds, shape)))
    return failure();
  return lds;
}

static FailureOr<Value>
materializeShapedResult(OpBuilder &builder, Location loc, Type convertedType,
                        Value vector, ArrayRef<int64_t> shape) {
  if (auto vectorType = dyn_cast_if_present<mlir::VectorType>(convertedType)) {
    if (vector.getType() != vectorType)
      return failure();
    return vector;
  }
  if (auto ptrType = dyn_cast_if_present<PtrType>(convertedType)) {
    if (ptrType.getAddrSpace() != AddrSpace::Workgroup)
      return failure();
    return writeVectorToFreshLDS(builder, loc, ptrType, vector, shape);
  }
  return failure();
}

// Read converted shaped value as multi-dim vector. Vector
// passthrough; LDS -> per-element loads via `loadVectorFromPtrView`.
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

// Per-axis index list for one result lane. Slice: `offset +
// coord*stride`. Scalar: `offset`.
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

// Result bundle for a load-like op.
struct LoadLikeResultShape {
  mlir::VectorType vectorType;
  SmallVector<int64_t> shape;
  Type elementType;
};

// Bare-vector result -> `(VectorType, shape, elementType)`.
static FailureOr<LoadLikeResultShape>
unpackLoadLikeResultShape(const TypeConverter &converter, Type bareResultType) {
  Type converted = converter.convertType(bareResultType);
  auto vectorType = dyn_cast_if_present<mlir::VectorType>(converted);
  if (!vectorType)
    return failure();
  auto bareVector = dyn_cast<BareVectorType>(bareResultType);
  if (!bareVector)
    return failure();
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bareVector));
  if (failed(dims))
    return failure();
  return LoadLikeResultShape{vectorType, std::move(*dims),
                             vectorType.getElementType()};
}

// Per-axis axes for a kernel-arg access. Post-flatten synthesizer
// first; falls back to generic collector.
static FailureOr<SmallVector<SliceAxis>>
resolveAccessAxes(Operation *op, ValueRange indices,
                  ConversionPatternRewriter &rewriter,
                  const KernelArgSource &kernelArg, ArrayRef<int64_t> shape) {
  if (auto synthesized = synthesizePostFlattenAxes(rewriter, op->getLoc(),
                                                   kernelArg, indices, shape))
    return std::move(*synthesized);
  return collectAxes(op, indices, rewriter, /*requireUnitStride=*/false);
}

// Per-lane scalar load chain: `hc.ptr_offset` + `hc.ptr_load` +
// `vector.insert` per coord. LLVM SLP recombines unit-stride.
static Value emitLoadLikeLanes(OpBuilder &rewriter, Location loc,
                               const KernelArgSource &kernelArg,
                               PtrType sourcePtrType,
                               const LoadLikeResultShape &result,
                               ArrayRef<SliceAxis> axes) {
  Value laneVec =
      arith::ConstantOp::create(rewriter, loc, result.vectorType,
                                rewriter.getZeroAttr(result.vectorType))
          .getResult();
  for (ArrayRef<int64_t> resultCoord : staticVectorCoordinates(result.shape)) {
    SmallVector<Value> indices =
        kernelArgLaneIndices(rewriter, loc, axes, resultCoord);
    Value flat = linearizeKernelArgOffset(rewriter, loc, kernelArg, indices);
    Value addr =
        HCPtrOffsetOp::create(rewriter, loc, sourcePtrType, kernelArg.ptr, flat)
            .getResult();
    Value elem = HCPtrLoadOp::create(rewriter, loc, result.elementType, addr)
                     .getResult();
    laneVec =
        vector::InsertOp::create(rewriter, loc, elem, laneVec, resultCoord)
            .getResult();
  }
  return laneVec;
}

// Per-lane vector load from kernel-arg `!hc.ptr<global, T>`. Per-
// element form covers unit + non-unit stride uniformly and sidesteps
// the `vector<Nxi1>` packed/byte mismatch. Bare-tensor loads
// (workgroup tiles) route through the generic pipeline upstream.
template <typename OpT>
struct ConvertLoadLikeOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LoadLikeResultShape> result = unpackLoadLikeResultShape(
        *this->typeConverter, op.getResult().getType());
    if (failed(result))
      return failure();

    Value source = [&]() -> Value {
      if constexpr (std::is_same_v<OpT, HCLoadOp>)
        return adaptor.getBuffer();
      else
        return adaptor.getSource();
    }();
    std::optional<KernelArgSource> kernelArg =
        resolveAccessKernelArg(rewriter, op.getLoc(), source);
    if (!kernelArg)
      return op.emitOpError(
          "expected load source to be a kernel-arg ptr ABI cast");
    if (kernelArg->rank() != static_cast<unsigned>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected kernel-arg source rank to match index rank");
    auto sourcePtrType = cast<PtrType>(kernelArg->ptr.getType());

    FailureOr<SmallVector<SliceAxis>> axes =
        resolveAccessAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                          *kernelArg, result->shape);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != static_cast<int64_t>(result->shape.size()))
      return op.emitOpError("load result rank must match slice subscript rank");

    Value laneVec = emitLoadLikeLanes(rewriter, op.getLoc(), *kernelArg,
                                      sourcePtrType, *result, *axes);
    rewriter.replaceOp(op, laneVec);
    return success();
  }
};

// Static shape from any symbolically shaped type. Drives
// per-element materialization across bare and semantic carriers.
static FailureOr<SmallVector<int64_t>> shapedResultShape(Type type) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type))
    return staticIntegerShape(shaped);
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
    if (failed(shape))
      return failure();
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
    if (failed(shape))
      return failure();
    auto origShaped =
        cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
    Type elementType = convertElementType(origShaped.getSymbolicElementType());
    if (!elementType)
      return failure();
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
    auto origShaped =
        cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
    Type elementType = convertElementType(origShaped.getSymbolicElementType());
    if (!elementType)
      return failure();
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

    // `bare_tensor` (-> workgroup ptr) -> `bare_vector`: per-element
    // load through any `hc.buffer_view` chain.
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

    // Bare-tensor result: load condition + true as vectors, blend,
    // write back to fresh LDS.
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

// Store source as `(vector, type)` via `shapedValueAsVector`.
static FailureOr<std::pair<Value, mlir::VectorType>>
lowerStoreSource(HCStoreOp op, HCStoreOp::Adaptor adaptor,
                 ConversionPatternRewriter &rewriter,
                 const TypeConverter &converter) {
  Type convertedSource = converter.convertType(op.getSource().getType());
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
  return std::make_pair(*source, sourceType);
}

// Optional store mask -> `i1` vector matching `sourceType`. Null when absent.
static FailureOr<Value> lowerStoreMask(HCStoreOp op, HCStoreOp::Adaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       const TypeConverter &converter,
                                       mlir::VectorType sourceType) {
  Value originalMask = op.getMask();
  if (!originalMask)
    return Value{};
  Type convertedMask = converter.convertType(originalMask.getType());
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
  return *maskVector;
}

// Per-lane scalar stores. Null mask -> `hc.ptr_store`; non-null ->
// `hc.ptr_store_pred` (mask as first-class operand).
static void emitStoreLanes(OpBuilder &rewriter, Location loc,
                           const KernelArgSource &kernelArg,
                           PtrType destPtrType, Value source,
                           mlir::VectorType sourceType, Value mask,
                           ArrayRef<SliceAxis> axes) {
  for (ArrayRef<int64_t> coordinate :
       staticVectorCoordinates(sourceType.getShape())) {
    Value element = extractVectorElement(rewriter, loc, source, coordinate);
    SmallVector<Value> indices =
        storeIndicesForCoordinate(rewriter, loc, axes, coordinate);
    Value flat = linearizeKernelArgOffset(rewriter, loc, kernelArg, indices);
    Value addr =
        HCPtrOffsetOp::create(rewriter, loc, destPtrType, kernelArg.ptr, flat)
            .getResult();
    if (!mask) {
      HCPtrStoreOp::create(rewriter, loc, element, addr);
      continue;
    }
    Value guard = extractVectorElement(rewriter, loc, mask, coordinate);
    HCPtrStorePredOp::create(rewriter, loc, element, addr, guard);
  }
}

struct ConvertStoreOp : public OpConversionPattern<HCStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<KernelArgSource> kernelArg =
        resolveAccessKernelArg(rewriter, op.getLoc(), adaptor.getDest());
    if (!kernelArg)
      return op.emitOpError(
          "expected store destination to be a kernel-arg ptr ABI cast");
    if (kernelArg->rank() != static_cast<unsigned>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected kernel-arg destination rank to match index rank");
    auto destPtrType = cast<PtrType>(kernelArg->ptr.getType());

    FailureOr<std::pair<Value, mlir::VectorType>> source =
        lowerStoreSource(op, adaptor, rewriter, *typeConverter);
    if (failed(source))
      return failure();
    auto [sourceValue, sourceType] = *source;
    SmallVector<int64_t> sourceShape(sourceType.getShape().begin(),
                                     sourceType.getShape().end());

    FailureOr<SmallVector<SliceAxis>> axes =
        resolveAccessAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                          *kernelArg, sourceShape);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != sourceType.getRank())
      return op.emitOpError(
          "store source rank must match slice subscript rank");

    FailureOr<Value> mask =
        lowerStoreMask(op, adaptor, rewriter, *typeConverter, sourceType);
    if (failed(mask))
      return failure();

    emitStoreLanes(rewriter, op.getLoc(), *kernelArg, destPtrType, sourceValue,
                   sourceType, *mask, *axes);
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

// `bare_tensor` view of workgroup ptr: pass source ptr through. Op
// survives as metadata anchor for `loadVectorFromPtrView`'s walk;
// DCE'd once every consumer has lowered.
static LogicalResult
lowerWorkgroupBufferView(HCBufferViewOp op, HCBufferViewOp::Adaptor adaptor,
                         ConversionPatternRewriter &rewriter, Type converted,
                         PtrType sourcePtrType) {
  if (sourcePtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  auto resultPtrType = dyn_cast_if_present<PtrType>(converted);
  if (!resultPtrType || resultPtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  if (resultPtrType.getElementType() != sourcePtrType.getElementType())
    return failure();
  rewriter.replaceOp(op, adaptor.getBuffer());
  return success();
}

// Permutation pulling scalar axes front (in order), slice axes
// after. `positions` collects the scalar `offset` values for extract.
static void
buildVectorViewPermutation(ArrayRef<SliceAxis> localAxes,
                           SmallVectorImpl<int64_t> &permutation,
                           SmallVectorImpl<OpFoldResult> &positions) {
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
}

// Reshape `result` -> `converted`. Both must be vectors.
static FailureOr<Value> reshapeToConvertedVector(OpBuilder &rewriter,
                                                 Location loc, Value result,
                                                 Type converted) {
  if (result.getType() == converted)
    return result;
  if (!isa<mlir::VectorType>(result.getType()) ||
      !isa<mlir::VectorType>(converted))
    return failure();
  return vector::ShapeCastOp::create(rewriter, loc, converted, result)
      .getResult();
}

// Vector-shape view: `vector.transpose` (scalar front) +
// `vector.extract` (drop scalar prefix) + shape cast.
static LogicalResult lowerVectorBufferView(HCBufferViewOp op,
                                           HCBufferViewOp::Adaptor adaptor,
                                           ConversionPatternRewriter &rewriter,
                                           Type converted) {
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
  buildVectorViewPermutation(localAxes, permutation, positions);

  Value source = adaptor.getBuffer();
  if (!llvm::equal(permutation, llvm::seq<int64_t>(0, localAxes.size())))
    source =
        vector::TransposeOp::create(rewriter, op.getLoc(), source, permutation)
            .getResult();

  Value result = source;
  if (!positions.empty())
    result = vector::ExtractOp::create(rewriter, op.getLoc(), source, positions)
                 .getResult();

  FailureOr<Value> reshaped =
      reshapeToConvertedVector(rewriter, op.getLoc(), result, converted);
  if (failed(reshaped))
    return failure();
  rewriter.replaceOp(op, *reshaped);
  return success();
}

struct ConvertBufferViewOp : public OpConversionPattern<HCBufferViewOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBufferViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (auto srcPtr = dyn_cast<PtrType>(adaptor.getBuffer().getType()))
      return lowerWorkgroupBufferView(op, adaptor, rewriter, converted, srcPtr);
    return lowerVectorBufferView(op, adaptor, rewriter, converted);
  }
};

struct ConvertDivOp : public OpConversionPattern<HCDivOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (converted.isIntOrIndex()) {
      rewriter.replaceOpWithNewOp<arith::DivUIOp>(
          op, converted, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    if (isa<FloatType>(converted)) {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(
          op, converted, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    return failure();
  }
};

struct ConvertNegOp : public OpConversionPattern<HCNegOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCNegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (converted.isIntOrIndex()) {
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
    if (isa<FloatType>(converted)) {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, converted,
                                                 adaptor.getValue());
      return success();
    }
    return failure();
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

// Resolve `!hc.buffer` operands to underlying `!hc.ptr<global>` via
// the kernel-arg UCC chain. Non-buffer untouched.
static void resolveBuffersInPlace(MutableArrayRef<Value> operands,
                                  bool &changed) {
  for (Value &v : operands) {
    if (!isa<BufferType>(v.getType()))
      continue;
    auto info = resolveKernelArg(v);
    if (!info)
      continue;
    v = info->ptr;
    changed = true;
  }
}

// LDS-backed `bare_tensor` ins -> workgroup ptr SSA the adaptor hands
// back (producer conversion already planted it).
static void swapInsLDSCarriersToPtrs(MutableArrayRef<Value> newIns,
                                     ValueRange adaptedIns, bool &changed) {
  for (auto [idx, v] : llvm::enumerate(newIns)) {
    if (!isa<BareTensorType>(v.getType()))
      continue;
    Value adapted = adaptedIns[idx];
    if (!isa<PtrType>(adapted.getType()))
      continue;
    newIns[idx] = adapted;
    changed = true;
  }
}

// `hc.generic` operand resolve: kernel-arg buffer carriers -> ptr<global>,
// LDS bare_tensor ins -> ptr<workgroup>. Body, bounds, offsets verbatim.
// Composed offsets keep ABI sym refs (`$STRIDE_0_<buf>`); ambient
// resolution happens at apply-lowering time.
struct AdaptGenericOp : public OpConversionPattern<HCGenericOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCGenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Two HC-to-HC retypes: kernel-arg buffer -> ptr<global> via UCC
    // walk; LDS bare_tensor ins -> ptr<workgroup> via adaptor. Outs
    // stays bare_tensor (swapping it would collapse the SSA result
    // the verifier matches against value-typed outs count). Iter
    // bounds take the index conversion to fold the trailing UCC chain.
    SmallVector<Value> newIns(op.getIns());
    SmallVector<Value> newOuts(op.getOuts());
    bool changed = false;
    resolveBuffersInPlace(newIns, changed);
    resolveBuffersInPlace(newOuts, changed);
    swapInsLDSCarriersToPtrs(newIns, adaptor.getIns(), changed);
    if (!changed)
      return failure();

    auto newOp = HCGenericOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), op.getIterSymsAttr(),
        adaptor.getIterBounds(), op.getIterKindsAttr(), ValueRange(newIns),
        ValueRange(newOuts), /*ambient_idxs=*/adaptor.getAmbientIdxs(),
        op.getAmbientIdxSymsAttr(), op.getInsOffsetsAttr(),
        op.getOutsOffsetsAttr());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// Clone `hc.yield`-terminated region into an `scf` region: map
// block args 1:1, clone non-terminator ops, emit `scf.yield` with
// cast-if-needed values. Used by `scf.for` and `scf.if`.
static LogicalResult cloneHCYieldRegion(Location loc, Region &srcRegion,
                                        Region &dstRegion,
                                        TypeRange resultTypes,
                                        ConversionPatternRewriter &rewriter) {
  Block &src = srcRegion.front();
  Block &dst = dstRegion.front();
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
    return failure();
  for (Operation &nested : src) {
    if (&nested == yield.getOperation())
      break;
    rewriter.clone(nested, mapping);
  }

  SmallVector<Value> yielded;
  for (auto [index, value] : llvm::enumerate(yield.getValues())) {
    Value mapped = mapping.lookupOrDefault(value);
    yielded.push_back(castIfNeeded(rewriter, loc, mapped, resultTypes[index]));
  }
  if (dstTerminator)
    rewriter.replaceOpWithNewOp<scf::YieldOp>(dstTerminator, yielded);
  else
    scf::YieldOp::create(rewriter, loc, yielded);
  return success();
}

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
    if (failed(cloneHCYieldRegion(op.getLoc(), op.getBody(), loop.getRegion(),
                                  loop.getResultTypes(), rewriter)))
      return op.emitOpError("body must end with `hc.yield`");
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
    if (failed(cloneHCYieldRegion(op.getLoc(), op.getThenRegion(),
                                  ifOp.getThenRegion(), results, rewriter)))
      return op.emitOpError("then region must end with `hc.yield`");
    if (!op.getElseRegion().empty() &&
        failed(cloneHCYieldRegion(op.getLoc(), op.getElseRegion(),
                                  ifOp.getElseRegion(), results, rewriter)))
      return op.emitOpError("else region must end with `hc.yield`");
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

static void populateLaunchBodyLoweringPatterns(TypeConverter &converter,
                                               MLIRContext *ctx,
                                               RewritePatternSet &patterns) {
  // `ConvertIdxApplyOp` / `ConvertPredApplyOp` still register here for
  // the first launch-body invocation (applies emitted by load_mask /
  // shaped-compute rewriters live outside `hc.generic` and need
  // immediate lowering). The post-`hc-lower-generic` applies are
  // handled by the standalone `hc-lower-apply` pass.
  patterns
      .add<ConvertIdxApplyOp, ConvertPredApplyOp, ConvertCastOp,
           ConvertBufferDimOp, ConvertLoadLikeOp<HCLoadOp>,
           ConvertLoadLikeOp<HCVLoadOp>, ConvertVecOp, ConvertSelectOp,
           ConvertStoreOp, ConvertBufferViewOp, ConvertForRangeOp, ConvertIfOp>(
          converter, ctx);

  patterns.add<AdaptRegionlessOp<HCTupleOp>, AdaptRegionlessOp<HCSliceExprOp>,
               AdaptRegionlessOp<HCGetItemOp>>(converter, ctx);
}

static bool regionsAreLegal(Operation *op, const TypeConverter &converter) {
  return llvm::all_of(op->getRegions(), [&](Region &region) {
    return converter.isLegal(&region);
  });
}

// `hc.generic` operand legal unless still a kernel-arg buffer
// awaiting `AdaptGenericOp` resolution.
static bool isHCGenericOperandLegal(Value v) {
  return !isa<BufferType>(v.getType()) || !resolveKernelArg(v);
}

static bool isHCGenericLegalAtLaunchBoundary(HCGenericOp op) {
  if (!llvm::all_of(op.getIns(), isHCGenericOperandLegal))
    return false;
  if (!llvm::all_of(op.getOuts(), isHCGenericOperandLegal))
    return false;
  // `bare_tensor` ins must lower via `AdaptGenericOp`. Outs skipped
  // -- not swapped (see `AdaptGenericOp`).
  for (Value v : op.getIns()) {
    if (isa<BareTensorType>(v.getType()))
      return false;
  }
  return true;
}

// Legal iff signature matches the converter projection. Missing
// signature -> legal (verifier flags separately).
static bool isHCIntrinsicSignatureLegal(HCIntrinsicOp op,
                                        const TypeConverter &converter) {
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
}

// Legal iff every operand/result type equals its boundary
// projection. Unprojectable -> illegal.
static bool isHCCallIntrinsicLegal(HCCallIntrinsicOp op,
                                   const TypeConverter &converter) {
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
}

// HC ptr-family + bodily generic ops survive for `hc-lower-to-llvm`
// / `hc-lower-generic` downstream.
static void registerLaunchBodyHCLegality(ConversionTarget &target) {
  // `hc.generic` rides through launch-body untouched -- operand
  // reconciliation is `hc-reconcile-generic-operands`' job, runs in
  // a follow-up pass before barriers / `hc-lower-generic`. `hc.yield`
  // legal only inside `hc.generic` (root-level `hc.for_range` lowers
  // to `scf.for` here).
  target.addLegalOp<HCUndefValueOp, UnrealizedConversionCastOp, HCAllocOp,
                    HCPtrOffsetOp, HCPtrLoadOp, HCPtrStoreOp, HCPtrLoadPredOp,
                    HCPtrStorePredOp, HCYieldPredicatedOp, HCGenericOp>();
  target.addDynamicallyLegalOp<HCYieldOp>([](HCYieldOp op) {
    return isa_and_nonnull<HCGenericOp>(op->getParentOp());
  });
  // `hc.load_mask` already lowered upstream by
  // `hc-load-store-to-generic`; surviving op is a producer bug.
  // Scalar / control-flow ops (const, int/float arith, cmp, cast,
  // for_range, if) are pre-lowered by `hc-lower-launch-scalar-ops`;
  // any survivor here surfaces as a missing-pattern diagnostic on the
  // residual op.
  target.addIllegalOp<HCCastOp, HCBufferDimOp, HCLoadOp, HCVLoadOp,
                      HCLoadMaskOp, HCBufferViewOp, HCVecOp, HCSelectOp,
                      HCStoreOp, HCForRangeOp, HCIfOp>();
}

// `hc.idx_apply` / `hc.pred_apply` / `hc.predicate` inside `hc.generic`
// body deferred to the post-unroll pass invocation (hc-lower-generic
// materialises the producer, hc-fold-predicates folds the predicate);
// outside, illegal here.
static void registerLaunchBodyApplyLegality(ConversionTarget &target) {
  auto applyLegalInsideGeneric = [](Operation *op) {
    return op->getParentOfType<HCGenericOp>() != nullptr;
  };
  target.addDynamicallyLegalOp<HCIdxApplyOp>(applyLegalInsideGeneric);
  target.addDynamicallyLegalOp<HCPredApplyOp>(applyLegalInsideGeneric);
  target.addDynamicallyLegalOp<HCPredicateOp>(applyLegalInsideGeneric);
}

static ConversionTarget
makeLaunchBodyLoweringTarget(MLIRContext *ctx, const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target
      .addLegalDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                       scf::SCFDialect, vector::VectorDialect>();
  registerLaunchBodyHCLegality(target);
  registerLaunchBodyApplyLegality(target);
  target.addDynamicallyLegalOp<HCIntrinsicOp>([&](HCIntrinsicOp op) {
    return isHCIntrinsicSignatureLegal(op, converter);
  });
  target.addDynamicallyLegalOp<HCCallIntrinsicOp>([&](HCCallIntrinsicOp op) {
    return isHCCallIntrinsicLegal(op, converter);
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
    // Bare-carrier + semantic-shape preflights moved to
    // `hc-verify-bare-carriers`. Runs once before the first
    // launch-body invocation instead of per invocation.
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

namespace mlir::hc {
std::unique_ptr<TypeConverter> makeLaunchBodyTypeConverter() {
  return std::make_unique<HCLaunchBodyTypeConverter>();
}

// Populator exposed for `hc-lower-apply` so the post-`hc-lower-generic`
// pass reuses the same `ExprLowerer` plumbing without duplicating the
// patterns. See `HCLowerApplyPass.cpp`.
void populateLaunchBodyApplyPatterns(TypeConverter &converter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<ConvertIdxApplyOp, ConvertPredApplyOp>(converter, ctx);
}

// `hc.idx_apply` / `hc.pred_apply` / `hc.predicate` are dyn-legal
// inside an `hc.generic` body (lowered when the body is cloned out
// by `hc-lower-generic`); illegal everywhere else.
void registerLaunchBodyApplyDynamicLegality(ConversionTarget &target) {
  registerLaunchBodyApplyLegality(target);
}

void populateIntrinsicBridgingPatterns(TypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<ConvertIntrinsicSignatureOp, ConvertCallIntrinsicOp>(converter,
                                                                    ctx);
}

void registerIntrinsicBridgingLegality(const TypeConverter &converter,
                                       ConversionTarget &target) {
  target.addDynamicallyLegalOp<HCIntrinsicOp>([&](HCIntrinsicOp op) {
    return isHCIntrinsicSignatureLegal(op, converter);
  });
  target.addDynamicallyLegalOp<HCCallIntrinsicOp>([&](HCCallIntrinsicOp op) {
    return isHCCallIntrinsicLegal(op, converter);
  });
}

void populateGenericReconciliationPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<AdaptGenericOp>(converter, ctx);
}

void registerGenericReconciliationLegality(ConversionTarget &target) {
  target.addDynamicallyLegalOp<HCGenericOp>(isHCGenericLegalAtLaunchBoundary);
}

void populateLaunchScalarOpsPatterns(TypeConverter &converter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  // Pure arith: const, int/float arith, cmp, div, neg, mod. `Cast`
  // is type-bridging over the converter (shaped flavours included)
  // and `ForRange` / `If` clone bodies whose block args carry shaped
  // types -- both leak cross-pass UCCs when split, so they stay in
  // launch-body.
  patterns.add<ConvertConstOp, ConvertIntBinaryOp<HCAddOp, arith::AddIOp>,
               ConvertIntBinaryOp<HCSubOp, arith::SubIOp>,
               ConvertIntBinaryOp<HCMulOp, arith::MulIOp>,
               ConvertIntBinaryOp<HCAndOp, arith::AndIOp>,
               ConvertIntBinaryOp<HCOrOp, arith::OrIOp>,
               ConvertFloatBinaryOp<HCAddOp, arith::AddFOp>,
               ConvertFloatBinaryOp<HCSubOp, arith::SubFOp>,
               ConvertFloatBinaryOp<HCMulOp, arith::MulFOp>, ConvertDivOp,
               ConvertIntBinaryOp<HCModOp, arith::RemUIOp>, ConvertNegOp,
               ConvertCmpOp<HCCmpLtOp>, ConvertCmpOp<HCCmpLeOp>,
               ConvertCmpOp<HCCmpGtOp>, ConvertCmpOp<HCCmpGeOp>,
               ConvertCmpOp<HCCmpEqOp>, ConvertCmpOp<HCCmpNeOp>>(converter,
                                                                 ctx);
}

void registerLaunchScalarOpsLegality(ConversionTarget &target) {
  target.addIllegalOp<HCConstOp, HCAddOp, HCSubOp, HCMulOp, HCDivOp, HCModOp,
                      HCAndOp, HCOrOp, HCNegOp, HCCmpLtOp, HCCmpLeOp, HCCmpGtOp,
                      HCCmpGeOp, HCCmpEqOp, HCCmpNeOp>();
}

void populateLaunchShapedConstantsPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<ConvertFullMaskOp, ConvertNullaryShapedConstantOp<HCVZerosOp, 0>,
               ConvertNullaryShapedConstantOp<HCVOnesOp, 1>,
               ConvertNullaryShapedConstantOp<HCZerosOp, 0>,
               ConvertNullaryShapedConstantOp<HCOnesOp, 1>,
               ConvertFillShapedConstantOp<HCVFullOp>,
               ConvertFillShapedConstantOp<HCFullOp>, ConvertEmptyOp>(converter,
                                                                      ctx);
}

void registerLaunchShapedConstantsLegality(ConversionTarget &target) {
  target.addIllegalOp<HCVZerosOp, HCVOnesOp, HCVFullOp, HCFullMaskOp, HCZerosOp,
                      HCOnesOp, HCFullOp, HCEmptyOp>();
}
} // namespace mlir::hc
