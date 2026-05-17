// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-kernels-to-gpu-launch`: first wrapper slice of HC →
// upstream GPU lowering.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringMap.h"

#include <limits>

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERKERNELSTOGPULAUNCH
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static ixs_node *rawNode(ExprAttr expr) {
  return const_cast<ixs_node *>(expr.getNode());
}

static bool isOneNode(ixs_node *node) {
  return ixs_node_tag(node) == IXS_INT && ixs_node_int_val(node) == 1;
}

static std::optional<StringRef> exactSymbolName(Attribute attr) {
  auto expr = dyn_cast<ExprAttr>(attr);
  if (!expr)
    return std::nullopt;
  ixs_node *node = rawNode(expr);
  if (ixs_node_tag(node) != IXS_SYM)
    return std::nullopt;
  return StringRef(ixs_node_sym_name(node));
}

static Type convertScalarABIType(Type type) {
  if (isa<IdxType>(type))
    return IndexType::get(type.getContext());
  if (isa<PredType>(type))
    return IntegerType::get(type.getContext(), 1);
  if (type.isIntOrIndexOrFloat())
    return type;
  return {};
}

static Type convertABIType(Type type);

// LLVM is opaque-pointers-only; element type is HC-side convenience.
// `!hc.undef` drops to null.
static PtrType convertBufferABIType(BufferType type) {
  Type elementType = type.getElementType();
  if (isa<UndefType>(elementType))
    elementType = nullptr;
  else if (Type converted = convertScalarABIType(elementType))
    elementType = converted;
  return PtrType::get(type.getContext(), AddrSpace::Global, elementType);
}

static Type convertABIType(Type type) {
  if (auto buffer = dyn_cast<BufferType>(type))
    return convertBufferABIType(buffer);
  if (Type scalar = convertScalarABIType(type))
    return scalar;
  if (auto tuple = dyn_cast<TupleType>(type)) {
    SmallVector<Type> elements;
    elements.reserve(tuple.size());
    for (Type element : tuple.getTypes()) {
      Type converted = convertABIType(element);
      if (!converted)
        return {};
      elements.push_back(converted);
    }
    return TupleType::get(type.getContext(), elements);
  }
  return {};
}

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

class ExprLowerer {
public:
  ExprLowerer(OpBuilder &builder, Location loc, const BoundValues &boundValues)
      : builder(builder), loc(loc), boundValues(boundValues) {}

  FailureOr<Value> lower(Attribute attr) {
    auto expr = dyn_cast<ExprAttr>(attr);
    if (!expr)
      return failure();
    return lowerNode(rawNode(expr));
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
      FailureOr<Value> coeff = lowerNode(ixs_node_add_term_coeff(node, index));
      FailureOr<Value> term = lowerNode(ixs_node_add_term(node, index));
      if (failed(coeff) || failed(term))
        return failure();
      Value scaled = *term;
      if (!isOneNode(ixs_node_add_term_coeff(node, index)))
        scaled = arith::MulIOp::create(builder, loc, *coeff, *term);
      *result = arith::AddIOp::create(builder, loc, *result, scaled);
    }
    return *result;
  }

  FailureOr<Value> lowerMul(ixs_node *node) {
    FailureOr<Value> result = lowerNode(ixs_node_mul_coeff(node));
    if (failed(result))
      return failure();

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
        *result = arith::MulIOp::create(builder, loc, *result, *factor);
    }
    return *result;
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

  OpBuilder &builder;
  Location loc;
  const BoundValues &boundValues;
};

static void bindScalarSymbol(Type originalType, Value hostArg,
                             BoundValues &boundValues) {
  auto idx = dyn_cast<IdxType>(originalType);
  if (!idx || !idx.getExpr())
    return;
  ixs_node *node = rawNode(idx.getExpr());
  if (ixs_node_tag(node) != IXS_SYM)
    return;
  boundValues.bind(StringRef(ixs_node_sym_name(node)), hostArg);
}

// Idempotent extern "C" helper decl. `llvm.emit_c_interface` makes
// convert-func-to-llvm emit the `_mlir_ciface_<name>` decl + a private
// `<name>` forwarder. No sret — call sites use the unmangled `@<name>`.
static func::FuncOp ensureRuntimeHelper(ModuleOp module, StringRef name,
                                        ArrayRef<Type> inputs,
                                        ArrayRef<Type> results) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = FunctionType::get(ctx, inputs, results);
  auto func = func::FuncOp::create(builder, module.getLoc(), name, fnType);
  func.setVisibility(SymbolTable::Visibility::Private);
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  return func;
}

static void ensureRuntimeHelpers(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  Type ptr = LLVM::LLVMPointerType::get(ctx);
  Type i32 = IntegerType::get(ctx, 32);
  Type i64 = IntegerType::get(ctx, 64);
  Type f64 = Float64Type::get(ctx);
  // `hc_get_ptr` returns the raw `data_ptr()`; dims/strides via the
  // scalar helpers below.
  ensureRuntimeHelper(module, "hc_get_ptr", {ptr}, {ptr});
  ensureRuntimeHelper(module, "hc_get_dim", {ptr, i32}, {i64});
  ensureRuntimeHelper(module, "hc_get_stride", {ptr, i32}, {i64});
  ensureRuntimeHelper(module, "hc_get_int64", {ptr}, {i64});
  ensureRuntimeHelper(module, "hc_get_float64", {ptr}, {f64});
}

// `_mlir_ciface_hc_get_*(pyobj, axis)` → i64 element units, cast to
// `index`. Shared so dim and stride don't drift.
static Value callIndexAccessor(OpBuilder &builder, Location loc,
                               ModuleOp module, StringRef name, Value pyArg,
                               unsigned axisIndex) {
  auto fn = module.lookupSymbol<func::FuncOp>(name);
  Type i32 = IntegerType::get(builder.getContext(), 32);
  Value axis = arith::ConstantIntOp::create(builder, loc, i32,
                                            static_cast<int64_t>(axisIndex));
  auto call = func::CallOp::create(builder, loc, fn, ValueRange{pyArg, axis});
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    call.getResult(0))
      .getResult();
}

static Value callGetDim(OpBuilder &builder, Location loc, ModuleOp module,
                        Value pyArg, unsigned dimIndex) {
  return callIndexAccessor(builder, loc, module, "hc_get_dim", pyArg, dimIndex);
}

static Value callGetStride(OpBuilder &builder, Location loc, ModuleOp module,
                           Value pyArg, unsigned dimIndex) {
  return callIndexAccessor(builder, loc, module, "hc_get_stride", pyArg,
                           dimIndex);
}

// Kernel-boundary expansion of `!hc.buffer<T, [dims]>`:
//   1. `!hc.ptr<global, T?>` from `data_ptr()`.
//   2. per-axis dim (`_mlir_ciface_hc_get_dim` or ExprLowerer).
//   3. per-axis stride (`_mlir_ciface_hc_get_stride`) — without this
//      transposed / sliced inputs silently fall back to contiguous.
// Stitched through a 1-to-N UCC back to the original buffer type so
// the kernel body is unchanged; launch-body unwinds the UCC at lower-
// to `hc.ptr_offset` + load/store.
struct BufferABIPack {
  Value ptr;
  SmallVector<Value> dims;
  SmallVector<Value> strides;
};

// Flat (ptr, dims..., strides...) at host scope. The matching UCC
// is built inside the launch region (so gpu-kernel-outlining captures
// the llvm-translatable scalars, not the buffer type).
static BufferABIPack buildBufferPack(OpBuilder &builder, Location loc,
                                     ModuleOp module, Value pyArg,
                                     PtrType ptrType,
                                     ArrayRef<Value> shapeValues) {
  BufferABIPack pack;

  auto getPtr = module.lookupSymbol<func::FuncOp>("hc_get_ptr");
  auto rawCall = func::CallOp::create(builder, loc, getPtr, ValueRange{pyArg});
  Value rawPtr = rawCall.getResult(0);
  pack.ptr = UnrealizedConversionCastOp::create(builder, loc, ptrType, rawPtr)
                 .getResult(0);

  pack.dims.assign(shapeValues.begin(), shapeValues.end());
  pack.strides.reserve(shapeValues.size());
  for (unsigned axis = 0; axis < shapeValues.size(); ++axis)
    pack.strides.push_back(callGetStride(builder, loc, module, pyArg, axis));
  return pack;
}

static Value buildBufferUCC(OpBuilder &builder, Location loc, BufferType target,
                            const BufferABIPack &pack) {
  SmallVector<Value> inputs;
  inputs.reserve(1 + pack.dims.size() + pack.strides.size());
  inputs.push_back(pack.ptr);
  inputs.append(pack.dims.begin(), pack.dims.end());
  inputs.append(pack.strides.begin(), pack.strides.end());
  return UnrealizedConversionCastOp::create(builder, loc, target, inputs)
      .getResult(0);
}

// PyObject → scalar of `targetType`. Wire is i64/f64; narrow in-IR.
static FailureOr<Value> buildScalar(OpBuilder &builder, Location loc,
                                    ModuleOp module, Value pyArg,
                                    Type targetType) {
  if (isa<FloatType>(targetType)) {
    auto fn = module.lookupSymbol<func::FuncOp>("hc_get_float64");
    auto call = func::CallOp::create(builder, loc, fn, ValueRange{pyArg});
    Value f64Value = call.getResult(0);
    if (isa<Float64Type>(targetType))
      return f64Value;
    return arith::TruncFOp::create(builder, loc, targetType, f64Value)
        .getResult();
  }
  if (isa<IndexType>(targetType) || isa<IntegerType>(targetType)) {
    auto fn = module.lookupSymbol<func::FuncOp>("hc_get_int64");
    auto call = func::CallOp::create(builder, loc, fn, ValueRange{pyArg});
    Value i64Value = call.getResult(0);
    if (isa<IndexType>(targetType))
      return arith::IndexCastOp::create(builder, loc, targetType, i64Value)
          .getResult();
    auto intType = cast<IntegerType>(targetType);
    if (intType.getWidth() == 64)
      return i64Value;
    if (intType.getWidth() < 64)
      return arith::TruncIOp::create(builder, loc, targetType, i64Value)
          .getResult();
    return arith::ExtSIOp::create(builder, loc, targetType, i64Value)
        .getResult();
  }
  return failure();
}

static LogicalResult lowerShapeDim(ExprLowerer &lowerer, ShapeAttr shape,
                                   unsigned dim, Value &result) {
  if (!shape || dim >= shape.getDims().size()) {
    result = lowerer.constant(1);
    return success();
  }
  FailureOr<Value> lowered = lowerer.lower(shape.getDims()[dim]);
  if (failed(lowered))
    return failure();
  result = *lowered;
  return success();
}

static LogicalResult lowerLaunchGeometry(OpBuilder &builder, Location loc,
                                         HCKernelOp kernel,
                                         const BoundValues &boundValues,
                                         SmallVectorImpl<Value> &grid,
                                         SmallVectorImpl<Value> &block) {
  std::optional<ShapeAttr> workShape = kernel.getWorkShape();
  std::optional<ShapeAttr> groupShape = kernel.getGroupShape();
  if (!workShape)
    return kernel.emitOpError("requires work_shape for GPU launch lowering");

  ExprLowerer lowerer(builder, loc, boundValues);
  for (unsigned dim = 0; dim != 3; ++dim) {
    Value workDim;
    Value blockDim;
    if (failed(lowerShapeDim(lowerer, *workShape, dim, workDim)))
      return kernel.emitOpError("failed to lower launch geometry dimension #")
             << dim;
    if (groupShape) {
      if (failed(lowerShapeDim(lowerer, *groupShape, dim, blockDim)))
        return kernel.emitOpError("failed to lower launch geometry dimension #")
               << dim;
    } else {
      blockDim = lowerer.constant(1);
    }
    grid.push_back(arith::CeilDivUIOp::create(builder, loc, workDim, blockDim)
                       .getResult());
    block.push_back(blockDim);
  }
  return success();
}

// Trailing `hc.return` or null. Non-empty return is a hard error —
// `gpu.launch` terminator has no operands.
static FailureOr<Operation *> findTrailingReturn(Block &kernelBlock) {
  if (kernelBlock.empty())
    return nullptr;
  auto returnOp = dyn_cast<HCReturnOp>(&kernelBlock.back());
  if (!returnOp)
    return nullptr;
  if (!returnOp.getValues().empty())
    return returnOp.emitOpError(
        "cannot lower kernel return values to gpu.launch");
  return returnOp.getOperation();
}

// Returned op is the body-clone insertion point.
static Operation *ensureLaunchTerminator(OpBuilder &builder,
                                         gpu::LaunchOp launch, Location loc) {
  Block &launchBlock = launch.getBody().front();
  Operation *term = launchBlock.empty() ? nullptr : &launchBlock.back();
  if (!term || !isa<gpu::TerminatorOp>(term)) {
    builder.setInsertionPointToEnd(&launchBlock);
    term = gpu::TerminatorOp::create(builder, loc);
  }
  return term;
}

// Group: undef-fed UCC (no runtime concept). Buffer: UCC pack.
// Scalar: passthrough or UCC-bridged on ABI type change.
static Value materializeKernelArgReplacement(
    OpBuilder &builder, MLIRContext *ctx, BlockArgument arg, unsigned index,
    ArrayRef<Value> kernelABIArgs,
    ArrayRef<std::optional<BufferABIPack>> bufferPacks) {
  if (isa<GroupType>(arg.getType())) {
    Value undef =
        HCUndefValueOp::create(builder, arg.getLoc(), UndefType::get(ctx))
            .getResult();
    return UnrealizedConversionCastOp::create(builder, arg.getLoc(),
                                              arg.getType(), undef)
        .getResult(0);
  }
  if (auto buffer = dyn_cast<BufferType>(arg.getType()))
    return buildBufferUCC(builder, arg.getLoc(), buffer, *bufferPacks[index]);
  Value abiValue = kernelABIArgs[index];
  if (abiValue.getType() == arg.getType())
    return abiValue;
  return UnrealizedConversionCastOp::create(builder, arg.getLoc(),
                                            arg.getType(), abiValue)
      .getResult(0);
}

static LogicalResult
cloneKernelBodyIntoLaunch(OpBuilder &builder, HCKernelOp kernel,
                          gpu::LaunchOp launch, ArrayRef<Value> kernelABIArgs,
                          ArrayRef<std::optional<BufferABIPack>> bufferPacks) {
  Block &kernelBlock = kernel.getBody().front();
  FailureOr<Operation *> returnLikeOr = findTrailingReturn(kernelBlock);
  if (failed(returnLikeOr))
    return failure();
  Operation *returnLike = *returnLikeOr;

  Operation *launchTerminator =
      ensureLaunchTerminator(builder, launch, kernel.getLoc());
  builder.setInsertionPoint(launchTerminator);

  IRMapping mapping;
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments()))
    mapping.map(arg, materializeKernelArgReplacement(
                         builder, kernel.getContext(), arg, index,
                         kernelABIArgs, bufferPacks));

  for (Operation &op : kernelBlock) {
    if (&op == returnLike)
      break;
    builder.clone(op, mapping);
  }
  return success();
}

// `aux_of`: parent buffer arg index. `axis`: axis in parent's pre-
// flatten shape. `kind`: "dim" vs "stride" accessor.
struct FlattenAuxInfo {
  unsigned auxOf;
  unsigned axis;
  StringRef kind;
};

static std::optional<FlattenAuxInfo> lookupFlattenAux(DictionaryAttr meta,
                                                      unsigned argIndex) {
  if (!meta)
    return std::nullopt;
  SmallString<8> key;
  Twine(argIndex).toVector(key);
  auto dict = dyn_cast_or_null<DictionaryAttr>(meta.get(key));
  if (!dict)
    return std::nullopt;
  auto auxOfAttr = dyn_cast_or_null<IntegerAttr>(dict.get("aux_of"));
  auto axisAttr = dyn_cast_or_null<IntegerAttr>(dict.get("axis"));
  auto kindAttr = dyn_cast_or_null<StringAttr>(dict.get("kind"));
  if (!auxOfAttr || !axisAttr || !kindAttr)
    return std::nullopt;
  return FlattenAuxInfo{static_cast<unsigned>(auxOfAttr.getInt()),
                        static_cast<unsigned>(axisAttr.getInt()),
                        kindAttr.getValue()};
}

// Group + flatten-aux slots aren't user-visible (sentinel host slot).
// Wrapper sig becomes `(stream, arg0..argN)` with N = userArgCount.
static LogicalResult
convertKernelABISignature(HCKernelOp kernel, DictionaryAttr auxMeta,
                          SmallVectorImpl<Type> &kernelABITypes,
                          SmallVectorImpl<unsigned> &hostArgFor,
                          unsigned &userArgCount) {
  Block &kernelBlock = kernel.getBody().front();
  userArgCount = 0;
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    if (isa<GroupType>(arg.getType()))
      continue;
    if (lookupFlattenAux(auxMeta, index))
      continue;
    Type converted = convertABIType(arg.getType());
    if (!converted)
      return kernel.emitOpError("unsupported kernel ABI argument type ")
             << arg.getType();
    kernelABITypes[index] = converted;
    hostArgFor[index] = 1 + userArgCount++;
  }
  return success();
}

// Pass 1: scalar idx args. Explicit `M: idx` wins over buffer-dim
// derivation of `M`.
static LogicalResult materializeScalarABIArgs(
    OpBuilder &builder, Location loc, ModuleOp module, HCKernelOp kernel,
    Block *entry, ArrayRef<Type> kernelABITypes, ArrayRef<unsigned> hostArgFor,
    DictionaryAttr auxMeta, MutableArrayRef<Value> kernelABIArgs,
    BoundValues &boundValues) {
  Block &kernelBlock = kernel.getBody().front();
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    if (isa<GroupType>(arg.getType()))
      continue;
    if (isa<BufferType>(arg.getType()))
      continue;
    if (lookupFlattenAux(auxMeta, index))
      continue;
    Value pyArg = entry->getArgument(hostArgFor[index]);
    FailureOr<Value> scalar =
        buildScalar(builder, loc, module, pyArg, kernelABITypes[index]);
    if (failed(scalar))
      return kernel.emitOpError("unsupported scalar ABI argument type ")
             << arg.getType();
    kernelABIArgs[index] = *scalar;
    bindScalarSymbol(arg.getType(), *scalar, boundValues);
  }
  return success();
}

// Pass 2: flatten-aux `!hc.idx<sym>` from parent buffer's `hc_get_dim`
// / `hc_get_stride`. Aux sym binds like any scalar idx so pre- and
// post-flatten launches share `boundValues`.
static LogicalResult materializeFlattenAuxArgs(
    OpBuilder &builder, Location loc, ModuleOp module, HCKernelOp kernel,
    Block *entry, ArrayRef<unsigned> hostArgFor, DictionaryAttr auxMeta,
    MutableArrayRef<Value> kernelABIArgs, BoundValues &boundValues) {
  Block &kernelBlock = kernel.getBody().front();
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    std::optional<FlattenAuxInfo> info = lookupFlattenAux(auxMeta, index);
    if (!info)
      continue;
    if (info->auxOf >= kernelBlock.getNumArguments())
      return kernel.emitOpError("flatten aux arg #")
             << index << " references out-of-range parent arg #" << info->auxOf;
    Value parentPyArg = entry->getArgument(hostArgFor[info->auxOf]);
    Value value =
        info->kind == "stride"
            ? callGetStride(builder, loc, module, parentPyArg, info->axis)
            : callGetDim(builder, loc, module, parentPyArg, info->axis);
    kernelABIArgs[index] = value;
    bindScalarSymbol(arg.getType(), value, boundValues);
  }
  return success();
}

// Returns true on aux hit (post-flatten layout). Missing dim slots
// stay null for caller to probe via `hc_get_dim`.
static bool collectFlattenAuxForBuffer(HCKernelOp kernel, unsigned bufferIdx,
                                       DictionaryAttr auxMeta,
                                       ArrayRef<Value> kernelABIArgs,
                                       SmallVectorImpl<Value> &shapeValues,
                                       SmallVectorImpl<Value> &strideValues,
                                       LogicalResult &status) {
  Block &kernelBlock = kernel.getBody().front();
  bool postFlatten = false;
  status = success();
  for (unsigned j = 0; j != kernelBlock.getNumArguments(); ++j) {
    std::optional<FlattenAuxInfo> auxInfo = lookupFlattenAux(auxMeta, j);
    if (!auxInfo || auxInfo->auxOf != bufferIdx)
      continue;
    postFlatten = true;
    Value auxValue = kernelABIArgs[j];
    if (!auxValue) {
      status = kernel.emitOpError("flatten aux arg #")
               << j << " unresolved before buffer #" << bufferIdx;
      return postFlatten;
    }
    auto &slots = auxInfo->kind == "stride" ? strideValues : shapeValues;
    if (auxInfo->axis >= slots.size())
      slots.resize(auxInfo->axis + 1);
    slots[auxInfo->axis] = auxValue;
  }
  return postFlatten;
}

// Post-flatten contiguous view: `(ptr, total_elements, 1)`. Per-axis
// strides already rode in via aux idx slots.
static BufferABIPack
buildPostFlattenBufferPack(OpBuilder &builder, Location loc, ModuleOp module,
                           Value pyArg, PtrType ptrType,
                           SmallVectorImpl<Value> &shapeValues,
                           ArrayRef<Value> strideValues) {
  // Constant-dim axes carry no aux — probe at runtime.
  unsigned rank = std::max(shapeValues.size(), strideValues.size());
  shapeValues.resize(rank);
  for (auto [axis, slot] : llvm::enumerate(shapeValues))
    if (!slot)
      slot = callGetDim(builder, loc, module, pyArg, axis);

  Value total;
  for (Value d : shapeValues)
    total =
        total ? arith::MulIOp::create(builder, loc, total, d).getResult() : d;
  Value one = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  if (!total)
    total = one;

  auto getPtr = module.lookupSymbol<func::FuncOp>("hc_get_ptr");
  auto rawCall = func::CallOp::create(builder, loc, getPtr, ValueRange{pyArg});
  Value bridgedPtr = UnrealizedConversionCastOp::create(builder, loc, ptrType,
                                                        rawCall.getResult(0))
                         .getResult(0);
  BufferABIPack pack;
  pack.ptr = bridgedPtr;
  pack.dims.push_back(total);
  pack.strides.push_back(one);
  return pack;
}

// Harvest first-occurrence shape syms (one `_get_dim` per name), then
// lower the full shape attr via `ExprLowerer`.
static LogicalResult materializePreFlattenBufferPack(
    OpBuilder &builder, Location loc, ModuleOp module, HCKernelOp kernel,
    unsigned bufferIdx, BufferType buffer, Value pyArg, PtrType ptrType,
    BoundValues &boundValues, std::optional<BufferABIPack> &outPack) {
  for (auto [dimIndex, dimAttr] :
       llvm::enumerate(buffer.getShape().getDims())) {
    std::optional<StringRef> symbol = exactSymbolName(dimAttr);
    if (!symbol)
      continue;
    if (boundValues.lookup(*symbol))
      continue;
    Value dimValue = callGetDim(builder, loc, module, pyArg, dimIndex);
    boundValues.bind(*symbol, dimValue);
  }

  ExprLowerer lowerer(builder, loc, boundValues);
  SmallVector<Value> shapeValues;
  shapeValues.reserve(buffer.getShape().getDims().size());
  for (Attribute dimAttr : buffer.getShape().getDims()) {
    FailureOr<Value> dimValue = lowerer.lower(dimAttr);
    if (failed(dimValue))
      return kernel.emitOpError("failed to lower buffer shape dim for arg #")
             << bufferIdx;
    shapeValues.push_back(*dimValue);
  }
  outPack = buildBufferPack(builder, loc, module, pyArg, ptrType, shapeValues);
  return success();
}

// Pass 3: buffer args, dispatched on aux-meta presence.
static LogicalResult materializeBufferABIArgs(
    OpBuilder &builder, Location loc, ModuleOp module, HCKernelOp kernel,
    Block *entry, ArrayRef<Type> kernelABITypes, ArrayRef<unsigned> hostArgFor,
    DictionaryAttr auxMeta, ArrayRef<Value> kernelABIArgs,
    BoundValues &boundValues,
    MutableArrayRef<std::optional<BufferABIPack>> bufferPacks) {
  Block &kernelBlock = kernel.getBody().front();
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    auto buffer = dyn_cast<BufferType>(arg.getType());
    if (!buffer)
      continue;
    Value pyArg = entry->getArgument(hostArgFor[index]);
    auto ptrType = cast<PtrType>(kernelABITypes[index]);

    SmallVector<Value> shapeValues;
    SmallVector<Value> strideValues;
    LogicalResult auxStatus = success();
    bool postFlatten =
        collectFlattenAuxForBuffer(kernel, index, auxMeta, kernelABIArgs,
                                   shapeValues, strideValues, auxStatus);
    if (failed(auxStatus))
      return failure();

    if (postFlatten) {
      bufferPacks[index] = buildPostFlattenBufferPack(
          builder, loc, module, pyArg, ptrType, shapeValues, strideValues);
      continue;
    }

    if (failed(materializePreFlattenBufferPack(
            builder, loc, module, kernel, index, buffer, pyArg, ptrType,
            boundValues, bufferPacks[index])))
      return failure();
  }
  return success();
}

static LogicalResult lowerKernel(HCKernelOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  Location loc = kernel.getLoc();
  Block &kernelBlock = kernel.getBody().front();
  ModuleOp module = kernel->getParentOfType<ModuleOp>();
  if (!module)
    return kernel.emitOpError("must be nested in a module");

  // Post-flatten kernels carry `hc.flatten_aux_args` pinning each aux
  // slot to (parent buffer arg, axis, accessor kind). Absent pre-flatten.
  DictionaryAttr auxMeta =
      kernel->getAttrOfType<DictionaryAttr>("hc.flatten_aux_args");

  // `hostArgFor[i]` indexes the host wrapper's `PyObject *` slot
  // (sentinel for group / aux). +1 offset reserves slot 0 for stream.
  SmallVector<Type> kernelABITypes(kernelBlock.getNumArguments());
  SmallVector<unsigned> hostArgFor(kernelBlock.getNumArguments(),
                                   std::numeric_limits<unsigned>::max());
  unsigned userArgCount = 0;
  if (failed(convertKernelABISignature(kernel, auxMeta, kernelABITypes,
                                       hostArgFor, userArgCount)))
    return failure();

  ensureRuntimeHelpers(module);

  // Host sig: `(stream: !llvm.ptr, arg0: !llvm.ptr, ...)`. Stream
  // threads to `hc_rt_{load,launch}_kernel`; null = HIP default.
  OpBuilder builder(kernel);
  Type ptrType = LLVM::LLVMPointerType::get(ctx);
  SmallVector<Type> hostInputTypes(1 + userArgCount, ptrType);
  auto fnType = FunctionType::get(ctx, hostInputTypes, {});
  auto hostFunc =
      func::FuncOp::create(builder, loc, kernel.getSymName(), fnType);
  Block *entry = hostFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // Three-pass arg materialization: scalar idx (authoritative for any
  // sym they bind), aux idx, then buffers.
  SmallVector<Value> kernelABIArgs(kernelBlock.getNumArguments());
  SmallVector<std::optional<BufferABIPack>> bufferPacks(
      kernelBlock.getNumArguments());
  BoundValues boundValues;
  if (failed(materializeScalarABIArgs(builder, loc, module, kernel, entry,
                                      kernelABITypes, hostArgFor, auxMeta,
                                      kernelABIArgs, boundValues)))
    return failure();
  if (failed(materializeFlattenAuxArgs(builder, loc, module, kernel, entry,
                                       hostArgFor, auxMeta, kernelABIArgs,
                                       boundValues)))
    return failure();
  if (failed(materializeBufferABIArgs(builder, loc, module, kernel, entry,
                                      kernelABITypes, hostArgFor, auxMeta,
                                      kernelABIArgs, boundValues, bufferPacks)))
    return failure();

  SmallVector<Value> grid;
  SmallVector<Value> block;
  if (failed(
          lowerLaunchGeometry(builder, loc, kernel, boundValues, grid, block)))
    return failure();

  auto launch = gpu::LaunchOp::create(builder, loc, grid[0], grid[1], grid[2],
                                      block[0], block[1], block[2]);
  if (failed(cloneKernelBodyIntoLaunch(builder, kernel, launch, kernelABIArgs,
                                       bufferPacks)))
    return failure();

  builder.setInsertionPointAfter(launch);
  func::ReturnOp::create(builder, loc);
  kernel.erase();
  return success();
}

struct HCLowerKernelsToGPULaunchPass
    : public hc::impl::HCLowerKernelsToGPULaunchBase<
          HCLowerKernelsToGPULaunchPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<HCKernelOp> kernels;
    getOperation()->walk([&](HCKernelOp kernel) { kernels.push_back(kernel); });

    for (HCKernelOp kernel : kernels) {
      if (failed(lowerKernel(kernel))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

// `createHCLowerKernelsToGPULaunchPass()` is tablegen-generated.
