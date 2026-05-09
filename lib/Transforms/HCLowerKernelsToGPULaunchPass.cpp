// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-kernels-to-gpu-launch`, the first wrapper slice of
// HC-to-upstream GPU lowering.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

static std::optional<int64_t> integerLiteral(Attribute attr) {
  auto expr = dyn_cast<ExprAttr>(attr);
  if (!expr)
    return std::nullopt;
  return sym::getIntegerLiteralValue(expr.getValue());
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

static MemRefType convertBufferABIType(BufferType type) {
  Type elementType = type.getElementType();
  if (isa<UndefType>(elementType))
    elementType = IntegerType::get(type.getContext(), 8);
  else if (Type converted = convertScalarABIType(elementType))
    elementType = converted;

  SmallVector<int64_t> dims;
  dims.reserve(type.getShape().getDims().size());
  for (Attribute dim : type.getShape().getDims()) {
    std::optional<int64_t> literal = integerLiteral(dim);
    dims.push_back(literal && *literal >= 0 ? *literal : ShapedType::kDynamic);
  }
  return MemRefType::get(dims, elementType);
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

// Idempotently declare an `extern "C"` runtime helper at module scope. The
// `llvm.emit_c_interface` attribute is what makes `convert-func-to-llvm`
// (a) emit an external `_mlir_ciface_<name>` decl that matches the C ABI
// exported by `libhc_rt_helpers.so`, and (b) generate a private body for
// `<name>` that handles memref descriptor sret packing and forwards to the
// cwrapper. Call sites in this pass therefore use the unmangled `@<name>`.
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

// Pre-declare every helper the host wrapper might reach for. Cheap and idem-
// potent; cleaner than per-call existence checks scattered through the body.
static void ensureRuntimeHelpers(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  Type ptr = LLVM::LLVMPointerType::get(ctx);
  Type i32 = IntegerType::get(ctx, 32);
  Type i64 = IntegerType::get(ctx, 64);
  Type f64 = Float64Type::get(ctx);
  Type byteRef =
      MemRefType::get({ShapedType::kDynamic}, IntegerType::get(ctx, 8));
  ensureRuntimeHelper(module, "hc_get_buffer", {ptr}, {byteRef});
  ensureRuntimeHelper(module, "hc_get_dim", {ptr, i32}, {i64});
  ensureRuntimeHelper(module, "hc_get_stride", {ptr, i32}, {i64});
  ensureRuntimeHelper(module, "hc_get_int64", {ptr}, {i64});
  ensureRuntimeHelper(module, "hc_get_float64", {ptr}, {f64});
}

// `_mlir_ciface_hc_get_dim(%pyobj, %dim_idx)` returns i64; we want index for
// downstream symbol arithmetic and `memref.view`. The cast is folded away by
// `arith-to-llvm` later when both sides are the platform pointer width.
static Value callGetDim(OpBuilder &builder, Location loc, ModuleOp module,
                        Value pyArg, unsigned dimIndex) {
  auto fn = module.lookupSymbol<func::FuncOp>("hc_get_dim");
  Type i32 = IntegerType::get(builder.getContext(), 32);
  Value dimAttr = arith::ConstantIntOp::create(builder, loc, i32,
                                               static_cast<int64_t>(dimIndex));
  auto call =
      func::CallOp::create(builder, loc, fn, ValueRange{pyArg, dimAttr});
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    call.getResult(0))
      .getResult();
}

// Materialize a typed memref from a `PyObject *` argument. Two-step: first
// pull a 1D byte buffer descriptor via `_mlir_ciface_hc_get_buffer`, then
// re-type via `memref.view` using the previously-resolved per-dim index
// values. v0 assumes contiguous row-major packing — non-contiguous strides
// will silently produce wrong indexing; tracked separately for follow-up.
static Value buildTypedBuffer(OpBuilder &builder, Location loc, ModuleOp module,
                              Value pyArg, MemRefType resultType,
                              ArrayRef<Value> shapeValues) {
  auto fn = module.lookupSymbol<func::FuncOp>("hc_get_buffer");
  auto call = func::CallOp::create(builder, loc, fn, ValueRange{pyArg});
  Value byteBuf = call.getResult(0);
  Value byteOffset =
      arith::ConstantIndexOp::create(builder, loc, 0).getResult();

  // Only the dynamic axes are passed as `sizes` operands to `memref.view`;
  // static literals are baked into the result type. Walk `shapeValues` in
  // step with the result type's shape to filter accordingly.
  SmallVector<Value> dynamicSizes;
  for (auto [dim, value] :
       llvm::zip_equal(resultType.getShape(), shapeValues)) {
    if (dim == ShapedType::kDynamic)
      dynamicSizes.push_back(value);
  }
  return memref::ViewOp::create(builder, loc, resultType, byteBuf, byteOffset,
                                dynamicSizes)
      .getResult();
}

// Pull a scalar argument of arbitrary HC scalar ABI type out of a PyObject.
// `targetType` is the post-`convertScalarABIType` MLIR type expected by the
// kernel body (e.g. `index`, `i32`, `f16`). We always go through i64/f64 on
// the wire and then narrow / convert in-IR — matches wave's helper surface
// and keeps the C ABI tiny.
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

static LogicalResult cloneKernelBodyIntoLaunch(OpBuilder &builder,
                                               HCKernelOp kernel,
                                               gpu::LaunchOp launch,
                                               ArrayRef<Value> kernelABIArgs) {
  Block &kernelBlock = kernel.getBody().front();
  Operation *returnLike = nullptr;
  if (!kernelBlock.empty()) {
    if (auto returnOp = dyn_cast<HCReturnOp>(&kernelBlock.back())) {
      if (!returnOp.getValues().empty())
        return returnOp.emitOpError("cannot lower kernel return values to "
                                    "gpu.launch");
      returnLike = returnOp;
    }
  }

  Block &launchBlock = launch.getBody().front();
  Operation *launchTerminator = nullptr;
  if (!launchBlock.empty())
    launchTerminator = &launchBlock.back();
  if (!launchTerminator || !isa<gpu::TerminatorOp>(launchTerminator)) {
    builder.setInsertionPointToEnd(&launchBlock);
    launchTerminator = gpu::TerminatorOp::create(builder, kernel.getLoc());
  }
  builder.setInsertionPoint(launchTerminator);

  IRMapping mapping;
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    Value replacement;
    if (isa<GroupType>(arg.getType())) {
      Value undef = HCUndefValueOp::create(builder, arg.getLoc(),
                                           UndefType::get(kernel.getContext()))
                        .getResult();
      replacement = UnrealizedConversionCastOp::create(builder, arg.getLoc(),
                                                       arg.getType(), undef)
                        .getResult(0);
    } else {
      Value abiValue = kernelABIArgs[index];
      replacement = abiValue.getType() == arg.getType()
                        ? abiValue
                        : UnrealizedConversionCastOp::create(
                              builder, arg.getLoc(), arg.getType(), abiValue)
                              .getResult(0);
    }
    mapping.map(arg, replacement);
  }

  for (Operation &op : kernelBlock) {
    if (&op == returnLike)
      break;
    builder.clone(op, mapping);
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

  // Per-arg conversion bookkeeping. `kernelABITypes[i]` is the post-
  // `convertABIType` type the kernel body expects; `hostArgFor[i]` is the
  // index of the matching `PyObject *` slot in the host wrapper signature
  // (or sentinel for `!hc.group` args, which aren't user-visible).
  SmallVector<Type> kernelABITypes(kernelBlock.getNumArguments());
  SmallVector<unsigned> hostArgFor(kernelBlock.getNumArguments(),
                                   std::numeric_limits<unsigned>::max());
  unsigned hostArgCount = 0;
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    if (isa<GroupType>(arg.getType()))
      continue;
    Type converted = convertABIType(arg.getType());
    if (!converted)
      return kernel.emitOpError("unsupported kernel ABI argument type ")
             << arg.getType();
    kernelABITypes[index] = converted;
    hostArgFor[index] = hostArgCount++;
  }

  ensureRuntimeHelpers(module);

  // Host wrapper takes one `PyObject *` per non-group kernel arg. We don't
  // expose a leading stream pointer yet — the launcher uses a default stream
  // baked into the HIP runtime shim. Threading an explicit stream is a
  // separate piece of plumbing once we have a Python side that materializes
  // one per-context.
  OpBuilder builder(kernel);
  Type ptrType = LLVM::LLVMPointerType::get(ctx);
  SmallVector<Type> hostInputTypes(hostArgCount, ptrType);
  auto fnType = FunctionType::get(ctx, hostInputTypes, {});
  auto hostFunc =
      func::FuncOp::create(builder, loc, kernel.getSymName(), fnType);
  Block *entry = hostFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // Two-pass arg materialization: scalar `idx` args first so a kernel that
  // declares `M: idx` alongside `Buffer[M, K, ...]` binds `M` from the
  // explicit scalar (authoritative) rather than from the buffer's dim. This
  // matches the previous "first wins" behaviour now that we're free of
  // lexical kernel-arg order.
  SmallVector<Value> kernelABIArgs(kernelBlock.getNumArguments());
  BoundValues boundValues;
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    if (isa<GroupType>(arg.getType()))
      continue;
    if (isa<BufferType>(arg.getType()))
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

  // Buffer args. For each, harvest any free shape symbols from this buffer's
  // shape (calling `_get_dim` once per first-occurrence symbol) and lower the
  // full shape attr — including non-trivial exprs — through `ExprLowerer` so
  // we can pass concrete dim values to `memref.view`.
  for (auto [index, arg] : llvm::enumerate(kernelBlock.getArguments())) {
    auto buffer = dyn_cast<BufferType>(arg.getType());
    if (!buffer)
      continue;
    Value pyArg = entry->getArgument(hostArgFor[index]);
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

    SmallVector<Value> shapeValues;
    ExprLowerer lowerer(builder, loc, boundValues);
    for (Attribute dimAttr : buffer.getShape().getDims()) {
      FailureOr<Value> dimValue = lowerer.lower(dimAttr);
      if (failed(dimValue))
        return kernel.emitOpError("failed to lower buffer shape dim for arg #")
               << index;
      shapeValues.push_back(*dimValue);
    }
    auto memrefType = cast<MemRefType>(kernelABITypes[index]);
    kernelABIArgs[index] =
        buildTypedBuffer(builder, loc, module, pyArg, memrefType, shapeValues);
  }

  SmallVector<Value> grid;
  SmallVector<Value> block;
  if (failed(
          lowerLaunchGeometry(builder, loc, kernel, boundValues, grid, block)))
    return failure();

  auto launch = gpu::LaunchOp::create(builder, loc, grid[0], grid[1], grid[2],
                                      block[0], block[1], block[2]);
  if (failed(cloneKernelBodyIntoLaunch(builder, kernel, launch, kernelABIArgs)))
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

// `createHCLowerKernelsToGPULaunchPass()` is emitted by tablegen.
