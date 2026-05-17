// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-launch-func-to-runtime`. Each `gpu.launch_func`
// becomes `hc_rt_load_kernel` + `hc_rt_launch_kernel` calls into the HIP
// shim; the matching `gpu.binary`'s HSACO blob is inlined as an LLVM
// global so the JIT'd module is self-contained — no filesystem touches
// at launch time.
//
// Stream is the host wrapper's arg(0) by convention. Cluster dims
// hard-coded to 0 (pipeline never carries them). `gpu.binary` must
// carry exactly one `#gpu.object`.

#include "hc/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLAUNCHFUNCTORUNTIME
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;

namespace {

// Per-launch global uniqueness scoped to the module symbol table.
static SmallString<128> getUniqueLLVMGlobalName(ModuleOp mod,
                                                SymbolTable &table,
                                                const llvm::Twine &srcName) {
  unsigned counter = 0;
  return SymbolTable::generateSymbolName<128>(
      srcName.str(),
      [&](StringRef candidate) { return table.lookupSymbolIn(mod, candidate); },
      counter);
}

// Lazy decl — trivial payloads with no launches don't carry runtime decls.
struct FunctionCallBuilder {
  FunctionCallBuilder(StringRef functionName, Type returnType,
                      ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}

  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ValueRange arguments) const {
    Operation *moduleOp = builder.getBlock()
                              ->getParentOp()
                              ->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "module not found");
    SymbolTable symbolTable(moduleOp);
    auto function = [&] {
      if (auto fn = symbolTable.lookup<LLVM::LLVMFuncOp>(functionName))
        return fn;
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&moduleOp->getRegion(0).front());
      return LLVM::LLVMFuncOp::create(builder, loc, functionName, functionType);
    }();
    return LLVM::CallOp::create(builder, loc, function, arguments);
  }

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

// Per-callsite `hipFunction_t` cache slot; runtime fills on first launch.
static Value createKernelHandle(OpBuilder &builder, SymbolTable &symbolTable,
                                Type globalType, ModuleOp mod,
                                const llvm::Twine &name) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Location loc = builder.getUnknownLoc();
  LLVM::GlobalOp handle;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(mod.getBody());
    SmallString<128> handleName =
        getUniqueLLVMGlobalName(mod, symbolTable, name);
    handle = LLVM::GlobalOp::create(
        builder, loc, globalType, /*isConstant=*/false, LLVM::Linkage::Internal,
        handleName, LLVM::ZeroAttr::get(builder.getContext()));
  }
  return LLVM::AddressOfOp::create(builder, loc, ptrType, handle.getSymName());
}

// Exactly one `#gpu.object` required (single rocdl target).
static gpu::ObjectAttr getSelectedObject(gpu::BinaryOp op) {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();
  if (objects.size() != 1) {
    op->emitError("hc-lower-launch-func-to-runtime: gpu.binary must carry "
                  "exactly one object (got ")
        << objects.size() << ")";
    return nullptr;
  }
  auto result = dyn_cast<gpu::ObjectAttr>(objects[0]);
  if (!result)
    op->emitError("hc-lower-launch-func-to-runtime: invalid object type on "
                  "gpu.binary");
  return result;
}

static gpu::ObjectAttr getBinary(gpu::LaunchFuncOp op) {
  auto kernelBinary = SymbolTable::lookupNearestSymbolFrom<gpu::BinaryOp>(
      op, op.getKernelModuleName());
  if (!kernelBinary) {
    op.emitError("hc-lower-launch-func-to-runtime: couldn't find the "
                 "gpu.binary holding the kernel: ")
        << op.getKernelModuleName().getValue();
    return nullptr;
  }
  return getSelectedObject(kernelBinary);
}

class HCLowerLaunchFuncToRuntimePass
    : public hc::impl::HCLowerLaunchFuncToRuntimeBase<
          HCLowerLaunchFuncToRuntimePass> {
public:
  using Base::Base;
  void runOnOperation() final;

private:
  // Binary erased after all launches consumed — inline erase invalidates
  // SymbolTable lookups for other launches into the same kernel.
  LogicalResult lowerOne(gpu::LaunchFuncOp op, ModuleOp mod,
                         SymbolTable &symbolTable,
                         const FunctionCallBuilder &loadFuncBuilder,
                         const FunctionCallBuilder &launchFuncBuilder);
};

static Value emitConstInt(OpBuilder &builder, Location loc, Type type,
                          int64_t val) {
  return LLVM::ConstantOp::create(builder, loc, type,
                                  builder.getIntegerAttr(type, val));
}

static Value emitAlloca(OpBuilder &builder, Location loc, Type ptrType,
                        Type elemType, int64_t size) {
  Type i64Type = IntegerType::get(builder.getContext(), 64);
  Value sizeVal = emitConstInt(builder, loc, i64Type, size);
  return LLVM::AllocaOp::create(builder, loc, ptrType, elemType, sizeVal,
                                /*alignment=*/0);
}

// HIP `kernelParams`: `void**` (alloca-per-arg, array via insertvalue, spill
// array).
static Value packKernelArgs(OpBuilder &builder, Location loc, ValueRange args,
                            Type ptrType) {
  auto argsPtrArrayType = LLVM::LLVMArrayType::get(ptrType, args.size());
  Value argsArray = LLVM::PoisonOp::create(builder, loc, argsPtrArrayType);
  for (auto &&[i, arg] : llvm::enumerate(args)) {
    Value argData = emitAlloca(builder, loc, ptrType, arg.getType(), 1);
    LLVM::StoreOp::create(builder, loc, arg, argData);
    argsArray =
        LLVM::InsertValueOp::create(builder, loc, argsArray, argData, i);
  }
  Value argsArrayPtr = emitAlloca(builder, loc, ptrType, argsPtrArrayType, 1);
  LLVM::StoreOp::create(builder, loc, argsArray, argsArrayPtr);
  return argsArrayPtr;
}

// Pipeline never carries cluster dims; null → 0 (runtime: <=1 means "no
// cluster").
static std::array<Value, 3> resolveClusterDims(OpBuilder &builder, Location loc,
                                               gpu::LaunchFuncOp op,
                                               Type i64Type) {
  auto orZero = [&](Value v) -> Value {
    return v ? v : emitConstInt(builder, loc, i64Type, 0);
  };
  return {orZero(op.getClusterSizeX()), orZero(op.getClusterSizeY()),
          orZero(op.getClusterSizeZ())};
}

LogicalResult HCLowerLaunchFuncToRuntimePass::lowerOne(
    gpu::LaunchFuncOp op, ModuleOp mod, SymbolTable &symbolTable,
    const FunctionCallBuilder &loadFuncBuilder,
    const FunctionCallBuilder &launchFuncBuilder) {
  MLIRContext *context = &getContext();
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type ptrType = LLVM::LLVMPointerType::get(context);

  gpu::ObjectAttr object = getBinary(op);
  if (!object)
    return failure();
  StringRef objData = object.getObject();

  IRRewriter builder(context);
  builder.setInsertionPoint(op);
  Location loc = op.getLoc();

  // Stream is the host wrapper's arg(0).
  auto enclosingFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!enclosingFunc || enclosingFunc.getNumArguments() == 0)
    return op->emitError(
        "hc-lower-launch-func-to-runtime: gpu.launch_func is not nested "
        "inside an llvm.func with a leading stream argument");
  Value stream = enclosingFunc.getArgument(0);

  // Per-callsite cache slot — runtime's atomic single-flight handles the race.
  StringRef kernelName = op.getKernelName();
  Value kernelHandle = createKernelHandle(builder, symbolTable, ptrType, mod,
                                          kernelName + "_handle");

  // NUL-terminate name for `hipModuleGetFunction`; blob is sized separately
  // (ELF).
  SmallString<64> nameBuf(kernelName);
  nameBuf.push_back('\0');
  Value kernelNameStr = LLVM::createGlobalString(
      loc, builder, getUniqueLLVMGlobalName(mod, symbolTable, kernelName),
      nameBuf, LLVM::Linkage::Internal);
  Value dataPtr = LLVM::createGlobalString(
      loc, builder,
      getUniqueLLVMGlobalName(mod, symbolTable, kernelName + "_data"), objData,
      LLVM::Linkage::Internal);
  Value dataSize = emitConstInt(builder, loc, i64Type, objData.size());

  Value funcObject =
      loadFuncBuilder
          .create(loc, builder,
                  {stream, kernelHandle, dataPtr, dataSize, kernelNameStr})
          ->getResult(0);

  Value sharedMemoryBytes = emitConstInt(builder, loc, i32Type, 0);
  ValueRange args = op.getKernelOperands();
  Value argsArrayPtr = packKernelArgs(builder, loc, args, ptrType);
  Value argsCount = emitConstInt(builder, loc, i32Type, args.size());

  std::array<Value, 3> cluster = resolveClusterDims(builder, loc, op, i64Type);
  launchFuncBuilder.create(loc, builder,
                           {stream, funcObject, sharedMemoryBytes,
                            op.getGridSizeX(), op.getGridSizeY(),
                            op.getGridSizeZ(), op.getBlockSizeX(),
                            op.getBlockSizeY(), op.getBlockSizeZ(), cluster[0],
                            cluster[1], cluster[2], argsArrayPtr, argsCount});
  builder.eraseOp(op);
  return success();
}

void HCLowerLaunchFuncToRuntimePass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *context = &getContext();

  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type voidType = LLVM::LLVMVoidType::get(context);

  // C ABI from `include/hc/Runtime/HipRuntime.h`; arg count and order
  // load-bearing.
  FunctionCallBuilder loadFuncBuilder(
      "hc_rt_load_kernel", ptrType,
      {ptrType, ptrType, ptrType, i64Type, ptrType});
  FunctionCallBuilder launchFuncBuilder(
      "hc_rt_launch_kernel", voidType,
      {ptrType, ptrType, i32Type, i64Type, i64Type, i64Type, i64Type, i64Type,
       i64Type, i64Type, i64Type, i64Type, ptrType, i32Type});

  SymbolTable symbolTable(mod);

  // Snapshot — erase inside `lowerOne` would invalidate the walk.
  SmallVector<gpu::LaunchFuncOp> launches;
  mod.walk([&](gpu::LaunchFuncOp op) { launches.push_back(op); });

  for (gpu::LaunchFuncOp op : launches) {
    if (failed(lowerOne(op, mod, symbolTable, loadFuncBuilder,
                        launchFuncBuilder))) {
      signalPassFailure();
      return;
    }
  }

  // Binaries — consumers gone, blob in per-callsite `_data` globals.
  SmallVector<gpu::BinaryOp> binaries;
  mod.walk([&](gpu::BinaryOp op) { binaries.push_back(op); });
  for (gpu::BinaryOp op : binaries)
    op.erase();
}

} // namespace
