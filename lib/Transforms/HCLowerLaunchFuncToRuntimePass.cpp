// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-launch-func-to-runtime`. Mirrors wave's
// `water-gpu-to-gpu-runtime` recipe (water/lib/Transforms/GPUToGPURuntime.cpp):
// each `gpu.launch_func` becomes a pair of calls into our HIP shim
// (`hc_rt_load_kernel` + `hc_rt_launch_kernel`) and the matching
// `gpu.binary`'s HSACO blob is materialised inline as an LLVM
// global so the JIT'd module is fully self-contained — no
// filesystem touches at launch time.
//
// The stream pointer is the first argument of the enclosing host
// wrapper (laid down by `hc-lower-kernels-to-gpu-launch`). We grab it
// from `parentOfType<LLVM::LLVMFuncOp>().getArgument(0)` rather than
// re-deriving it from a side-channel attribute, because by the time
// this pass runs the host wrapper has already been lowered through
// convert-func-to-llvm and the convention "host wrapper's arg(0) is
// the stream" is the cheapest invariant to enforce.
//
// We deliberately ship a smaller surface than wave's version: no
// cluster size (gpu.launch_func today never carries cluster dims
// through our pipeline; we hard-code 0 for cluster_{x,y,z}), and the
// binary always carries exactly one `#gpu.object` (we only emit one
// per binary in the schedule).

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

// Wave's helper, copied verbatim: keep symbol uniqueness scoped to the
// module's symbol table so multiple launches of the same kernel each
// get their own `_data` / `_handle` globals.
static SmallString<128> getUniqueLLVMGlobalName(ModuleOp mod,
                                                SymbolTable &table,
                                                const llvm::Twine &srcName) {
  unsigned counter = 0;
  return SymbolTable::generateSymbolName<128>(
      srcName.str(),
      [&](StringRef candidate) { return table.lookupSymbolIn(mod, candidate); },
      counter);
}

// Lazy declaration helper for the runtime entry points. We don't hard-code
// the function decls in the schedule because then every payload (including
// the trivial ones with no launches) would have to carry them; resolving
// on demand keeps the surface minimal and the lookup is cheap.
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

// Mint a fresh internal-linkage global with the given element type and a
// `zeroinitializer` body, return its address. Used for the per-callsite
// `hipFunction_t` cache slot the runtime fills on first launch.
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

// Pull the raw blob out of a `gpu.binary`. We require exactly one object
// (the schedule produces one `#gpu.object` per binary — single rocdl
// target). Anything else is a hard error rather than picking blindly.
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
  // One launch_func -> two runtime calls + per-callsite globals. Reads the
  // HSACO blob from the matching gpu.binary; the binary itself is erased
  // in a second pass after every launch_func has been consumed (erasing
  // it inline would invalidate the SymbolTable lookups for any other
  // launch into the same kernel).
  LogicalResult lowerOne(gpu::LaunchFuncOp op, ModuleOp mod,
                         SymbolTable &symbolTable,
                         const FunctionCallBuilder &loadFuncBuilder,
                         const FunctionCallBuilder &launchFuncBuilder);
};

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

  auto createConst = [&](Type type, int64_t val) -> Value {
    return LLVM::ConstantOp::create(builder, loc, type,
                                    builder.getIntegerAttr(type, val));
  };
  auto createAlloca = [&](Type elemType, int64_t size) -> Value {
    Value sizeVal = createConst(i64Type, size);
    return LLVM::AllocaOp::create(builder, loc, ptrType, elemType, sizeVal,
                                  /*alignment=*/0);
  };

  // Stream comes from the host wrapper's leading `!llvm.ptr` arg. The
  // host wrapper is the only function that contains gpu.launch_func ops
  // in our pipeline, so an enclosing-LLVMFunc lookup is well-defined;
  // missing one is a programmer error (the launch_func escaped its
  // wrapper somehow), not a user-input failure.
  auto enclosingFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!enclosingFunc || enclosingFunc.getNumArguments() == 0)
    return op->emitError(
        "hc-lower-launch-func-to-runtime: gpu.launch_func is not nested "
        "inside an llvm.func with a leading stream argument");
  Value stream = enclosingFunc.getArgument(0);

  // Per-callsite globals. Wave keeps the cache slot per-callsite (rather
  // than per-binary) to sidestep cross-callsite synchronisation; we
  // follow the same shape — the runtime's atomic single-flight handles
  // the actual race.
  StringRef kernelName = op.getKernelName();
  Value kernelHandle = createKernelHandle(builder, symbolTable, ptrType, mod,
                                          kernelName + "_handle");

  // `kernel_name + "\0"` so the runtime can pass it straight through to
  // `hipModuleGetFunction`, which expects NUL-terminated input.
  SmallString<64> nameBuf(kernelName);
  nameBuf.push_back('\0');
  Value kernelNameStr = LLVM::createGlobalString(
      loc, builder, getUniqueLLVMGlobalName(mod, symbolTable, kernelName),
      nameBuf, LLVM::Linkage::Internal);

  // The HSACO blob ships as a raw byte string — `createGlobalString`
  // gives us back a pointer to its first byte, which is exactly what
  // `hipModuleLoadData` wants. We do NOT NUL-terminate: ELF blobs are
  // self-describing and the size is passed alongside the pointer.
  Value dataPtr = LLVM::createGlobalString(
      loc, builder,
      getUniqueLLVMGlobalName(mod, symbolTable, kernelName + "_data"), objData,
      LLVM::Linkage::Internal);
  Value dataSize = createConst(i64Type, objData.size());

  Value funcObject =
      loadFuncBuilder
          .create(loc, builder,
                  {stream, kernelHandle, dataPtr, dataSize, kernelNameStr})
          ->getResult(0);

  // HIP `kernelParams` convention: an array of `void*`, each pointing
  // at the storage holding one kernel argument. `alloca` per arg, store
  // the value, then build the array via insertvalue and spill the array
  // itself to one more alloca. Matches wave's recipe; matches what HIP's
  // `hipModuleLaunchKernel` documents.
  Value sharedMemoryBytes = createConst(i32Type, 0);
  ValueRange args = op.getKernelOperands();
  auto argsPtrArrayType = LLVM::LLVMArrayType::get(ptrType, args.size());
  Value argsArray = LLVM::PoisonOp::create(builder, loc, argsPtrArrayType);
  for (auto &&[i, arg] : llvm::enumerate(args)) {
    Value argData = createAlloca(arg.getType(), 1);
    LLVM::StoreOp::create(builder, loc, arg, argData);
    argsArray =
        LLVM::InsertValueOp::create(builder, loc, argsArray, argData, i);
  }
  Value argsArrayPtr = createAlloca(argsPtrArrayType, 1);
  LLVM::StoreOp::create(builder, loc, argsArray, argsArrayPtr);
  Value argsCount = createConst(i32Type, args.size());

  // Cluster dims: the GPU dialect wires these as optional operands.
  // `gpu.launch_func` produced by our pipeline today never carries
  // cluster dims (we never call the cluster-aware overload of
  // `gpu-kernel-outlining`), so the Optional getters return null. Pass
  // 0 in that case — the runtime treats <=1 cluster as "no cluster" and
  // routes to the simpler launch entry point.
  Value clusterX =
      op.getClusterSizeX() ? op.getClusterSizeX() : createConst(i64Type, 0);
  Value clusterY =
      op.getClusterSizeY() ? op.getClusterSizeY() : createConst(i64Type, 0);
  Value clusterZ =
      op.getClusterSizeZ() ? op.getClusterSizeZ() : createConst(i64Type, 0);

  launchFuncBuilder.create(loc, builder,
                           {stream, funcObject, sharedMemoryBytes,
                            op.getGridSizeX(), op.getGridSizeY(),
                            op.getGridSizeZ(), op.getBlockSizeX(),
                            op.getBlockSizeY(), op.getBlockSizeZ(), clusterX,
                            clusterY, clusterZ, argsArrayPtr, argsCount});
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

  // Match the C ABI in include/hc/Runtime/HipRuntime.h. Argument count
  // and order are load-bearing: the runtime trusts the JIT to hand it
  // the right shape and there's no MLIR-side type check beyond what
  // LLVM::CallOp's verifier enforces against the function decl above.
  FunctionCallBuilder loadFuncBuilder(
      "hc_rt_load_kernel", ptrType,
      {ptrType, ptrType, ptrType, i64Type, ptrType});
  FunctionCallBuilder launchFuncBuilder(
      "hc_rt_launch_kernel", voidType,
      {ptrType, ptrType, i32Type, i64Type, i64Type, i64Type, i64Type, i64Type,
       i64Type, i64Type, i64Type, i64Type, ptrType, i32Type});

  SymbolTable symbolTable(mod);

  // Snapshot the launch ops up front so erasing them inside `lowerOne`
  // doesn't invalidate the walk.
  SmallVector<gpu::LaunchFuncOp> launches;
  mod.walk([&](gpu::LaunchFuncOp op) { launches.push_back(op); });

  for (gpu::LaunchFuncOp op : launches) {
    if (failed(lowerOne(op, mod, symbolTable, loadFuncBuilder,
                        launchFuncBuilder))) {
      signalPassFailure();
      return;
    }
  }

  // Now the binaries — every consumer is gone, the blob ships in the
  // per-callsite `_data` globals we just minted. Snapshot first for the
  // same reason as the launches.
  SmallVector<gpu::BinaryOp> binaries;
  mod.walk([&](gpu::BinaryOp op) { binaries.push_back(op); });
  for (gpu::BinaryOp op : binaries)
    op.erase();
}

} // namespace
