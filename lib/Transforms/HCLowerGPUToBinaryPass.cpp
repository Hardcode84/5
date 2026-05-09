// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-gpu-to-binary`. Compiles each `gpu.module`
// in the payload into a HSACO blob attached as a sibling `gpu.binary`
// op, then erases the source `gpu.module`. Mirrors wave's
// `water-gpu-module-to-binary` pass — minus the dump/override knobs
// (only useful when iterating on device-libs linking, which we don't
// do yet) and minus any ROCm-install fallback (we use our own pinned
// `ld.lld` exclusively, per the toolchain bead).

#include "hc/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>
#include <string>

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERGPUTOBINARY
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;

namespace {

// AMDGPU initializers are gated on the build's target list; calling
// them more than once is safe and they bail out cheaply if the target
// is already registered.
void initializeAMDGPUTargetOnce() {
  static const bool done = []() {
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    return true;
  }();
  (void)done;
}

class HCLowerGPUToBinaryPass
    : public hc::impl::HCLowerGPUToBinaryBase<HCLowerGPUToBinaryPass> {
public:
  using Base::Base;
  void runOnOperation() final;

private:
  LogicalResult lowerOne(gpu::GPUModuleOp module);

  // Resolve ld.lld in option → HC_LLD → $PATH order. Returns an empty
  // string and emits a diagnostic on `module` if nothing usable is
  // found. We deliberately do not look at ROCM_PATH — the toolchain
  // bead is explicit that we own our own lld and that consumers
  // should fail loudly rather than silently pick up a host install.
  std::string resolveLldPath(gpu::GPUModuleOp module);
};

std::string HCLowerGPUToBinaryPass::resolveLldPath(gpu::GPUModuleOp module) {
  if (!lldPath.empty())
    return lldPath;

  if (const char *fromEnv = std::getenv("HC_LLD"); fromEnv && *fromEnv)
    return std::string(fromEnv);

  llvm::ErrorOr<std::string> found = llvm::sys::findProgramByName("ld.lld");
  if (found)
    return *found;

  module.emitError("hc-lower-gpu-to-binary: ld.lld not found").attachNote()
      << "set --lld-path=, the HC_LLD env var, or put ld.lld on $PATH; "
         "we deliberately do not look at ROCM_PATH";
  return {};
}

LogicalResult HCLowerGPUToBinaryPass::lowerOne(gpu::GPUModuleOp module) {
  auto targets = module.getTargetsAttr();
  if (!targets || targets.size() != 1)
    return module.emitError("hc-lower-gpu-to-binary: gpu.module must carry "
                            "exactly one target "
                            "attribute (got ")
           << (targets ? targets.size() : 0) << ")";

  Attribute targetAttr = targets[0];
  auto rocdlTarget = dyn_cast_if_present<ROCDL::ROCDLTargetAttr>(targetAttr);
  if (!rocdlTarget)
    return module.emitError(
               "hc-lower-gpu-to-binary: only #rocdl.target<...> is supported "
               "today (got ")
           << targetAttr << ")";

  initializeAMDGPUTargetOnce();

  // Step 1: gpu.module → llvm::Module. The ROCDL/LLVM/Builtin dialect
  // translation interfaces must already be registered on the context's
  // dialect registry; that's `mlir::registerAllToLLVMIRTranslations`,
  // wired up by every entry point that runs this pass (hc-opt main +
  // any future Python driver).
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return module.emitError(
        "hc-lower-gpu-to-binary: failed to translate gpu.module to LLVM IR");

  // Step 2: build target machine from the rocdl target attr.
  std::string lookupError;
  llvm::Triple triple(llvm::Triple::normalize(rocdlTarget.getTriple()));
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, lookupError);
  if (!target)
    return module.emitError(
               "hc-lower-gpu-to-binary: TargetRegistry lookup failed for ")
           << rocdlTarget.getTriple() << ": " << lookupError;

  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(triple, rocdlTarget.getChip(),
                                  rocdlTarget.getFeatures(), {}, {}));
  if (!targetMachine)
    return module.emitError(
        "hc-lower-gpu-to-binary: failed to create AMDGPU target machine");
  targetMachine->setOptLevel(
      static_cast<llvm::CodeGenOptLevel>(rocdlTarget.getO()));

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetMachine->getTargetTriple());

  // Step 3: optimize. Plain wrapper around LLVM's standard pipeline
  // at the target's opt level; matches wave's `optimizeModule`.
  auto optimizer =
      makeOptimizingTransformer(static_cast<int>(targetMachine->getOptLevel()),
                                /*sizeLevel=*/0, targetMachine.get());
  if (auto err = optimizer(llvmModule.get())) {
    InFlightDiagnostic diag =
        module.emitError("hc-lower-gpu-to-binary: failed to optimize LLVM IR");
    llvm::handleAllErrors(std::move(err),
                          [&diag](const llvm::ErrorInfoBase &info) {
                            diag.attachNote() << info.message();
                          });
    return failure();
  }

  // Step 4: LLVM IR → ISA text.
  auto emitOpError = [&]() -> InFlightDiagnostic { return module.emitError(); };
  FailureOr<llvm::SmallString<0>> isa =
      LLVM::ModuleToObject::translateModuleToISA(*llvmModule, *targetMachine,
                                                 emitOpError);
  if (failed(isa))
    return failure();

  // Step 5: ISA → ELF object via the AMDGPU MC stack.
  FailureOr<SmallVector<char, 0>> object = ROCDL::assembleIsa(
      llvm::StringRef((*isa).data(), (*isa).size()),
      targetMachine->getTargetTriple().str(), targetMachine->getTargetCPU(),
      targetMachine->getTargetFeatureString(), emitOpError);
  if (failed(object))
    return failure();

  // Step 6: ELF → HSACO via ld.lld.
  std::string lld = resolveLldPath(module);
  if (lld.empty())
    return failure();

  FailureOr<SmallVector<char, 0>> binary =
      ROCDL::linkObjectCode(*object, lld, emitOpError);
  if (failed(binary))
    return failure();

  // Attach the blob as a gpu.binary sibling and drop the source module.
  // The `Binary` compilation target tells the GPU dialect that the
  // attached string is the final device blob (no further translation
  // needed downstream); the original rocdl target attr rides along so
  // the launch-time pieces can still introspect the chip if they want.
  Builder b(module.getContext());
  StringAttr blob = b.getStringAttr(StringRef(binary->data(), binary->size()));
  Attribute objectAttr = gpu::ObjectAttr::get(
      module.getContext(), targetAttr, gpu::CompilationTarget::Binary, blob,
      /*properties=*/DictionaryAttr{},
      /*kernels=*/gpu::KernelTableAttr{});

  OpBuilder builder(module.getContext());
  builder.setInsertionPointAfter(module);
  gpu::BinaryOp::create(builder, module.getLoc(), module.getName(),
                        /*offloadingHandler=*/nullptr,
                        builder.getArrayAttr({objectAttr}));

  module.erase();
  return success();
}

void HCLowerGPUToBinaryPass::runOnOperation() {
  ModuleOp root = getOperation();
  // Materialize the module list up front; lowerOne erases entries as
  // it walks. early_inc_range would also work but we'll stop on the
  // first failure so a vector keeps the control flow obvious.
  SmallVector<gpu::GPUModuleOp> modules(root.getOps<gpu::GPUModuleOp>());
  for (gpu::GPUModuleOp module : modules) {
    if (failed(lowerOne(module))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace
