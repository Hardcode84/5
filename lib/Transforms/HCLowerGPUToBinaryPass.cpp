// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-gpu-to-binary`. Compile each `gpu.module` to
// an HSACO blob, attach as a sibling `gpu.binary`, erase the source.
// Uses our pinned `ld.lld` -- no ROCm-install fallback.

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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
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

// Safe to call repeatedly; bail out cheap if already registered.
static void initializeAMDGPUTargetOnce() {
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

  // Search order: option -> `HC_LLD` -> `$PATH`. ROCM_PATH excluded --
  // consumers should fail loudly, not silently pick up a host install.
  std::string resolveLldPath(gpu::GPUModuleOp module);

  // Dump to `<dumpIntermediates>/<moduleName>.<stage>`; stage names
  // sort by pipeline order. Write failures surface as pass errors.
  LogicalResult dumpStage(gpu::GPUModuleOp module, StringRef stage,
                          StringRef bytes);

  // Serialize `m` to text; forward to `dumpStage`.
  LogicalResult dumpLLVMModule(gpu::GPUModuleOp module, StringRef stage,
                               const llvm::Module &m);

  // Exactly one `#rocdl.target` required; else hard error.
  FailureOr<ROCDL::ROCDLTargetAttr>
  validateAndExtractRocdlTarget(gpu::GPUModuleOp module);

  // Side-effects `llvmModule`: data layout and triple must match the
  // target machine for downstream emit / optimize.
  FailureOr<std::unique_ptr<llvm::TargetMachine>>
  buildTargetMachineFor(gpu::GPUModuleOp module,
                        ROCDL::ROCDLTargetAttr rocdlTarget,
                        llvm::Module &llvmModule);

  // LLVM standard pipeline at the target opt level; dumps `1-post-opt.ll`.
  LogicalResult optimizeLLVMModule(gpu::GPUModuleOp module,
                                   llvm::Module &llvmModule,
                                   llvm::TargetMachine &targetMachine);

  // Translate optimized LLVM IR to ISA text. Dumps `2-isa.s`.
  FailureOr<llvm::SmallString<0>>
  emitISAFromLLVMModule(gpu::GPUModuleOp module, llvm::Module &llvmModule,
                        llvm::TargetMachine &targetMachine);

  // ISA -> ELF (AMDGPU MC) -> HSACO (`ld.lld`); dumps `2b-object.o` and
  // `3-binary.hsaco`.
  FailureOr<SmallVector<char, 0>>
  assembleAndLinkBinary(gpu::GPUModuleOp module, llvm::SmallString<0> &isa,
                        llvm::TargetMachine &targetMachine);

  // Target attr rides along -- launch-time can still introspect the chip.
  void attachBinaryAndEraseModule(gpu::GPUModuleOp module, Attribute targetAttr,
                                  ArrayRef<char> binary);
};

LogicalResult HCLowerGPUToBinaryPass::dumpStage(gpu::GPUModuleOp module,
                                                StringRef stage,
                                                StringRef bytes) {
  if (dumpIntermediates.empty())
    return success();

  llvm::SmallString<128> path(dumpIntermediates);
  llvm::sys::path::append(path, module.getName().str() + "." + stage.str());

  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return module.emitError("hc-lower-gpu-to-binary: failed to open ")
           << path << " for dump-intermediates: " << ec.message();
  out.write(bytes.data(), bytes.size());
  out.flush();
  if (out.has_error())
    return module.emitError("hc-lower-gpu-to-binary: write failed for ")
           << path << " (dump-intermediates)";
  return success();
}

LogicalResult HCLowerGPUToBinaryPass::dumpLLVMModule(gpu::GPUModuleOp module,
                                                     StringRef stage,
                                                     const llvm::Module &m) {
  if (dumpIntermediates.empty())
    return success();
  std::string text;
  llvm::raw_string_ostream os(text);
  m.print(os, /*AAW=*/nullptr);
  os.flush();
  return dumpStage(module, stage, text);
}

std::string HCLowerGPUToBinaryPass::resolveLldPath(gpu::GPUModuleOp module) {
  // Order: `--lld-path` / `HC_LLD` / `$PATH`. Existence-check first --
  // `ExecuteAndWait` swallows missing-binary into "lld invocation
  // failed". Explicit candidates fail loud on a miss; PATH fallback
  // only fires when neither was set.
  auto failMissing = [&](StringRef where, StringRef path) {
    module.emitError("hc-lower-gpu-to-binary: ld.lld not executable at ")
        << path << " (via " << where << ")";
  };

  if (!lldPath.empty()) {
    if (llvm::sys::fs::can_execute(lldPath))
      return lldPath;
    failMissing("--lld-path=", lldPath);
    return {};
  }

  if (const char *fromEnv = std::getenv("HC_LLD"); fromEnv && *fromEnv) {
    if (llvm::sys::fs::can_execute(fromEnv))
      return std::string(fromEnv);
    failMissing("HC_LLD env", fromEnv);
    return {};
  }

  llvm::ErrorOr<std::string> found = llvm::sys::findProgramByName("ld.lld");
  if (found && llvm::sys::fs::can_execute(*found))
    return *found;

  module.emitError("hc-lower-gpu-to-binary: ld.lld not found").attachNote()
      << "set --lld-path=, the HC_LLD env var, or put ld.lld on $PATH; "
         "we deliberately do not look at ROCM_PATH";
  return {};
}

FailureOr<ROCDL::ROCDLTargetAttr>
HCLowerGPUToBinaryPass::validateAndExtractRocdlTarget(gpu::GPUModuleOp module) {
  auto targets = module.getTargetsAttr();
  if (!targets || targets.size() != 1) {
    module.emitError("hc-lower-gpu-to-binary: gpu.module must carry exactly "
                     "one target attribute (got ")
        << (targets ? targets.size() : 0) << ")";
    return failure();
  }
  Attribute targetAttr = targets[0];
  auto rocdlTarget = dyn_cast_if_present<ROCDL::ROCDLTargetAttr>(targetAttr);
  if (!rocdlTarget) {
    module.emitError("hc-lower-gpu-to-binary: only #rocdl.target<...> is "
                     "supported today (got ")
        << targetAttr << ")";
    return failure();
  }
  return rocdlTarget;
}

FailureOr<std::unique_ptr<llvm::TargetMachine>>
HCLowerGPUToBinaryPass::buildTargetMachineFor(
    gpu::GPUModuleOp module, ROCDL::ROCDLTargetAttr rocdlTarget,
    llvm::Module &llvmModule) {
  std::string lookupError;
  llvm::Triple triple(llvm::Triple::normalize(rocdlTarget.getTriple()));
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, lookupError);
  if (!target) {
    module.emitError(
        "hc-lower-gpu-to-binary: TargetRegistry lookup failed for ")
        << rocdlTarget.getTriple() << ": " << lookupError;
    return failure();
  }

  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(triple, rocdlTarget.getChip(),
                                  rocdlTarget.getFeatures(), {}, {}));
  if (!targetMachine) {
    module.emitError(
        "hc-lower-gpu-to-binary: failed to create AMDGPU target machine");
    return failure();
  }
  targetMachine->setOptLevel(
      static_cast<llvm::CodeGenOptLevel>(rocdlTarget.getO()));

  llvmModule.setDataLayout(targetMachine->createDataLayout());
  llvmModule.setTargetTriple(targetMachine->getTargetTriple());
  return targetMachine;
}

LogicalResult
HCLowerGPUToBinaryPass::optimizeLLVMModule(gpu::GPUModuleOp module,
                                           llvm::Module &llvmModule,
                                           llvm::TargetMachine &targetMachine) {
  // LLVM standard pipeline at target opt level.
  auto optimizer =
      makeOptimizingTransformer(static_cast<int>(targetMachine.getOptLevel()),
                                /*sizeLevel=*/0, &targetMachine);
  if (auto err = optimizer(&llvmModule)) {
    InFlightDiagnostic diag =
        module.emitError("hc-lower-gpu-to-binary: failed to optimize LLVM IR");
    llvm::handleAllErrors(std::move(err),
                          [&diag](const llvm::ErrorInfoBase &info) {
                            diag.attachNote() << info.message();
                          });
    return failure();
  }
  return dumpLLVMModule(module, "1-post-opt.ll", llvmModule);
}

FailureOr<llvm::SmallString<0>> HCLowerGPUToBinaryPass::emitISAFromLLVMModule(
    gpu::GPUModuleOp module, llvm::Module &llvmModule,
    llvm::TargetMachine &targetMachine) {
  auto emitOpError = [&]() -> InFlightDiagnostic { return module.emitError(); };
  FailureOr<llvm::SmallString<0>> isa =
      LLVM::ModuleToObject::translateModuleToISA(llvmModule, targetMachine,
                                                 emitOpError);
  if (failed(isa))
    return failure();
  if (failed(dumpStage(module, "2-isa.s",
                       StringRef((*isa).data(), (*isa).size()))))
    return failure();
  return isa;
}

FailureOr<SmallVector<char, 0>> HCLowerGPUToBinaryPass::assembleAndLinkBinary(
    gpu::GPUModuleOp module, llvm::SmallString<0> &isa,
    llvm::TargetMachine &targetMachine) {
  auto emitOpError = [&]() -> InFlightDiagnostic { return module.emitError(); };
  FailureOr<SmallVector<char, 0>> object = ROCDL::assembleIsa(
      llvm::StringRef(isa.data(), isa.size()),
      targetMachine.getTargetTriple().str(), targetMachine.getTargetCPU(),
      targetMachine.getTargetFeatureString(), emitOpError);
  if (failed(object))
    return failure();
  // `2b` sorts between `2-isa.s` and `3-binary.hsaco`. `linkObjectCode`
  // swallows lld stderr; without this dump every linker bug needs a
  // re-run with extra instrumentation.
  if (failed(dumpStage(module, "2b-object.o",
                       StringRef(object->data(), object->size()))))
    return failure();

  std::string lld = resolveLldPath(module);
  if (lld.empty())
    return failure();

  FailureOr<SmallVector<char, 0>> binary =
      ROCDL::linkObjectCode(*object, lld, emitOpError);
  if (failed(binary))
    return failure();
  if (failed(dumpStage(module, "3-binary.hsaco",
                       StringRef(binary->data(), binary->size()))))
    return failure();
  return binary;
}

void HCLowerGPUToBinaryPass::attachBinaryAndEraseModule(gpu::GPUModuleOp module,
                                                        Attribute targetAttr,
                                                        ArrayRef<char> binary) {
  // `Binary` compilation target = final device blob (no further translation).
  Builder b(module.getContext());
  StringAttr blob = b.getStringAttr(StringRef(binary.data(), binary.size()));
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
}

LogicalResult HCLowerGPUToBinaryPass::lowerOne(gpu::GPUModuleOp module) {
  FailureOr<ROCDL::ROCDLTargetAttr> rocdlTargetOr =
      validateAndExtractRocdlTarget(module);
  if (failed(rocdlTargetOr))
    return failure();
  ROCDL::ROCDLTargetAttr rocdlTarget = *rocdlTargetOr;

  initializeAMDGPUTargetOnce();

  // Translation interfaces must be registered on the context
  // (`mlir::registerAllToLLVMIRTranslations`).
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return module.emitError(
        "hc-lower-gpu-to-binary: failed to translate gpu.module to LLVM IR");

  FailureOr<std::unique_ptr<llvm::TargetMachine>> targetMachineOr =
      buildTargetMachineFor(module, rocdlTarget, *llvmModule);
  if (failed(targetMachineOr))
    return failure();
  std::unique_ptr<llvm::TargetMachine> targetMachine =
      std::move(*targetMachineOr);

  // Pre-opt dump first -- survives an optimizer crash.
  if (failed(dumpLLVMModule(module, "0-pre-opt.ll", *llvmModule)))
    return failure();
  if (failed(optimizeLLVMModule(module, *llvmModule, *targetMachine)))
    return failure();

  FailureOr<llvm::SmallString<0>> isa =
      emitISAFromLLVMModule(module, *llvmModule, *targetMachine);
  if (failed(isa))
    return failure();

  FailureOr<SmallVector<char, 0>> binary =
      assembleAndLinkBinary(module, *isa, *targetMachine);
  if (failed(binary))
    return failure();

  attachBinaryAndEraseModule(module, rocdlTarget, *binary);
  return success();
}

void HCLowerGPUToBinaryPass::runOnOperation() {
  ModuleOp root = getOperation();
  // Materialize first -- `lowerOne` erases as it walks.
  SmallVector<gpu::GPUModuleOp> modules(root.getOps<gpu::GPUModuleOp>());
  for (gpu::GPUModuleOp module : modules) {
    if (failed(lowerOne(module))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace
