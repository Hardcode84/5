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

// AMDGPU initializers are gated on the build's target list; calling
// them more than once is safe and they bail out cheaply if the target
// is already registered.
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

  // Resolve ld.lld in option → HC_LLD → $PATH order. Returns an empty
  // string and emits a diagnostic on `module` if nothing usable is
  // found. We deliberately do not look at ROCM_PATH — the toolchain
  // bead is explicit that we own our own lld and that consumers
  // should fail loudly rather than silently pick up a host install.
  std::string resolveLldPath(gpu::GPUModuleOp module);

  // Write `bytes` to `<dumpIntermediates>/<moduleName>.<stage>` if the
  // option is non-empty; no-op otherwise. Stage names embed an order
  // prefix (`0-pre-opt.ll`, `1-post-opt.ll`, ...) so `ls` shows the
  // pipeline order. Diagnostics on write failure attach to `module`
  // and surface as a pass-level error — nobody opts into dumping
  // expecting silent partial output.
  LogicalResult dumpStage(gpu::GPUModuleOp module, StringRef stage,
                          StringRef bytes);

  // LLVM module convenience: serialises `m` to text and forwards to
  // `dumpStage`. Kept separate so the call sites stay readable.
  LogicalResult dumpLLVMModule(gpu::GPUModuleOp module, StringRef stage,
                               const llvm::Module &m);

  // Validate the module's `targets` attr and unwrap the single
  // `#rocdl.target` element. Anything else is a hard error with the
  // offending attr surfaced in the diagnostic.
  FailureOr<ROCDL::ROCDLTargetAttr>
  validateAndExtractRocdlTarget(gpu::GPUModuleOp module);

  // Build the AMDGPU target machine from the rocdl target attr and
  // imprint its data layout / triple onto the just-translated LLVM
  // module. Side effects on `llvmModule` are intentional — the data
  // layout and triple must match the target machine for downstream
  // emit / optimize to be well-defined.
  FailureOr<std::unique_ptr<llvm::TargetMachine>>
  buildTargetMachineFor(gpu::GPUModuleOp module,
                        ROCDL::ROCDLTargetAttr rocdlTarget,
                        llvm::Module &llvmModule);

  // Run LLVM's standard optimization pipeline at the target machine's
  // opt level. Dumps `1-post-opt.ll` on success.
  LogicalResult optimizeLLVMModule(gpu::GPUModuleOp module,
                                   llvm::Module &llvmModule,
                                   llvm::TargetMachine &targetMachine);

  // Translate optimized LLVM IR to ISA text. Dumps `2-isa.s`.
  FailureOr<llvm::SmallString<0>>
  emitISAFromLLVMModule(gpu::GPUModuleOp module, llvm::Module &llvmModule,
                        llvm::TargetMachine &targetMachine);

  // Assemble ISA text to an ELF object via the AMDGPU MC stack, then
  // link with our own ld.lld into an HSACO. Dumps `2b-object.o` and
  // `3-binary.hsaco` along the way.
  FailureOr<SmallVector<char, 0>>
  assembleAndLinkBinary(gpu::GPUModuleOp module, llvm::SmallString<0> &isa,
                        llvm::TargetMachine &targetMachine);

  // Attach the HSACO blob as a sibling `gpu.binary` op and drop the
  // source `gpu.module`. The original rocdl target attr rides along
  // on the object so launch-time pieces can still introspect the chip.
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
  // Three candidate sources in priority order:
  //   1. `--lld-path=` option (populated Python-side by `_substitute_lld`
  //      from `_native_paths.lld_path()`, which points at the bundled
  //      `hc/_native/bin/ld.lld` by default).
  //   2. `HC_LLD` env (source-tree dev override, undocumented but kept
  //      working for users who run pre-staged toolchain builds).
  //   3. `findProgramByName("ld.lld")` — last-ditch PATH search; not what
  //      the production pipeline should rely on but harmless as a fallback
  //      when the wheel is misconfigured.
  //
  // Each candidate is existence-checked before we hand it to
  // `linkObjectCode` — the MLIR `ExecuteAndWait` wrapper surfaces an
  // `execve` failure as the same unhelpful "lld invocation failed"
  // diagnostic as a real linker error, and a missing-binary is by far
  // the more common failure mode for source-tree devs and broken wheel
  // installs. Validating up front lets the diagnostic name the missing
  // path so the user knows exactly which file to stage.
  //
  // Explicit candidates (option, env) are authoritative — when the
  // user sets one we fail loudly on a miss rather than cascading to
  // PATH. Silently substituting a different `ld.lld` than was asked
  // for would defeat the point of the override and surface as the
  // next mysterious "why is my linker doing X" bug. Only the implicit
  // PATH fallback fires when nothing was set.
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
  // Plain wrapper around LLVM's standard pipeline at the target's opt
  // level; matches wave's `optimizeModule`.
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
  // Stage `2b` because we want this between `2-isa.s` and `3-binary.hsaco`
  // in `ls | sort` — the assembled ELF is the input lld actually sees,
  // so when the linker step fails the .o is what you reach for first.
  // The MLIR `linkObjectCode` wrapper swallows lld's stderr and surfaces
  // only a generic "lld invocation failed", so without this dump every
  // linker bug starts with a re-run-with-extra-instrumentation step.
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
  // The `Binary` compilation target tells the GPU dialect that the
  // attached string is the final device blob (no further translation
  // needed downstream); the original rocdl target attr rides along so
  // the launch-time pieces can still introspect the chip if they want.
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

  // gpu.module → llvm::Module. The ROCDL/LLVM/Builtin dialect translation
  // interfaces must already be registered on the context's dialect
  // registry; that's `mlir::registerAllToLLVMIRTranslations`, wired up by
  // every entry point that runs this pass (hc-opt main + any future
  // Python driver).
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

  // Dump the pre-optimization LLVM module first so it survives an
  // optimizer crash — historically the most informative artifact when
  // the backend miscompiles, since it shows what the optimizer was
  // handed before fold-the-world set in.
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
