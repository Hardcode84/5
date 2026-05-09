// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Two ingestion paths share one backend:
//   `loadLLVMIR(text)` → parseAssembly → addLLVMModule
//   `loadMLIR(text)`   → parseSourceString + translateModuleToLLVMIR →
//   addLLVMModule
// The MLIR path is what the post-pipeline host wrapper goes through;
// the LLVM-text path stays around for hand-written IR experiments and
// tests that don't want the dialect-translation cost.
//
// The compiler creator wires in `mlir::makeOptimizingTransformer` even
// though the LLVM-IR ingestion path doesn't go through MLIR otherwise.
// It's just the convenient way to reach LLVM's standard
// optimize-then-codegen pipeline at a chosen opt level; we depend on
// `MLIRExecutionEngineUtils` for that helper alone.

#include "hc/Runtime/ExecutionEngine.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace {

// Run host-target initializers exactly once. The JIT needs the native
// target's codegen + asm printer + asm parser registered before any
// compile happens; doing it lazily here keeps callers from having to
// remember it.
void initializeNativeTargetOnce() {
  static const bool done = []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
  }();
  (void)done;
}

llvm::Error makeStringError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

// Minimum dialect set the post-pipeline host wrapper needs: LLVM (for
// the body) and the builtin/LLVM translations to lower the parsed
// `ModuleOp` to `llvm::Module`. We deliberately stop there rather than
// `registerAllDialects` — the lowering pipeline guarantees the IR is
// pure LLVM dialect by the time it reaches us, and shrinking the
// dialect surface keeps the wheel size honest. If a future pipeline
// stage starts emitting something else, the parser will surface a
// "unregistered dialect" diagnostic that points right at the gap.
std::unique_ptr<mlir::MLIRContext> makeMLIRContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  // Allow unknown attributes (e.g. `gpu.container_module` left on the
  // module by upstream passes) to pass through the parser without us
  // having to load the corresponding dialect — the attribute is just a
  // marker, the translation step ignores it.
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->allowUnregisteredDialects(true);
  return context;
}

// Apply LLVM's standard optimize-then-emit pipeline at the JTM's chosen
// opt level. This mirrors `mlir::ExecutionEngine`'s default plumbing
// without dragging in the rest of `mlir::ExecutionEngine`.
class HcCompiler : public llvm::orc::SimpleCompiler {
public:
  explicit HcCompiler(std::unique_ptr<llvm::TargetMachine> tm)
      : SimpleCompiler(*tm), optimizer(mlir::makeOptimizingTransformer(
                                 static_cast<unsigned>(tm->getOptLevel()),
                                 /*sizeLevel=*/0, tm.get())),
        targetMachine(std::move(tm)) {}

  llvm::Expected<CompileResult> operator()(llvm::Module &m) override {
    m.setDataLayout(targetMachine->createDataLayout());
    m.setTargetTriple(targetMachine->getTargetTriple());
    if (auto err = optimizer(&m))
      return std::move(err);
    return llvm::orc::SimpleCompiler::operator()(m);
  }

private:
  std::function<llvm::Error(llvm::Module *)> optimizer;
  std::shared_ptr<llvm::TargetMachine> targetMachine;
};

} // namespace

namespace hc {

ExecutionEngine::ExecutionEngine(const ExecutionEngineOptions &options)
    : symbolMap(options.symbolMap) {
  initializeNativeTargetOnce();

  auto tmBuilder =
      llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());

  // SectionMemoryManager is the conservative choice — slower than JITLink
  // but bulletproof on every host LLVM supports. We can switch later if
  // measurements ever flag the manager as a bottleneck.
  auto objectLinkingLayerCreator = [](llvm::orc::ExecutionSession &session)
      -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
    auto getMemoryManager = [](const llvm::MemoryBuffer &) {
      return std::make_unique<llvm::SectionMemoryManager>();
    };
    return std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        session, getMemoryManager);
  };

  auto compileFunctionCreator = [optLevel = options.jitCodeGenOptLevel](
                                    llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    if (optLevel)
      jtmb.setCodeGenOptLevel(*optLevel);
    auto tm = jtmb.createTargetMachine();
    if (!tm)
      return tm.takeError();
    return std::make_unique<HcCompiler>(std::move(*tm));
  };

  jit = llvm::cantFail(
      llvm::orc::LLJITBuilder()
          .setCompileFunctionCreator(compileFunctionCreator)
          .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
          .setJITTargetMachineBuilder(tmBuilder)
          .create());
}

ExecutionEngine::~ExecutionEngine() = default;

llvm::Expected<ExecutionEngine::ModuleHandle>
ExecutionEngine::loadLLVMIR(llvm::StringRef text) {
  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic diag;
  auto memoryBuffer = llvm::MemoryBuffer::getMemBuffer(
      text, "hc-jit-input", /*RequiresNullTerminator=*/false);
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssembly(memoryBuffer->getMemBufferRef(), diag, *context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream os(message);
    diag.print("hc-jit", os);
    return makeStringError("failed to parse LLVM IR: " + os.str());
  }
  return addLLVMModule(std::move(module), std::move(context));
}

llvm::Expected<ExecutionEngine::ModuleHandle>
ExecutionEngine::loadMLIR(llvm::StringRef text) {
  if (!mlirContext)
    mlirContext = makeMLIRContext();

  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::parseSourceString<mlir::ModuleOp>(text, mlirContext.get());
  if (!moduleOp)
    return makeStringError("failed to parse MLIR text into a ModuleOp");

  // Translation needs its own `LLVMContext` so the resulting
  // `llvm::Module` has the same lifetime story as the LLVM-IR ingestion
  // path — owned by the `ThreadSafeModule` and released when the JIT
  // dylib is torn down.
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*moduleOp, *llvmContext);
  if (!llvmModule)
    return makeStringError(
        "failed to translate MLIR module to LLVM IR (check that the input is "
        "in LLVM dialect or any other dialect with a registered translation)");

  return addLLVMModule(std::move(llvmModule), std::move(llvmContext));
}

llvm::Expected<ExecutionEngine::ModuleHandle>
ExecutionEngine::addLLVMModule(std::unique_ptr<llvm::Module> module,
                               std::unique_ptr<llvm::LLVMContext> context) {
  // Fresh dylib per module so we can release them independently. The
  // counter is monotonic — we never reuse names because deleting a
  // dylib doesn't immediately free the name on the ExecutionSession
  // side.
  std::string dylibName;
  llvm::orc::JITDylib *dylib = nullptr;
  while (true) {
    dylibName = ("hc_module" + llvm::Twine(uniqueNameCounter++)).str();
    if (jit->getJITDylibByName(dylibName))
      continue;
    auto created = jit->createJITDylib(dylibName);
    if (!created)
      return created.takeError();
    dylib = &created.get();
    break;
  }

  // Process-wide symbol resolution lets the JIT find anything currently
  // visible in the host process (libc, the runtime helpers the caller
  // dlopens before invoke, etc.) without us having to enumerate them.
  auto dataLayout = jit->getDataLayout();
  dylib->addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  if (symbolMap) {
    llvm::orc::MangleAndInterner mangle(dylib->getExecutionSession(),
                                        dataLayout);
    if (auto err = dylib->define(absoluteSymbols(symbolMap(mangle))))
      return std::move(err);
  }

  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
  if (auto err = jit->addIRModule(*dylib, std::move(tsm)))
    return std::move(err);

  if (auto err = jit->initialize(*dylib))
    return std::move(err);

  return static_cast<ModuleHandle>(dylib);
}

void ExecutionEngine::releaseModule(ModuleHandle handle) {
  auto *dylib = static_cast<llvm::orc::JITDylib *>(handle);
  llvm::cantFail(jit->deinitialize(*dylib));
  llvm::cantFail(jit->getExecutionSession().removeJITDylib(*dylib));
}

llvm::Expected<void *> ExecutionEngine::lookup(ModuleHandle handle,
                                               llvm::StringRef name) const {
  auto *dylib = static_cast<llvm::orc::JITDylib *>(handle);
  auto sym = jit->lookup(*dylib, name);
  if (!sym) {
    // The error returned by ORC may reference internal string tables
    // that get freed when later code calls `consumeError` (or similar).
    // Eagerly serialize the message into a fresh StringError so the
    // caller can route it across the C++/Python boundary safely.
    std::string message;
    llvm::raw_string_ostream os(message);
    llvm::handleAllErrors(sym.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return makeStringError("lookup '" + name + "' failed: " + os.str());
  }
  if (void *fptr = sym->toPtr<void *>())
    return fptr;
  return makeStringError("lookup '" + name + "' returned a null address");
}

} // namespace hc
