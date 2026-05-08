// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implementation lives behind a tiny surface so we can grow it without
// disturbing callers. v0 supports raw LLVM IR text only; the MLIR-text
// path (parse + translate + load) follows once the host wrapper lands —
// keeping it out of v0 lets the engine link without MLIR libs at all.
//
// The compiler creator wires in `mlir::makeOptimizingTransformer` even
// though we don't go through MLIR otherwise. It's just the convenient
// path to the standard LLVM optimize-then-codegen pipeline at the
// requested opt level; we depend on `MLIRExecutionEngineUtils` for that
// helper alone.

#include "hc/Runtime/ExecutionEngine.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "mlir/ExecutionEngine/OptUtils.h"

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
  // Fresh context per load — TSM owns it, so it gets dropped when the
  // module is released and concurrent loads don't contend on a shared
  // context. Parsing has to happen against this owned context too, so
  // the resulting Module's reference is valid for the TSM's lifetime.
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

  // Each module gets its own JITDylib so we can release them
  // independently. The counter is monotonic — we never reuse names
  // because deleting a dylib doesn't immediately free the name on the
  // ExecutionSession side.
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
  // visible in the host process (libc, the runtime helpers we'll dlsym
  // before construction, etc.) without us having to enumerate them.
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
