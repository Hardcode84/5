// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Thin C++ wrapper over `llvm::orc::LLJIT`. Avoids `mlir::ExecutionEngine`
// because it rewrites every public function with a packed-args wrapper --
// we want one ctypes thunk per launch with no `void**` repacking.
// Per-module `LLVMContext` lives inside `ThreadSafeModule`, so releasing
// a module also disposes of its context.

#ifndef HC_RUNTIME_EXECUTIONENGINE_H
#define HC_RUNTIME_EXECUTIONENGINE_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>
#include <optional>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace llvm::orc {
class LLJIT;
class MangleAndInterner;
} // namespace llvm::orc

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace hc {

struct ExecutionEngineOptions {
  // Unset -> `JITTargetMachineBuilder::detectHost` picks.
  std::optional<llvm::CodeGenOptLevel> jitCodeGenOptLevel;

  // Callback over precomputed `SymbolMap`: `MangleAndInterner` needs the
  // JIT's data layout, which isn't available until after `LLJIT::Create`.
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;
};

class ExecutionEngine {
public:
  using ModuleHandle = void *;

  explicit ExecutionEngine(const ExecutionEngineOptions &options);
  ~ExecutionEngine();

  ExecutionEngine(const ExecutionEngine &) = delete;
  ExecutionEngine &operator=(const ExecutionEngine &) = delete;

  /// Parse LLVM IR text and add to the JIT in a fresh `JITDylib`.
  /// Handle is opaque; pass back only to `lookup` / `releaseModule`.
  llvm::Expected<ModuleHandle> loadLLVMIR(llvm::StringRef text);

  /// Parse MLIR (LLVM dialect or anything translatable to LLVM IR),
  /// translate, then add. Same handle semantics as `loadLLVMIR`. The
  /// MLIR context is created lazily and reused across calls.
  llvm::Expected<ModuleHandle> loadMLIR(llvm::StringRef text);

  /// Tear down the module's JITDylib. Handle is invalid after return.
  void releaseModule(ModuleHandle handle);

  /// Resolve `name` inside `handle`'s dylib. ORC's native error refs
  /// internal string tables that vanish under late `Error::log`, so the
  /// error is rewrapped as `StringError`.
  llvm::Expected<void *> lookup(ModuleHandle handle,
                                llvm::StringRef name) const;

private:
  // Wrap an already-built module + its context into a `ThreadSafeModule`
  // and add it to a fresh JITDylib with configured symbol map +
  // process-symbol resolution.
  llvm::Expected<ModuleHandle>
  addLLVMModule(std::unique_ptr<llvm::Module> module,
                std::unique_ptr<llvm::LLVMContext> context);

  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;
  // Lazy; parsing is read-only on it, so cross-load sharing is safe.
  std::unique_ptr<mlir::MLIRContext> mlirContext;
  // Per-module JITDylib name suffix (`hc_module0`, `hc_module1`, ...).
  int uniqueNameCounter = 0;
};

} // namespace hc

#endif // HC_RUNTIME_EXECUTIONENGINE_H
