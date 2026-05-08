// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Thin C++ wrapper around `llvm::orc::LLJIT`. We don't reach for
// `mlir::ExecutionEngine` because it bakes in MLIR pass pipelines and a
// fixed set of dialect translations; the host wrapper hands us already-
// lowered IR (LLVM dialect text later, raw LLVM IR text in v0) and we
// only want the JIT plumbing — symbol resolution against a per-process
// process-symbol generator plus an explicit caller-supplied symbol map.
//
// Per-module LLVMContext ownership: each load creates a fresh
// `LLVMContext` and hands it to `ThreadSafeModule`. The TSM keeps the
// context alive as long as ORC needs it; releasing the module also
// disposes of the context. This avoids the "context outlives the JIT"
// destruction footgun and lets concurrent loads use independent
// contexts.

#ifndef HC_RUNTIME_EXECUTIONENGINE_H
#define HC_RUNTIME_EXECUTIONENGINE_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>
#include <optional>

namespace llvm::orc {
class LLJIT;
class MangleAndInterner;
} // namespace llvm::orc

namespace hc {

struct ExecutionEngineOptions {
  // Code-gen optimization level for the JIT target machine. Defaulting
  // to "unset" lets `JITTargetMachineBuilder::detectHost` pick.
  std::optional<llvm::CodeGenOptLevel> jitCodeGenOptLevel;

  // Symbols to inject into every loaded module's JITDylib. The callback
  // form (rather than a precomputed `SymbolMap`) is what wave uses and
  // it sidesteps an awkward dependency: `MangleAndInterner` needs the
  // JIT's data layout, which we don't have until after `LLJIT::Create`.
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;
};

class ExecutionEngine {
public:
  using ModuleHandle = void *;

  explicit ExecutionEngine(const ExecutionEngineOptions &options);
  ~ExecutionEngine();

  ExecutionEngine(const ExecutionEngine &) = delete;
  ExecutionEngine &operator=(const ExecutionEngine &) = delete;

  // Parse `text` as LLVM IR and add it to the JIT. Each call gets a
  // fresh `JITDylib` so loaded modules are isolated and individually
  // releasable. The returned handle is opaque (it points at the dylib
  // entry inside the JIT) and must only be passed back to `lookup` /
  // `releaseModule`.
  llvm::Expected<ModuleHandle> loadLLVMIR(llvm::StringRef text);

  // Tear down a module's JITDylib. After this returns the handle is
  // invalid; lookups against it are UB.
  void releaseModule(ModuleHandle handle);

  // Resolve `name` inside `handle`'s dylib. Returns the function/object
  // address as a `void*`; ctypes / cffi / our host wrapper reinterprets
  // that to a callable. The returned error is rewrapped into a
  // `StringError` because ORC's native errors hold references into
  // internal string tables that go away when `Error::log` is called
  // late (or in a different thread); see wave's notes for details.
  llvm::Expected<void *> lookup(ModuleHandle handle,
                                llvm::StringRef name) const;

private:
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;
  // Each new module gets a unique JITDylib name like `hc_module0`,
  // `hc_module1` ... so we can ship multiple compiled host wrappers
  // concurrently without ORC complaining about duplicate dylibs.
  int uniqueNameCounter = 0;
};

} // namespace hc

#endif // HC_RUNTIME_EXECUTIONENGINE_H
