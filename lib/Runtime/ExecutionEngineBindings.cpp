// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Nanobind module exposing `hc::ExecutionEngine` to Python. The whole
// module is one translation unit because we want a single "C++ surface"
// the linker can hide everything-but-`PyInit_*` from via the version
// script — keeping LLVM internals from leaking into other extensions
// loaded in the same process.

#include "hc/Runtime/ExecutionEngine.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace {

// Translate `llvm::Expected<T>` to a Python-friendly exception. We
// pre-format the error into a string before constructing the exception
// because the LLVM error chain holds references into ORC-internal
// storage that may go out of scope by the time Python's exception
// formatter runs (see ExecutionEngine::lookup for the long-form
// explanation).
template <typename T>
static T unwrapExpected(llvm::Expected<T> expected, const char *context) {
  if (expected)
    return std::move(*expected);
  std::string message;
  llvm::raw_string_ostream os(message);
  llvm::logAllUnhandledErrors(expected.takeError(), os);
  throw std::runtime_error(std::string(context) + ": " + os.str());
}

} // namespace

NB_MODULE(hc_execution_engine, m) {
  m.doc() =
      "LLVM ORC LLJIT bindings for hc — JITs LLVM IR text emitted by the "
      "compiler pipeline and resolves caller-supplied symbols against the "
      "loaded modules.";

  nb::enum_<llvm::CodeGenOptLevel>(m, "CodeGenOptLevel")
      .value("O0", llvm::CodeGenOptLevel::None)
      .value("O1", llvm::CodeGenOptLevel::Less)
      .value("O2", llvm::CodeGenOptLevel::Default)
      .value("O3", llvm::CodeGenOptLevel::Aggressive)
      .export_values();

  nb::class_<hc::ExecutionEngineOptions>(m, "ExecutionEngineOptions")
      .def(nb::init<>())
      .def_rw("jit_code_gen_opt_level",
              &hc::ExecutionEngineOptions::jitCodeGenOptLevel,
              "JIT codegen optimization level; None lets the host detector "
              "pick.")
      .def(
          "set_symbol_map",
          [](hc::ExecutionEngineOptions &self,
             const std::map<std::string, std::uintptr_t> &symbols) {
            // Snapshot the dict by value so the captured lambda doesn't
            // dangle when the caller reuses or mutates the source dict.
            self.symbolMap = [symbols](llvm::orc::MangleAndInterner mangle) {
              llvm::orc::SymbolMap result;
              for (const auto &[name, address] : symbols) {
                auto flags = llvm::JITSymbolFlags::Exported |
                             llvm::JITSymbolFlags::Callable;
                result[mangle(name)] = llvm::orc::ExecutorSymbolDef(
                    llvm::orc::ExecutorAddr(address), flags);
              }
              return result;
            };
          },
          nb::arg("symbols"),
          "Register `{name: address}` pairs to be resolvable from any "
          "module loaded into this engine. Addresses are integers (e.g. "
          "ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value).");

  nb::class_<hc::ExecutionEngine>(m, "ExecutionEngine")
      .def(nb::init<const hc::ExecutionEngineOptions &>(), nb::arg("options"))
      .def(
          "load_llvm_ir",
          [](hc::ExecutionEngine &self, const std::string &text) {
            auto handle = unwrapExpected(self.loadLLVMIR(text),
                                         "ExecutionEngine.load_llvm_ir");
            return reinterpret_cast<std::uintptr_t>(handle);
          },
          nb::arg("text"),
          "Parse LLVM IR text, JIT-compile it, and return an opaque "
          "module handle (an int).")
      .def(
          "load_mlir",
          [](hc::ExecutionEngine &self, const std::string &text) {
            auto handle = unwrapExpected(self.loadMLIR(text),
                                         "ExecutionEngine.load_mlir");
            return reinterpret_cast<std::uintptr_t>(handle);
          },
          nb::arg("text"),
          "Parse MLIR LLVM-dialect text, translate to LLVM IR, "
          "JIT-compile it, and return an opaque module handle (an int). "
          "The translation registry is set up for the builtin and LLVM "
          "dialects; other dialect attributes pass through opaquely.")
      .def(
          "lookup",
          [](const hc::ExecutionEngine &self, std::uintptr_t handle,
             const std::string &name) {
            auto ptr = unwrapExpected(
                self.lookup(reinterpret_cast<void *>(handle), name),
                "ExecutionEngine.lookup");
            return reinterpret_cast<std::uintptr_t>(ptr);
          },
          nb::arg("handle"), nb::arg("name"),
          "Return the address of `name` inside the module identified by "
          "`handle`. Wrap the result with ctypes.CFUNCTYPE to call.")
      .def(
          "release_module",
          [](hc::ExecutionEngine &self, std::uintptr_t handle) {
            self.releaseModule(reinterpret_cast<void *>(handle));
          },
          nb::arg("handle"),
          "Tear down the JITDylib backing `handle`. After this returns "
          "the handle is invalid.");
}
