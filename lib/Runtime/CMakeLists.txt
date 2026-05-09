## SPDX-FileCopyrightText: 2026 hc contributors
##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Self-contained Python discovery. We use `Python3::Python` (full libpython
# link) rather than `Python3::Module` because the MLIR/LLVM build inherits
# `-Wl,--no-undefined` from `HandleLLVMOptions`; with `Module` the GIL/refcount
# entry points would be left unresolved at link time even though dlopen would
# resolve them at runtime against the loading interpreter. Linking libpython
# costs us no flexibility here — the helpers are only ever loaded into a
# Python process.
find_package(Python3 COMPONENTS Development REQUIRED)

add_library(hc_rt_helpers SHARED BufferUtils.cpp)
target_include_directories(
  hc_rt_helpers
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(hc_rt_helpers PRIVATE Python3::Python)

install(TARGETS hc_rt_helpers LIBRARY DESTINATION lib)

# HIP launcher shim — Linux-only v0. We deliberately do NOT depend on a
# system ROCm install (no `find_package(HIP)`, no `-lamdhip64`); the
# shim dlopen's libamdhip64.so at runtime via `hc_rt_init`. That keeps
# the wheel buildable on hosts without ROCm and importable on hosts
# that have neither ROCm nor a GPU.
add_library(hc_hip_runtime SHARED HipRuntime.cpp)
target_include_directories(
  hc_hip_runtime
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(hc_hip_runtime PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS hc_hip_runtime LIBRARY DESTINATION lib)

# JIT execution engine — nanobind module wrapping `llvm::orc::LLJIT`.
# Built with NB_STATIC + a version script so the embedded LLVM cannot
# collide with another LLVM-using Python extension loaded in the same
# process. -fno-rtti matches LLVM's own build (which is no-rtti per the
# toolchain lock); without it, the linker warns about missing typeinfo.
if(NOT TARGET nanobind-static)
  find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(hc_execution_engine NB_STATIC
  ExecutionEngine.cpp
  ExecutionEngineBindings.cpp
)
target_include_directories(
  hc_execution_engine
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
set_source_files_properties(ExecutionEngine.cpp PROPERTIES
  COMPILE_FLAGS "-fno-rtti"
)
target_link_libraries(hc_execution_engine PRIVATE
  # libpython link — same trick as the helpers/HIP shim, needed because
  # `-Wl,--no-undefined` would otherwise flag the Python C API entry
  # points reached through nanobind. The version script hides everything
  # except `PyInit_*`, so no symbol conflicts with the host interpreter.
  Python3::Python

  LLVM${LLVM_NATIVE_ARCH}AsmParser
  LLVM${LLVM_NATIVE_ARCH}CodeGen
  LLVM${LLVM_NATIVE_ARCH}Desc
  LLVM${LLVM_NATIVE_ARCH}Info
  LLVMAsmParser
  LLVMExecutionEngine
  LLVMOrcJIT
  LLVMTarget

  # MLIR ingestion path: parse LLVM-dialect text, translate to LLVM IR.
  # Kept to the minimum dialect surface the post-pipeline host wrapper
  # needs (LLVM only) so the wheel doesn't drag the rest of MLIR in.
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngineUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRTargetLLVMIRExport
)
target_link_options(hc_execution_engine PRIVATE
  -Wl,--no-undefined
  -Wl,-z,defs
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/hc_execution_engine.lds"
)

# Land the .so under the bundled python_packages tree so the Python
# wrapper can find it via a single sys.path prepend; mirroring how
# hc_mlir is laid out keeps the install layout uniform.
install(TARGETS hc_execution_engine
  LIBRARY DESTINATION python_packages/hc_runtime
)
