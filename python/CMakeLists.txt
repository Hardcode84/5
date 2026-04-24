# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

declare_mlir_python_sources(HCPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HCPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hc_mlir"
  TD_FILE dialects/HCFrontOps.td
  SOURCES
    dialects/hc_front.py
  DIALECT_NAME hc_front
)

# `hc` dialect has no Python-side op tablegen yet — we only need the Python
# shim that re-exports `register_dialects` / `register_passes` bound by the
# nanobind extension below. `declare_mlir_python_sources` drops the file
# into the package layout without requiring a `.td` input.
declare_mlir_python_sources(HCPythonSources.HCDialect
  ADD_TO_PARENT HCPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hc_mlir"
  SOURCES
    dialects/hc.py
)

declare_mlir_python_extension(HCPythonSources.NanobindExtension
  MODULE_NAME _hcFrontDialectsNanobind
  ADD_TO_PARENT HCPythonSources
  SOURCES
    HCFrontExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    HCFrontCAPI
    HCCAPI
)

add_mlir_python_common_capi_library(HCPythonCAPI
  INSTALL_COMPONENT HCPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    HCPythonSources
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
)

set(HCPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

add_mlir_python_modules(HCPythonModules
  ROOT_PREFIX "${HCPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    HCPythonSources
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    HCPythonCAPI
)
