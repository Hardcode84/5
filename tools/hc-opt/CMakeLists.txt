# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(hc-opt hc-opt.cpp)
llvm_update_compile_flags(hc-opt)
target_link_libraries(
  hc-opt
  PRIVATE
  MLIRHCFrontDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)
mlir_check_all_link_libraries(hc-opt)

install(TARGETS hc-opt RUNTIME DESTINATION bin)
