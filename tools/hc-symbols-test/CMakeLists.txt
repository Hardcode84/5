# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(hc-symbols-test hc-symbols-test.cpp)
llvm_update_compile_flags(hc-symbols-test)
target_link_libraries(
  hc-symbols-test
  PRIVATE
  MLIRHCDialect
)
mlir_check_all_link_libraries(hc-symbols-test)
