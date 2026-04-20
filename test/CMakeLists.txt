# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT EXISTS "${LLVM_DEFAULT_EXTERNAL_LIT}" AND NOT LLVM_EXTERNAL_LIT)
  set(
    LLVM_EXTERNAL_LIT
    "${PROJECT_SOURCE_DIR}/test/llvm-lit.py"
    CACHE STRING "Command used to spawn lit"
  )
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(HC_TEST_DEPENDS
  hc-opt
)

add_lit_testsuite(check-hc "Running the hc regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${HC_TEST_DEPENDS}
)
set_target_properties(check-hc PROPERTIES FOLDER "Tests")

add_lit_testsuites(HC ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${HC_TEST_DEPENDS})
