# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS HCTransformOps.td)
mlir_tablegen(HCTransformOps.h.inc -gen-op-decls)
mlir_tablegen(HCTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRHCTransformOpsIncGen)

add_mlir_doc(HCTransformOps HCTransformOps hc/ -gen-op-doc -dialect=transform)
