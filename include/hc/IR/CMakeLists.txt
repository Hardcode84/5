# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect(HCOps hc)

set(LLVM_TARGET_DEFINITIONS HCAttrs.td)
mlir_tablegen(HCEnums.h.inc -gen-enum-decls)
mlir_tablegen(HCEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HCAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hc)
mlir_tablegen(HCAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hc)
add_public_tablegen_target(MLIRHCAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS HCTypes.td)
mlir_tablegen(HCTypes.h.inc -gen-typedef-decls -typedefs-dialect=hc)
mlir_tablegen(HCTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hc)
add_public_tablegen_target(MLIRHCTypesIncGen)

add_mlir_doc(HCDialect HCDialect hc/ -gen-dialect-doc)
add_mlir_doc(HCOps HCOps hc/ -gen-op-doc)
