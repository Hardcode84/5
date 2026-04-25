# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS HCOpsInterfaces.td)
mlir_tablegen(HCOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HCOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRHCOpsInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS HCTypesInterfaces.td)
mlir_tablegen(HCTypesInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(HCTypesInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRHCTypesInterfacesIncGen)

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
