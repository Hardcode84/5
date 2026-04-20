# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect(HCFrontOps hc)
add_mlir_doc(HCFrontDialect HCFrontDialect hc/Front/ -gen-dialect-doc)
add_mlir_doc(HCFrontOps HCFrontOps hc/Front/ -gen-op-doc)
