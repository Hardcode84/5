# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=HCFrontTransforms)
add_public_tablegen_target(MLIRHCFrontTransformsPassIncGen)

add_mlir_doc(Passes HCFrontTransformsPasses hc/Front/Transforms/ -gen-pass-doc)
