// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_C_HC_H
#define HC_C_HC_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HC, hc);

// Register hc's three pass families (front transforms, front→hc conversion,
// hc transforms). Idempotent. Does NOT register upstream MLIR passes — Python
// bindings do that via `_mlirRegisterEverything`; non-Python callers must
// call `mlir::registerAllPasses()` themselves first.
MLIR_CAPI_EXPORTED void mlirRegisterHCAllPasses(void);

// Append the `transform.hc.*` extension. Required before parsing transform
// libraries that use those ops.
MLIR_CAPI_EXPORTED void
mlirRegisterHCTransformDialectExtension(MlirDialectRegistry registry);

// Force-load `dlti` alongside HC. Avoids the multi-threaded dialect-load
// trip when `gpu-to-llvm` queries `dlti.dl_spec` from inside
// `transform-interpreter`.
MLIR_CAPI_EXPORTED void
mlirRegisterHCDependentDialectExtensions(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // HC_C_HC_H
