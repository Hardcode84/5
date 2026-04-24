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

// Dialect handle for the `hc` semantic dialect. Load it onto an MLIR
// context alongside `hc_front` before parsing modules that have already
// been lowered past `-convert-hc-front-to-hc`.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HC, hc);

// Register hc's own three pass families (hc-front transforms, hc-front→hc
// conversion, hc transforms) into the process-wide pass registry.
// Idempotent — repeated calls from the same process are safe.
//
// Upstream MLIR passes (transform-interpreter, canonicalize, ...) are NOT
// registered here: the Python bindings already do that via
// `_mlirRegisterEverything`'s site initialization, and re-registering
// would double-register pipelines and abort. Callers driving this from
// outside a Python process should instead call `mlir::registerAllPasses()`
// themselves once before `mlirRegisterHCAllPasses`.
MLIR_CAPI_EXPORTED void mlirRegisterHCAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // HC_C_HC_H
