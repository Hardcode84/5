// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_C_FRONT_H
#define HC_C_FRONT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HC, hc);

MLIR_CAPI_EXPORTED MlirType mlirHCFrontValueTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirHCTypeIsAFrontValueType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirHCFrontValueTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirHCFrontTypeExprTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirHCTypeIsAFrontTypeExprType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirHCFrontTypeExprTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // HC_C_FRONT_H
