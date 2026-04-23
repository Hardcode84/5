// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-c/Front.h"

#include "hc/Front/IR/HCFrontDialect.h"
#include "hc/Front/IR/HCFrontTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "llvm/Support/Casting.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HCFront, hc_front,
                                      mlir::hc::front::HCFrontDialect)

MlirType mlirHCFrontValueTypeGet(MlirContext ctx) {
  return wrap(mlir::hc::front::ValueType::get(unwrap(ctx)));
}

bool mlirHCTypeIsAFrontValueType(MlirType type) {
  return llvm::isa<mlir::hc::front::ValueType>(unwrap(type));
}

MlirTypeID mlirHCFrontValueTypeGetTypeID(void) {
  return wrap(mlir::hc::front::ValueType::getTypeID());
}

MlirType mlirHCFrontTypeExprTypeGet(MlirContext ctx) {
  return wrap(mlir::hc::front::TypeExprType::get(unwrap(ctx)));
}

bool mlirHCTypeIsAFrontTypeExprType(MlirType type) {
  return llvm::isa<mlir::hc::front::TypeExprType>(unwrap(type));
}

MlirTypeID mlirHCFrontTypeExprTypeGetTypeID(void) {
  return wrap(mlir::hc::front::TypeExprType::getTypeID());
}
