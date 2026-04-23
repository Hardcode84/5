// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCTypes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir::hc;

#define GET_TYPEDEF_CLASSES
#include "hc/IR/HCTypes.cpp.inc"

void HCDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/IR/HCTypes.cpp.inc"
      >();
}

mlir::LogicalResult
mlir::hc::BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

mlir::LogicalResult
mlir::hc::TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}
