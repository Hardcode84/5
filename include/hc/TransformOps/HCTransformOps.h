// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMOPS_HCTRANSFORMOPS_H
#define HC_TRANSFORMOPS_HCTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "hc/TransformOps/HCTransformOps.h.inc"

namespace mlir::hc::transform {

void registerTransformDialectExtension(::mlir::DialectRegistry &registry);

} // namespace mlir::hc::transform

#endif // HC_TRANSFORMOPS_HCTRANSFORMOPS_H
