// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_IR_HCFRONTOPS_H
#define HC_FRONT_IR_HCFRONTOPS_H

#include "hc/Front/IR/HCFrontTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "hc/Front/IR/HCFrontOps.h.inc"

#endif // HC_FRONT_IR_HCFRONTOPS_H
