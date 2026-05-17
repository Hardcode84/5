// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPS_H
#define HC_IR_HCOPS_H

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCOpsInterfaces.h"
#include "hc/IR/HCTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::hc {

/// `parameters` minus const-only kwargs; the residue names the SSA operand
/// slots of `hc.call_intrinsic`.
ArrayAttr filterIntrinsicOperandParameters(ArrayAttr parameters,
                                           ArrayAttr constKwargs);

/// Operand signature for an intrinsic whose operands share one staging type
/// (typically `!hc.undef` pre-inference).
FunctionType getIntrinsicOperandFunctionType(ArrayAttr parameters,
                                             ArrayAttr constKwargs,
                                             TypeRange resultTypes,
                                             Type uniformOperandType);

/// Post-flatten lift relation between yield and result types: both must be
/// the 1D bare carrier with `result_storage == yield_storage *
/// product(suffix)` under ixsimpl. `TupleType` recurses; scalar yields
/// synthesise `yield_storage = 1`. False on shape/kind/elem disagreement
/// or pre-flatten rank-N inputs.
bool postFlattenLiftMatches(Type yieldedType, Type resultType,
                            ArrayRef<Attribute> suffix);

} // namespace mlir::hc

#define GET_OP_CLASSES
#include "hc/IR/HCOps.h.inc"

#endif // HC_IR_HCOPS_H
