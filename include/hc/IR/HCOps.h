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

/// Return the full parameter list with const-only kwargs removed. The
/// remaining names are the SSA operand slots of `hc.call_intrinsic`.
ArrayAttr filterIntrinsicOperandParameters(ArrayAttr parameters,
                                           ArrayAttr constKwargs);

/// Build the runtime operand signature for an intrinsic whose operands all
/// currently share one staging type, usually `!hc.undef` before inference.
FunctionType getIntrinsicOperandFunctionType(ArrayAttr parameters,
                                             ArrayAttr constKwargs,
                                             TypeRange resultTypes,
                                             Type uniformOperandType);

/// Match the post-flatten collective-lift relationship between a region's
/// yielded value type and its result type. Both must have collapsed to the
/// 1D bare carrier `hc-flatten-with-layouts` produces; the lift survives as
/// the ixsimpl identity `result_storage == yield_storage * product(suffix)`.
/// `TupleType` recurses element-wise; collective scalar yields synthesise a
/// `yield_storage = 1`. Returns false on shape, kind, or element-type
/// disagreement, and on pre-flatten rank-N inputs.
bool postFlattenLiftMatches(Type yieldedType, Type resultType,
                            ArrayRef<Attribute> suffix);

} // namespace mlir::hc

#define GET_OP_CLASSES
#include "hc/IR/HCOps.h.inc"

#endif // HC_IR_HCOPS_H
