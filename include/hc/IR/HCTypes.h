// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPES_H
#define HC_IR_HCTYPES_H

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCTypesInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "hc/IR/HCTypes.h.inc"

namespace mlir::hc {

IdxType getUnpinnedIdxType(MLIRContext *ctx);
PredType getUnpinnedPredType(MLIRContext *ctx);

/// Returns true when `type` is HC's erased refinement placeholder.
bool isHCUndefType(Type type);

/// Join two concrete HC type facts, recursively joining tuple elements.
/// Returns null when the facts are incompatible.
Type joinHCTypes(Type lhs, Type rhs);

/// Compatibility used for progressive signatures: exact match, `!hc.undef`,
/// or recursive tuple compatibility.
bool areHCProgressiveTypesCompatible(Type source, Type dest);

/// Compatibility used across region/control-flow edges. This includes
/// progressive compatibility plus recursive `HCJoinableTypeInterface` joins.
bool areHCBranchTypesCompatible(Type source, Type dest);

/// Returns true when replacing `current` with `inferred` is a monotonic HC type
/// refinement.
bool shouldRefineHCType(Type current, Type inferred);

} // namespace mlir::hc

#endif // HC_IR_HCTYPES_H
