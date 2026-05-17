// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPES_H
#define HC_IR_HCTYPES_H

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCTypesInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

#define GET_TYPEDEF_CLASSES
#include "hc/IR/HCTypes.h.inc"

namespace mlir::hc {

IdxType getUnpinnedIdxType(MLIRContext *ctx);
PredType getUnpinnedPredType(MLIRContext *ctx);

enum class LaunchGeoMethod {
  GroupId,
  LocalId,
  SubgroupId,
  GroupShape,
  WorkOffset,
  WorkShape,
  GroupSize,
  WaveSize,
};

enum class LaunchGeoArity {
  MultiAxis,
  Scalar,
};

enum class LaunchGeoRankDomain {
  WorkGridWithGroupFallback,
  WorkGrid,
  Workgroup,
  Scalar,
};

struct LaunchGeoMethodInfo {
  LaunchGeoMethod method;
  llvm::StringRef name;
  llvm::StringRef symbolPrefix;
  LaunchGeoArity arity;
  LaunchGeoRankDomain rankDomain;

  bool isScalar() const { return arity == LaunchGeoArity::Scalar; }
};

LaunchGeoMethodInfo getLaunchGeoMethodInfo(LaunchGeoMethod method);
std::optional<LaunchGeoMethodInfo>
classifyLaunchGeoMethod(llvm::StringRef method);

struct LaunchContextMetadata {
  ShapeAttr workShape;
  ShapeAttr groupShape;
  ExprAttr subgroupSize;
};

std::optional<LaunchContextMetadata> getLaunchContextMetadata(Type contextType);

/// True when `type` is HC's erased refinement placeholder.
bool isHCUndefType(Type type);

/// Static shape from tuple of pinned `!hc.idx`. Null when absent,
/// erased, non-tuple, or contains a dynamic/non-index dim.
ShapeAttr getStaticShapeFromTupleType(Type shapeType);

/// Diagnostic form of `getStaticShapeFromTupleType`.
FailureOr<ShapeAttr> verifyStaticShapeFromTupleType(Type shapeType,
                                                    Operation *diagOp);

/// Join two concrete HC type facts, recursing through tuples. Null on
/// incompatibility.
Type joinHCTypes(Type lhs, Type rhs);

/// Progressive-signature compat: exact, `!hc.undef`, or recursive
/// tuple compat.
bool areHCProgressiveTypesCompatible(Type source, Type dest);

/// Region / control-flow edge compat: progressive + recursive
/// `HCJoinableTypeInterface` joins.
bool areHCBranchTypesCompatible(Type source, Type dest);

/// True when replacing `current` with `inferred` is monotonic
/// refinement.
bool shouldRefineHCType(Type current, Type inferred);

} // namespace mlir::hc

#endif // HC_IR_HCTYPES_H
