// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPESINTERFACES
#define HC_IR_HCTYPESINTERFACES

include "mlir/IR/OpBase.td"

def HCJoinableTypeInterface : TypeInterface<"HCJoinableTypeInterface"> {
  let description = [{
    Type-local join hook for HC's progressive type lattice.

    The hook returns the conservative common type for two distinct concrete
    facts, or null when this type has no opinion. It is used by dataflow
    analyses that need a sound over-approximation without hard-coding every
    HC type pair in the analysis lattice.
  }];
  let cppNamespace = "::mlir::hc";
  let methods = [
    InterfaceMethod<
      "Return a conservative common type with `other`, or null.",
      "::mlir::Type",
      "joinHCType",
      (ins "::mlir::Type":$other)
    >
  ];
}

#endif // HC_IR_HCTYPESINTERFACES
