// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPESINTERFACES
#define HC_IR_HCTYPESINTERFACES

include "mlir/IR/OpBase.td"

def HCJoinableTypeInterface : TypeInterface<"HCJoinableTypeInterface"> {
  let description = [{
    Join hook for HC's progressive type lattice.

    Returns the conservative common type of two concrete facts, or null
    when the type has no opinion. Used by dataflow analyses that need a
    sound over-approximation without enumerating every HC type pair.
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

def SymbolicallyShapedTypeInterface
    : TypeInterface<"SymbolicallyShapedTypeInterface"> {
  let description = [{
    Symbolic-shape hook for HC shaped types.

    Consumers wanting just element type or rank/dim expressions depend on
    this interface instead of enumerating buffer/tensor/vector shells.

    Layout accessors are nullable: null `LayoutAttr` = no explicit slot
    (identity-layout contract). The clone hook is the canonical
    "rebuild this type with a different layout" entry — passing null
    strips the slot, which is how `hc-canonicalize-layouts` collapses
    explicit identity layouts to absent.
  }];
  let cppNamespace = "::mlir::hc";
  let methods = [
    InterfaceMethod<
      "Return the symbolic shape carried by this type.",
      "::mlir::hc::ShapeAttr",
      "getSymbolicShape"
    >,
    InterfaceMethod<
      "Return the element type carried by this shaped type.",
      "::mlir::Type",
      "getSymbolicElementType"
    >,
    InterfaceMethod<
      "Return the explicit layout carried by this type, or null if absent.",
      "::mlir::hc::LayoutAttr",
      "getSymbolicLayout"
    >,
    InterfaceMethod<
      "Return a copy of this type with the given layout (null = strip).",
      "::mlir::Type",
      "cloneWithSymbolicLayout",
      (ins "::mlir::hc::LayoutAttr":$layout)
    >,
    InterfaceMethod<
      "Return a copy of this type with the given symbolic shape; the "
      "existing element type and layout slot are preserved.",
      "::mlir::Type",
      "cloneWithSymbolicShape",
      (ins "::mlir::hc::ShapeAttr":$shape)
    >
  ];
}

#endif // HC_IR_HCTYPESINTERFACES
