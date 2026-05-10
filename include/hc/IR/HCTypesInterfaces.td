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

def SymbolicallyShapedTypeInterface
    : TypeInterface<"SymbolicallyShapedTypeInterface"> {
  let description = [{
    Type-local hook for HC types that carry symbolic shape metadata.

    Consumers that only need element type or rank/dimension expressions should
    depend on this interface instead of enumerating buffer/tensor/vector shells.

    The layout accessors are nullable: a null `LayoutAttr` means the type has
    no explicit layout slot (the v0 identity-row-major contract). The clone
    hook is the canonical "rebuild this type with a different layout" entry
    point — passing a null `LayoutAttr` strips the slot and returns the
    layout-less form, which is how `hc-canonicalize-layouts` collapses
    explicit identity layouts back to absent.
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
