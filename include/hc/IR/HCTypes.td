// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPES
#define HC_IR_HCTYPES

include "hc/IR/HCDialect.td"
include "hc/IR/HCAttrs.td"
include "mlir/IR/AttrTypeBase.td"

class HC_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HC_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def HC_BufferType : HC_Type<"Buffer", "buffer"> {
  let summary = "Semantic buffer type";
  let description = [{
    Initial semantic buffer type carrying an element type and a shape
    attribute. The shape is intentionally modeled as an attribute payload so the
    symbolic surface can evolve independently from the type shell.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    AttrParameter<"::mlir::Attribute", "shape attribute">:$shape
  );
  let assemblyFormat = "`<` $elementType `,` $shape `>`";
}

def HC_TensorType : HC_Type<"Tensor", "tensor"> {
  let summary = "Semantic tensor type";
  let description = [{
    Initial semantic tensor type carrying an element type and a symbolic shape
    attribute.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    AttrParameter<"::mlir::Attribute", "shape attribute">:$shape
  );
  let assemblyFormat = "`<` $elementType `,` $shape `>`";
}

#endif // HC_IR_HCTYPES
