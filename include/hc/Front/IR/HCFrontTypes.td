// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_IR_HCFRONTTYPES
#define HC_FRONT_IR_HCFRONTTYPES

include "hc/Front/IR/HCFrontDialect.td"
include "mlir/IR/AttrTypeBase.td"

class HCFront_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HCFront_Dialect, name, traits> {
  let mnemonic = "front." # typeMnemonic;
}

def HCFront_ValueType : HCFront_Type<"Value", "value"> {
  let summary = "Opaque frontend value placeholder type.";
}

def HCFront_TypeExprType : HCFront_Type<"TypeExpr", "typeexpr"> {
  let summary = "Opaque frontend type-expression placeholder type.";
}

#endif // HC_FRONT_IR_HCFRONTTYPES
