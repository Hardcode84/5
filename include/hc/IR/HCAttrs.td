// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCATTRS
#define HC_IR_HCATTRS

include "hc/IR/HCDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"

class HC_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HC_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def HC_ExprAttr : HC_Attr<"Expr", "expr"> {
  let summary = "Symbolic expression attribute";
  let description = [{
    Initial attribute for semantic symbolic expressions. The current
    implementation stores textual payload so the IR surface can stabilize before
    the direct `ixsimpl` handle-backed implementation lands.
  }];
  let parameters = (ins StringRefParameter<"expression text">:$text);
  let assemblyFormat = "`<` $text `>`";
}

def HC_PredAttr : HC_Attr<"Pred", "pred"> {
  let summary = "Symbolic predicate attribute";
  let description = [{
    Initial attribute for semantic symbolic predicates. As with `#hc.expr`, the
    current implementation keeps textual payload for now and is intended to be
    replaced by direct `ixsimpl` handles in a follow-on change.
  }];
  let parameters = (ins StringRefParameter<"predicate text">:$text);
  let assemblyFormat = "`<` $text `>`";
}

def HC_ShapeAttr : HC_Attr<"Shape", "shape"> {
  let summary = "Shape tuple built from expression-like attributes";
  let parameters = (ins ArrayRefParameter<"Attribute">:$dims);
  let assemblyFormat = "`<` `[` $dims `]` `>`";
}

def HC_ConstraintSetAttr : HC_Attr<"ConstraintSet", "constraints"> {
  let summary = "Set of predicate-like attributes";
  let parameters = (ins ArrayRefParameter<"Attribute">:$predicates);
  let assemblyFormat = "`<` `[` $predicates `]` `>`";
}

#endif // HC_IR_HCATTRS
