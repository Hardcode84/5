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

def HC_ExprHandleParameter
    : AttrParameter<"::mlir::hc::sym::ExprHandle",
                    "symbolic expression handle"> {
  let parser = "::mlir::hc::sym::parseExprHandle($_parser)";
  let printer = "::mlir::hc::sym::printExprHandle($_printer, $_self)";
}

def HC_PredHandleParameter
    : AttrParameter<"::mlir::hc::sym::PredHandle",
                    "symbolic predicate handle"> {
  let parser = "::mlir::hc::sym::parsePredHandle($_parser)";
  let printer = "::mlir::hc::sym::printPredHandle($_printer, $_self)";
}

def HC_ExprAttr : HC_Attr<"Expr", "expr"> {
  let summary = "Symbolic expression attribute";
  let description = [{
    Semantic symbolic expression attribute backed by an imported `ixsimpl`
    expression node in the dialect-owned symbolic store.
  }];
  let parameters = (ins HC_ExprHandleParameter:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    const ixs_node *getNode() const { return getValue().raw(); }
  }];
}

def HC_PredAttr : HC_Attr<"Pred", "pred"> {
  let summary = "Symbolic predicate attribute";
  let description = [{
    Semantic symbolic predicate attribute backed by an imported `ixsimpl`
    predicate node in the dialect-owned symbolic store.
  }];
  let parameters = (ins HC_PredHandleParameter:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    const ixs_node *getNode() const { return getValue().raw(); }
  }];
}

def HC_ShapeAttr : HC_Attr<"Shape", "shape"> {
  let summary = "Shape tuple built from symbolic expression dimensions";
  let description = [{
    Stores `#hc.expr` dimension attributes internally while printing them as a
    compact list of symbolic strings, for example `#hc.shape<["M", "N + 1"]>`.
  }];
  let parameters = (ins ArrayRefParameter<"Attribute">:$dims);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def HC_ConstraintSetAttr : HC_Attr<"ConstraintSet", "constraints"> {
  let summary = "Set of `#hc.pred` attributes";
  let parameters = (ins ArrayRefParameter<"Attribute">:$predicates);
  let assemblyFormat = "`<` `[` $predicates `]` `>`";
  let genVerifyDecl = 1;
}

def HC_ScopeAttr : HC_Attr<"Scope", "scope"> {
  let summary = "Kernel execution scope.";
  let description = [{
    One of `"WorkGroup"`, `"SubGroup"`, or `"WorkItem"`. Used by
    `hc.intrinsic` declarations and by the `hc.workitem_region` /
    `hc.subgroup_region` structuring ops so later passes can dispatch per-scope
    without string-matching mnemonics.
  }];
  let parameters = (ins StringRefParameter<"scope name">:$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

def HC_EffectsAttr : HC_Attr<"Effects", "effects"> {
  let summary = "Effect class advertised by an `hc.intrinsic` declaration.";
  let description = [{
    One of `"Pure"`, `"Read"`, `"Write"`, `"ReadWrite"`. Passes that care about
    reordering intrinsics (e.g. `hc.wave_barrier`, a hypothetical pure dot
    intrinsic) read this attribute instead of reasoning about op traits.
  }];
  let parameters = (ins StringRefParameter<"effect class">:$kind);
  let assemblyFormat = "`<` $kind `>`";
  let genVerifyDecl = 1;
}

#endif // HC_IR_HCATTRS
