// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCATTRS
#define HC_IR_HCATTRS

include "hc/IR/HCDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

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

// `HC_EffectsAttr` used to be a stringly-typed whitelist; it is now an enum so
// wrong spellings fail at parse time and `hc.call` / `hc.call_intrinsic`
// can dispatch on a typed handle instead of a `StringRef` comparison.

//===----------------------------------------------------------------------===//
// Typed enum attributes for operand kinds. Using `I32EnumAttr` gates wrong
// spellings at parse time (no `hc.reduce %v, kind = avg` making it to the
// verifier) and gives lowerings a typed handle instead of a `StringRef`.
//===----------------------------------------------------------------------===//

def HC_ReduceKindSum : I32EnumAttrCase<"Sum", 0, "sum">;
def HC_ReduceKindMax : I32EnumAttrCase<"Max", 1, "max">;
def HC_ReduceKindMin : I32EnumAttrCase<"Min", 2, "min">;

def HC_ReduceKind : I32EnumAttr<"ReduceKind",
    "Reduction combinator identity (sum/max/min).",
    [HC_ReduceKindSum, HC_ReduceKindMax, HC_ReduceKindMin]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hc";
}

def HC_ReduceKindAttr
    : EnumAttr<HC_Dialect, HC_ReduceKind, "reduce_kind"> {
  let summary = "Reduction combinator identity (sum/max/min).";
  let assemblyFormat = "$value";
}

def HC_LayoutRowMajor : I32EnumAttrCase<"RowMajor", 0, "row_major">;
def HC_LayoutColMajor : I32EnumAttrCase<"ColMajor", 1, "col_major">;

def HC_Layout : I32EnumAttr<"Layout",
    "Named tensor/vector layout (row_major/col_major).",
    [HC_LayoutRowMajor, HC_LayoutColMajor]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hc";
}

def HC_LayoutAttr : EnumAttr<HC_Dialect, HC_Layout, "layout"> {
  let summary = "Named tensor/vector layout descriptor.";
  let assemblyFormat = "$value";
}

def HC_EffectClassPure : I32EnumAttrCase<"Pure", 0, "pure">;
def HC_EffectClassRead : I32EnumAttrCase<"Read", 1, "read">;
def HC_EffectClassWrite : I32EnumAttrCase<"Write", 2, "write">;
def HC_EffectClassReadWrite
    : I32EnumAttrCase<"ReadWrite", 3, "read_write">;

def HC_EffectClass : I32EnumAttr<"EffectClass",
    "Declared effect class for hc.func / hc.intrinsic.",
    [HC_EffectClassPure, HC_EffectClassRead, HC_EffectClassWrite,
     HC_EffectClassReadWrite]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hc";
}

def HC_EffectsAttr : EnumAttr<HC_Dialect, HC_EffectClass, "effects"> {
  let summary = "Declared effect class (pure/read/write/read_write).";
  let assemblyFormat = "$value";
}

#endif // HC_IR_HCATTRS
