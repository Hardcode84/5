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

def HC_DynSizeAttr : HC_Attr<"DynSize", "dyn"> {
  let summary = "Sentinel `?` for an unknown / host-owned shape entry.";
  let description = [{
    Marker for a shape slot whose size isn't knowable in-compiler — e.g. a
    buffer argument whose backing storage lives on the host descriptor.
    Renders as `?` in `#hc.shape<[...]>` and on shaped-type surfaces.

    Lives in `ShapeAttr.dims` alongside `ExprAttr`; every shape walker must
    discriminate explicitly. No `composeExprBinary` / `walkSymbolNames`
    overload for `DynSize` — consumers needing a concrete size must reach
    the host descriptor.
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

def HC_ShapeAttr : HC_Attr<"Shape", "shape"> {
  let summary = "Shape tuple built from symbolic expression dimensions";
  let description = [{
    Stores per-dimension attributes internally while printing them as a compact
    list, for example `#hc.shape<["M", "N + 1", "?"]>`. Each entry is either a
    concrete `#hc.expr` (printed without the `#hc.expr<...>` wrapper) or a
    `#hc.dyn` sentinel rendered as `?`.
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
    One of `"WorkGroup"`, `"SubGroup"`, `"WorkItem"`. Carried by
    `hc.intrinsic` and the `hc.{workitem,subgroup}_region` ops so passes
    dispatch per-scope without mnemonic matching.
  }];
  let parameters = (ins StringRefParameter<"scope name">:$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Typed enum attributes for operand kinds. `I32EnumAttr` rejects wrong
// spellings at parse time and gives lowerings a typed handle.
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

def HC_LayoutAttr : HC_Attr<"Layout", "layout"> {
  let summary = "Symbolic layout descriptor for shaped values.";
  let description = [{
    Source-level `index_map(...)` as a quintuple of bound-name lists +
    `#hc.expr` payloads:

      #hc.layout<shape_syms = ["W", "H"],
                 index_syms = ["i", "j"],
                 params = {row_stride = #hc.expr<"H + 4">},
                 storage_size = #hc.expr<"W * row_stride">,
                 offset = #hc.expr<"i * row_stride + j">>

    `shape_syms` bind positionally to the attached type's shape;
    `index_syms` bind at access sites; `params` are derived expressions
    over `shape_syms`. `shape_syms` and `index_syms` names must be
    unique within each list and disjoint; `params` keys must not collide
    with either — substitution at the use site relies on disjointness.

    `shape_syms.size() == index_syms.size() == rank`. `index_syms[k]` is
    the access coordinate for axis `k`; `shape_syms[k]` its extent. The
    N positional indices on `hc.load` / `hc.vload` / `hc.store` /
    `hc.load_mask` bind to `index_syms` in order; each may be a scalar
    pin or a slice iter sweep.

    Non-injective offsets are allowed (per-lane fragments, broadcast
    layouts). `storage_size` is the flat element extent and may be
    smaller than the logical-shape product when offset is non-injective;
    colliding writes overwrite, by design.

    All `#hc.expr` payloads go through the dialect store at construction
    so ixsimpl-canonical equality gives layout equality.

    No grammar restriction on `offset` / `storage_size`: we accept what
    ixsimpl accepts and leave injectivity / size proofs to consumers
    (canonicalize, flatten). See `doc/layouts.md` for the full design.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Attribute">:$shapeSyms,
    ArrayRefParameter<"::mlir::Attribute">:$indexSyms,
    "::mlir::DictionaryAttr":$params,
    "::mlir::hc::ExprAttr":$storageSize,
    "::mlir::hc::ExprAttr":$offset
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// Address space tag for `!hc.ptr`. Workgroup / global / private are the
// three classes the AMDGPU and SPIR-V backends distinguish.
def HC_AddrSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 0, "workgroup">;
def HC_AddrSpaceGlobal    : I32EnumAttrCase<"Global", 1, "global">;
def HC_AddrSpacePrivate   : I32EnumAttrCase<"Private", 2, "private">;

def HC_AddrSpace : I32EnumAttr<"AddrSpace",
    "Memory address space (workgroup/global/private).",
    [HC_AddrSpaceWorkgroup, HC_AddrSpaceGlobal, HC_AddrSpacePrivate]> {
  // No specialized attr companion: surfaces only as a typed parameter on
  // `!hc.ptr`. No stray `#hc<addr_space ...>` attributes.
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hc";
}

// Iteration kind for `hc.generic`. `parallel` iters are independent and
// vectorizable; `reduction` iters thread a sequential carry.
def HC_IterKindParallel  : I32EnumAttrCase<"Parallel",  0, "parallel">;
def HC_IterKindReduction : I32EnumAttrCase<"Reduction", 1, "reduction">;

def HC_IterKind : I32EnumAttr<"IterKind",
    "Iteration kind for `hc.generic` (parallel/reduction).",
    [HC_IterKindParallel, HC_IterKindReduction]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hc";
}

def HC_IterKindAttr : EnumAttr<HC_Dialect, HC_IterKind, "iter_kind"> {
  let summary = "Iteration kind for `hc.generic` (parallel/reduction).";
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
