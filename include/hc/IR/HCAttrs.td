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
    Marker attribute that fills the slot of a regular `#hc.expr` dimension when
    the size is not knowable inside the compiler — e.g. a buffer argument whose
    backing storage is owned by the host descriptor and not derivable from the
    in-IR symbol set. Renders as `?` inside `#hc.shape<[...]>` and on the
    surface form of any shaped type.

    The attribute participates in `ShapeAttr.dims` alongside `ExprAttr` so
    every shape-entry walker has to discriminate explicitly: there is no
    `composeExprBinary` / `walkSymbolNames` overload for `DynSize`, by design.
    Consumers that need a concrete size must reach for the host descriptor.
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

def HC_LayoutAttr : HC_Attr<"Layout", "layout"> {
  let summary = "Symbolic layout descriptor for shaped values.";
  let description = [{
    Captures `index_map(...)` from the source layer as a quintuple of
    bound-name lists and `#hc.expr` payloads. Textual form, single-lined
    here for clarity:

      #hc.layout<shape_syms = ["W", "H"],
                 index_syms = ["i", "j"],
                 params = {row_stride = #hc.expr<"H + 4">},
                 storage_size = #hc.expr<"W * row_stride">,
                 offset = #hc.expr<"i * row_stride + j">>

    Bound-name conventions: `shape_syms` are bound positionally to the
    shape entries of the attached type; `index_syms` are bound at access
    sites; `params` are derived expressions over `shape_syms` whose
    names `storage_size` and `offset` may reference. Names listed in
    `shape_syms` and `index_syms` must be unique within their list and
    disjoint from each other; `params` keys must not collide with
    either. Symbolic-engine-side substitution at the use site relies on
    that disjointness.

    `shape_syms` and `index_syms` must have the same length and both
    match the attached type's rank. Every `index_syms[k]` is the access
    coordinate for logical axis `k`; every `shape_syms[k]` names the
    extent of that axis. The N positional indices on `hc.load` /
    `hc.vload` / `hc.store` / `hc.load_mask` bind to `index_syms` in
    order; each may be a scalar (pin) or a slice (iter sweep over that
    axis).

    Non-injective offsets are allowed: distinct logical-index tuples
    may map to the same flat storage offset. Per-lane fragments,
    broadcast layouts, and similar patterns express this directly —
    the logical shape carries the lane / wave extent and the layout's
    offset collapses it. `storage_size` is the flat extent in elements
    and may be smaller than the logical-shape product when the offset
    is non-injective; consumers reading through the layout get the
    canonical slot, consumers writing accept that lanes which collide
    overwrite each other (use that intentionally or don't).

    All `#hc.expr` payloads go through the dialect symbolic store at
    construction so type uniquing across ixsimpl-canonical equality
    gives free layout equality.

    No restriction on the offset / storage_size grammar in v1: we
    accept what ixsimpl can chew and leave injectivity / size proofs
    to the consumers (canonicalize, flatten). See `doc/layouts.md` for
    the full design.
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

// Address space tag carried by `!hc.ptr` (and any future memory carrier
// in this dialect). Workgroup / global / private are the three classes
// the AMDGPU and SPIR-V backends actually distinguish; richer schemes
// (constant, image, named driver classes) are not in v1 — extending the
// enum later is additive.
def HC_AddrSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 0, "workgroup">;
def HC_AddrSpaceGlobal    : I32EnumAttrCase<"Global", 1, "global">;
def HC_AddrSpacePrivate   : I32EnumAttrCase<"Private", 2, "private">;

def HC_AddrSpace : I32EnumAttr<"AddrSpace",
    "Memory address space (workgroup/global/private).",
    [HC_AddrSpaceWorkgroup, HC_AddrSpaceGlobal, HC_AddrSpacePrivate]> {
  // No specialized attr companion: the enum surfaces only as a typed type
  // parameter on `!hc.ptr`; we don't want stray `#hc<addr_space ...>`
  // attributes growing other surfaces.
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hc";
}

// Iteration kind tag for `hc.generic`: parallel iters are independent and
// vectorizable; reduction iters carry a sequential init across the loop.
// The split mirrors `linalg.generic`'s `iterator_types` but stays inside the
// dialect so the symbolic engine — not affine maps — owns offset analysis.
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
