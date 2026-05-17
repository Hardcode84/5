// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPES
#define HC_IR_HCTYPES

include "hc/IR/HCAttrs.td"
include "hc/IR/HCDialect.td"
include "hc/IR/HCTypesInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

class HC_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HC_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def HC_InlineShapeAttrParameter
    : AttrParameter<"::mlir::hc::ShapeAttr", "shape attribute"> {
  let parser = "::mlir::hc::parseInlineShapeAttr($_parser)";
  let printer = "::mlir::hc::printInlineShapeAttr($_printer, $_self)";
}

def HC_UndefType : HC_Type<"Undef", "undef"> {
  let summary = "Placeholder type for values awaiting type inference.";
  let description = [{
    Type-not-yet-known state produced by `hc_front` -> `hc`. Inference later
    refines via `unrealized_conversion_cast`.
  }];
}

def HC_IdxType : HC_Type<"Idx", "idx",
    [DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Index-kind value, optionally pinned to a symbolic expression.";
  let description = [{
    Scalar index. Pinned form (`!hc.idx<"M + 1">`) uses ixsimpl-canonical
    expression uniquing — two values typed identically denote the same
    index.

    Expression text inlines as a quoted string to dodge ambiguous-looking
    `<<...>>` nesting from the default parameter printer.
  }];
  let parameters = (ins OptionalParameter<"::mlir::hc::ExprAttr">:$expr);
  let hasCustomAssemblyFormat = 1;
}

def HC_PredType : HC_Type<"Pred", "pred",
    [DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Boolean-kind value, optionally pinned to a symbolic predicate.";
  let description = [{
    Boolean counterpart of `!hc.idx`. Pinned form (`!hc.pred<"M < N">`)
    uniques on the symbolic predicate; text inlined the same way.
  }];
  let parameters = (ins OptionalParameter<"::mlir::hc::PredAttr">:$pred);
  let hasCustomAssemblyFormat = 1;
}

def HC_SliceType : HC_Type<"Slice", "slice",
    [DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "First-class slice value.";
  let description = [{
    Produced by `hc.slice_expr`, consumed by `hc.load` / `hc.store` /
    `hc.buffer_view` / friends. Optional `lower` / `upper` / `step` mirror
    syntactically-present operands. Missing parts carry Python slice
    semantics (`None`); present-but-unknown parts use `!hc.undef`.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::Type">:$lowerType,
    OptionalParameter<"::mlir::Type">:$upperType,
    OptionalParameter<"::mlir::Type">:$stepType
  );
  let hasCustomAssemblyFormat = 1;
}

// Shaped types carry an optional `#hc.layout<...>` after the shape. Absent
// layout = identity contract (canonical identity offset, product-of-shape
// storage, rightmost axis varies fastest); explicit layouts go through
// canonicalize / flatten (`doc/layouts.md`). Both semantic
// (`!hc.tensor` / `!hc.vector`) and bare (`!hc.bare_tensor` /
// `!hc.bare_vector`) carry a layout slot because flatten needs them on
// either side of the decompose-shaped-values boundary.
//
// Each shaped type below has a `get(ctx, elem, shape)` overload alongside
// the auto-generated 4-arg form so layout-less callers skip `LayoutAttr{}`.

def HC_BufferType : HC_Type<"Buffer", "buffer",
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>,
     DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Semantic buffer type";
  let description = [{
    Element type + shape attribute. Shape inlines as `["M", "N + 1"]`. Empty
    shape is a rank-0 mutable scalar buffer, not missing metadata. Optional
    `#hc.layout<...>` payload follows; absent = default fully-strided
    np/torch layout once host wrappers feed strides in (see
    `doc/layouts.md`).
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape,
    OptionalParameter<"::mlir::hc::LayoutAttr">:$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape (`,` $layout^)? `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType,
                     "::mlir::hc::ShapeAttr":$shape), [{
      return $_get($_ctxt, elementType, shape, ::mlir::hc::LayoutAttr{});
    }]>
  ];
}

def HC_TensorType : HC_Type<"Tensor", "tensor",
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>,
     DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Semantic tensor type";
  let description = [{
    Element type + symbolic shape attribute. Shape inlines as
    `["M", "N + 1"]`. Optional `#hc.layout<...>` follows; absent = identity.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape,
    OptionalParameter<"::mlir::hc::LayoutAttr">:$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape (`,` $layout^)? `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType,
                     "::mlir::hc::ShapeAttr":$shape), [{
      return $_get($_ctxt, elementType, shape, ::mlir::hc::LayoutAttr{});
    }]>
  ];
}

def HC_VectorType : HC_Type<"Vector", "vector",
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>,
     DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Semantic vector type";
  let description = [{
    Immutable fixed-size vector. Element type + symbolic shape. Printed as
    `!hc.vector<elem, ["A", "B"]>`. Optional `#hc.layout<...>` follows;
    absent = identity.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape,
    OptionalParameter<"::mlir::hc::LayoutAttr">:$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape (`,` $layout^)? `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType,
                     "::mlir::hc::ShapeAttr":$shape), [{
      return $_get($_ctxt, elementType, shape, ::mlir::hc::LayoutAttr{});
    }]>
  ];
}

def HC_BareTensorType : HC_Type<"BareTensor", "bare_tensor",
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>,
     DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Compiler-internal tensor data without semantic validity.";
  let description = [{
    Raw shaped tensor produced by HC normalization. Carries data only;
    validity rides on a parallel `!hc.bare_tensor<!hc.pred, ...>` of the
    same shape. Optional `#hc.layout<...>` follows; absent = identity.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape,
    OptionalParameter<"::mlir::hc::LayoutAttr">:$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape (`,` $layout^)? `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType,
                     "::mlir::hc::ShapeAttr":$shape), [{
      return $_get($_ctxt, elementType, shape, ::mlir::hc::LayoutAttr{});
    }]>
  ];
}

def HC_BareVectorType : HC_Type<"BareVector", "bare_vector",
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>,
     DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Compiler-internal vector data without semantic validity.";
  let description = [{
    Raw shaped vector produced by HC normalization. Carries data only;
    validity rides on a parallel `!hc.bare_vector<!hc.pred, ...>` of the
    same shape. Optional `#hc.layout<...>` follows; absent = identity.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape,
    OptionalParameter<"::mlir::hc::LayoutAttr">:$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape (`,` $layout^)? `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType,
                     "::mlir::hc::ShapeAttr":$shape), [{
      return $_get($_ctxt, elementType, shape, ::mlir::hc::LayoutAttr{});
    }]>
  ];
}

// Memory carrier — post-flatten replacement for memref-shaped storage.
// Element type optional: typed pointers preserve it through the SPIR-V
// boundary; opaque pointers match the LLVM-17 opaque surface AMDGPU
// expects. Address space required — no "default" memory class on a GPU.
def HC_PtrType : HC_Type<"Ptr", "ptr"> {
  let summary = "Memory pointer in a named address space (workgroup/global/private).";
  let description = [{
    Pointer to a single `$elementType` element (or opaque memory when
    `$elementType` is absent) in `$addrSpace`. Element type carries through
    to `hc-lower-to-llvm`, which drops it on the AMDGPU path and keeps it
    on SPIR-V. See `doc/layouts.md` for the post-flatten role.
  }];
  let parameters = (ins
    EnumParameter<HC_AddrSpace>:$addrSpace,
    OptionalParameter<"::mlir::Type">:$elementType
  );
  // Surface mirrors `!llvm.ptr<3>`: required addrspace keyword, optional
  // element type after the comma. Optional group needs a literal anchor
  // (the `,`) so the optional element can't precede the required AS.
  let assemblyFormat = "`<` $addrSpace (`,` $elementType^)? `>`";
  let builders = [
    TypeBuilder<(ins "::mlir::hc::AddrSpace":$addrSpace), [{
      return $_get($_ctxt, addrSpace, ::mlir::Type{});
    }]>
  ];
}

def HC_GroupType : HC_Type<"Group", "group"> {
  let summary = "Kernel launch-context token.";
  let description = [{
    Invariant launch metadata for the kernel `group` parameter. Dynamic
    queries (`hc.group_id`, `hc.local_id`, ...) still produce `!hc.idx`;
    this type only records the static facts those queries derive rank and
    shape from.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::hc::ShapeAttr">:$workShape,
    OptionalParameter<"::mlir::hc::ShapeAttr">:$groupShape,
    OptionalParameter<"::mlir::hc::ExprAttr">:$subgroupSize
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def HC_WorkitemType : HC_Type<"Workitem", "workitem"> {
  let summary = "Workitem launch-context token.";
  let description = [{
    Static workgroup metadata visible from a workitem-scoped Python handle.
    `hc.local_id` uses this to pick arity and synthesize axis-specific
    `!hc.idx<"$WI...">` results instead of falling back to unknown-rank.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::hc::ShapeAttr">:$groupShape,
    OptionalParameter<"::mlir::hc::ExprAttr">:$subgroupSize
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def HC_SubgroupType : HC_Type<"Subgroup", "subgroup"> {
  let summary = "Subgroup launch-context token.";
  let description = [{
    Static workgroup metadata visible from a subgroup-scoped Python handle.
    `hc.subgroup_id` picks arity from it; `hc.wave_size` refines from the
    subgroup-size slot.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::hc::ShapeAttr">:$groupShape,
    OptionalParameter<"::mlir::hc::ExprAttr">:$subgroupSize
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// Open value-type constraint for generic ops (`hc.add`, `hc.cmp.lt`,
// `hc.cast`, ...). Wide on purpose: pre-inference may only see `!hc.undef`;
// post-inference any concrete HC type, builtin tuple, or builtin scalar.
// Folders / verifiers dispatch on concrete combinations.
def HC_ValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_PredType,
  HC_SliceType,
  HC_BufferType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
  HC_PtrType,
  HC_GroupType,
  HC_WorkitemType,
  HC_SubgroupType,
  AnyInteger,
  AnyFloat,
  Index,
  AnyTuple,
]>;

// Tuple-rooted values. Uses MLIR's builtin `tuple<...>` so tuples carry
// arbitrary HC element types without a dialect-specific aggregate.
def HC_TupleValueType : AnyTypeOf<[
  HC_UndefType,
  AnyTuple,
]>;

// Numeric / symbolic subset. Excludes `!hc.pred`, `!hc.slice`, `!hc.buffer`.
// `!hc.undef` admitted pre-inference. Bare carriers ride alongside semantic
// counterparts so decomposition can plant arith on the data half before
// the *-to-generic rewriters funnel into `hc.generic`.
def HC_NumericValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
  AnyInteger,
  AnyFloat,
  Index,
]>;

// Scalar numeric / symbolic values (e.g. inactive fills) — SSA values
// never shaped aggregates.
def HC_ScalarValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_PredType,
  AnyInteger,
  AnyFloat,
  Index,
]>;

// Value-side type for `hc.ptr_load` / `hc.ptr_store`: scalar (single
// element) or `vector<NxT>` (contiguous N elements from the pointer).
// Vector width unconstrained at op level; LLVM lowering splits oversized
// vectors into hardware chunks. Kept distinct from `HC_ScalarValueType`
// because only pointer access ops grow this surface.
def HC_PtrAccessValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_PredType,
  AnyInteger,
  AnyFloat,
  Index,
  FixedVectorOfAnyRank<[AnyInteger, AnyFloat, Index]>,
]>;

// Mask operand type for predicated ops (`hc.ptr_load_pred`,
// `hc.ptr_store_pred`, `hc.yield_predicated`, `hc.predicate`). Concrete
// `i1` / `vector<Nxi1>` is the post-lowering surface; `!hc.pred` is the
// pre-lowering body-level surface (mask element type before
// `hc-lower-bare-predicates`). Shape parity with the value operand is on
// the per-op verifier; this constraint only restricts the flavour.
def HC_PtrPredicateType : AnyTypeOf<[
  HC_UndefType,
  HC_PredType,
  I1,
  FixedVectorOfAnyRank<[I1]>,
]>;

// SSA operand type for `hc.idx_apply` / `hc.pred_apply`: the runtime
// integer for one bound free symbol. `index` is the natural form;
// `!hc.idx<...>` accepted so typed-symbolic SSA skips a UCC round-trip.
// Lowering folds both to `index` via the launch-body type converter
// before expression substitution.
def HC_SymBindingValueType : AnyTypeOf<[
  Index,
  HC_IdxType,
]>;

// Shaped values for tensor/vector-only ops (`hc.matmul`, `hc.reduce`,
// `hc.vec`, `hc.with_inactive`).
def HC_ShapedValueType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_VectorType,
]>;

// Raw data/mask containers used post-decomposition. Separate from
// `HC_ShapedValueType` so source-semantic ops opt into the split
// deliberately.
def HC_BareShapedValueType : AnyTypeOf<[
  HC_BareTensorType,
  HC_BareVectorType,
]>;

// Both flavours + the memory carrier. `hc.as_layout` is a pure relabel
// that composes with decomposition (semantic ↔ bare) and applies to
// `!hc.buffer` too (WMMA / multibuffered LDS). The storage_size verifier
// is relaxed buffer-side — see `HCAsLayoutOp::verify`.
def HC_AnyShapedValueType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
  HC_BufferType,
]>;

def HC_TensorLikeValueType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_BareTensorType,
]>;

def HC_VectorLikeValueType : AnyTypeOf<[
  HC_UndefType,
  HC_VectorType,
  HC_BareVectorType,
]>;

def HC_DecomposableShapedValueType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
]>;

// Operand type for `hc.generic`. Widens `HC_DecomposableShapedValueType`
// with `!hc.ptr<...>` / `!hc.buffer<...>` so a generic body mixes value
// results with in-place memory writes — one op covers `linalg.generic`'s
// tensor + memref regimes; mixed-flavour outs are legal. Per-operand
// effects derive in `getEffects`: value = pure, ptr/buffer ins = Read,
// ptr/buffer outs = Read+Write.
def HC_GenericOperandType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
  HC_BufferType,
  HC_PtrType,
]>;

// Buffer-rooted handle for data-movement / subscript ops. Bars a stray
// scalar or `!hc.pred` from sliding in pre-inference.
def HC_BufferValueType : AnyTypeOf<[
  HC_UndefType,
  HC_BufferType,
]>;

// Buffer or tensor root: ops that address either storage class
// (`hc.vload` reads, `hc.store` writes). `hc.buffer_view` widens to
// `HC_ViewRootValueType` because vector subscripts may refine.
def HC_BufferOrTensorValueType : AnyTypeOf<[
  HC_UndefType,
  HC_BufferType,
  HC_TensorType,
  HC_BareTensorType,
]>;

// Roots for the generic subscript form lowering to `hc.buffer_view`.
// Vector subscripts refine to a vector fragment or element type once
// inference knows root and index types.
def HC_ViewRootValueType : AnyTypeOf<[
  HC_UndefType,
  HC_BufferType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
]>;

// Launch-context root. `!hc.undef` admitted pre-inference.
def HC_GroupValueType : AnyTypeOf<[
  HC_UndefType,
  HC_GroupType,
  HC_WorkitemType,
  HC_SubgroupType,
]>;

// Pointer-shaped operand. `!hc.undef` admitted pre-inference.
def HC_PtrValueType : AnyTypeOf<[
  HC_UndefType,
  HC_PtrType,
]>;

#endif // HC_IR_HCTYPES
