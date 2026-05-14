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
    `!hc.undef` is the "type not yet known" state produced by the mechanical
    `hc_front` -> `hc` rewrite. A later inference pipeline refines it into a
    concrete `hc` type (or builtin) via `unrealized_conversion_cast`.
  }];
}

def HC_IdxType : HC_Type<"Idx", "idx",
    [DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Index-kind value, optionally pinned to a symbolic expression.";
  let description = [{
    `!hc.idx` marks a value as a scalar index. When the symbolic expression is
    pinned (`!hc.idx<"M + 1">`), type uniquing over the ixsimpl-canonicalized
    expression gives free expression equality: two SSA values typed
    `!hc.idx<"M + 1">` are known to denote the same index.

    Textually the expression text is inlined as a quoted string to avoid the
    ambiguous-looking `<<...>>` nesting the default attribute-parameter
    printer would produce.
  }];
  let parameters = (ins OptionalParameter<"::mlir::hc::ExprAttr">:$expr);
  let hasCustomAssemblyFormat = 1;
}

def HC_PredType : HC_Type<"Pred", "pred",
    [DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "Boolean-kind value, optionally pinned to a symbolic predicate.";
  let description = [{
    Boolean analogue of `!hc.idx`. When pinned (`!hc.pred<"M < N">`), the type
    uniquely identifies the symbolic predicate; the text is inlined the same
    way as for `!hc.idx`.
  }];
  let parameters = (ins OptionalParameter<"::mlir::hc::PredAttr">:$pred);
  let hasCustomAssemblyFormat = 1;
}

def HC_SliceType : HC_Type<"Slice", "slice",
    [DeclareTypeInterfaceMethods<HCJoinableTypeInterface>]> {
  let summary = "First-class slice value.";
  let description = [{
    Produced by `hc.slice_expr` and consumed by `hc.load`, `hc.store`,
    `hc.buffer_view`, and friends. Optional lower/upper/step parameters mirror
    the syntactically present operands of the producing `hc.slice_expr`.
    Missing parameters carry Python slice semantics (`None`-equivalent);
    present-but-unknown parts use `!hc.undef`.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::Type">:$lowerType,
    OptionalParameter<"::mlir::Type">:$upperType,
    OptionalParameter<"::mlir::Type">:$stepType
  );
  let hasCustomAssemblyFormat = 1;
}

// All five shaped types carry an optional `#hc.layout<...>` payload after
// the shape. Absent layout is the v0 identity contract (canonical
// identity offset, product-of-shape storage, rightmost axis varies
// fastest); explicit layouts are interpreted by the canonicalize / flatten
// passes documented in `doc/layouts.md`. Layouts attach to both semantic
// (`!hc.tensor`, `!hc.vector`) and bare (`!hc.bare_tensor`,
// `!hc.bare_vector`) shapes because the flatten pass needs them on either
// side of the decompose-shaped-values boundary.

// Each of the five shaped types below adds a `get(ctx, elem, shape)`
// overload alongside the auto-generated 4-arg form so existing callers
// that don't carry a layout don't need to spell `LayoutAttr{}`. Absent
// layout is the v0 default and the canonical-fold target.

def HC_BufferType : HC_Type<"Buffer", "buffer",
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>]> {
  let summary = "Semantic buffer type";
  let description = [{
    Initial semantic buffer type carrying an element type and a shape
    attribute. The shape is intentionally modeled as an attribute payload so the
    symbolic surface can evolve independently from the type shell, while the
    textual type syntax inlines the shape dimensions as `["M", "N + 1"]`.
    An empty shape is a rank-0 mutable scalar buffer, not missing metadata.
    An optional `#hc.layout<...>` payload follows the shape; absent means
    "default fully-strided np/torch layout" once host wrappers start
    feeding strides in (see `doc/layouts.md`).
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
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>]> {
  let summary = "Semantic tensor type";
  let description = [{
    Initial semantic tensor type carrying an element type and a symbolic shape
    attribute. The textual type syntax inlines the shape dimensions as
    `["M", "N + 1"]`. An optional `#hc.layout<...>` payload follows the
    shape; absent means the identity layout.
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
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>]> {
  let summary = "Semantic vector type";
  let description = [{
    Immutable fixed-size vector value. Carries an element type and a symbolic
    shape; activity mask semantics, the optional collective-return suffix, and
    layout metadata sit on top of this shell as they settle. Printed as
    `!hc.vector<elem, ["A", "B"]>`. An optional `#hc.layout<...>` payload
    follows the shape; absent means the identity layout.
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
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>]> {
  let summary = "Compiler-internal tensor data without semantic validity.";
  let description = [{
    Raw shaped tensor container produced by HC normalization before upstream
    lowering. Unlike `!hc.tensor`, this type carries only data; validity travels
    as a parallel `!hc.bare_tensor<!hc.pred, ...>` value with the same shape.
    An optional `#hc.layout<...>` payload follows the shape; absent means
    the identity layout.
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
    [DeclareTypeInterfaceMethods<SymbolicallyShapedTypeInterface>]> {
  let summary = "Compiler-internal vector data without semantic validity.";
  let description = [{
    Raw shaped vector container produced by HC normalization before upstream
    lowering. Unlike `!hc.vector`, this type carries only data; validity travels
    as a parallel `!hc.bare_vector<!hc.pred, ...>` value with the same shape.
    An optional `#hc.layout<...>` payload follows the shape; absent means
    the identity layout.
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

// Memory carrier type, the post-flatten replacement for the
// memref-shaped storage we used to hand to the launch-body lowering.
// Element type is optional: typed pointers (`!hc.ptr<f16, addrspace = ...>`)
// preserve the element through to the SPIR-V boundary; opaque pointers
// (`!hc.ptr<addrspace = ...>`) match the LLVM-17 opaque-pointer surface
// the AMDGPU backend expects. Address space is required — there's no
// "default" memory class on a GPU, so omitting it would just hide bugs.
def HC_PtrType : HC_Type<"Ptr", "ptr"> {
  let summary = "Memory pointer in a named address space (workgroup/global/private).";
  let description = [{
    Pointer to a single element of `$elementType` (or to opaque memory
    when `$elementType` is absent), living in `$addrSpace`. See
    `doc/layouts.md` for the role this plays in the post-flatten
    pipeline. Element type carries through the dialect until the
    `hc-lower-to-llvm` boundary drops it on the AMDGPU path; the
    SPIR-V path keeps it.
  }];
  let parameters = (ins
    EnumParameter<HC_AddrSpace>:$addrSpace,
    OptionalParameter<"::mlir::Type">:$elementType
  );
  // Surface mirrors `!llvm.ptr<3>`: required address space first as a
  // bare keyword, optional element type after the comma. Optional
  // groups need a literal anchor for the parser to probe presence
  // (the trailing `,` does that), so the optional positional element
  // can't sit before the required one without going custom.
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
    Carries invariant launch metadata for the special kernel `group`
    parameter. Dynamic launch queries (`hc.group_id`, `hc.local_id`, ...)
    still produce ordinary `!hc.idx` values; this type only records the
    static metadata those queries can derive ranks and shape facts from.
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
    Carries the static workgroup metadata visible from a workitem-scoped
    Python handle. `hc.local_id` uses this type to choose arity and synthesize
    axis-specific `!hc.idx<"$WI...">` results without falling back to an
    unknown-rank maximum.
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
    Carries the static workgroup metadata visible from a subgroup-scoped
    Python handle. `hc.subgroup_id` uses this type to choose arity and
    `hc.wave_size` can refine from the subgroup size metadata.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::hc::ShapeAttr">:$groupShape,
    OptionalParameter<"::mlir::hc::ExprAttr">:$subgroupSize
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// AnyOf constraint covering every type that a generic `hc` op (`hc.add`,
// `hc.cmp.lt`, `hc.cast`, ...) is willing to see. Intentionally wide: an op
// emitted by `hc_front` -> `hc` may see only `!hc.undef`; after inference it
// may see any of the concrete `hc` types, builtin tuple, or builtin scalar. Folding and
// verification dispatch on the concrete-type combinations within.
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

// Tuple-rooted values. The dialect uses MLIR's builtin `tuple<...>` shell so
// tuple values can carry arbitrary HC semantic element types without inventing
// a dialect-specific aggregate type.
def HC_TupleValueType : AnyTypeOf<[
  HC_UndefType,
  AnyTuple,
]>;

// Numeric/symbolic subset. Excludes `!hc.pred`, `!hc.slice`, `!hc.buffer` —
// types that have no business flowing through arithmetic ops. `!hc.undef`
// is kept for pre-inference compatibility.
def HC_NumericValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_TensorType,
  HC_VectorType,
  AnyInteger,
  AnyFloat,
  Index,
]>;

// Scalar numeric/symbolic values. Used for operands such as inactive fills
// that must be ordinary SSA values but never shaped aggregates.
def HC_ScalarValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_PredType,
  AnyInteger,
  AnyFloat,
  Index,
]>;

// Value-side type for `hc.ptr_load` / `hc.ptr_store`: scalars (single-
// element access) or upstream `vector<NxT>` (contiguous N-element access
// starting at the pointer). Vector width is unconstrained at the op; the
// LLVM lowering at the dialect boundary splits oversized vectors into
// hardware-sized chunks. Kept distinct from `HC_ScalarValueType` because
// only the pointer access ops grow this surface — the rest of `hc` keeps
// its end-to-end scalar/HC-shaped invariant.
def HC_PtrAccessValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_PredType,
  AnyInteger,
  AnyFloat,
  Index,
  FixedVectorOfAnyRank<[AnyInteger, AnyFloat, Index]>,
]>;

// Predicate-side type for the mask operand of predicated ops
// (`hc.ptr_load_pred` / `hc.ptr_store_pred` at ptr level,
// `hc.yield_predicated` / `hc.predicate` at body level). Concrete
// `i1` / `vector<Nxi1>` is the post-lowering surface; `!hc.pred` is
// the pre-lowering body-level surface (the element type of a
// `!hc.bare_tensor<!hc.pred, ...>` mask operand when an `hc.generic`
// loads one element at a time, before `hc-lower-bare-predicates`
// converts the carrier to `vector<Nxi1>`). Shape parity against the
// matching value operand (scalar value <-> scalar mask, vector<NxT>
// value <-> vector<Nxi1> of the same N) is enforced by the per-op
// verifier; this constraint only restricts the type to one of the
// flavours.
def HC_PtrPredicateType : AnyTypeOf<[
  HC_UndefType,
  HC_PredType,
  I1,
  FixedVectorOfAnyRank<[I1]>,
]>;

// Operand type for `hc.idx_apply` / `hc.pred_apply`: the SSA Value that
// supplies the runtime integer for one explicitly-bound free symbol of
// the carried expression / predicate. `index` is the natural form
// (loop induction vars, kernel-arg dim/stride values, builtin arithmetic);
// `!hc.idx<...>` is accepted so producers that already have a typed
// symbolic SSA in hand don't have to round-trip through an
// `unrealized_conversion_cast` to pass it as an operand. Lowering
// converts both to `index` via the launch-body type converter before
// substituting into the expression walk.
def HC_SymBindingValueType : AnyTypeOf<[
  Index,
  HC_IdxType,
]>;

// Shaped values for ops that only make sense on tensors/vectors
// (`hc.matmul`, `hc.reduce`, `hc.vec`, `hc.with_inactive`).
// `!hc.undef` is admitted for the same pre-inference reason as above.
def HC_ShapedValueType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_VectorType,
]>;

// Compiler-internal raw data/mask containers used after semantic shaped values
// are decomposed. Kept separate from `HC_ShapedValueType` so source-semantic ops
// opt into the split representation deliberately.
def HC_BareShapedValueType : AnyTypeOf<[
  HC_BareTensorType,
  HC_BareVectorType,
]>;

// Both flavors at once plus the memory carrier: `hc.as_layout` is a
// pure relabel that composes with decomposition (semantic ↔ bare) and
// applies equally to pointer-rooted shaped storage (`!hc.buffer`),
// since reinterpreting a buffer's access pattern is the motivating
// shape for WMMA / multibuffered LDS. The storage_size verifier is
// relaxed for the buffer side — see `HCAsLayoutOp::verify` — because
// a buffer's named dims are a proxy for the actual allocation, not a
// declaration of its byte extent.
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

// Operand-type predicate for `hc.generic`. Widens
// `HC_DecomposableShapedValueType` with the memory carriers
// (`!hc.ptr<...>`, `!hc.buffer<...>`) so a generic body may freely
// mix value-semantic results with in-place memory writes — the same
// op covers both `linalg.generic`'s tensor regime (value outs) and
// memref regime (ptr outs), and a mixed-flavor outs list is legal
// (e.g. compute a result tensor while spilling a debug trace into LDS
// in one fused loop nest). Per-operand effects are derived from
// operand types in `getEffects`: value slots are pure, ptr/buffer
// slots contribute Read (ins) or Read+Write (outs).
def HC_GenericOperandType : AnyTypeOf<[
  HC_UndefType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
  HC_BufferType,
  HC_PtrType,
]>;

// Buffer-rooted values. Used on the "handle" operand of data-movement and
// subscript ops so a stray `!hc.pred` or scalar value cannot stand in for a
// buffer even pre-inference.
def HC_BufferValueType : AnyTypeOf<[
  HC_UndefType,
  HC_BufferType,
]>;

// Buffer or tensor root. Used by ops that can address either storage class:
// `hc.vload` reads from either, and `hc.store` writes into either.
// `hc.buffer_view` uses `HC_ViewRootValueType` because generic subscripts may
// carry vectors and specialize them during type inference.
def HC_BufferOrTensorValueType : AnyTypeOf<[
  HC_UndefType,
  HC_BufferType,
  HC_TensorType,
  HC_BareTensorType,
]>;

// Roots accepted by the generic subscript form that still lowers to
// `hc.buffer_view` today. Vector subscripts refine to either a vector fragment
// or the scalar element type once inference knows the root and index types.
def HC_ViewRootValueType : AnyTypeOf<[
  HC_UndefType,
  HC_BufferType,
  HC_TensorType,
  HC_VectorType,
  HC_BareTensorType,
  HC_BareVectorType,
]>;

// Launch-context root. `!hc.undef` keeps hand-written pre-inference IR legal;
// concrete tokens are seeded from kernel/group/workitem/subgroup metadata.
def HC_GroupValueType : AnyTypeOf<[
  HC_UndefType,
  HC_GroupType,
  HC_WorkitemType,
  HC_SubgroupType,
]>;

// Pointer-shaped operand. `!hc.undef` is admitted so the pointer ops can
// be emitted before type inference fills the addrspace/element in.
def HC_PtrValueType : AnyTypeOf<[
  HC_UndefType,
  HC_PtrType,
]>;

#endif // HC_IR_HCTYPES
