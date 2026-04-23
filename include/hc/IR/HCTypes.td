// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCTYPES
#define HC_IR_HCTYPES

include "hc/IR/HCDialect.td"
include "hc/IR/HCAttrs.td"
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

def HC_IdxType : HC_Type<"Idx", "idx"> {
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

def HC_PredType : HC_Type<"Pred", "pred"> {
  let summary = "Boolean-kind value, optionally pinned to a symbolic predicate.";
  let description = [{
    Boolean analogue of `!hc.idx`. When pinned (`!hc.pred<"M < N">`), the type
    uniquely identifies the symbolic predicate; the text is inlined the same
    way as for `!hc.idx`.
  }];
  let parameters = (ins OptionalParameter<"::mlir::hc::PredAttr">:$pred);
  let hasCustomAssemblyFormat = 1;
}

def HC_SliceType : HC_Type<"Slice", "slice"> {
  let summary = "First-class slice value.";
  let description = [{
    Produced by `hc.slice_expr` with optional low/high/step operands; consumed
    by `hc.load`, `hc.store`, `hc.buffer_view`, and friends. Which operands are
    actually present is recorded as `has_lower`/`has_upper`/`has_step` on the
    producer op, mirroring `hc_front.slice`.
  }];
}

def HC_BufferType : HC_Type<"Buffer", "buffer"> {
  let summary = "Semantic buffer type";
  let description = [{
    Initial semantic buffer type carrying an element type and a shape
    attribute. The shape is intentionally modeled as an attribute payload so the
    symbolic surface can evolve independently from the type shell, while the
    textual type syntax inlines the shape dimensions as `["M", "N + 1"]`.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape
  );
  let assemblyFormat = "`<` $elementType `,` $shape `>`";
  let genVerifyDecl = 1;
}

def HC_TensorType : HC_Type<"Tensor", "tensor"> {
  let summary = "Semantic tensor type";
  let description = [{
    Initial semantic tensor type carrying an element type and a symbolic shape
    attribute. The textual type syntax inlines the shape dimensions as
    `["M", "N + 1"]`.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape
  );
  let assemblyFormat = "`<` $elementType `,` $shape `>`";
  let genVerifyDecl = 1;
}

def HC_VectorType : HC_Type<"Vector", "vector"> {
  let summary = "Semantic vector type";
  let description = [{
    Immutable fixed-size vector value. Carries an element type and a symbolic
    shape; activity mask semantics, the optional collective-return suffix, and
    layout metadata sit on top of this shell as they settle. Printed as
    `!hc.vector<elem, ["A", "B"]>`.
  }];
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    HC_InlineShapeAttrParameter:$shape
  );
  let assemblyFormat = "`<` $elementType `,` $shape `>`";
  let genVerifyDecl = 1;
}

// AnyOf constraint covering every type that a generic `hc` op (`hc.add`,
// `hc.cmp.lt`, `hc.cast`, ...) is willing to see. Intentionally wide: an op
// emitted by `hc_front` -> `hc` may see only `!hc.undef`; after inference it
// may see any of the concrete `hc` types or a builtin scalar. Folding and
// verification dispatch on the concrete-type combinations within.
def HC_ValueType : AnyTypeOf<[
  HC_UndefType,
  HC_IdxType,
  HC_PredType,
  HC_SliceType,
  HC_BufferType,
  HC_TensorType,
  HC_VectorType,
  AnyInteger,
  AnyFloat,
  Index,
]>;

#endif // HC_IR_HCTYPES
