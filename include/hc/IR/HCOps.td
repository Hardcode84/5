// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPS
#define HC_IR_HCOPS

include "hc/IR/HCTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class HC_RegionOp<string mnemonic, list<Trait> traits = []>
    : HC_Op<mnemonic, !listconcat([NoTerminator, SingleBlock], traits)>;

// Pure binary op whose operands and result are any `HC_ValueType`. The
// combination is left deliberately loose: a just-lowered `hc_front` module
// passes all `!hc.undef`, later inference narrows to `!hc.idx`, `!hc.tensor`,
// or a builtin scalar, and folders dispatch on the observed combination.
class HC_GenericBinaryOp<string mnemonic, list<Trait> traits = []>
    : HC_Op<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins HC_ValueType:$lhs, HC_ValueType:$rhs);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

class HC_GenericUnaryOp<string mnemonic, list<Trait> traits = []>
    : HC_Op<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins HC_ValueType:$value);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def HCKernelOp : HC_RegionOp<"kernel", [Symbol]> {
  let summary = "Semantic kernel region.";
  let description = [{
    Top-level compiled unit. Carries a module-scoped symbol name (`@name`) so
    call sites and specialization metadata can reference it by flat symbol
    ref instead of string lookup.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name (`requirements` `=` $requirements^)?
    attr-dict-with-keyword $body
  }];
}

def HCFuncOp : HC_RegionOp<"func", [Symbol]> {
  let summary = "Semantic helper region.";
  let description = [{
    Helper callable referenced by `hc.call`. Carries a symbol name so symbol
    resolution can catch typos and wrong-kind references at verify time;
    signatures live on operand/result types of the call sites until
    `hc.func` grows an explicit signature.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name (`requirements` `=` $requirements^)?
    attr-dict-with-keyword $body
  }];
}

def HCSubgroupRegionOp : HC_RegionOp<"subgroup_region"> {
  let summary = "Semantic subgroup region.";
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "(`captures` `=` $captures^)? attr-dict-with-keyword $body";
}

def HCWorkitemRegionOp : HC_RegionOp<"workitem_region"> {
  let summary = "Semantic workitem region.";
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "(`captures` `=` $captures^)? attr-dict-with-keyword $body";
}

def HCReturnOp : HC_Op<"return"> {
  let summary = "Semantic return.";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Scalar producers and casts.
//===----------------------------------------------------------------------===//

def HCConstOp : HC_Op<"const", [Pure]> {
  let summary = "Literal scalar producer (int, float, bool, string).";
  let description = [{
    Carries a builtin attribute payload lowered from `hc_front.constant`.
    The result type starts as `!hc.undef` after the mechanical frontend
    rewrite and is refined by later inference (e.g., to `!hc.idx<"7">` for
    integer literals that appear in an index context).
  }];
  let arguments = (ins AnyAttr:$value);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "`<` $value `>` attr-dict `:` type($result)";
}

def HCSymbolOp : HC_Op<"symbol", [Pure]> {
  let summary = "Introduce a symbolic capture bound to a named symbol.";
  let description = [{
    Produces a pinned `!hc.idx<"name">`. The expression lives in the result
    type; this op carries no operands or attributes so the bound symbol is
    visible in one place only. The verifier rejects a bare `!hc.idx` or a
    non-idx result type.

    `HC_ValueType` rather than `HC_IdxType` is used as the result constraint
    solely so the assembly printer keeps the full `!hc.idx<"M">` spelling
    instead of stripping the dialect prefix for a single-type constraint.
  }];
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def HCCastOp : HC_Op<"cast", [Pure]> {
  let summary = "User-visible value-level type conversion.";
  let description = [{
    Generic value-changing cast: refinement-lattice transitions
    (`!hc.undef -> !hc.idx<E>`), symbolic-to-builtin exits
    (`!hc.idx<E> -> index`), and layout changes on vectors all lower to
    `hc.cast`. Distinct from `unrealized_conversion_cast`, which is
    compiler-internal scaffolding that later passes are expected to erase,
    and from `hc.astype`, which is the source-level Python `.astype`
    surface carrying a target element type attribute on the op.
  }];
  let arguments = (ins HC_ValueType:$source);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Generic arithmetic, boolean, and comparison ops.
//
// These accept any `HC_ValueType` combination. The chosen op is picked by the
// frontend based on the Python surface (`+`, `-`, `*`, `//`, `%`, `not`, ...).
// Concrete meaning — symbolic folding via ixsimpl, scalar arith, broadcast
// tensor arith — is resolved by type inference and per-type folders.
//===----------------------------------------------------------------------===//

def HCAddOp : HC_GenericBinaryOp<"add", [Commutative]> {
  let summary = "Generic addition.";
}

def HCSubOp : HC_GenericBinaryOp<"sub"> {
  let summary = "Generic subtraction.";
}

def HCMulOp : HC_GenericBinaryOp<"mul", [Commutative]> {
  let summary = "Generic multiplication.";
}

def HCDivOp : HC_GenericBinaryOp<"div"> {
  let summary = "Generic integer/float division (Python `//` for ints).";
}

def HCModOp : HC_GenericBinaryOp<"mod"> {
  let summary = "Generic remainder (Python `%`).";
}

def HCNegOp : HC_GenericUnaryOp<"neg"> {
  let summary = "Generic unary minus.";
}

def HCAndOp : HC_GenericBinaryOp<"and", [Commutative]> {
  let summary = "Generic logical AND (Python `and`, `&` on bools).";
}

def HCOrOp : HC_GenericBinaryOp<"or", [Commutative]> {
  let summary = "Generic logical OR (Python `or`, `|` on bools).";
}

def HCNotOp : HC_GenericUnaryOp<"not"> {
  let summary = "Generic logical NOT (Python `not`).";
}

class HC_CmpOp<string suffix>
    : HC_GenericBinaryOp<"cmp." # suffix> {
  let summary = "Generic comparison (`" # suffix # "`).";
  let description = [{
    Two symbolic index operands fold to `!hc.pred<"...">`; concrete scalars
    produce `i1`; tensors broadcast to `tensor<...xi1>`.
  }];
}

def HCCmpLtOp : HC_CmpOp<"lt">;
def HCCmpLeOp : HC_CmpOp<"le">;
def HCCmpGtOp : HC_CmpOp<"gt">;
def HCCmpGeOp : HC_CmpOp<"ge">;
def HCCmpEqOp : HC_CmpOp<"eq">;
def HCCmpNeOp : HC_CmpOp<"ne">;

//===----------------------------------------------------------------------===//
// Intrinsic declaration and structured control flow.
//===----------------------------------------------------------------------===//

def HCIntrinsicOp : HC_RegionOp<"intrinsic", [Symbol]> {
  let summary = "Scope-gated semantic intrinsic declaration.";
  let description = [{
    Declares an intrinsic with its scope and (optional) effect class.
    Analogous to `hc_front.intrinsic`; later passes dispatch on `scope` and
    `effects` without string-matching the mnemonic.

    The body region holds verify/infer hooks, lowering recipes, or a
    fallback body once those settle. An empty body is legal and denotes a
    declaration-only intrinsic that later lowering patterns handle
    directly.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    HC_ScopeAttr:$scope,
    OptionalAttr<HC_EffectsAttr>:$effects
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name `scope` `=` $scope (`effects` `=` $effects^)?
    attr-dict-with-keyword $body
  }];
}

def HCYieldOp : HC_Op<"yield",
    [Pure, Terminator,
     ParentOneOf<["HCForRangeOp", "HCIfOp"]>]> {
  let summary = "Terminator for `hc.for_range` and `hc.if` regions.";
  let description = [{
    Yields loop-carried or if-produced values back to the enclosing op.
    Value count must match the parent op's result count; verification of
    that happens on the parent op so the error message references the
    right operation.
  }];
  let arguments = (ins Variadic<HC_ValueType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [
    // Needed by `SingleBlockImplicitTerminator` on the parent ops.
    OpBuilder<(ins), [{ /* empty yield */ }]>,
  ];
}

def HCForRangeOp : HC_Op<"for_range",
    [SingleBlockImplicitTerminator<"HCYieldOp">]> {
  let summary = "Structured for-range loop over symbolic bounds.";
  let description = [{
    Parallels `scf.for`. Operands are typed `HC_ValueType` so the op is
    usable before type inference runs: a just-lowered frontend module
    passes `!hc.undef` for lb/ub/step, and inference later narrows them
    to `!hc.idx` / builtin integers.

    The body block takes the induction variable followed by the iter_args
    values; the block-argument count therefore equals
    `1 + iter_inits.size()` and each loop-carried result type matches the
    corresponding `iter_inits` operand.
  }];
  let arguments = (ins
    HC_ValueType:$lower,
    HC_ValueType:$upper,
    HC_ValueType:$step,
    Variadic<HC_ValueType>:$iter_inits
  );
  let results = (outs Variadic<HC_ValueType>:$iter_results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lower `to` $upper `step` $step
    (`iter_args` `(` $iter_inits^ `)`)? attr-dict `:`
    functional-type(operands, results) $body
  }];
}

def HCIfOp : HC_Op<"if",
    [SingleBlockImplicitTerminator<"HCYieldOp">]> {
  let summary = "Structured conditional with optional else region.";
  let description = [{
    Parallels `scf.if`. The condition is any `HC_ValueType` (typically an
    `i1`, `!hc.pred`, or `!hc.undef`). Result types are annotated so the
    op is usable before its yield operands have concrete types; the
    verifier matches each non-empty region's yield count and types against
    the op's result signature, and requires an `else` region when the op
    produces results.
  }];
  let arguments = (ins HC_ValueType:$cond);
  let results = (outs Variadic<HC_ValueType>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                        AnyRegion:$else_region);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $cond (`->` `(` type($results)^ `)`)? attr-dict `:` type($cond)
    $then_region (`else` $else_region^)?
  }];
}

//===----------------------------------------------------------------------===//
// Launch-geometry queries.
//
// Single group-handle operand in, one or more `HC_ValueType` values out.
// Pre-inference results are all `!hc.undef`; inference pins them to
// `!hc.idx<"_symN">` with fresh symbols. Marked `Pure`: launch state is
// invariant for a single kernel instance (pipeline invariant; see
// `doc/lowering.md` "Launch-geometry invariant").
//===----------------------------------------------------------------------===//

class HC_LaunchGeoOp<string mnemonic>
    : HC_Op<mnemonic, [Pure]> {
  let arguments = (ins HC_ValueType:$group);
  let results = (outs Variadic<HC_ValueType>:$values);
  let assemblyFormat =
      "$group attr-dict `:` functional-type(operands, results)";
}

def HCGroupIdOp : HC_LaunchGeoOp<"group_id"> {
  let summary = "Workgroup index along each axis.";
}

def HCLocalIdOp : HC_LaunchGeoOp<"local_id"> {
  let summary = "Workitem index inside the workgroup along each axis.";
}

def HCSubgroupIdOp : HC_LaunchGeoOp<"subgroup_id"> {
  let summary = "Subgroup index along each axis.";
}

def HCGroupShapeOp : HC_LaunchGeoOp<"group_shape"> {
  let summary = "Workgroup shape along each axis.";
}

def HCGroupSizeOp : HC_LaunchGeoOp<"group_size"> {
  let summary = "Total workitem count in the workgroup.";
  let description = [{
    Single-result geometry query; the result is the product of
    `hc.group_shape` components.
  }];
}

def HCWorkOffsetOp : HC_LaunchGeoOp<"work_offset"> {
  let summary = "Global workitem offset along each axis.";
}

def HCWorkShapeOp : HC_LaunchGeoOp<"work_shape"> {
  let summary = "Global launch shape along each axis.";
}

def HCWaveSizeOp : HC_LaunchGeoOp<"wave_size"> {
  let summary = "Subgroup size (wave width) for this launch.";
}

//===----------------------------------------------------------------------===//
// Buffer views and subscript construction.
//===----------------------------------------------------------------------===//

def HCBufferDimOp : HC_Op<"buffer_dim", [Pure]> {
  let summary = "Symbolic dimension of a buffer (matches `a.shape[N]`).";
  let arguments = (ins HC_ValueType:$buffer, I64Attr:$axis);
  let results = (outs HC_ValueType:$dim);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $buffer `,` `axis` `=` $axis attr-dict `:` type($buffer) `->` type($dim)
  }];
}

def HCSliceExprOp : HC_Op<"slice_expr",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "Build a first-class slice value.";
  let description = [{
    Produces an `!hc.slice` from any subset of `lower`, `upper`, and `step`.
    Each part is an *optional* SSA operand rather than a positional argument
    — the frontend emits the subset syntactically present in the source
    (`x[1:]` vs `x[1:10:2]`), and `AttrSizedOperandSegments` makes the
    mapping between operand groups and parts exact at verify time.

    Missing parts default to Python slice semantics (`None`): `lower` omitted
    means "from the start", `upper` omitted means "to the end", `step`
    omitted means "step of 1".
  }];
  let arguments = (ins
    Optional<HC_ValueType>:$lower,
    Optional<HC_ValueType>:$upper,
    Optional<HC_ValueType>:$step
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    `(` (`lower` `=` $lower^)? (`upper` `=` $upper^)? (`step` `=` $step^)?
    `)` attr-dict `:` functional-type(operands, results)
  }];
}

def HCBufferViewOp : HC_Op<"buffer_view", [Pure]> {
  let summary = "Sub-buffer (no data movement).";
  let description = [{
    Returns a view of `$buffer` reduced by the variadic index/slice
    operands. The variadic list carries a mix of scalar indices and
    `!hc.slice` values, matching Python-style multi-axis subscripts.
  }];
  let arguments = (ins HC_ValueType:$buffer, Variadic<HC_ValueType>:$indices);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $buffer `[` $indices `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Data movement.
//===----------------------------------------------------------------------===//

def HCLoadOp : HC_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Buffer -> tensor load.";
  let description = [{
    Materializes a tensor from a buffer at a variadic subscript. Shape
    metadata (optional `#hc.shape`) travels in the attribute dict rather
    than inline because `#hc.shape`'s free-form printer confuses an inline
    `:` terminator; keeping it inside `{}` keeps the assembly format
    unambiguous at the cost of slightly more verbose textual IR.

    When present, the shape attribute must have the same rank as the index
    list; the verifier rejects rank mismatches. The op carries a `MemRead`
    effect so effect-aware passes cannot reorder it past a store.
  }];
  let arguments = (ins
    HC_ValueType:$buffer,
    Variadic<HC_ValueType>:$indices,
    OptionalAttr<HC_ShapeAttr>:$shape
  );
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $buffer `[` $indices `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCVLoadOp : HC_Op<"vload", [MemoryEffects<[MemRead]>]> {
  let summary = "Buffer or tensor -> vector load.";
  let description = [{
    Vector counterpart of `hc.load`; same attribute-dict shape carrier, same
    rank-match verifier, same `MemRead` effect. Only the source operand's
    admissible type broadens (tensor is also legal) and the result is a
    vector instead of a tensor.
  }];
  let arguments = (ins
    HC_ValueType:$source,
    Variadic<HC_ValueType>:$indices,
    OptionalAttr<HC_ShapeAttr>:$shape
  );
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCStoreOp : HC_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tensor or vector into a buffer or tensor.";
  let arguments = (ins
    HC_ValueType:$dest,
    Variadic<HC_ValueType>:$indices,
    HC_ValueType:$source
  );
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $source attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCVecOp : HC_Op<"vec", [MemoryEffects<[MemRead]>]> {
  let summary = "Tensor -> vector materialization.";
  let description = [{
    Broadcasts a tensor (workgroup-local storage) into a per-lane vector.
    Carries a `MemRead` effect because the source tensor lives in addressable
    memory; two `hc.vec` ops on the same tensor may observe different contents
    if a store is interleaved, so CSE must not collapse them blindly.
  }];
  let arguments = (ins HC_ValueType:$value);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def HCWithInactiveOp : HC_Op<"with_inactive", [Pure]> {
  let summary = "Replace inactive (masked-out) elements with a literal value.";
  let description = [{
    `$inactive` is carried in the attribute dict so later passes can pattern
    match on its payload without worrying about operand discovery. The
    payload must be a typed scalar literal (int, float, or bool) whose
    numeric domain matches the value's element type once inference pins it;
    pre-inference `!hc.undef` operands skip the check.
  }];
  let arguments = (ins
    HC_ValueType:$value,
    AnyAttrOf<[I1Attr, I8Attr, I16Attr, I32Attr, I64Attr,
               F16Attr, F32Attr, F64Attr, IndexAttr, BoolAttr]>:$inactive
  );
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$value attr-dict `:` type($value) `->` type($result)";
}

def HCAsLayoutOp : HC_Op<"as_layout", [Pure]> {
  let summary = "Change the layout of a tensor or vector value.";
  let description = [{
    `$layout` is a named layout descriptor carried in the attribute dict.
    v0 accepts only `"row_major"` and `"col_major"`; the verifier rejects
    any other string. Later revisions will widen the constraint to a typed
    `#hc.layout<...>` attribute without changing call sites.
  }];
  let arguments = (ins HC_ValueType:$value, StrAttr:$layout);
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$value attr-dict `:` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Allocators. Vector variants are scope-free; tensor variants are workgroup
// scope (checked by a later pass). All carry `MemAlloc` so CSE does not
// collapse sibling allocations. Shape travels in the attribute dict to keep
// the assembly format unambiguous. See `doc/lowering.md` "Data movement"
// for the full rationale.
//===----------------------------------------------------------------------===//

class HC_NullaryShapedAllocOp<string mnemonic>
    : HC_Op<mnemonic, [MemoryEffects<[MemAlloc]>]> {
  let arguments = (ins HC_ShapeAttr:$shape);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class HC_FillShapedAllocOp<string mnemonic>
    : HC_Op<mnemonic, [MemoryEffects<[MemAlloc]>]> {
  let arguments = (ins HC_ValueType:$value, HC_ShapeAttr:$shape);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat =
      "$value attr-dict `:` type($value) `->` type($result)";
}

def HCVZerosOp : HC_NullaryShapedAllocOp<"vzeros"> {
  let summary = "Zero-initialized vector.";
}

def HCVOnesOp : HC_NullaryShapedAllocOp<"vones"> {
  let summary = "Ones-initialized vector.";
}

def HCVFullOp : HC_FillShapedAllocOp<"vfull"> {
  let summary = "Vector filled with a scalar value.";
}

def HCZerosOp : HC_NullaryShapedAllocOp<"zeros"> {
  let summary = "Zero-initialized tensor (workgroup scope).";
}

def HCOnesOp : HC_NullaryShapedAllocOp<"ones"> {
  let summary = "Ones-initialized tensor (workgroup scope).";
}

def HCFullOp : HC_FillShapedAllocOp<"full"> {
  let summary = "Tensor filled with a scalar value (workgroup scope).";
}

def HCEmptyOp : HC_NullaryShapedAllocOp<"empty"> {
  let summary = "Uninitialized tensor (workgroup scope).";
}

//===----------------------------------------------------------------------===//
// Reductions, matmul, astype.
//===----------------------------------------------------------------------===//

def HCMatmulOp : HC_Op<"matmul", [Pure]> {
  let summary = "Generic matmul over tensors or vectors.";
  let arguments = (ins HC_ValueType:$lhs, HC_ValueType:$rhs);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def HCReduceOp : HC_Op<"reduce", [Pure]> {
  let summary = "Reduce `$value` along `$axis` using `$kind`.";
  let description = [{
    `$kind` must be one of `"sum"`, `"max"`, `"min"`; the verifier rejects
    any other spelling so typos fail at verify time rather than in some
    later lowering pattern. `$keepdims` matches the NumPy surface and is
    optional with default `false`. `$axis` must be non-negative.
  }];
  let arguments = (ins
    HC_ValueType:$value,
    StrAttr:$kind,
    I64Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` `kind` `=` $kind `,` `axis` `=` $axis
    (`,` `keepdims` `=` $keepdims^)? attr-dict `:`
    type($value) `->` type($result)
  }];
}

def HCAsTypeOp : HC_Op<"astype", [Pure]> {
  let summary = "Explicit numeric conversion (Python `.astype`).";
  let description = [{
    User-visible cast that retains element-type information on the op. Later
    lowering rewrites `hc.astype` into `hc.cast` or a target arithmetic
    conversion op depending on the operand types. The verifier rejects
    non-numeric target types so nonsense spellings (`target = !hc.slice`)
    fail early.
  }];
  let arguments = (ins HC_ValueType:$value, TypeAttr:$target);
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` `target` `=` $target attr-dict `:`
    type($value) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Calls.
//
// Both ops carry a `FlatSymbolRefAttr` callee and verify symbol uses
// through `SymbolUserOpInterface`. `verifySymbolUses` resolves the callee
// against the nearest symbol table and checks the target op *kind*:
// `hc.call` must point at an `hc.func`, `hc.call_intrinsic` at an
// `hc.intrinsic`. Full signature matching waits for `hc.func` /
// `hc.intrinsic` to grow explicit arg/result signatures; wrong-kind
// references and typos are already caught.
//
// Memory effects: conservative `MemRead + MemWrite` by default because the
// callee body is opaque at verify time. Without an effect declaration,
// effect-aware passes (CSE/LICM/speculation) treat opaque calls as
// reorderable with loads/stores, which is unsound as soon as a helper does
// anything beyond `Pure` compute. Once `hc.func` / `hc.intrinsic` carry
// explicit effect class attrs, the narrower effect can be picked up from
// the target op; for now the pessimistic default is the safe choice.
//===----------------------------------------------------------------------===//

def HCCallOp : HC_Op<"call",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call into an `hc.func`.";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<HC_ValueType>:$args);
  let results = (outs Variadic<HC_ValueType>:$results);
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}

def HCCallIntrinsicOp : HC_Op<"call_intrinsic",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call into an `hc.intrinsic` declaration.";
  let description = [{
    Specialization-required keyword arguments live on this op as
    attributes (e.g. `{wave_size = 32 : i64}`), not SSA operands, so they
    are available to verify/specialize hooks without operand bookkeeping.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<HC_ValueType>:$args);
  let results = (outs Variadic<HC_ValueType>:$results);
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}

#endif // HC_IR_HCOPS
