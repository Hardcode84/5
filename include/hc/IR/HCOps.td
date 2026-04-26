// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPS
#define HC_IR_HCOPS

include "hc/IR/HCOpsInterfaces.td"
include "hc/IR/HCTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class HC_RegionOp<string mnemonic, list<Trait> traits = []>
    : HC_Op<mnemonic, !listconcat([NoTerminator, SingleBlock], traits)>;

// Pure binary op parameterized on operand/result constraints. The default
// `HC_ValueType` is deliberately loose for ops whose operand mix is open
// (cmp, and/or, cast); narrower bases below tighten it for arithmetic /
// shaped / boolean flavours. Folders dispatch on concrete-type combinations
// regardless.
class HC_GenericBinaryOp<string mnemonic, list<Trait> traits = [],
                         TypeConstraint operand = HC_ValueType,
                         TypeConstraint result = HC_ValueType>
    : HC_Op<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins operand:$lhs, operand:$rhs);
  let results = (outs result:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

class HC_GenericUnaryOp<string mnemonic, list<Trait> traits = [],
                        TypeConstraint operand = HC_ValueType,
                        TypeConstraint result = HC_ValueType>
    : HC_Op<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins operand:$value);
  let results = (outs result:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

class HCBranchCompatibleTypes<string lhs, string rhs> : TypesMatchWith<
    lhs # " and " # rhs # " have branch-compatible types",
    lhs, rhs, "$_self",
    [{
      [](::mlir::TypeRange lhsTypes, ::mlir::TypeRange rhsTypes) {
        return ::llvm::equal(
            lhsTypes, rhsTypes,
            [&](::mlir::Type lhsType, ::mlir::Type rhsType) {
              return ::mlir::hc::areHCBranchTypesCompatible(lhsType, rhsType);
            });
      }
    }]>;

def HCKernelOp : HC_RegionOp<"kernel", [IsolatedFromAbove, Symbol]> {
  let summary = "Semantic kernel region.";
  let description = [{
    Top-level compiled unit. Carries a module-scoped symbol name (`@name`) so
    call sites and specialization metadata can reference it by flat symbol
    ref instead of string lookup.

    Kernel parameters live as SSA block arguments on the body's entry block;
    the optional `function_type` attribute mirrors their types one-to-one
    (when present) and declares the callable signature. Kernel result types
    are forbidden — kernels terminate with an operand-less `hc.return`.

    Decorator metadata arrives as optional attributes: symbolic `work_shape`
    and `group_shape` as `#hc.shape`, `subgroup_size` as an `i32`, and
    `literals` as a `StrArrayAttr` listing symbols that specialization is
    allowed to bind. All are optional so the `hc_front -> hc` pass can fill
    them in progressively.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
    OptionalAttr<HC_ShapeAttr>:$work_shape,
    OptionalAttr<HC_ShapeAttr>:$group_shape,
    OptionalAttr<I32Attr>:$subgroup_size,
    OptionalAttr<StrArrayAttr>:$literals,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HCFuncOp : HC_RegionOp<"func",
    [IsolatedFromAbove, Symbol,
     DeclareOpInterfaceMethods<CallableOpInterface,
                               ["getCallableRegion", "getArgumentTypes",
                                "getResultTypes"]>]> {
  let summary = "Semantic helper region.";
  let description = [{
    Helper callable referenced by `hc.call`. Carries a symbol name and an
    optional `FunctionType` signature; when the signature is present, call
    sites get arity and per-arg/result type parity checked at verify time
    (`!hc.undef` on either side escapes the parity check, matching the
    progressive-typing policy).

    Parameters live as block arguments of the body's entry block; their
    types mirror `function_type` inputs one-to-one. Legacy frontend output
    may omit the signature entirely, in which case the body block has no
    arguments — that keeps the op parseable while types are inferred. Once
    inference finalizes a signature, it is added and stricter checks kick
    in.

    An optional `effects` enum (`pure | read | write | read_write`)
    advertises the callee's effect class. `hc.call` sites read it via
    `MemoryEffectOpInterface` and narrow their effects accordingly;
    absence defaults to `read_write` — the same conservative fallback
    that kept effect-aware passes from reordering opaque calls before.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements,
    OptionalAttr<HC_EffectsAttr>:$effects,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region SizedRegion<1>:$body);
  let builders = [
    OpBuilder<(ins "StringAttr":$sym_name,
                   CArg<"TypeAttr", "{}">:$function_type,
                   CArg<"ConstraintSetAttr", "{}">:$requirements,
                   CArg<"EffectClassAttr", "{}">:$effects), [{
      build($_builder, $_state, sym_name, function_type, requirements, effects,
            /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// `hc.subgroup_region` and `hc.workitem_region` model Python-style
// nested name scopes: writes inside shadow outer bindings and don't
// leak, reads capture outer bindings (see `NameStoreRegionOpInterface`
// and `-hc-promote-names`). The `$results` list surfaces values that
// the body elects to push out — pre-promotion the body terminates with
// `hc.region_return ["name", ...]` naming the bindings to lift; the
// promoter resolves those names, rewrites the terminator into
// `hc.yield`, and extends the op's results to match. An empty
// `$results` list plus no terminator (or a `hc.region_return`
// terminator to be lowered) is the legal "side-effect only" /
// "pre-promotion" form the frontend emits before promotion; the
// verifier enforces that `$results`-declaring shape always pairs with
// a concrete `hc.yield` of matching arity and type.
def HCSubgroupRegionOp : HC_RegionOp<"subgroup_region",
    [NameStoreRegionOpInterface]> {
  let summary = "Semantic subgroup region.";
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let results = (outs Variadic<HC_ValueType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    (`captures` `=` $captures^)? (`->` `(` type($results)^ `)`)?
      attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def HCWorkitemRegionOp : HC_RegionOp<"workitem_region",
    [NameStoreRegionOpInterface]> {
  let summary = "Semantic workitem region.";
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let results = (outs Variadic<HC_ValueType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    (`captures` `=` $captures^)? (`->` `(` type($results)^ `)`)?
      attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def HCReturnOp : HC_Op<"return", [ReturnLike, Terminator]> {
  let summary = "Semantic return.";
  let description = [{
    Terminator for the nearest enclosing `hc.kernel` / `hc.func` /
    `hc.intrinsic`. `hc.subgroup_region`, `hc.workitem_region`, and the
    structured control-flow ops are transparent — `hc.return` falls through
    them and terminates the callable.

    Kernels never produce a value: `hc.return` inside a `hc.kernel` must be
    operand-less. Funcs and intrinsics with a declared `function_type` must
    return operands matching the declared result types (the `!hc.undef`
    escape still applies for progressive typing).
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Scalar producers and casts.
//===----------------------------------------------------------------------===//

def HCConstOp : HC_Op<"const",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
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
  let hasFolder = 1;
}

def HCUndefValueOp : HC_Op<"undef_value", [ConstantLike, Pure]> {
  let summary = "Typed undefined seed for name-promotion scaffolding.";
  let description = [{
    Produces a `!hc.undef` value with no source operand. The name-promotion
    pass uses it to seed `hc.for_range` iter_args for write-first names: the
    body overwrites the carried value before any in-body read, while a
    zero-trip loop leaves the binding undefined in the output. Keeping this
    placeholder in the `hc` dialect makes surviving scaffolding obvious in IR
    dumps and avoids leaking `builtin.unrealized_conversion_cast` into the
    semantic pipeline.
  }];
  let results = (outs HC_UndefType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasFolder = 1;
}

def HCTupleOp : HC_Op<"tuple",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "First-class tuple value.";
  let description = [{
    Builds a single SSA aggregate using MLIR's builtin `tuple<...>` type.
    This preserves Python tuple structure in `hc` IR so later passes can
    project with `hc.getitem` or destructure explicitly instead of relying on
    converter-local side tables.
  }];
  let arguments = (ins Variadic<HC_ValueType>:$elements);
  let results = (outs HC_TupleValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $elements `)` attr-dict `:` functional-type(operands, results)
  }];
}

def HCSymbolOp : HC_Op<"symbol",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Introduce a symbolic capture bound to a named symbol.";
  let description = [{
    Produces a pinned `!hc.idx<"name">`. The bound name lives in the result
    type; this op carries no operands or attributes so the binding is
    visible in one place only. The verifier rejects a bare `!hc.idx` or a
    non-idx result type. Uniquing on `!hc.idx<E>` gives symbol equality for
    free.
  }];
  let results = (outs HC_IdxType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def HCCastOp : HC_Op<"cast",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
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

// Arithmetic ops reject anything that can't carry a number: `!hc.pred`,
// `!hc.slice`, `!hc.buffer`. Symbolic (`!hc.idx`), scalar (int/float/index),
// tensor/vector, and `!hc.undef` remain legal.
def HCAddOp : HC_GenericBinaryOp<"add",
    [Commutative, DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic addition.";
}

def HCSubOp : HC_GenericBinaryOp<"sub",
    [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic subtraction.";
}

def HCMulOp : HC_GenericBinaryOp<"mul",
    [Commutative, DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic multiplication.";
}

def HCDivOp : HC_GenericBinaryOp<"div",
    [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic integer/float division (Python `//` for ints).";
}

def HCModOp : HC_GenericBinaryOp<"mod",
    [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic remainder (Python `%`).";
}

def HCNegOp : HC_GenericUnaryOp<"neg",
    [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic unary minus.";
}

// Boolean ops stay on `HC_ValueType` for now: `i1` is an `AnyInteger`, and a
// cleanly-scoped `HC_BooleanValueType` would require structural support for
// "integer width == 1" / "tensor element is i1". File as follow-up.
def HCAndOp : HC_GenericBinaryOp<"and", [Commutative]> {
  let summary = "Generic logical AND (Python `and`, `&` on bools).";
}

def HCOrOp : HC_GenericBinaryOp<"or", [Commutative]> {
  let summary = "Generic logical OR (Python `or`, `|` on bools).";
}

def HCNotOp : HC_GenericUnaryOp<"not"> {
  let summary = "Generic logical NOT (Python `not`).";
}

// Comparisons take numeric operands; result stays wide (may be `!hc.pred`,
// `i1`, `tensor<i1>`, `!hc.undef`).
class HC_CmpOp<string suffix>
    : HC_GenericBinaryOp<"cmp." # suffix,
                         [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
                         HC_NumericValueType, HC_ValueType> {
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

def HCIntrinsicOp : HC_RegionOp<"intrinsic", [IsolatedFromAbove, Symbol]> {
  let summary = "Scope-gated semantic intrinsic declaration.";
  let description = [{
    Declares an intrinsic with its scope and (optional) effect class.
    Analogous to `hc_front.intrinsic`; later passes dispatch on `scope` and
    `effects` without string-matching the mnemonic.

    `parameters` records the full declared Python parameter order, while
    `keyword_only` names the entries that Python would not allow to be bound
    positionally. When a signature is present, only non-`const_kwargs`
    parameters live as block arguments of the body's entry block; their types
    mirror `function_type` inputs one-to-one. The body region holds
    verify/infer hooks, lowering recipes, or a fallback body once those settle.
    An empty body is legal and denotes a declaration-only intrinsic that later
    lowering patterns handle directly. Intrinsics with `const_kwargs`,
    `parameters`, or a non-empty runtime input signature must provide enough
    of the trio for the verifier to check their agreement.

    `const_kwargs` declares keyword-only names that every
    `hc.call_intrinsic` for this callee must carry as specialization
    attributes (e.g. `wave_size`, `arch`). `verifySymbolUses` on the call site
    checks every name is present — value type/shape is up to the intrinsic's
    own verify hook because there is no universal "kwarg type" story yet.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
    HC_ScopeAttr:$scope,
    OptionalAttr<HC_EffectsAttr>:$effects,
    OptionalAttr<StrArrayAttr>:$const_kwargs,
    OptionalAttr<StrArrayAttr>:$parameters,
    OptionalAttr<StrArrayAttr>:$keyword_only
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HCYieldOp : HC_Op<"yield",
    [Pure, Terminator,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                               ["getSuccessorRegions"]>,
     ParentOneOf<["HCForRangeOp", "HCIfOp",
                  "HCSubgroupRegionOp", "HCWorkitemRegionOp"]>]> {
  let summary =
      "Terminator for `hc.for_range` / `hc.if` / nested-scope regions.";
  let description = [{
    Yields loop-carried, if-produced, or region-return values back to
    the enclosing op. Value count must match the parent op's result
    count; verification of that runs on the parent op (`hc.for_range`,
    `hc.if`, `hc.workitem_region`, `hc.subgroup_region`) so the
    diagnostic points at the op that declares the result signature,
    not at the terminator.

    For `hc.workitem_region` / `hc.subgroup_region` the yield is the
    post-`-hc-promote-names` form — pre-promotion the body terminates
    with `hc.region_return`, which the pass resolves and rewrites into
    this op.
  }];
  let arguments = (ins Variadic<HC_ValueType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [
    // Needed by `SingleBlockImplicitTerminator` on the parent ops.
    OpBuilder<(ins), [{ /* empty yield */ }]>,
  ];
}

def HCRegionReturnOp : HC_Op<"region_return",
    [Terminator,
     ParentOneOf<["HCSubgroupRegionOp", "HCWorkitemRegionOp"]>]> {
  let summary =
      "Transient terminator: names the bindings this region lifts as results.";
  let description = [{
    Pre-promotion marker emitted by the frontend. `$names` lists, in
    order, the imperative name-store slots whose current binding should
    surface as the enclosing region's result. `-hc-promote-names`
    resolves each name against the body's binding map, rewrites this op
    into `hc.yield` with the resolved SSA values, and extends the
    parent region op's `$results` to match. Does not survive past
    `-hc-promote-names`; any `hc.region_return` that reaches a later
    pass is a driver bug.

    Each name must appear at most once: the promoter produces one
    result per entry and threads one writeback assign per entry into
    the enclosing name store, so duplicates would mean two results
    sharing a name — ambiguous writeback order, and nothing good on
    the read side either.
  }];
  let arguments = (ins StrArrayAttr:$names);
  let assemblyFormat = "$names attr-dict";
  let hasVerifier = 1;
}

def HCForRangeOp : HC_Op<"for_range",
    [SingleBlockImplicitTerminator<"HCYieldOp">,
     HCBranchCompatibleTypes<"iter_results", "iter_inits">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
                               ["getEntrySuccessorOperands",
                                "getSuccessorInputs",
                                "areTypesCompatible"]>,
     DeclareOpInterfaceMethods<HCInferRegionArgTypeOpInterface>,
     NameStoreRegionOpInterface]> {
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
    `(` type($lower) `,` type($upper) `,` type($step) `)`
    (`->` `(` type($iter_results)^ `)`)? $body
  }];
}

def HCIfOp : HC_Op<"if",
    [SingleBlockImplicitTerminator<"HCYieldOp">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
                               ["getSuccessorInputs",
                                "areTypesCompatible"]>,
     NameStoreRegionOpInterface]> {
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
// Frontend lowering gives generated queries internal `$`-prefixed symbols
// (`!hc.idx<"$WG0">`, etc.) that cannot collide with Python-source symbols.
// Marked `Pure`: launch state is invariant for a single kernel instance
// (pipeline invariant; see `doc/lowering.md`
// "Launch-geometry invariant").
//===----------------------------------------------------------------------===//

class HC_LaunchGeoOp<string mnemonic>
    : HC_Op<mnemonic,
            [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let arguments = (ins HC_GroupValueType:$group);
  let results = (outs Variadic<HC_ValueType>:$values);
  let hasVerifier = 1;
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

def HCBufferDimOp : HC_Op<"buffer_dim",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Symbolic dimension of a buffer (matches `a.shape[N]`).";
  let arguments = (ins HC_BufferValueType:$buffer, I64Attr:$axis);
  let results = (outs HC_ValueType:$dim);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $buffer `,` `axis` `=` $axis attr-dict `:` type($buffer) `->` type($dim)
  }];
}

def HCSliceExprOp : HC_Op<"slice_expr",
    [Pure, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
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

def HCBufferViewOp : HC_Op<"buffer_view",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Sub-view into a buffer, tensor, or vector root.";
  let description = [{
    Returns a view of the root reduced by the variadic index/slice operands.
    The variadic list carries a mix of scalar indices and `!hc.slice` values,
    matching Python-style multi-axis subscripts. Vector roots are legal as a
    deferred generic-subscript form; inference leaves them unrefined until a
    later view/getitem specialization can choose element or fragment semantics.
  }];
  let arguments = (ins HC_ViewRootValueType:$buffer,
                       Variadic<HC_ValueType>:$indices);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $buffer `[` $indices `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCGetItemOp : HC_Op<"getitem",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Generic Python square-bracket indexing.";
  let description = [{
    Preserves Python `base[index...]` structure before the base kind is known.
    Later inference/refinement can specialize tuple item, buffer/tensor view,
    or element cases without requiring the frontend conversion to choose early.
  }];
  let arguments = (ins AnyType:$base, Variadic<HC_ValueType>:$indices);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Data movement.
//===----------------------------------------------------------------------===//

def HCLoadOp : HC_Op<"load",
    [MemoryEffects<[MemRead]>,
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Buffer -> tensor load.";
  let description = [{
    Materializes a tensor from a buffer at a variadic subscript. The result
    shape is an ordinary SSA value, normally a tuple of idx-typed dimensions.
    A missing source-level shape is represented as `!hc.undef`; a later
    verifier pass owns static-shape diagnostics. The op carries a `MemRead`
    effect so effect-aware passes cannot reorder it past a store.
  }];
  let arguments = (ins
    HC_BufferValueType:$buffer,
    Variadic<HC_ValueType>:$indices,
    HC_TupleValueType:$shape
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $buffer `[` $indices `]` `,` `shape` $shape attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCVLoadOp : HC_Op<"vload",
    [MemoryEffects<[MemRead]>,
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Buffer or tensor -> vector load.";
  let description = [{
    Vector counterpart of `hc.load`; same SSA shape carrier and `MemRead`
    effect. Only the source operand's admissible type broadens (tensor is also
    legal) and the result is a vector instead of a tensor.
  }];
  let arguments = (ins
    HC_BufferOrTensorValueType:$source,
    Variadic<HC_ValueType>:$indices,
    HC_TupleValueType:$shape
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $source `[` $indices `]` `,` `shape` $shape attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCStoreOp : HC_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tensor or vector into a buffer or tensor.";
  let description = [{
    Stores the shaped `$source` value into `$dest`. Unlike load and allocator
    ops, store does not need a separate shape operand because the stored value's
    type already carries the tensor/vector shape.
  }];
  let arguments = (ins
    HC_BufferOrTensorValueType:$dest,
    Variadic<HC_ValueType>:$indices,
    HC_ShapedValueType:$source
  );
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $source attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCVecOp : HC_Op<"vec",
    [MemoryEffects<[MemRead]>,
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Tensor -> vector materialization.";
  let description = [{
    Broadcasts a tensor (workgroup-local storage) into a per-lane vector.
    Carries a `MemRead` effect because the source tensor lives in addressable
    memory; two `hc.vec` ops on the same tensor may observe different contents
    if a store is interleaved, so CSE must not collapse them blindly.
  }];
  let arguments = (ins HC_ShapedValueType:$value);
  let results = (outs HC_ShapedValueType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def HCWithInactiveOp : HC_Op<"with_inactive",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Replace inactive (masked-out) elements with an SSA value.";
  let description = [{
    `$inactive` is an ordinary scalar operand. When inference pins both
    operands, its numeric domain must match the shaped value's element domain;
    pre-inference `!hc.undef` operands skip the check.
  }];
  let arguments = (ins
    HC_ShapedValueType:$value,
    HC_ScalarValueType:$inactive
  );
  let results = (outs HC_ShapedValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$value `,` $inactive attr-dict `:` functional-type(operands, results)";
}

def HCAsLayoutOp : HC_Op<"as_layout",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Change the layout of a tensor or vector value.";
  let description = [{
    `$layout` is a typed `#hc<layout ...>` enum attribute. v0 admits
    `row_major` and `col_major`; wrong spellings fail at parse, not verify.
    Lifted out of `attr-dict` so the surface reads like the other explicit
    keyword-argument ops (`hc.reduce`, `hc.astype`).
  }];
  let arguments = (ins HC_ShapedValueType:$value, HC_LayoutAttr:$layout);
  let results = (outs HC_ShapedValueType:$result);
  let assemblyFormat = [{
    $value `,` `layout` `=` $layout attr-dict `:`
    type($value) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Allocators. Vector variants are scope-free; tensor variants are workgroup
// scope (checked by a later pass). All carry `MemAlloc` so CSE does not
// collapse sibling allocations. Shape is an ordinary SSA value so frontend
// `shape=` kwargs do not need a dialect-side attribute escape hatch.
//===----------------------------------------------------------------------===//

class HC_NullaryShapedAllocOp<string mnemonic>
    : HC_Op<mnemonic, [
        MemoryEffects<[MemAlloc]>,
        DeclareOpInterfaceMethods<HCInferTypeOpInterface>
      ]> {
  let arguments = (ins
    HC_TupleValueType:$shape,
    OptionalAttr<TypeAttr>:$dtype
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    `shape` $shape (`,` `dtype` `=` $dtype^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

class HC_FillShapedAllocOp<string mnemonic>
    : HC_Op<mnemonic, [
        MemoryEffects<[MemAlloc]>,
        DeclareOpInterfaceMethods<HCInferTypeOpInterface>
      ]> {
  let arguments = (ins
    HC_ValueType:$value,
    HC_TupleValueType:$shape,
    OptionalAttr<TypeAttr>:$dtype
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $value `,` `shape` $shape (`,` `dtype` `=` $dtype^)? attr-dict `:`
    functional-type(operands, results)
  }];
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
  let hasVerifier = 1;
}

def HCOnesOp : HC_NullaryShapedAllocOp<"ones"> {
  let summary = "Ones-initialized tensor (workgroup scope).";
  let hasVerifier = 1;
}

def HCFullOp : HC_FillShapedAllocOp<"full"> {
  let summary = "Tensor filled with a scalar value (workgroup scope).";
  let hasVerifier = 1;
}

def HCEmptyOp : HC_NullaryShapedAllocOp<"empty"> {
  let summary = "Uninitialized tensor (workgroup scope).";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reductions, matmul, astype.
//===----------------------------------------------------------------------===//

def HCMatmulOp : HC_Op<"matmul", [Pure]> {
  let summary = "Generic matmul over tensors or vectors.";
  let arguments = (ins HC_ShapedValueType:$lhs, HC_ShapedValueType:$rhs);
  let results = (outs HC_ShapedValueType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def HCReduceOp : HC_Op<"reduce", [Pure]> {
  let summary = "Reduce `$value` along `$axis` using `$kind`.";
  let description = [{
    `$kind` is the typed `#hc<reduce_kind ...>` enum (`sum` / `max` / `min`);
    wrong spellings fail at parse rather than verify. `$keepdims` matches the
    NumPy surface and is optional with default `false`. `$axis` must be
    non-negative and, when `$value` has a concrete rank, less than that rank.
  }];
  let arguments = (ins
    HC_ShapedValueType:$value,
    HC_ReduceKindAttr:$kind,
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
// against the nearest symbol table, checks the target op *kind* (`hc.call`
// → `hc.func`, `hc.call_intrinsic` → `hc.intrinsic`), and — when the
// callee carries a `function_type` — checks arity and per-arg/result
// parity with `!hc.undef` as the progressive-typing escape hatch.
//
// Memory effects are read from the callee's declared `effects` enum when
// present and default to `MemRead + MemWrite` otherwise — an opaque helper
// that may read *and* write is the conservative assumption, and it keeps
// effect-aware passes (CSE/LICM/speculation) from reordering calls past
// loads/stores until the callee opts into something narrower.
//===----------------------------------------------------------------------===//

def HCCallOp : HC_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface,
                               ["getCallableForCallee",
                                "setCalleeFromCallable", "getArgOperands",
                                "getArgOperandsMutable"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Call into an `hc.func`.";
  let description = [{
    Memory effects are inherited from the callee's declared `effects`
    class (if present) and default to `MemRead + MemWrite` otherwise —
    an opaque helper that may read *and* write is the conservative
    assumption, and it keeps effect-aware passes from reordering calls
    past loads/stores until the callee opts into something narrower.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<HC_ValueType>:$args,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<HC_ValueType>:$results);
  let builders = [
    OpBuilder<(ins "TypeRange":$results, "FlatSymbolRefAttr":$callee,
                   CArg<"ValueRange", "{}">:$args), [{
      build($_builder, $_state, results, callee, args,
            /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    }]>,
  ];
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}

def HCCallIntrinsicOp : HC_Op<"call_intrinsic",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Call into an `hc.intrinsic` declaration.";
  let description = [{
    Specialization-required keyword arguments live on this op as
    attributes (e.g. `{wave_size = 32 : i64}`), not SSA operands, so they
    are available to verify/specialize hooks without operand bookkeeping.

    Memory effects are inherited from the callee intrinsic's `effects`
    class; absence defaults to `MemRead + MemWrite`, same policy as
    `hc.call`.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<HC_ValueType>:$args);
  let results = (outs Variadic<HC_ValueType>:$results);
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}

//===----------------------------------------------------------------------===//
// Imperative name store (transient).
//
// Python uses names-as-slots: `acc = ...` inside a loop rebinds a name
// that outlives the iteration. Every region-carrying op in this dialect
// (`hc.for_range`, `hc.if`, `hc.{workitem,subgroup}_region`) has to join
// those rebinds back into SSA — iter_args + yields, results + branch
// yields, region captures + region results.
//
// Rather than doing that analysis inline in `-convert-hc-front-to-hc`
// (and repeating it per region kind), the frontend pass emits these two
// ops to preserve the name-store shape, and a dedicated pass
// `-hc-promote-names` reconstructs SSA uniformly.
//
// Both ops are transient: they must not survive past `-hc-promote-names`.
// The verifier contract enforcing that belongs on the consumer pass, not
// on these ops themselves.
//===----------------------------------------------------------------------===//

def HCAssignOp : HC_Op<"assign", [MemoryEffects<[MemWrite]>]> {
  let summary = "Imperative write to a named slot; pre-SSA placeholder.";
  let description = [{
    `hc.assign "x", %v` records a write to the Python-level name `x` at
    this program point. Produces no SSA result — consumers read through
    `hc.name_load` and `-hc-promote-names` rewrites each such read into
    a direct SSA use (or into an iter_arg / block arg manufactured for
    the enclosing region op).

    Carries `MemWrite` on `DefaultResource`. The effect pairs with
    `hc.name_load`'s `MemRead` so that effect-aware passes (`-cse`,
    speculation, LICM) see a read/write pair on the same resource and
    keep same-name reads in program order around the assign. `MemWrite`
    does not drive DCE — writes are never trivially dead even with no
    SSA result — so as far as DCE is concerned the trait is bookkeeping;
    the symmetry with `hc.store` is cosmetic but keeps the pair easy
    to pattern on.

    Aliasing against real buffer traffic (`hc.load` / `hc.store`) is
    necessarily pessimistic: we reuse `DefaultResource` rather than
    declaring a dedicated name-store resource kind. That is safe for
    these transient ops; if a later pass wants to hoist across
    name-store boundaries, a dedicated resource is the documented
    escape hatch.

    Cross-kernel / cross-function name capture is out of scope: each
    `hc.kernel` / `hc.func` owns its own name store, and promotion
    operates on one function region tree at a time.

    ### Induction-variable self-bind

    The one shape worth calling out explicitly — because it looks
    redundant and isn't — is the induction-variable self-assign at the
    top of an `hc.for_range` body:

    ```
    hc.for_range %lb, %ub, %step : !hc.undef -> () {
    ^bb0(%iv: !hc.undef):
      hc.assign "i", %iv : !hc.undef    // <-- IV self-bind
      ...
      hc.region_return
    }
    ```

    The frontend emits this pattern unconditionally so every Python
    name has a reaching `hc.assign` before it is read, including the
    loop-counter name, without the frontend having to reason about
    whether the counter is used in the body. `-hc-promote-names`
    detects the pattern (any `hc.assign "<name>", %iv` where `%iv` is
    the `hc.for_range` block argument) and erases it, then resolves
    reads of `"i"` directly to `%iv`. Specifically it is *not* treated
    as a carried-name / iter_arg candidate. See
    `extractIvSelfBinds` / `promoteForRange` in
    `lib/Transforms/HCPromoteNamesPass.cpp` for the matcher, and
    `test/HC/promote-names-regions.mlir`'s `@for_range_iv_self_bind*`
    cases for the pinned behaviour.
  }];
  let arguments = (ins StrAttr:$name, HC_ValueType:$value);
  let assemblyFormat = "$name `,` $value attr-dict `:` type($value)";
}

def HCNameLoadOp : HC_Op<"name_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Imperative read from a named slot; pre-SSA placeholder.";
  let description = [{
    `%v = hc.name_load "x"` materializes the current binding of name `x`
    at this program point. The result type is the usual open
    `HC_ValueType` so the op is usable pre-inference with `!hc.undef`;
    `-hc-promote-names` replaces each use with the reaching definition
    (the nearest dominating `hc.assign`, or a new block/iter arg when
    the reach crosses a region boundary) and erases the load.

    Carries `MemRead` on `DefaultResource`. Two generic passes key off
    this effect. Standard `-cse` skips ops whose effect list is
    non-empty, so two `hc.name_load` ops of the same name — with or
    without an intervening `hc.assign` — are never merged. MLIR's
    `wouldOpBeTriviallyDead` treats a read-only op with no uses as
    dead, so `-canonicalize` erases unused loads without waiting on
    `-hc-promote-names`. `test/HC/name-store-effects.mlir` pins both
    behaviours against the pinned MLIR revision.

    Aliasing is pessimistic against real buffer traffic for the same
    reason as `hc.assign`; the same escape hatch (a dedicated
    name-store resource kind) applies if it ever matters.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

#endif // HC_IR_HCOPS
