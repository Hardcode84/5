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

// Pure binary op base. Default `HC_ValueType` is loose for open-mix ops
// (cmp, and/or, cast); narrower bases below tighten it. Folders dispatch
// on concrete-type combinations either way.
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
    Top-level compiled unit. Carries a module-scoped symbol name (`@name`).

    Kernel parameters live as block args on the body's entry block.
    Optional `function_type` mirrors them and declares the callable
    signature. Kernel results are forbidden — kernels terminate with an
    operand-less `hc.return`.

    Decorator metadata (all optional, filled in progressively):

      * `work_shape`, `group_shape`: `#hc.shape`
      * `subgroup_size`: `i32`
      * `bound_symbols`: `StrArrayAttr` of launch / kernel-input ABI
        symbols allowed to materialize.
      * `literals`: `StrArrayAttr` of symbols specialization may bind.
      * `literal_bindings`: `DictionaryAttr` (`{name = i64}`) pinning
        integer values for `hc-specialize-literals`.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
    OptionalAttr<HC_ShapeAttr>:$work_shape,
    OptionalAttr<HC_ShapeAttr>:$group_shape,
    OptionalAttr<I32Attr>:$subgroup_size,
    OptionalAttr<StrArrayAttr>:$bound_symbols,
    OptionalAttr<StrArrayAttr>:$literals,
    OptionalAttr<DictionaryAttr>:$literal_bindings,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HCFuncOp : HC_Op<"func",
    [IsolatedFromAbove, Symbol,
     SingleBlockImplicitTerminator<"HCReturnOp">,
     DeclareOpInterfaceMethods<CallableOpInterface,
                               ["getCallableRegion", "getArgumentTypes",
                                "getResultTypes"]>]> {
  let summary = "Semantic helper region.";
  let description = [{
    Helper callable for `hc.call`. Symbol name + optional `FunctionType`.
    With signature present, call sites get arity + per-arg/result parity
    at verify time; `!hc.undef` on either side escapes parity
    (progressive typing).

    Parameters are block args mirroring `function_type` inputs. A missing
    signature leaves the body block argless — parseable while inference
    runs; stricter checks engage once a signature is added.

    Optional `effects` enum (`pure | read | write | read_write`) advertises
    the callee's effect class to `hc.call` sites via
    `MemoryEffectOpInterface`. Absent = `read_write` (conservative).
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

// Python-style nested name scopes: inner writes shadow outer bindings
// without leaking; inner reads capture outer bindings. `$results` lifts
// values out — pre-promotion the body terminates with
// `hc.region_return ["name", ...]`, post-promotion with `hc.yield`. Empty
// `$results` + no terminator (or a `hc.region_return` to be lowered) is
// the pre-promotion side-effect-only form. Verifier requires
// `$results`-declaring shape to pair with `hc.yield` of matching arity
// and type.
def HCSubgroupRegionOp : HC_RegionOp<"subgroup_region",
    [NameStoreRegionOpInterface,
     DeclareOpInterfaceMethods<HCYieldedResultsOpInterface>]> {
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
    [NameStoreRegionOpInterface,
     DeclareOpInterfaceMethods<HCYieldedResultsOpInterface>]> {
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
    Terminator for the nearest `hc.kernel` / `hc.func` / `hc.intrinsic`.
    `hc.{subgroup,workitem}_region` and the structured CFG ops are
    transparent — `hc.return` falls through to the callable.

    `hc.return` in a kernel must be operand-less. With a declared
    `function_type`, operands must match declared result types; `!hc.undef`
    escapes (progressive typing).
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
    Builtin attribute payload lowered from `hc_front.constant`. Result type
    starts `!hc.undef`; inference refines (e.g. `!hc.idx<"7">` for integer
    literals in index context).
  }];
  let arguments = (ins AnyAttr:$value);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "`<` $value `>` attr-dict `:` type($result)";
  let hasFolder = 1;
}

def HCUndefValueOp : HC_Op<"undef_value", [ConstantLike, Pure]> {
  let summary = "Typed undefined seed for name-promotion scaffolding.";
  let description = [{
    Produces a `!hc.undef` value with no source operand. Name promotion
    seeds `hc.for_range` iter_args for write-first names with this — body
    overwrites before any in-body read, zero-trip loop leaves binding
    undefined. Lives in `hc` so surviving scaffolding is visible in dumps
    instead of leaking as `builtin.unrealized_conversion_cast`.
  }];
  let results = (outs HC_UndefType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasFolder = 1;
}

def HCTupleOp : HC_Op<"tuple",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "First-class tuple value.";
  let description = [{
    SSA aggregate built on MLIR's builtin `tuple<...>`. Preserves Python
    tuple structure so passes can project with `hc.getitem` or
    destructure, instead of leaning on converter-local side tables.
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
    Pinned `!hc.idx<"name">`. Bound name lives on the result type — op has
    no operands or attributes, binding visible in one place. Verifier
    rejects bare `!hc.idx` and non-idx results. `!hc.idx<E>` uniquing
    gives symbol equality.
  }];
  let results = (outs HC_IdxType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def HCIdxApplyOp : HC_Op<"idx_apply",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary =
      "Materialize an `!hc.idx<expr>` from explicitly bound operand values.";
  let description = [{
    Symbolic expression on the result type (`!hc.idx<expr>`) plus explicit
    bindings of `expr`'s free symbols to SSA operands. `symbols[k]` names
    the free symbol supplied by `operands[k]`. Unlisted free symbols are
    ambient — resolved at lowering from the surrounding launch context
    (`$WG*` / `$WI*` / `$WGS*`).

    Operands are `index` or `!hc.idx<...>`; both fold to `index` via the
    launch-body type converter before substitution.

    Empty `symbols` / empty `operands` is the severing form — type-only,
    falls entirely through ambient binding. Producers use it when SSA
    values for free symbols aren't yet available (kernel ABI shape syms,
    stride params); sites holding the runtime SSA bind it so every
    user-named symbol becomes a dataflow edge.
  }];
  let arguments = (ins
    Variadic<HC_SymBindingValueType>:$operands,
    StrArrayAttr:$symbols
  );
  let results = (outs HC_IdxType:$result);
  let assemblyFormat = [{
    custom<ApplyBindings>($operands, $symbols) attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def HCPredApplyOp : HC_Op<"pred_apply",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary =
      "Materialize an `!hc.pred<pred>` from explicitly bound operand values.";
  let description = [{
    Predicate-side `hc.idx_apply`. Same operand semantics, same listed-
    vs-ambient split, same lowering convention; result type is
    `!hc.pred<...>`. Operands remain integer-kind because predicate
    leaves are still integer symbols.
  }];
  let arguments = (ins
    Variadic<HC_SymBindingValueType>:$operands,
    StrArrayAttr:$symbols
  );
  let results = (outs HC_PredType:$result);
  let assemblyFormat = [{
    custom<ApplyBindings>($operands, $symbols) attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def HCCastOp : HC_Op<"cast",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "User-visible value-level type conversion.";
  let description = [{
    Generic value-changing cast: refinement transitions
    (`!hc.undef -> !hc.idx<E>`), symbolic-to-builtin exits
    (`!hc.idx<E> -> index`), vector layout changes. Distinct from
    `unrealized_conversion_cast` (compiler scaffolding that later passes
    erase) and from `hc.astype` (Python `.astype` surface carrying a
    target element-type attribute).
  }];
  let arguments = (ins HC_ValueType:$source);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Generic arithmetic, boolean, and comparison ops.
//
// Accept any `HC_ValueType` combination. Frontend picks the op from the
// Python surface. Concrete meaning — symbolic via ixsimpl, scalar arith,
// broadcast tensor — resolves through type inference and per-type folders.
//===----------------------------------------------------------------------===//

// Arithmetic rejects non-numeric kinds (`!hc.pred`, `!hc.slice`,
// `!hc.buffer`). Symbolic, scalar, shaped, and `!hc.undef` legal.
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

// Mechanical carrier for Python `**`. `-hc-lower-pow` rewrites supported
// shapes (positive-integer-literal rhs) into mul chains the rest of the
// pipeline understands and diagnoses anything else at the carrier rather
// than letting downstream guess from an opaque mul chain.
def HCPowOp : HC_GenericBinaryOp<"pow",
    [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic exponentiation (Python `**`).";
}

def HCNegOp : HC_GenericUnaryOp<"neg",
    [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
    HC_NumericValueType, HC_NumericValueType> {
  let summary = "Generic unary minus.";
}

// Open-ended named-function call (NumPy ufuncs and similar). Distinct
// from Python-operator ops (`hc.add` / `hc.neg` / ...) which canonicalize
// at parse time.
//
//   * `name`: qualified dotted Python name; lowering dispatches on the
//     full path.
//   * `args`: variadic — unary, binary, n-ary uniformly.
//   * Single result. Multi-result builtins (`np.divmod`) need their own
//     op; none are supported today.
//
// Invariant: only names matching `isElementwiseBuiltinCall` reach this
// op, so readers may assume elementwise homogeneous semantics. Reshape
// / where / advanced indexing belong elsewhere — don't widen.
def HCBuiltinCallOp : HC_Op<"builtin_call", [Pure,
    DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Call to a named library function (e.g. `numpy.sqrt`).";
  let arguments = (ins StrAttr:$name, Variadic<HC_NumericValueType>:$args);
  let results = (outs HC_NumericValueType:$result);
  let assemblyFormat =
      "$name `(` $args `)` attr-dict `:` functional-type(operands, results)";
}

// Boolean ops stay on `HC_ValueType`: `i1` falls under `AnyInteger`, and
// a tight `HC_BooleanValueType` needs "width == 1" / "elem is i1"
// structural constraints we don't have yet.
def HCAndOp : HC_GenericBinaryOp<"and", [Commutative]> {
  let summary = "Generic logical AND (Python `and`, `&` on bools).";
}

def HCOrOp : HC_GenericBinaryOp<"or", [Commutative]> {
  let summary = "Generic logical OR (Python `or`, `|` on bools).";
}

def HCNotOp : HC_GenericUnaryOp<"not"> {
  let summary = "Generic logical NOT (Python `not`).";
}

// Comparisons: numeric operands, wide result (`!hc.pred` / `i1` /
// `tensor<i1>` / `!hc.undef`).
class HC_CmpOp<string suffix>
    : HC_GenericBinaryOp<"cmp." # suffix,
                         [DeclareOpInterfaceMethods<HCInferTypeOpInterface>],
                         HC_NumericValueType, HC_ValueType> {
  let summary = "Generic comparison (`" # suffix # "`).";
  let description = [{
    Symbolic index operands fold to `!hc.pred<"...">`; scalars produce
    `i1`; tensors broadcast to `tensor<...xi1>`.
  }];
}

def HCCmpLtOp : HC_CmpOp<"lt">;
def HCCmpLeOp : HC_CmpOp<"le">;
def HCCmpGtOp : HC_CmpOp<"gt">;
def HCCmpGeOp : HC_CmpOp<"ge">;
def HCCmpEqOp : HC_CmpOp<"eq">;
def HCCmpNeOp : HC_CmpOp<"ne">;

def HCSelectOp : HC_Op<"select", [Pure]> {
  let summary = "Select between bare shaped data and a scalar inactive value.";
  let description = [{
    Post-decomposition form. `$condition` is a bare `!hc.pred`
    tensor/vector mask; `$true_value` is matching bare data; `$false_value`
    is the scalar fill for inactive elements.
  }];
  let arguments = (ins
    HC_BareShapedValueType:$condition,
    HC_BareShapedValueType:$true_value,
    HC_ScalarValueType:$false_value
  );
  let results = (outs HC_BareShapedValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$condition `,` $true_value `,` $false_value attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Intrinsic declaration and structured control flow.
//===----------------------------------------------------------------------===//

def HCIntrinsicOp : HC_RegionOp<"intrinsic", [IsolatedFromAbove, Symbol]> {
  let summary = "Scope-gated semantic intrinsic declaration.";
  let description = [{
    Intrinsic declaration with scope and optional effect class. Passes
    dispatch on `scope` / `effects` without mnemonic matching.

    `parameters` is the full declared Python parameter order;
    `keyword_only` names entries Python forbids binding positionally.
    Block args mirror `function_type` minus `const_kwargs`. Body holds
    verify/infer hooks, lowering recipes, or a fallback; an empty body is
    a declaration-only intrinsic handled by lowering patterns directly.
    The `const_kwargs` / `parameters` / signature trio must agree if any
    is set.

    `const_kwargs` lists keyword-only specialization names every
    `hc.call_intrinsic` must carry (`wave_size`, `arch`, ...).
    `verifySymbolUses` checks presence; value type/shape is the
    intrinsic's own verify hook (no universal kwarg-type story).
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
                  "HCSubgroupRegionOp", "HCWorkitemRegionOp",
                  "HCGenericOp"]>]> {
  let summary =
      "Terminator for `hc.for_range` / `hc.if` / nested-scope regions.";
  let description = [{
    Yields values to the enclosing op. Value count must match parent
    result count; the check lives on the parent op so the diagnostic
    points at the result-declaring op, not the terminator.

    For `hc.{workitem,subgroup}_region` this is the post-promotion form
    — pre-promotion uses `hc.region_return`.

    Inside `hc.generic` this is the unconditional terminator publishing
    into outs slots every iteration. `hc.yield_predicated` covers the
    per-value mask-gated case without an `scf.if` in the body.
  }];
  let arguments = (ins Variadic<HC_ValueType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [
    // Required by `SingleBlockImplicitTerminator` on the parent ops.
    OpBuilder<(ins), [{ /* empty yield */ }]>,
  ];
}

def HCYieldPredicatedOp : HC_Op<"yield_predicated",
    [Pure, Terminator, SameVariadicOperandSize,
     ParentOneOf<["HCGenericOp"]>]> {
  let summary =
      "Predicated terminator for `hc.generic` bodies: per-value masked publish.";
  let description = [{
    Per-value masked `hc.yield`. Position `i` publishes `$values[i]` into
    outs slot `i` where `$masks[i]` is true; masked-out lanes keep the
    outs-as-init carry. Strict parity: one mask per value.

    Mask shape tracks the value: scalar value pairs `i1`, `vector<NxT>`
    pairs `vector<Nxi1>` of matching N. `!hc.undef` on either side
    escapes the parity check.

    Always-true positions take `arith.constant true : i1`; lowering folds
    those back to unpredicated stores. Single verifier rule — every mask
    is an `i1` SSA.

    Consumers (masked store rewrite, `hc.load_mask` rewrite, OOB-tile
    guard) share the handoff: compute mask from iter syms + ambient
    bounds, ride it on `$masks`, lowering picks predicated mem op or
    `arith.select`.
  }];
  let arguments = (ins Variadic<HC_ValueType>:$values,
                       Variadic<HC_PtrPredicateType>:$masks);
  let assemblyFormat = [{
    $values `mask` $masks attr-dict `:`
    `(` type($values) `)` `,` `(` type($masks) `)`
  }];
  let hasVerifier = 1;
}

def HCRegionReturnOp : HC_Op<"region_return",
    [Terminator,
     ParentOneOf<["HCSubgroupRegionOp", "HCWorkitemRegionOp"]>]> {
  let summary =
      "Transient terminator: names the bindings this region lifts as results.";
  let description = [{
    Pre-promotion marker. `$names` lists, in order, the imperative
    name-store slots to lift as enclosing region results.
    `-hc-promote-names` resolves each against the body's binding map,
    rewrites to `hc.yield` with the resolved SSA, and extends the parent
    op's `$results`. Must not survive past `-hc-promote-names`.

    Each name appears at most once — duplicates would produce results
    sharing a name (ambiguous writeback order).
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
    Mirrors `scf.for`. Operands typed `HC_ValueType` so the op survives
    pre-inference (`!hc.undef` for lb/ub/step); inference narrows to
    `!hc.idx` / builtin integers.

    Body block args: induction variable + iter_args. Block-arg count
    = `1 + iter_inits.size()`; each result type matches the corresponding
    `iter_inits` operand.
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
    Mirrors `scf.if`. Condition is any `HC_ValueType` (typically `i1` /
    `!hc.pred` / `!hc.undef`). Result types annotated so the op survives
    pre-inference; verifier matches each non-empty region's yield count
    and types against the op's signature and requires `else` when the op
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
// Single group-handle in, one or more `HC_ValueType` out. Generated queries
// carry `$`-prefixed symbols (`!hc.idx<"$WG0">`, ...) that can't collide
// with Python-source names. `Pure` — launch state is invariant per kernel
// instance (`doc/lowering.md`).
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
    Single result = product of `hc.group_shape` components.
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
    `!hc.slice` from any subset of `lower` / `upper` / `step`. Each part
    is optional; the frontend emits only syntactically-present parts
    (`x[1:]` vs `x[1:10:2]`). `AttrSizedOperandSegments` pins the
    operand-group mapping at verify time.

    Missing parts use Python slice defaults (`None`): omitted `lower` =
    "from start", `upper` = "to end", `step` = 1.
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
    View of the root reduced by variadic index/slice operands. The list
    mixes scalar indices and `!hc.slice` values (Python multi-axis
    subscripts). Vector roots follow the same shape-reduction rules:
    scalar indices consume axes, slices produce vector fragments, indexing
    every axis yields the element type.

    `$unit_axes`: output positions where NumPy `None` / `np.newaxis`
    inserts a unit dim. `x[:, None, :]` becomes
    `hc.buffer_view %x[%full, %full] {unit_axes = [1]}`; positions are in
    output-rank coordinates. `$indices` carries no `None` placeholder —
    indices consume source axes 1:1; unit dims are interleaved into the
    result shape by inference.
  }];
  let arguments = (ins HC_ViewRootValueType:$buffer,
                       Variadic<HC_ValueType>:$indices,
                       OptionalAttr<DenseI64ArrayAttr>:$unit_axes);
  let results = (outs HC_ValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $buffer `[` $indices `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCGetItemOp : HC_Op<"getitem",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Generic Python square-bracket indexing.";
  let description = [{
    Carrier for `base[index...]` before the base kind is known. Inference
    specializes into tuple item / buffer or tensor view / element access
    without forcing the frontend to commit early.
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
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
     DeclareOpInterfaceMethods<HCStaticShapeOpInterface>]> {
  let summary = "Buffer -> tensor load.";
  let description = [{
    Materializes a tensor from a buffer at a variadic subscript. Result
    shape is an SSA tuple of idx-typed dims; missing source-level shape
    is `!hc.undef`, verified later. `MemRead` keeps reorder past a store
    out.
  }];
  let arguments = (ins
    HC_BufferValueType:$buffer,
    Variadic<HC_ValueType>:$indices,
    HC_TupleValueType:$shape,
    OptionalAttr<HC_LayoutAttr>:$layout
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $buffer `[` $indices `]` `,` `shape` $shape attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCVLoadOp : HC_Op<"vload",
    [MemoryEffects<[MemRead]>,
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
     DeclareOpInterfaceMethods<HCStaticShapeOpInterface>]> {
  let summary = "Buffer or tensor -> vector load.";
  let description = [{
    Vector `hc.load`. Same SSA shape carrier and `MemRead`. Source widens
    to admit tensors; result is a vector.
  }];
  let arguments = (ins
    HC_BufferOrTensorValueType:$source,
    Variadic<HC_ValueType>:$indices,
    HC_TupleValueType:$shape,
    OptionalAttr<HC_LayoutAttr>:$layout
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $source `[` $indices `]` `,` `shape` $shape attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCLoadMaskOp : HC_Op<"load_mask", [Pure]> {
  let summary = "Compute the validity mask for a shaped load.";
  let description = [{
    Decomposed-`hc.load` / `hc.vload` companion. Same addressing operands
    as the data load; result is a bare `!hc.pred` tensor/vector of
    per-element validity.
  }];
  let arguments = (ins
    HC_BufferOrTensorValueType:$source,
    Variadic<HC_ValueType>:$indices,
    HC_TupleValueType:$shape
  );
  let results = (outs HC_BareShapedValueType:$mask);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `[` $indices `]` `,` `shape` $shape attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCFullMaskOp : HC_Op<"full_mask", [Pure]> {
  let summary = "Produce an all-true bare predicate mask.";
  let description = [{
    Seed for values whose whole static shape is valid after decomposition
    — allocators and inactive-filled vectors.
  }];
  let results = (outs HC_BareShapedValueType:$mask);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($mask)";
}

def HCStoreOp : HC_Op<"store", [MemoryEffects<[MemWrite]>,
                                 AttrSizedOperandSegments]> {
  let summary = "Store a tensor or vector into a buffer or tensor.";
  let description = [{
    Stores shaped `$source` into `$dest`. Post-decomposition, `$source`
    may be bare data with `$mask` as the matching bare `!hc.pred`
    validity. No separate shape operand — the source type carries it.
    Operand segments separate variadic indices from the optional mask.
  }];
  let arguments = (ins
    HC_BufferOrTensorValueType:$dest,
    Variadic<HC_ValueType>:$indices,
    HC_DecomposableShapedValueType:$source,
    Optional<HC_BareShapedValueType>:$mask
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $source (`,` `mask` $mask^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCVecOp : HC_Op<"vec",
    [MemoryEffects<[MemRead]>,
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Tensor -> vector materialization.";
  let description = [{
    Broadcasts a tensor (workgroup-local storage) into a per-lane vector.
    `MemRead` because the source lives in addressable memory — two
    `hc.vec` over the same tensor may see different contents if a store
    interleaves, so CSE must not collapse them blindly.
  }];
  let arguments = (ins
    HC_TensorLikeValueType:$value,
    OptionalAttr<HC_LayoutAttr>:$layout
  );
  let results = (outs HC_VectorLikeValueType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def HCWithInactiveOp : HC_Op<"with_inactive",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Replace inactive (masked-out) elements with an SSA value.";
  let description = [{
    `$inactive` is a scalar. Once inference pins both operands, its
    numeric domain must match the shaped value's element domain;
    `!hc.undef` operands skip the check.
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

def HCStripLayoutOp : HC_Op<"strip_layout",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Drop a layout-bearing value's layout, project to bare storage.";
  let description = [{
    User-marked boundary: operand's layout no longer applies, value
    becomes a standalone bare carrier. Surface spelling:
    ``as_layout(value, None)``.

    `hc.as_layout` preserves storage span (verifier matches effective
    ``storage_size``). Strip *changes* it: the operand may carry a
    non-injective layout (broadcast / per-lane WMMA fragment into
    wave-wide storage); the bare result's span is the dim product. The
    divergence is the point — "the wave-wide / broadcast backing was an
    addressing convenience at the source vload; the per-lane fragment
    is a standalone value from here on".

    Result and operand agree on element type, shape, and flavor: strip
    preserves carrier kind (vector -> vector, etc.) and only drops the
    layout slot. Buffers are rejected — buffer storage is host-side and
    `hc.as_layout` is the right buffer-level relabel.

    Lowered pre-flatten by ``hc-lower-strip-layout`` to `hc.vzeros` +
    `hc.generic` gather over the source's layout offset. No-op when the
    operand is layout-less or carries identity (bare forwarded directly).
  }];
  let arguments = (ins HC_AnyShapedValueType:$value);
  let results = (outs HC_AnyShapedValueType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
  let hasVerifier = 1;
}

def HCAsLayoutOp : HC_Op<"as_layout",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Change the layout of a tensor or vector value.";
  let description = [{
    `$layout` is the structured `#hc.layout<...>` from `doc/layouts.md`.
    The attribute is wrapped in `(...)` on the surface so the trailing
    `: type` separator is unambiguous (custom parse/print in
    `lib/IR/HCOps.cpp`).

    Operand and result may differ in rank/shape — relabel without
    touching storage. Verifier requires both to address the same
    physical span: each side's effective `storage_size` (layout's
    `storage_size` after binding `shape_syms` to the type's dims, or
    dim product when layout-less) must canonicalize to the same
    `#hc.expr`. ixsimpl hash-consing makes the comparison structural.
    A 1-D bare allocation reinterpreted as a 4-D layout-bearing view
    is the motivating shape (see `doc/layouts.md` and
    `@as_layout_shape_change_storage_match` in
    `test/HC/ops-buffer-data.mlir`).

    Both semantic and bare carriers accepted on either side — pure
    type op, composes with source decomposition.

    `!hc.buffer` operands also take optional `shape=` SSA to declare
    the reinterpreted extent: pointer storage and logical extent are
    independent. Tensor/vector operands carry dims as value semantics;
    `shape=` is meaningless and frontend rejects it.
  }];
  let arguments = (ins
    HC_AnyShapedValueType:$value,
    Optional<HC_TupleValueType>:$shape,
    HC_LayoutAttr:$layout
  );
  let results = (outs HC_AnyShapedValueType:$result);
  let assemblyFormat = [{
    $value `,` `layout` `=` custom<HCAsLayoutAttr>($layout)
    (`,` `shape` `=` $shape^ `:` type($shape))?
    attr-dict `:` type($value) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Allocators. Vectors are scope-free; tensors are workgroup-scope (later
// pass checks). `MemAlloc` stops CSE from collapsing siblings. Shape is
// an SSA value — no attribute escape hatch needed for `shape=` kwargs.
//===----------------------------------------------------------------------===//

class HC_NullaryShapedAllocOp<string mnemonic, bit vectorResult>
    : HC_Op<mnemonic, [
        MemoryEffects<[MemAlloc]>,
        DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
        HCStaticShapeOpInterface
      ]> {
  let arguments = (ins
    HC_TupleValueType:$shape,
    OptionalAttr<TypeAttr>:$dtype,
    OptionalAttr<HC_LayoutAttr>:$layout
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    `shape` $shape (`,` `dtype` `=` $dtype^)? attr-dict `:`
    functional-type(operands, results)
  }];
  // Shared interface hooks — all allocator shape carriers behave the
  // same; no per-mnemonic repetition.
  let extraClassDeclaration = [{
    ::mlir::Value getStaticShapeOperand() { return getShape(); }
    ::mlir::Type getStaticShapedResultType() { return getResult().getType(); }
    bool hasStaticVectorResult() { return }] # !if(vectorResult, "true", "false") # [{; }
    ::mlir::Value getStaticShapeSourceOperand() { return {}; }
    ::mlir::ValueRange getStaticShapeIndexOperands() { return {}; }
  }];
}

class HC_FillShapedAllocOp<string mnemonic, bit vectorResult>
    : HC_Op<mnemonic, [
        MemoryEffects<[MemAlloc]>,
        DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
        HCStaticShapeOpInterface
      ]> {
  let arguments = (ins
    HC_ValueType:$value,
    HC_TupleValueType:$shape,
    OptionalAttr<TypeAttr>:$dtype,
    OptionalAttr<HC_LayoutAttr>:$layout
  );
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = [{
    $value `,` `shape` $shape (`,` `dtype` `=` $dtype^)? attr-dict `:`
    functional-type(operands, results)
  }];
  // Same shape contract as nullary; only the leading fill value differs.
  let extraClassDeclaration = [{
    ::mlir::Value getStaticShapeOperand() { return getShape(); }
    ::mlir::Type getStaticShapedResultType() { return getResult().getType(); }
    bool hasStaticVectorResult() { return }] # !if(vectorResult, "true", "false") # [{; }
    ::mlir::Value getStaticShapeSourceOperand() { return {}; }
    ::mlir::ValueRange getStaticShapeIndexOperands() { return {}; }
  }];
}

def HCVZerosOp : HC_NullaryShapedAllocOp<"vzeros", 1> {
  let summary = "Zero-initialized vector.";
}

def HCVOnesOp : HC_NullaryShapedAllocOp<"vones", 1> {
  let summary = "Ones-initialized vector.";
}

def HCVFullOp : HC_FillShapedAllocOp<"vfull", 1> {
  let summary = "Vector filled with a scalar value.";
}

def HCZerosOp : HC_NullaryShapedAllocOp<"zeros", 0> {
  let summary = "Zero-initialized tensor (workgroup scope).";
  let hasVerifier = 1;
}

def HCOnesOp : HC_NullaryShapedAllocOp<"ones", 0> {
  let summary = "Ones-initialized tensor (workgroup scope).";
  let hasVerifier = 1;
}

def HCFullOp : HC_FillShapedAllocOp<"full", 0> {
  let summary = "Tensor filled with a scalar value (workgroup scope).";
  let hasVerifier = 1;
}

def HCEmptyOp : HC_NullaryShapedAllocOp<"empty", 0> {
  let summary = "Uninitialized tensor (workgroup scope).";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Pointer-flavored memory carriers. Post-flatten, nD shaped values collapse
// to a 1D bare carrier backed by one of these. Op surface stays tight
// (alloc / element-strided offset / scalar load / scalar store); vector
// loads/stores ride `hc.generic` once vectorization picks shapes.
//===----------------------------------------------------------------------===//

def HCAllocOp : HC_Op<"alloc", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "Allocate `count` elements in the result pointer's address space.";
  let description = [{
    Pointer to a fresh `count`-element buffer in the result type's
    address space. `count` is `index`; pre-inference may pass
    `!hc.undef`.

    Per `doc/layouts.md`, `count` must fold to a literal after the
    literal-binding pass — AMDGPU LDS lowering only materializes
    statically-sized regions; the LLVM-boundary verifier rejects
    unbounded counts. Op-level verifier only checks presence.
  }];
  let arguments = (ins HC_ValueType:$count);
  let results = (outs HC_PtrValueType:$result);
  let assemblyFormat = [{
    `count` `=` $count attr-dict `:`
    type($count) `->` qualified(type($result))
  }];
}

def HCPtrOffsetOp : HC_Op<"ptr_offset", [
    Pure
  ]> {
  let summary = "Offset a pointer by an integer count of elements.";
  let description = [{
    Element-strided pointer arithmetic: result points `index` elements
    past `source`. Element width comes from `$elementType`; opaque
    pointers stride by the abstract "one element" the producing alloc
    sized in (LLVM lowering reattaches the concrete type at codegen).

    Address space and element type preserved; mixing rejected.
  }];
  let arguments = (ins HC_PtrValueType:$source, HC_ValueType:$index);
  let results = (outs HC_PtrValueType:$result);
  let assemblyFormat = [{
    $source `,` $index attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def HCPtrLoadOp : HC_Op<"ptr_load", [
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "Load a scalar or contiguous-vector value from `$source`.";
  let description = [{
    Load at `$source`. Result is a scalar element type or `vector<NxT>`
    (contiguous N elements). Typed pointers require element-type parity
    with the result (`T` for vector, the result itself for scalar);
    opaque pointers accept any element and propagate it to codegen.

    Vector width unconstrained at op level; `hc-lower-to-llvm` splits
    oversized vectors into hardware chunks.
  }];
  let arguments = (ins HC_PtrValueType:$source);
  let results = (outs HC_PtrAccessValueType:$result);
  let assemblyFormat = [{
    $source attr-dict `:` qualified(type($source)) `->` type($result)
  }];
  let hasVerifier = 1;
}

def HCPtrStoreOp : HC_Op<"ptr_store", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "Store a scalar or contiguous-vector value at `$dest`.";
  let description = [{
    Store `$value` at `$dest`. Value is a scalar element type or
    `vector<NxT>` (contiguous N elements). Typed pointers require
    element-type parity with the value (`T` for vector, value type for
    scalar); opaque pointers accept any element and LLVM lowering
    reads the type off the store.

    Vector width unconstrained at op level; `hc-lower-to-llvm` splits
    oversized vectors into hardware chunks.
  }];
  let arguments = (ins HC_PtrAccessValueType:$value, HC_PtrValueType:$dest);
  let assemblyFormat = [{
    $value `,` $dest attr-dict `:`
    type($value) `,` qualified(type($dest))
  }];
  let hasVerifier = 1;
}

def HCPtrLoadPredOp : HC_Op<"ptr_load_pred", [
    MemoryEffects<[MemRead]>,
    AllTypesMatch<["passthrough", "result"]>
  ]> {
  let summary =
      "Predicated load: load from `$source` when `$predicate` is true, "
      "otherwise yield `$passthrough`.";
  let description = [{
    Predicated `hc.ptr_load`. Result = load value when predicate true,
    else `$passthrough`. Vector form: `$predicate` is `vector<Nxi1>`
    per-lane mask, result is elementwise blend (mirrors
    `vector.maskedload`).

    Shape parity: scalar result ↔ `i1` mask + scalar `$passthrough`;
    `vector<NxT>` result ↔ `vector<Nxi1>` mask + `vector<NxT>`
    `$passthrough`. Pointer element-type parity matches `hc.ptr_load`.

    `$passthrough` is required — no implicit poison on masked-out
    lanes; the producer always supplies the fill.
  }];
  let arguments = (ins
    HC_PtrValueType:$source,
    HC_PtrPredicateType:$predicate,
    HC_PtrAccessValueType:$passthrough
  );
  let results = (outs HC_PtrAccessValueType:$result);
  let assemblyFormat = [{
    $source `,` $predicate `passthrough` $passthrough attr-dict `:`
    qualified(type($source)) `,` type($predicate) `,` type($passthrough)
    `->` type($result)
  }];
  let hasVerifier = 1;
}

def HCPtrStorePredOp : HC_Op<"ptr_store_pred", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary =
      "Predicated store: write `$value` to `$dest` when `$predicate` is "
      "true, otherwise no-op.";
  let description = [{
    Predicated `hc.ptr_store`. Predicate true: store; false: no-op.
    Vector form: `$predicate` is `vector<Nxi1>` per-lane mask, only
    true lanes write (mirrors `vector.maskedstore`).

    Shape parity: scalar `$value` ↔ `i1` mask; `vector<NxT>` ↔
    `vector<Nxi1>` (same N). Pointer element-type parity matches
    `hc.ptr_store`.
  }];
  let arguments = (ins
    HC_PtrAccessValueType:$value,
    HC_PtrValueType:$dest,
    HC_PtrPredicateType:$predicate
  );
  let assemblyFormat = [{
    $value `,` $dest `,` $predicate attr-dict `:`
    type($value) `,` qualified(type($dest)) `,` type($predicate)
  }];
  let hasVerifier = 1;
}

def HCPredicateOp : HC_Op<"predicate",
    [Pure, AllTypesMatch<["value", "passthrough", "result"]>]> {
  let summary =
      "Predicated SSA value (`mask ? value : passthrough`) with producer-"
      "hoist lowering semantics.";
  let description = [{
    Per-lane select with a hoist contract: `result = mask ? value :
    passthrough`, but lowering rewrites the producer of `$value` into a
    predicated form rather than emitting a speculative load + blend.
    OOB / inactive lanes never materialise the read — that's why this
    is a distinct op, not `arith.select`.

    Generic surface (`Pure`, no parent restriction) shared by producer
    rewriters and `hc.generic` body lowering. Mask shape parity: scalar
    value ↔ `i1`, `vector<NxT>` ↔ `vector<Nxi1>` (same N). `!hc.undef`
    escapes parity. `$passthrough` is required — no implicit poison.

    Lowering by `hc-lower-generic`, dispatched on the def of `$value`
    (allow-list):

    * `hc.generic` body ins block-arg → implicit `hc.ptr_load` becomes
      `hc.ptr_load_pred`. No speculative read on OOB lanes.
    * `hc.ptr_load` / `hc.vload` / `hc.load` → clone the producer in
      predicated form at its def site. One clone per `hc.predicate` use.
    * `vector.extract` of a vector → `arith.select` at the predicate
      site; underlying vector load stays unconditional.
    * Anything else → lowering diagnostic. Allow-list extension is
      per-producer.

    Dominance: `$mask` / `$passthrough` must dominate the producer's
    def site (that's where the predicate physically lives in lowered
    IR). Violations diagnose at lowering.

    Trivial masks fold via `canonicalize`: `m_One()` drops the
    predicate (`result := value`), `m_Zero()` drops the load
    (`result := passthrough`). Fold runs before producer dispatch.
  }];
  let arguments = (ins HC_PtrAccessValueType:$value,
                       HC_PtrPredicateType:$mask,
                       HC_PtrAccessValueType:$passthrough);
  let results = (outs HC_PtrAccessValueType:$result);
  let assemblyFormat = [{
    $value `mask` $mask `passthrough` $passthrough attr-dict `:`
    type($value) `,` type($mask)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reductions, matmul, astype.
//===----------------------------------------------------------------------===//

def HCMatmulOp : HC_Op<"matmul", [Pure]> {
  let summary = "Generic matmul over tensors or vectors.";
  // Bare carriers accepted: `hc-decompose-shaped-values` splits semantic
  // operands into (data, mask) and re-plants matmul on the data half
  // (see `ConvertMatmulShapedOp`). Semantic-only would reject every
  // post-decompose program.
  let arguments = (ins HC_DecomposableShapedValueType:$lhs,
                       HC_DecomposableShapedValueType:$rhs);
  let results = (outs HC_DecomposableShapedValueType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def HCReduceOp : HC_Op<"reduce",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Reduce `$value` along `$axis` using `$kind`.";
  let description = [{
    `$kind` is the `#hc<reduce_kind ...>` enum (`sum` / `max` / `min`);
    wrong spellings fail at parse. `$keepdims` matches NumPy; default
    `false`. `$axis` is non-negative and, when `$value` has a concrete
    rank, less than it.

    Operand accepts bare too — `hc-decompose-shaped-values` plants
    `hc.reduce` on the data half; `hc-elementwise-to-generic` lowers
    either through the same `hc.generic` reduction.
  }];
  let arguments = (ins
    HC_DecomposableShapedValueType:$value,
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

//===----------------------------------------------------------------------===//
// Generic compute. Loosely mirrors `linalg.generic`: scalar body over
// named iters (parallel/reduction), addressed by per-operand `#hc.expr`
// offsets. The only op combining symbolic offset analysis with body
// codegen, and the only one that emits vector loads/stores once symbolic
// stride proves contiguity. See `doc/layouts.md`.
//===----------------------------------------------------------------------===//

def HCGenericOp : HC_Op<"generic",
    [SingleBlock,
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Generic compute with value or in-place outs.";
  let description = [{
    Body region over a Cartesian product of named iters. Each iter has a
    name, an upper bound (lb 0, step 1), and a kind: `parallel` iters are
    independent (vectorization candidate); `reduction` iters thread an
    output value across iterations as an accumulator (body reads the
    output block arg, yields the updated value).

    Outs are polymorphic: each output operand is either a value-semantic
    shaped HC type or a memory carrier (`!hc.ptr<...>`, `!hc.buffer<...>`);
    flavours mix freely on the same op. Subsumes `linalg.generic`'s
    tensor / memref split into one surface — fused compute-and-spill is
    one loop nest. Per-operand effects:

      * value slots: pure.
      * ptr/buffer ins: `MemRead`.
      * ptr/buffer outs: `MemRead + MemWrite` (read backs the carry).

    Outs-as-init contract: each output operand is the initial accumulator;
    result type matches operand type. SSA results track *value-typed*
    outs in declaration order; ptr/buffer outs produce no SSA result. A
    pure-store generic produces zero results, mixed one per value slot.

    Addressing is symbolic and per-axis. Each operand carries an array
    of `#hc.expr` offsets, length = operand rank. Each axis expr is
    evaluated with iter names bound to current iter values and other
    free symbols bound by the surrounding kernel scope. The op stays
    logically nD; flatten composes per-axis offsets through the
    operand's layout offset and rewrites to a 1D bare form. No grammar
    restriction on offsets — what ixsimpl accepts, this op accepts.
    Lowering owns contiguity / linearity proofs.

    Verifier rules:
      - `iter_syms`, `iter_bounds`, `iter_kinds` are parallel arrays;
        names unique and non-empty.
      - Per-operand offset arrays match operand counts; each entry is
        an array of `#hc.expr` (rank check is post-flatten).
      - Body block has `ins.size() + outs.size()` args; each type
        matches the element type of the corresponding operand
        (`!hc.undef` and opaque ptrs escape parity).
      - Result count = value-typed-outs count, in declaration order.
      - Output offsets reference only `parallel` iter names — a
        reduction iter on an out would mean writing the same slot
        twice without a combinator.

    Input-side masks are not on this op pre-decomposition (`hc.load_mask`
    / `hc.store(mask=...)` cover that); a future mask slot would be its
    own operand. Output-side gating rides on `hc.yield_predicated` —
    one `i1` per yielded value — so masked-store / OOB-tile / predicated-
    select avoid `scf.if` in the body. Body terminator: `hc.yield`
    (unconditional) or `hc.yield_predicated` (per-value mask).

    `ambient_idxs` / `ambient_idx_syms`: parallel SSA-operand / name-
    attr pair pinning every non-iter free symbol the offsets reference
    (`$STRIDE_*`, kernel-arg dim syms, `$WG*` / `$WI*` / `$WGS*`,
    structured-loop join syms, ...) to its runtime SSA at emission.
    `hc-flatten-with-layouts` is the canonical populator. Pre-flatten
    emitters leave the lists empty; lowering treats absence as
    "fall back to ambient-context resolution", same as `hc.idx_apply`'s
    severing form. Operand types are `!hc.idx<...>` (or `index` after
    the launch-body type converter strips the sym-typed payload — the
    name stays on `ambient_idx_syms`).
  }];
  let arguments = (ins
    StrArrayAttr:$iter_syms,
    Variadic<HC_ValueType>:$iter_bounds,
    TypedArrayAttrBase<HC_IterKindAttr,
                       "array of `#hc<iter_kind ...>` attributes">:$iter_kinds,
    Variadic<HC_GenericOperandType>:$ins,
    Variadic<HC_GenericOperandType>:$outs,
    Variadic<HC_SymBindingValueType>:$ambient_idxs,
    StrArrayAttr:$ambient_idx_syms,
    TypedArrayAttrBase<
        TypedArrayAttrBase<HC_ExprAttr, "per-axis `#hc.expr<...>` array">,
        "array of per-operand per-axis `#hc.expr<...>` offset arrays"
      >:$ins_offsets,
    TypedArrayAttrBase<
        TypedArrayAttrBase<HC_ExprAttr, "per-axis `#hc.expr<...>` array">,
        "array of per-operand per-axis `#hc.expr<...>` offset arrays"
      >:$outs_offsets
  );
  let results = (outs Variadic<HC_GenericOperandType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def HCAsTypeOp : HC_Op<"astype", [Pure]> {
  let summary = "Explicit numeric conversion (Python `.astype`).";
  let description = [{
    Cast that retains target element type on the op. Lowering rewrites to
    `hc.cast` or a target arithmetic conversion depending on operand
    types. Verifier rejects non-numeric targets (`target = !hc.slice`).
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
// `FlatSymbolRefAttr` callee. `verifySymbolUses` resolves against the
// nearest symbol table, checks op-kind parity (`hc.call` → `hc.func`,
// `hc.call_intrinsic` → `hc.intrinsic`), and with a callee
// `function_type` checks arity + per-arg/result parity (`!hc.undef`
// escapes — progressive typing).
//
// Memory effects come from the callee's `effects` enum when set;
// otherwise default to `MemRead + MemWrite` to block CSE / LICM /
// speculation from reordering opaque calls past loads/stores.
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
    Effects inherited from the callee's `effects` class; default
    `MemRead + MemWrite` (conservative — blocks reorder past loads/stores
    until the callee narrows).
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
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Call into an `hc.intrinsic` declaration.";
  let description = [{
    Specialization-required kwargs live as attributes
    (`{wave_size = 32 : i64}`), not SSA operands — available to
    verify/specialize hooks without operand bookkeeping.

    Result types flow through HC type inference via the callee
    `function_type`. Target lowering owns turning the call into
    backend-specific IR.

    Effects: same policy as `hc.call`.
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
// Python rebinds names across loop iterations; region ops join those
// rebinds back to SSA (iter_args + yields, results + branch yields,
// region captures + region results). The frontend emits these two ops
// to preserve the name-store shape so `-hc-promote-names` does the
// reconstruction uniformly — one pass instead of one per region kind.
// Both ops are transient: must not survive `-hc-promote-names`.
//===----------------------------------------------------------------------===//

def HCAssignOp : HC_Op<"assign", [MemoryEffects<[MemWrite]>]> {
  let summary = "Imperative write to a named slot; pre-SSA placeholder.";
  let description = [{
    `hc.assign "x", %v` records a write to Python name `x`. No SSA
    result — consumers go through `hc.name_load` and `-hc-promote-names`
    rewrites each read into a direct use (or an iter_arg / block arg).

    `MemWrite` on `DefaultResource`. Pairs with `hc.name_load`'s
    `MemRead` on the same resource so `-cse`, speculation, and LICM
    keep same-name reads in program order around the assign. Doesn't
    drive DCE.

    Aliasing against real buffer traffic is pessimistic (shared
    `DefaultResource`); safe for these transient ops. A dedicated
    name-store resource kind is the documented escape if a later pass
    needs to hoist across name-store boundaries.

    Cross-kernel / cross-function name capture is out of scope. Each
    `hc.kernel` / `hc.func` owns its own name store.

    ### Induction-variable self-bind

    Looks redundant, isn't:

    ```
    hc.for_range %lb, %ub, %step : !hc.undef -> () {
    ^bb0(%iv: !hc.undef):
      hc.assign "i", %iv : !hc.undef    // <-- IV self-bind
      ...
      hc.region_return
    }
    ```

    The frontend emits this unconditionally so every Python name has a
    reaching `hc.assign` before its first read, including the loop
    counter, without the frontend reasoning about counter use.
    `-hc-promote-names` matches `hc.assign "<name>", %iv` where `%iv`
    is the `hc.for_range` block arg, erases the assign, and resolves
    reads of the name directly to `%iv`. Not treated as a carried-name
    / iter_arg candidate. Matcher: `extractIvSelfBinds` /
    `promoteForRange` in `lib/Transforms/HCPromoteNamesPass.cpp`.
  }];
  let arguments = (ins StrAttr:$name, HC_ValueType:$value);
  let assemblyFormat = "$name `,` $value attr-dict `:` type($value)";
}

def HCNameLoadOp : HC_Op<"name_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Imperative read from a named slot; pre-SSA placeholder.";
  let description = [{
    `%v = hc.name_load "x"` materializes name `x`'s current binding.
    Result is open `HC_ValueType` (works pre-inference with
    `!hc.undef`). `-hc-promote-names` replaces uses with the reaching
    def (nearest dominating `hc.assign`, or a new block/iter arg when
    the reach crosses a region boundary) and erases the load.

    `MemRead` on `DefaultResource`. Two generic-pass interactions:
    `-cse` skips ops with non-empty effects (no merging two
    `hc.name_load` of the same name), and `wouldOpBeTriviallyDead`
    treats read-only-no-use as dead (`-canonicalize` erases unused
    loads before `-hc-promote-names`). Pinned in
    `test/HC/name-store-effects.mlir`.

    Aliasing is pessimistic against buffer traffic (same as
    `hc.assign`).
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs HC_ValueType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

#endif // HC_IR_HCOPS
