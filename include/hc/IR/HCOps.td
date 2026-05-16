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
    and `group_shape` as `#hc.shape`, `subgroup_size` as an `i32`,
    `bound_symbols` as a `StrArrayAttr` listing launch and kernel-input ABI
    symbols that may be materialized, `literals` as a `StrArrayAttr` listing
    symbols that specialization is allowed to bind, and `literal_bindings`
    as a `DictionaryAttr` (`{name = i64}`) pinning concrete integer values
    for the specialization point — the launcher (`hc.compile(symbols=...)`)
    stamps each requested binding here so `hc-specialize-literals` can fold
    them into the IR. All are optional so the `hc_front -> hc` pass can fill
    them in progressively.
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

def HCIdxApplyOp : HC_Op<"idx_apply",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary =
      "Materialize an `!hc.idx<expr>` from explicitly bound operand values.";
  let description = [{
    Carries a symbolic expression on the result type (`!hc.idx<expr>`) and an
    explicit binding of free symbols in `expr` to SSA operands. Each
    `symbols[k]` names one free symbol of `expr` whose runtime integer is
    supplied by `operands[k]`. Free symbols of `expr` not listed in `symbols`
    are treated as ambient — resolved by lowering from the surrounding launch
    context (`$WG*` / `$WI*` / `$WGS*`).

    Operands are `index` or `!hc.idx<...>`; the `!hc.idx` form is accepted so
    producers that already hold a typed symbolic SSA value can wire it in
    without an explicit cast. The launch-body type converter folds both to
    `index` before the expression walks the substituted operand.

    An empty `symbols` / empty `operands` apply is the value-less
    severing form: it carries the type only and falls entirely
    through ambient launch-context binding. Producer passes use this
    form when the SSA values for free symbols aren't yet available
    at the rewrite point (kernel ABI shape syms, stride params);
    sites that already hold the runtime SSA bind it explicitly so
    every user-named symbol becomes a real dataflow edge.
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
    Predicate-side analogue of `hc.idx_apply`. Carries a symbolic predicate
    on the result type (`!hc.pred<pred>`) and an explicit binding of free
    symbols in `pred` to SSA operands. Operand semantics, the
    listed-vs-ambient symbol split, and the lowering convention are identical
    to `hc.idx_apply`; the only difference is that the result type is a
    boolean-kind `!hc.pred<...>` rather than `!hc.idx<...>`. Operands remain
    integer-kind because expression / predicate leaves are integer symbols
    in either case.
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

// `hc.pow lhs, rhs` covers Python's `**`. Carrier is mechanical — the
// front-to-hc rewrite emits this for every `BinOp.op == Pow` it sees,
// and the dedicated `-hc-lower-pow` pass (`include/hc/Transforms/Passes.td`)
// rewrites supported shapes into the mul chain that the rest of the
// pipeline actually understands. Non-supported shapes (non-constant or
// non-integer-literal exponents) are diagnosed by `-hc-lower-pow`
// because the dialect doesn't yet have a `math.pow`-style runtime op to
// route them to; this op exists so the diagnostic fingers the original
// `**` rather than some downstream lowering trying to read intent off
// an opaque mul chain.
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

def HCSelectOp : HC_Op<"select", [Pure]> {
  let summary = "Select between bare shaped data and a scalar inactive value.";
  let description = [{
    Internal form used after mask decomposition. `$condition` is a bare
    `!hc.pred` tensor/vector mask, `$true_value` is matching bare data, and
    `$false_value` is the scalar fill for inactive elements.
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
                  "HCSubgroupRegionOp", "HCWorkitemRegionOp",
                  "HCGenericOp"]>]> {
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

    Inside `hc.generic` this is the unconditional terminator — every
    iteration of the body publishes its values into the outs slots.
    The masked counterpart `hc.yield_predicated` covers the
    per-value-gated case (masked store, OOB tile guard, predicated
    select) without an `scf.if` shape in the body.
  }];
  let arguments = (ins Variadic<HC_ValueType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [
    // Needed by `SingleBlockImplicitTerminator` on the parent ops.
    OpBuilder<(ins), [{ /* empty yield */ }]>,
  ];
}

def HCYieldPredicatedOp : HC_Op<"yield_predicated",
    [Pure, Terminator, SameVariadicOperandSize,
     ParentOneOf<["HCGenericOp"]>]> {
  let summary =
      "Predicated terminator for `hc.generic` bodies: per-value masked publish.";
  let description = [{
    Per-value masked counterpart of `hc.yield` for `hc.generic` bodies.
    Position `i` publishes `$values[i]` into outs slot `i` on lanes where
    `$masks[i]` is true; masked-out lanes leave the outs-as-init carry
    in place. Strict parity: `values.size() == masks.size()` — one mask
    per yielded value, no exceptions.

    Mask type tracks the value's shape (`hc-lower-generic` enforces it
    when it materialises the predicated store / select at the body
    boundary): scalar value pairs with `i1`, `vector<NxT>` pairs with
    `vector<Nxi1>` of the same N. Pre-inference `!hc.undef` operands on
    either side escape the parity check.

    Always-true positions materialise `arith.constant true : i1` (folding
    in the lowering turns those back into unpredicated stores). A
    sentinel "no mask" op is an opt-in follow-up if body bloat from
    constant-i1 ever measures; the foundational primitive stays one
    verifier rule — every mask is an `i1` SSA.

    The motivating consumers — masked store rewrite, `hc.load_mask`
    rewrite, OOB-tile guard at the workgroup-staged copy source — all
    share the same handoff: compute the mask in the body from iter syms
    and ambient bounds, ride it on `$masks`, let the lowering pick the
    predicated memory op or `arith.select` blend.
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
    matching Python-style multi-axis subscripts. Vector roots use the same
    shape-reduction rules: scalar indices consume axes, slices produce vector
    fragments, and indexing every vector axis yields the element type.

    `$unit_axes` carries the output positions where NumPy `None` / `np.newaxis`
    inserts a unit-size dim — `x[:, None, :]` becomes
    `hc.buffer_view %x[%full, %full] {unit_axes = [1]}`. Positions are in the
    output rank's coordinates (so two `None`s in the original subscript yield
    two `unit_axes` entries). The variadic `$indices` list never contains a
    `None` placeholder; the residual indices consume source axes one-for-one,
    and the unit dims are interleaved into the result shape on the
    type-inference side.
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
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
     DeclareOpInterfaceMethods<HCStaticShapeOpInterface>]> {
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
    Vector counterpart of `hc.load`; same SSA shape carrier and `MemRead`
    effect. Only the source operand's admissible type broadens (tensor is also
    legal) and the result is a vector instead of a tensor.
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
    Internal companion for decomposed `hc.load` / `hc.vload`. It carries the
    same addressing operands as the load that produced the data and returns a
    bare tensor/vector of `!hc.pred` values describing element validity.
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
    Internal mask seed for values whose whole static shape is known valid after
    decomposition, such as allocators and inactive-filled vectors.
  }];
  let results = (outs HC_BareShapedValueType:$mask);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($mask)";
}

def HCStoreOp : HC_Op<"store", [MemoryEffects<[MemWrite]>,
                                 AttrSizedOperandSegments]> {
  let summary = "Store a tensor or vector into a buffer or tensor.";
  let description = [{
    Stores the shaped `$source` value into `$dest`. After shaped-value
    decomposition, `$source` may be a bare data container and `$mask` carries the
    matching bare `!hc.pred` validity for masked writes. Unlike load and
    allocator ops, store does not need a separate shape operand because the
    stored value's type already carries the tensor/vector shape. The operand
    segment attribute keeps the variadic indices distinct from the optional mask.
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
    Carries a `MemRead` effect because the source tensor lives in addressable
    memory; two `hc.vec` ops on the same tensor may observe different contents
    if a store is interleaved, so CSE must not collapse them blindly.
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

def HCStripLayoutOp : HC_Op<"strip_layout",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
  let summary = "Drop a layout-bearing value's layout, project to bare storage.";
  let description = [{
    User-marked boundary that asserts the operand's layout no longer
    applies and the value becomes a standalone bare carrier from this
    point on. Spelt at the surface as ``as_layout(value, None)``.

    Where ``hc.as_layout`` reinterprets a value's logical structure
    while preserving the underlying storage span (operand and result
    must agree on effective ``storage_size`` under ixsimpl), this op
    *changes* that span: the operand's layout may declare a backing
    larger than the dim product (a non-injective layout — broadcast,
    per-lane WMMA fragment view into a wave-wide storage) and the
    result re-types as bare with no layout, whose effective span is
    the dim product. The storage-size divergence is the point — the
    user is saying "the wave-wide / broadcast backing was an
    addressing convenience at the source vload; the per-lane / per-
    workitem fragment is a standalone value from here on".

    Result and operand must agree on element type, shape, and flavor:
    strip preserves the carrier kind (vector -> vector, bare_vector ->
    bare_vector, tensor -> tensor, bare_tensor -> bare_tensor) and
    only drops the layout slot. Switching flavor at the strip site
    would clash with downstream inference at every peer-typed use
    (the K-tile loop's iter init / result pair against an
    intrinsic's vector return is the motivating example — see
    ``examples/amdgpu_gfx11_wmma_matmul.py``). Buffers are rejected
    outright: buffer storage is host-side and ``hc.as_layout`` is
    the right primitive for buffer-level relabel.

    Lowered pre-flatten by ``hc-lower-strip-layout`` into an
    ``hc.vzeros`` plus ``hc.generic`` gather over the source's layout
    offset, so the post-flatten pipeline only sees ordinary
    ``hc.generic`` ops. The lowering is a no-op when the operand has
    no layout (or carries the identity layout) — the bare type is
    forwarded directly.
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
    `$layout` is the structured `#hc.layout<...>` descriptor documented
    in `doc/layouts.md`.

    The attribute is wrapped in `(...)` in the surface IR so
    the trailing `: type` separator on the op signature is unambiguous;
    see the custom parse/print in `lib/IR/HCOps.cpp` for the lookahead
    rationale. Lifted out of `attr-dict` so the surface reads like the
    other explicit keyword-argument ops (`hc.reduce`, `hc.astype`).

    Operand and result may differ in rank/shape — `hc.as_layout` is the
    surface for reinterpreting a value's logical structure without
    touching the underlying storage. The verifier guards safety by
    requiring operand and result to address the same physical span:
    each side's effective `storage_size` (the layout's
    `storage_size` after binding `shape_syms` to the type's dim
    entries, or the product of dims when the type has no layout)
    must canonicalize to the same `#hc.expr` value. ixsimpl
    hash-consing gives that comparison structural meaning, not just
    textual. A 1-D bare allocation reinterpreted as a 4-D
    layout-bearing view (`storage_size = "b * m * n * l"` on the
    layout side, product `2 * M * N * L` on the bare side) is the
    motivating shape; see `doc/layouts.md` and
    `@as_layout_shape_change_storage_match` in
    `test/HC/ops-buffer-data.mlir`.

    Both semantic (`!hc.tensor` / `!hc.vector`) and bare
    (`!hc.bare_tensor` / `!hc.bare_vector`) shaped values are
    accepted on either side: the relabel is a pure type op and
    composes with the source decomposition that introduces the
    bare carriers.

    Pointer-rooted (`!hc.buffer`) operands additionally accept the
    optional `shape=` SSA operand to declare the reinterpreted
    extent. The buffer's named dims are a proxy for its logical
    extent — pointer storage and logical extent are independent
    concepts — so the new shape rides on the op rather than being
    derived from the operand's dims. Tensor/vector operands have
    dims as part of their value semantics; `shape=` is meaningless
    there and the frontend rejects it.
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
// Allocators. Vector variants are scope-free; tensor variants are workgroup
// scope (checked by a later pass). All carry `MemAlloc` so CSE does not
// collapse sibling allocations. Shape is an ordinary SSA value so frontend
// `shape=` kwargs do not need a dialect-side attribute escape hatch.
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
  // Allocator shape carriers all have the same layout; keep their static-shape
  // interface hooks on the shared ODS base instead of repeating per mnemonic.
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
  // Fill allocators share the same shape contract as nullary allocators; only
  // the leading fill value differs.
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
// Pointer-flavored memory carriers. The post-flatten pipeline drops nD shaped
// values to a 1D bare carrier whose backing storage is one of these. The op
// surface is intentionally tight (alloc / element-strided offset / scalar
// load / scalar store): vector loads/stores live on `hc.generic` once
// vectorization decides what to emit, not on per-element ops.
//===----------------------------------------------------------------------===//

def HCAllocOp : HC_Op<"alloc", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "Allocate `count` elements in the result pointer's address space.";
  let description = [{
    Produces a pointer to a fresh `count`-element buffer in the result
    type's address space. `count` is specified as `index` in the model;
    pre-inference IR may pass `!hc.undef` and let inference narrow it.

    `doc/layouts.md` requires `count` to fold to a literal constant
    after the literal-binding pass — the AMDGPU LDS lowering can only
    materialize statically-sized regions in v1, and the verifier on the
    LLVM-lowering boundary rejects allocations that didn't bind.
    Structurally-pre-binding IR is allowed (the verifier here checks
    only that a count is present); the lowering pass owns the post-bind
    constness check.
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
    Element-strided pointer arithmetic: the result points `index`
    elements past `source`, where the element width is fixed by the
    pointer type's `$elementType`. Opaque pointers stride by the
    abstract "one element" the alloc that produced them was sized in;
    the LLVM lowering reattaches a concrete element type at codegen.

    The address space and element type are preserved through the
    offset; mixing them is verifier-rejected.
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
    Load from the address pointed to by `$source`. The result type is
    either a scalar element type (single-element load) or
    `vector<NxT>` (contiguous N-element load starting at `$source`).
    When the pointer carries an explicit `$elementType`, the result's
    element type — `T` for vector results, the result type itself for
    scalars — must match; opaque pointers accept any element type and
    propagate the choice down to the load codegen.

    Vector width is unconstrained at this op; the `hc-lower-to-llvm`
    boundary splits oversized vectors into hardware-sized chunks.
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
    Store from `$value` into the address pointed to by `$dest`. The
    value type is either a scalar element type (single-element store)
    or `vector<NxT>` (contiguous N-element store starting at `$dest`).
    When `$dest` carries an explicit `$elementType`, the value's
    element type — `T` for vector values, the value type itself for
    scalars — must match; opaque pointers accept any element type and
    the LLVM lowering reads the element type off the store.

    Vector width is unconstrained at this op; the `hc-lower-to-llvm`
    boundary splits oversized vectors into hardware-sized chunks.
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
    Predicated counterpart of `hc.ptr_load`. The result is the load's
    value when the predicate is true; otherwise the result is the
    `$passthrough` operand. For vector forms `$predicate` is a
    per-lane mask (`vector<Nxi1>`) and the result is the elementwise
    blend of the load and the passthrough — equivalent to upstream
    `vector.maskedload`.

    Shape parity: scalar `$result` requires `$predicate : i1` and a
    scalar `$passthrough`; `vector<NxT>` `$result` requires
    `$predicate : vector<Nxi1>` (same N) and a `vector<NxT>`
    `$passthrough`. Pointer element-type parity follows the same rule
    as `hc.ptr_load`: typed pointers require an exact match against
    the result's element type.

    `$passthrough` is required (no implicit poison/undef on the
    masked-out lanes); the producer always supplies the desired
    fill — typically a zero constant when the inactive value is
    irrelevant.
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
    Predicated counterpart of `hc.ptr_store`. When `$predicate` is
    true, `$value` is stored at `$dest`; when false, the op is a
    no-op. For vector forms `$predicate` is a per-lane mask
    (`vector<Nxi1>`) and only the lanes whose mask bit is true are
    written — equivalent to upstream `vector.maskedstore`.

    Shape parity: scalar `$value` requires `$predicate : i1`;
    `vector<NxT>` `$value` requires `$predicate : vector<Nxi1>`
    (same N). Pointer element-type parity follows the same rule as
    `hc.ptr_store`: typed pointers require an exact match against
    the value's element type.
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
    passthrough`, but the lowering rewrites the producer of `$value` in
    place to a predicated form rather than emitting a speculative load
    followed by a blend. OOB / inactive lanes never materialise the
    underlying read — which is the whole point of having a distinct op
    instead of plain `arith.select`.

    The surface is intentionally generic — `Pure`, no parent restriction —
    so producer-side rewriters and the `hc.generic` body lowering share
    the same op. Mask shape parity is the same rule the predicated memory
    ops carry: scalar value pairs with `i1`, `vector<NxT>` value pairs
    with `vector<Nxi1>` of the same N. Pre-inference `!hc.undef` on
    either side escapes the parity check.

    `$passthrough` is required (no implicit poison / zero default), same
    rule `hc.ptr_load_pred` carries: the producer always supplies the
    inactive value.

    Lowering — `hc-lower-generic` dispatches on the def of `$value`
    (allow-list, not default-accept):
    * `hc.generic` body ins block-arg: the implicit `hc.ptr_load` at the
      ins-slot offset becomes `hc.ptr_load_pred` with `$mask` /
      `$passthrough`. Speculative reads on OOB lanes never happen.
    * `hc.ptr_load` / `hc.vload` / `hc.load`: clone the producer in place
      in its predicated form. Each `hc.predicate` use clones one
      predicated op at the producer's def site, so multiple decorators on
      the same load each get their own predicated load.
    * `vector.extract` of a vector value: emit `arith.select %mask,
      %value, %passthrough` at the predicate site. The underlying vector
      load stays unconditional; the predicate gates the lane only.
    * Anything else: lowering diagnostic. Extending the allow-list is a
      deliberate per-producer choice.

    Dominance: `$mask` and `$passthrough` must dominate the producer's
    def site when the lowering clones / rewrites there. The producer's
    site is where the predicate physically lives in the lowered IR, so
    the mask SSA has to be available there. Violations are lowering-time
    diagnostics, not silent drops.

    Always-true / always-false masks fold via `canonicalize`: `m_One()`
    drops the predicate (`result := value`), `m_Zero()` drops the load
    (`result := passthrough`). The fold path runs before the lowering's
    producer dispatch, so unpredicated emit stays the default for
    trivial masks. The folds land with the lowering wire-up, not on the
    foundation surface.
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
  let arguments = (ins HC_ShapedValueType:$lhs, HC_ShapedValueType:$rhs);
  let results = (outs HC_ShapedValueType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def HCReduceOp : HC_Op<"reduce",
    [Pure, DeclareOpInterfaceMethods<HCInferTypeOpInterface>]> {
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

//===----------------------------------------------------------------------===//
// Generic compute op. Loosely mirrors `linalg.generic`: a body region of
// scalar work, parameterized by a list of named iters with kinds
// (parallel/reduction), driven over inputs and outputs whose addressing is
// described by per-operand `#hc.expr` offsets resolved against the iter
// names plus any outer-scope symbols. The post-flatten lowering is the only
// op that combines symbolic offset analysis with body codegen, and the only
// op authorized to emit vector loads/stores once the symbolic-stride pass
// proves contiguity. See `doc/layouts.md` for the design.
//===----------------------------------------------------------------------===//

def HCGenericOp : HC_Op<"generic",
    [SingleBlock,
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Generic compute with value or in-place outs.";
  let description = [{
    Bodyful op that runs a scalar block over a Cartesian product of named
    iters. Each iter has a name, an upper bound (lower bound 0, step 1), and
    a kind: `parallel` iters are independent and a vectorization candidate;
    `reduction` iters thread an output value across iterations as an
    accumulator (the body reads the corresponding output block argument as
    the running value and yields the updated value).

    Outs are polymorphic: each output operand may be either a value-semantic
    shaped HC type (tensor / vector / bare carrier) or a memory carrier
    (`!hc.ptr<...>`, `!hc.buffer<...>`); the two flavors mix freely on the
    same op. This subsumes `linalg.generic`'s tensor / memref split into a
    single surface — a fused "compute and spill" lands as one loop nest
    instead of two. Per-operand effects:

      * value-typed slots are pure (no effect),
      * ptr/buffer in `ins` contributes `MemRead` on the operand,
      * ptr/buffer in `outs` contributes `MemRead + MemWrite` (the read
        backs the carry — same outs-as-init contract as the value form,
        but sourced from the destination's existing memory).

    Outputs follow `linalg.generic`'s "outs-as-init" contract: each output
    operand is the initial accumulator value and the produced result has
    the same type. SSA results track the *value-typed* outs in declaration
    order; ptr/buffer outs contribute no SSA result. A pure-store generic
    (all-ptr outs) produces zero results, mixed produces one per
    value-typed slot.

    Operand addressing is symbolic and per-axis: each input and output
    carries an array of `#hc.expr` offsets, one entry per operand axis
    (length = operand rank). Each axis expression is evaluated in a
    scope where the iter names are bound to the current iter values
    and any other free symbols are bound by the surrounding kernel
    scope. The op stays logically nD on its operand axes regardless of
    pipeline position; flatten composes per-axis offsets through the
    operand's `#hc.layout` offset to produce a single composed
    expression and rewrites the operand to its 1D bare form (where the
    axis array becomes single-element). The op does not constrain
    offset grammar; what `ixsimpl` accepts, the op accepts. The
    lowering owns contiguity / linearity proofs.

    Verifier rules:
      - `iter_syms`, `iter_bounds`, `iter_kinds` are parallel arrays;
        names must be unique and non-empty.
      - Per-operand offset arrays must match operand counts; each
        per-operand entry is itself an array of `#hc.expr` (rank check
        is deferred to a post-flatten verifier).
      - Body block has `ins.size() + outs.size()` arguments; each arg
        type must match the element type of the corresponding operand
        (operand element type for shaped types / pointee type for ptrs;
        `!hc.undef` operands and opaque ptrs escape the parity check).
      - Result count equals the number of value-typed outs in
        declaration order; each result type matches the corresponding
        value-typed out's operand type.
      - Output offsets must reference only `parallel` iter names: a
        reduction iter on an output offset would mean writing the same
        slot twice without specifying combinator semantics, which the op
        does not model. Applies to both value- and ptr-typed outs.

    Mask handling on the input side is deferred — pre-decomposition
    the input mask story rides on `hc.load_mask` / `hc.store(mask=...)`;
    if `hc.generic` grows mask operands it will be a separate slot, not
    piggybacked on the offset array. The body-output side is covered by
    the `hc.yield_predicated` terminator, which carries one `i1` mask
    per yielded value so masked-store / OOB-tile / predicated-select
    rewrites need not introduce `scf.if` shapes in the body. The body
    therefore terminates with either `hc.yield` (unconditional publish)
    or `hc.yield_predicated` (per-value mask gate).

    Ambient-symbol bindings ride on `ambient_idxs` / `ambient_idx_syms`:
    a parallel SSA-operand / name-attr pair that pins every non-iter
    free symbol the offset expressions reference (`$STRIDE_*`, kernel-
    arg dim syms, `$WG*` / `$WI*` / `$WGS*`, structured-loop join
    symbols, ...) to the SSA value that holds its runtime integer at
    the point of emission. `hc-flatten-with-layouts` is the canonical
    populator — it walks the kernel-arg-bundle / launch-context chain
    while every ambient sym is reachable as an HC-typed SSA value and
    captures the binding so downstream lowerings can pass it through
    as a real dataflow edge instead of re-discovering the resolution
    by name. Pre-flatten emitters leave the lists empty; the lowering
    treats absence as "fall back to ambient-context resolution",
    matching `hc.idx_apply`'s severing form. Operand types are
    `!hc.idx<...>` (or `index` after the launch-body type converter
    has stripped the sym-typed payload — the symbolic name is kept on
    `ambient_idx_syms`).
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
     DeclareOpInterfaceMethods<HCInferTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Call into an `hc.intrinsic` declaration.";
  let description = [{
    Specialization-required keyword arguments live on this op as
    attributes (e.g. `{wave_size = 32 : i64}`), not SSA operands, so they
    are available to verify/specialize hooks without operand bookkeeping.

    Result types participate in HC type inference through the callee
    `function_type`; intrinsic declarations are the target-independent
    contract surface, while target lowering remains responsible for turning the
    call into backend-specific IR.

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
