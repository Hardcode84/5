// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPS
#define HC_IR_HCOPS

include "hc/IR/HCTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
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

def HCKernelOp : HC_RegionOp<"kernel"> {
  let summary = "Semantic kernel region";
  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "$name (`requirements` `=` $requirements^)? attr-dict-with-keyword $body";
}

def HCFuncOp : HC_RegionOp<"func"> {
  let summary = "Semantic helper region";
  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<HC_ConstraintSetAttr>:$requirements
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "$name (`requirements` `=` $requirements^)? attr-dict-with-keyword $body";
}

def HCSubgroupRegionOp : HC_RegionOp<"subgroup_region"> {
  let summary = "Semantic subgroup region";
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "(`captures` `=` $captures^)? attr-dict-with-keyword $body";
}

def HCWorkitemRegionOp : HC_RegionOp<"workitem_region"> {
  let summary = "Semantic workitem region";
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "(`captures` `=` $captures^)? attr-dict-with-keyword $body";
}

def HCReturnOp : HC_Op<"return"> {
  let summary = "Semantic return";
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
  let summary = "User-visible type conversion (Python `.astype`, ...).";
  let description = [{
    Semantic value conversion. Distinct from `unrealized_conversion_cast`,
    which is reserved for compiler-internal type transitions in dialect
    conversions; `hc.cast` persists and later lowers to concrete arithmetic
    conversion ops.
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

#endif // HC_IR_HCOPS
