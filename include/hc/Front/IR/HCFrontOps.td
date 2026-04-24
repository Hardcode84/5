// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_IR_HCFRONTOPS
#define HC_FRONT_IR_HCFRONTOPS

include "hc/Front/IR/HCFrontTypes.td"
include "mlir/IR/CommonAttrConstraints.td"

class HCFront_ValueProducer<string mnemonic, list<Trait> traits = []>
    : HCFront_Op<mnemonic, traits> {
  let results = (outs HCFront_ValueType:$result);
}

class HCFront_RegionOp<string mnemonic, list<Trait> traits = []>
    : HCFront_Op<
        mnemonic,
        !listconcat([NoTerminator, SingleBlock], traits)
      >;

def HCFront_KernelOp : HCFront_RegionOp<"kernel"> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

def HCFront_FuncOp : HCFront_RegionOp<"func"> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

def HCFront_IntrinsicOp : HCFront_RegionOp<"intrinsic"> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

// Operands and results across this dialect are uniformly `!hc_front.value`
// (or, occasionally, `!hc_front.typeexpr`). Both are parameterless and therefore
// buildable, so the declarative assembly format synthesizes the types without
// needing explicit `type($x)` / `functional-type(...)` directives in the
// spellings below.

def HCFront_ConstantOp : HCFront_ValueProducer<"constant"> {
  let arguments = (ins AnyAttr:$value);
  let assemblyFormat = "`<` $value `>` attr-dict";
}

def HCFront_NameOp : HCFront_ValueProducer<"name"> {
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

def HCFront_AssignOp : HCFront_Op<"assign"> {
  let arguments = (ins HCFront_ValueType:$target, HCFront_ValueType:$value);
  let assemblyFormat = "$target `=` $value attr-dict";
}

def HCFront_AugAssignOp : HCFront_Op<"aug_assign"> {
  let arguments = (ins
    StrAttr:$kind,
    HCFront_ValueType:$target,
    HCFront_ValueType:$value
  );
  let assemblyFormat = "$kind $target `=` $value attr-dict";
}

def HCFront_TargetNameOp : HCFront_ValueProducer<"target_name"> {
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

def HCFront_TargetTupleOp : HCFront_ValueProducer<"target_tuple"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$elements);
  let assemblyFormat = "`(` $elements `)` attr-dict";
}

def HCFront_TargetSubscriptOp : HCFront_ValueProducer<"target_subscript"> {
  let arguments = (ins
    HCFront_ValueType:$base,
    Variadic<HCFront_ValueType>:$indices
  );
  let assemblyFormat = "$base `[` $indices `]` attr-dict";
}

def HCFront_AttrOp : HCFront_ValueProducer<"attr"> {
  let arguments = (ins HCFront_ValueType:$base, StrAttr:$name);
  let assemblyFormat = "$base `,` $name attr-dict";
}

def HCFront_SubscriptOp : HCFront_ValueProducer<"subscript"> {
  let arguments = (ins
    HCFront_ValueType:$base,
    Variadic<HCFront_ValueType>:$indices
  );
  let assemblyFormat = "$base `[` $indices `]` attr-dict";
}

def HCFront_SliceOp : HCFront_ValueProducer<"slice"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$parts);
  let assemblyFormat = "`(` $parts `)` attr-dict";
}

def HCFront_CallOp : HCFront_ValueProducer<"call"> {
  let arguments = (ins
    HCFront_ValueType:$callee,
    Variadic<HCFront_ValueType>:$arguments
  );
  let assemblyFormat = "$callee `(` $arguments `)` attr-dict";
}

def HCFront_KeywordOp : HCFront_ValueProducer<"keyword"> {
  let arguments = (ins StrAttr:$name, HCFront_ValueType:$value);
  let assemblyFormat = "$name `=` $value attr-dict";
}

def HCFront_TupleOp : HCFront_ValueProducer<"tuple"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$elements);
  let assemblyFormat = "`(` $elements `)` attr-dict";
}

def HCFront_ListOp : HCFront_ValueProducer<"list"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$elements);
  let assemblyFormat = "`[` $elements `]` attr-dict";
}

def HCFront_BinOp : HCFront_ValueProducer<"binop"> {
  let arguments = (ins
    StrAttr:$kind,
    HCFront_ValueType:$lhs,
    HCFront_ValueType:$rhs
  );
  let assemblyFormat = "$kind `(` $lhs `,` $rhs `)` attr-dict";
}

def HCFront_UnaryOp : HCFront_ValueProducer<"unaryop"> {
  let arguments = (ins StrAttr:$kind, HCFront_ValueType:$operand);
  let assemblyFormat = "$kind `(` $operand `)` attr-dict";
}

def HCFront_CompareOp : HCFront_ValueProducer<"compare"> {
  let arguments = (ins
    ArrayAttr:$predicates,
    Variadic<HCFront_ValueType>:$operands
  );
  let assemblyFormat = "$predicates `(` $operands `)` attr-dict";
}

def HCFront_IfOp : HCFront_RegionOp<"if"> {
  let regions = (region
    SizedRegion<1>:$condition,
    SizedRegion<1>:$thenRegion,
    SizedRegion<1>:$elseRegion
  );
  let assemblyFormat =
      "$condition `then` $thenRegion `else` $elseRegion attr-dict";
}

def HCFront_ForOp : HCFront_RegionOp<"for"> {
  let regions = (region
    SizedRegion<1>:$target,
    SizedRegion<1>:$iter,
    SizedRegion<1>:$body
  );
  let assemblyFormat = "$target `in` $iter `do` $body attr-dict";
}

def HCFront_ReturnOp : HCFront_Op<"return"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$values);
  let assemblyFormat = "($values^)? attr-dict";
}

def HCFront_SubgroupRegionOp : HCFront_RegionOp<"subgroup_region"> {
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "(`captures` `=` $captures^)? attr-dict-with-keyword $body";
}

def HCFront_WorkitemRegionOp : HCFront_RegionOp<"workitem_region"> {
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "(`captures` `=` $captures^)? attr-dict-with-keyword $body";
}

// Name-scope boundary around a cloned inline callee body. Stamped by
// `-hc-front-inline`, consumed by `-convert-hc-front-to-hc`; no runtime
// semantics. See the `HCFrontInline` pass description for the contract.
def HCFront_InlinedRegionOp : HCFront_RegionOp<"inlined_region"> {
  let arguments = (ins
    StrAttr:$callee,
    Variadic<HCFront_ValueType>:$arguments
  );
  let results = (outs Variadic<HCFront_ValueType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $callee `(` $arguments `)` (`->` `(` type($results)^ `)`)?
      attr-dict-with-keyword $body
  }];
}

#endif // HC_FRONT_IR_HCFRONTOPS
