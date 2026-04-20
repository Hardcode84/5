// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_IR_HCFRONTOPS
#define HC_FRONT_IR_HCFRONTOPS

include "hc/Front/IR/HCFrontTypes.td"
include "mlir/IR/CommonAttrConstraints.td"

class HCFront_ValueProducer<string mnemonic, list<Trait> traits = []>
    : HCFront_Op<"front." # mnemonic, traits> {
  let results = (outs HCFront_ValueType:$result);
}

class HCFront_RegionOp<string mnemonic, list<Trait> traits = []>
    : HCFront_Op<
        "front." # mnemonic,
        !listconcat([NoTerminator, SingleBlock], traits)
      >;

def HCFront_KernelOp : HCFront_RegionOp<"kernel"> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
}

def HCFront_FuncOp : HCFront_RegionOp<"func"> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
}

def HCFront_IntrinsicOp : HCFront_RegionOp<"intrinsic"> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
}

def HCFront_ConstantOp : HCFront_ValueProducer<"constant"> {
  let arguments = (ins AnyAttr:$value);
}

def HCFront_NameOp : HCFront_ValueProducer<"name"> {
  let arguments = (ins StrAttr:$name);
}

def HCFront_AssignOp : HCFront_Op<"front.assign"> {
  let arguments = (ins HCFront_ValueType:$target, HCFront_ValueType:$value);
}

def HCFront_AugAssignOp : HCFront_Op<"front.aug_assign"> {
  let arguments = (ins
    StrAttr:$kind,
    HCFront_ValueType:$target,
    HCFront_ValueType:$value
  );
}

def HCFront_TargetNameOp : HCFront_ValueProducer<"target_name"> {
  let arguments = (ins StrAttr:$name);
}

def HCFront_TargetTupleOp : HCFront_ValueProducer<"target_tuple"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$elements);
}

def HCFront_TargetSubscriptOp : HCFront_ValueProducer<"target_subscript"> {
  let arguments = (ins
    HCFront_ValueType:$base,
    Variadic<HCFront_ValueType>:$indices
  );
}

def HCFront_AttrOp : HCFront_ValueProducer<"attr"> {
  let arguments = (ins HCFront_ValueType:$base, StrAttr:$name);
}

def HCFront_SubscriptOp : HCFront_ValueProducer<"subscript"> {
  let arguments = (ins
    HCFront_ValueType:$base,
    Variadic<HCFront_ValueType>:$indices
  );
}

def HCFront_SliceOp : HCFront_ValueProducer<"slice"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$parts);
}

def HCFront_CallOp : HCFront_ValueProducer<"call"> {
  let arguments = (ins
    HCFront_ValueType:$callee,
    Variadic<HCFront_ValueType>:$arguments
  );
}

def HCFront_KeywordOp : HCFront_ValueProducer<"keyword"> {
  let arguments = (ins StrAttr:$name, HCFront_ValueType:$value);
}

def HCFront_TupleOp : HCFront_ValueProducer<"tuple"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$elements);
}

def HCFront_ListOp : HCFront_ValueProducer<"list"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$elements);
}

def HCFront_BinOp : HCFront_ValueProducer<"binop"> {
  let arguments = (ins
    StrAttr:$kind,
    HCFront_ValueType:$lhs,
    HCFront_ValueType:$rhs
  );
}

def HCFront_UnaryOp : HCFront_ValueProducer<"unaryop"> {
  let arguments = (ins StrAttr:$kind, HCFront_ValueType:$operand);
}

def HCFront_CompareOp : HCFront_ValueProducer<"compare"> {
  let arguments = (ins
    ArrayAttr:$predicates,
    Variadic<HCFront_ValueType>:$operands
  );
}

def HCFront_IfOp : HCFront_RegionOp<"if"> {
  let regions = (region
    SizedRegion<1>:$condition,
    SizedRegion<1>:$thenRegion,
    SizedRegion<1>:$elseRegion
  );
}

def HCFront_ForOp : HCFront_RegionOp<"for"> {
  let regions = (region
    SizedRegion<1>:$target,
    SizedRegion<1>:$iter,
    SizedRegion<1>:$body
  );
}

def HCFront_ReturnOp : HCFront_Op<"front.return"> {
  let arguments = (ins Variadic<HCFront_ValueType>:$values);
}

def HCFront_SubgroupRegionOp : HCFront_RegionOp<"subgroup_region"> {
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
}

def HCFront_WorkitemRegionOp : HCFront_RegionOp<"workitem_region"> {
  let arguments = (ins OptionalAttr<StrArrayAttr>:$captures);
  let regions = (region SizedRegion<1>:$body);
}

#endif // HC_FRONT_IR_HCFRONTOPS
