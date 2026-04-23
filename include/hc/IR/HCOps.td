// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPS
#define HC_IR_HCOPS

include "hc/IR/HCTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"

class HC_RegionOp<string mnemonic, list<Trait> traits = []>
    : HC_Op<mnemonic, !listconcat([NoTerminator, SingleBlock], traits)>;

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

#endif // HC_IR_HCOPS
