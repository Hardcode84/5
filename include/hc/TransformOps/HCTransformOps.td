// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMOPS_HCTRANSFORMOPS
#define HC_TRANSFORMOPS_HCTRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class HCTransform_Op<string mnemonic, list<Trait> traits = []>
    : Op<Transform_Dialect, "hc." # mnemonic,
         !listconcat([
           DeclareOpInterfaceMethods<TransformOpInterface>,
           DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
         ], traits)> {
  let cppNamespace = "::mlir::hc::transform";
}

def HCTransformMatchIntrinsicCallOp
    : HCTransform_Op<"match_intrinsic_call"> {
  let summary = "Match hc.call_intrinsic payload ops by callee";
  let description = [{
    Produces handles to `hc.call_intrinsic` payload operations under `root`
    whose callee matches `callee`. The optional `target` attribute is metadata
    carried by recipe dispatch; the payload call itself is target-independent.
  }];

  let arguments = (ins TransformHandleTypeInterface:$root,
                       FlatSymbolRefAttr:$callee,
                       OptionalAttr<StrAttr>:$target);
  let results = (outs Transform_AnyOpType:$matched);
  let assemblyFormat = [{
    $root $callee (`target` `=` $target^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def HCTransformGetIntrinsicOperandOp
    : HCTransform_Op<"get_intrinsic_operand"> {
  let summary = "Get a payload operand from an intrinsic call handle";
  let arguments = (ins TransformHandleTypeInterface:$call,
                       OptionalAttr<StrAttr>:$name,
                       OptionalAttr<I64Attr>:$index);
  let results = (outs Transform_AnyValue:$value);
  let assemblyFormat = [{
    $call attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def HCTransformGetIntrinsicResultTypeOp
    : HCTransform_Op<"get_intrinsic_result_type"> {
  let summary = "Get an intrinsic call result type as a transform parameter";
  let arguments = (ins TransformHandleTypeInterface:$call, I64Attr:$index);
  let results = (outs Transform_TypeParamType:$type);
  let assemblyFormat = [{
    $call attr-dict `:` functional-type(operands, results)
  }];
}

def HCTransformGetIntrinsicAttrOp
    : HCTransform_Op<"get_intrinsic_attr"> {
  let summary = "Get a constant attribute from an intrinsic call";
  let arguments = (ins TransformHandleTypeInterface:$call, StrAttr:$name);
  let results = (outs Transform_AnyParamType:$param);
  let assemblyFormat = [{
    $call attr-dict `:` functional-type(operands, results)
  }];
}

def HCTransformCreateOp
    : HCTransform_Op<"create_op", [AttrSizedOperandSegments]> {
  let summary = "Create a payload operation from intrinsic recipe handles";
  let description = [{
    Creates one payload operation named `op_name` using value handles as
    operands, type parameters as result types, dynamic attribute parameters
    named by `dynamic_attr_names`, and optional literal `static_attrs`.
  }];

  let arguments = (ins StrAttr:$op_name,
                       TransformHandleTypeInterface:$insertion_point,
                       Variadic<TransformValueHandleTypeInterface>:$inputs,
                       Variadic<TransformParamTypeInterface>:$result_types,
                       Variadic<TransformParamTypeInterface>:$dynamic_attrs,
                       StrArrayAttr:$dynamic_attr_names,
                       OptionalAttr<DictionaryAttr>:$static_attrs);
  let results = (outs Variadic<TransformValueHandleTypeInterface>:$results);
  let assemblyFormat = [{
    $op_name `at` $insertion_point `(` $inputs `)`
    `result_types` `(` $result_types `)`
    `dynamic_attrs` $dynamic_attr_names `(` $dynamic_attrs `)`
    (`static_attrs` `=` $static_attrs^)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def HCTransformReplaceIntrinsicCallOp
    : HCTransform_Op<"replace_intrinsic_call"> {
  let summary = "Replace an intrinsic call with payload values";
  let arguments = (ins TransformHandleTypeInterface:$call,
                       Variadic<TransformValueHandleTypeInterface>:$replacements);
  let assemblyFormat = [{
    $call `with` $replacements attr-dict `:`
    functional-type(operands, results)
  }];
}

#endif // HC_TRANSFORMOPS_HCTRANSFORMOPS
