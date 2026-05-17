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
    Handles to `hc.call_intrinsic` payload ops under `root` whose callee
    matches. Optional `target` is recipe-dispatch metadata; payload call
    itself is target-independent.
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
    Creates a payload op named `op_name` with value handles as operands,
    type params as result types, dynamic attribute params named by
    `dynamic_attr_names`, and optional literal `static_attrs`.
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

def HCTransformRequireIntrinsicAttrOp
    : HCTransform_Op<"require_intrinsic_attr"> {
  let summary = "Assert an intrinsic call's named attribute equals a literal";
  let description = [{
    Recipe pre-check: definite failure if the `hc.call_intrinsic` payload
    is missing the named attribute or carries a different value than
    `expected`. Comparison is attribute equality, so `i32` vs `i64`
    counts as a mismatch — write `expected` at the call site's width
    (`wave_size = 32 : i64` paired with `expected = 32 : i64`).

    No result handle. Definite (not silenceable) failure on purpose —
    silenceable would let the sequence fall through and surface a
    generic "no recipe matched", losing the wrong-arch / wrong-wave_size
    context.
  }];

  let arguments = (ins TransformHandleTypeInterface:$call,
                       StrAttr:$name,
                       AnyAttr:$expected);
  // `$expected` is `AnyAttr` whose textual form may contain `:`
  // (`32 : i64`), which clashes with any custom format's trailing
  // `:` for `type($call)`. Keep it inside attr-dict.
  let assemblyFormat = [{
    $call attr-dict `:` type($call)
  }];
}

def HCTransformConstantTypeOp
    : HCTransform_Op<"constant_type"> {
  let summary = "Materialize a literal type as a transform type parameter";
  let description = [{
    `!transform.type_param` carrying the literal type. Recipes feed this
    through `transform.hc.create_op`'s `result_types` or
    `transform.hc.cast_value`'s `target_type` to assert concrete upstream
    types (e.g. `vector<16xf16>` for `amdgpu.wmma` operands). Pure on
    the payload (no IR mutation); `getEffects` declares only the producer
    effect.
  }];

  let arguments = (ins TypeAttr:$value);
  let results = (outs Transform_TypeParamType:$result);
  let assemblyFormat = [{ $value attr-dict `:` type($result) }];
}

def HCTransformCastValueOp
    : HCTransform_Op<"cast_value"> {
  let summary = "Bridge an operand value to a target type via UCC";
  let description = [{
    Inserts a `builtin.unrealized_conversion_cast` at the source's def
    site and yields a handle to the cast result. If the source already
    has the target type, the source is forwarded.

    Bridges HC bare types ↔ upstream types at `hc.call_intrinsic`
    boundaries. The recipe-inserted cast pairs with the UCC
    `hc-lower-launch-body` plants on either side, so post-rewrite
    canonicalize collapses the upstream → bare → upstream chain.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$source,
                       TransformParamTypeInterface:$target_type);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat = [{
    $source `to` $target_type attr-dict `:`
    functional-type(operands, results)
  }];
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
