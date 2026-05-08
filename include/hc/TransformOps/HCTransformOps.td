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

def HCTransformRequireIntrinsicAttrOp
    : HCTransform_Op<"require_intrinsic_attr"> {
  let summary = "Assert an intrinsic call's named attribute equals a literal";
  let description = [{
    Recipe-level pre-check: fails the apply with a definite diagnostic if the
    `hc.call_intrinsic` payload does not carry the named attribute, or if it
    carries a different value than `expected`. Comparison is by attribute
    equality, so an `i32` vs `i64` mismatch counts as a value mismatch — write
    the expected literal at the same width as the call site (e.g.
    `wave_size = 32 : i64` paired with `expected = 32 : i64`). The op carries
    no result handle: the caller already has the call handle and only wants
    the apply to fail loudly when the assumption underlying the rewrite is
    violated. Definite (not silenceable) failure is intentional — silenceable
    would let the named-sequence application fall through and the surface
    diagnostic would degrade to the generic "no recipe matched" message,
    losing the "wrong arch" / "wrong wave_size" context.
  }];

  let arguments = (ins TransformHandleTypeInterface:$call,
                       StrAttr:$name,
                       AnyAttr:$expected);
  // Keep both attributes inside attr-dict: `$expected` is `AnyAttr` and the
  // built-in attribute printer can emit values whose textual form contains
  // `:` (e.g. `32 : i64`), which would clash with the trailing `:` literal
  // demanded by `type($call)` in any custom format.
  let assemblyFormat = [{
    $call attr-dict `:` type($call)
  }];
}

def HCTransformConstantTypeOp
    : HCTransform_Op<"constant_type"> {
  let summary = "Materialize a literal type as a transform type parameter";
  let description = [{
    Produces a `!transform.type_param` carrying the literal type attribute.
    Recipes that want to assert a concrete upstream type at create-op time
    (for example `vector<16xf16>` to satisfy `amdgpu.wmma`'s operand
    constraints) feed this through `transform.hc.create_op`'s `result_types`
    or `transform.hc.cast_value`'s `target_type`. The transform op is pure
    in payload terms (no IR mutation) but produces a result handle that
    downstream recipe ops consume; the explicit `getEffects` model declares
    only the producer effect so the verifier is happy.
  }];

  let arguments = (ins TypeAttr:$value);
  let results = (outs Transform_TypeParamType:$result);
  let assemblyFormat = [{ $value attr-dict `:` type($result) }];
}

def HCTransformCastValueOp
    : HCTransform_Op<"cast_value"> {
  let summary = "Bridge an operand value to a target type via UCC";
  let description = [{
    Inserts a `builtin.unrealized_conversion_cast` at the source value's
    definition site, casting it to the target type, and yields a value
    handle to the cast result. When the source value already has the
    target type the source is forwarded unchanged and no cast is inserted.

    Recipes lowering `hc.call_intrinsic` boundaries use this to bridge HC
    bare types ↔ upstream types: `hc-lower-launch-body` keeps
    `!hc.bare_vector` / `!hc.bare_tensor` operands across the call, but
    upstream payload ops (e.g. `amdgpu.wmma`) want plain `vector<NxF>`.
    The recipe-inserted cast pairs with the existing UCC the launch-body
    pass plants on either side of the call boundary, so the post-rewrite
    canonicalize collapses the upstream → bare → upstream chain to identity
    and the bridging UCC disappears entirely.
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
