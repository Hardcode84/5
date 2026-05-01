// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES
#define HC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCPromoteNames : Pass<"hc-promote-names"> {
  let summary = "Promote Python name-store ops to SSA values.";
  let description = [{
    Replaces `hc.assign` / `hc.name_load` with structured SSA carriers,
    rebuilding region-carrying ops so each Python-level name becomes a
    real SSA value or iter_arg / result / block-arg of the enclosing
    region op. This pass is the second half of the pipeline; the
    pipeline contract — what the placeholders are, who is allowed to
    look at them, what "supported invocation" means — lives on
    `ConvertHCFrontToHC` in `include/hc/Conversion/HCFrontToHC/Passes.td`.

    Runs bottom-up on every function region tree (`hc.kernel` / `hc.func`
    / `hc.intrinsic`). Region ops that tag themselves with
    `NameStoreRegionOpInterface` participate:

      - `hc.for_range`  : carried names become iter_args + yielded
                          values. A leading `hc.assign "<n>", %iv`
                          (the frontend's way of publishing the IV
                          under its Python-source name) is recognized
                          as a tautological self-bind, kept loop-local,
                          and folded to direct uses of the block arg.
      - `hc.if`         : carried names become op results + two-branch
                          yields; branches must agree on the write set.
      - `hc.workitem_region` / `hc.subgroup_region` : Python-style
                          nested scopes. Reads inside capture outer
                          bindings (lazy outer snapshot); writes
                          shadow without leaking. Out-bound
                          propagation is opt-in via an explicit
                          `hc.region_return <names>` terminator: the
                          promoter rebuilds the op with one result
                          per returned name and pushes writebacks out
                          to the enclosing scope's name store.

    Post-condition (on success): no `hc.assign` / `hc.name_load` survives
    anywhere under the processed callables. Cross-kernel / cross-function
    name capture is out of scope; each callable is its own promotion
    root.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInferTypes : Pass<"hc-infer-types"> {
  let summary = "Infer concrete HC value types from pre-inference `!hc.undef` IR.";
  let description = [{
    Runs a sparse forward dataflow analysis over `hc` IR and rewrites values
    whose type can be refined without changing program structure. The first
    slice focuses on scalar/index facts: constants, symbols, launch geometry,
    slice construction, buffer dimensions, index arithmetic/comparisons, and
    structured loop carriers. Inference is monotonic: `!hc.undef` refines to a
    concrete type, and conflicting concrete facts fail the pass instead of
    being reconciled with a guess.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCVerifyStaticShapes : Pass<"hc-verify-static-shapes"> {
  let summary = "Verify SSA shape operands have static HC shape metadata.";
  let description = [{
    Checks `hc.load`, `hc.vload`, and shaped allocator ops after type
    inference. Shape operands must be tuple values whose element types are
    pinned `!hc.idx<...>` dimensions, and the resulting tensor/vector type must
    carry the same static shape. Load-like ops also require a statically-ranked
    source and index/slice operands whose arity stays within that source rank.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCDecomposeShapedValues
    : Pass<"hc-decompose-shaped-values"> {
  let summary = "Split semantic tensor/vector values into bare data and masks.";
  let description = [{
    Rewrites supported semantic `!hc.tensor` / `!hc.vector` producers and users
    into compiler-internal `!hc.bare_tensor` / `!hc.bare_vector` data plus
    matching bare `!hc.pred` masks. This is an HC-to-HC normalization boundary
    for upstream lowering; semantic shaped values that cross unsupported ops are
    diagnosed instead of being silently dropped.
  }];
  let options = [
    Option<"strictMode", "strict", "bool", "true",
           "Fail unless every semantic shaped boundary is decomposed. Disable "
           "this to run a partial conversion that preserves unsupported "
           "boundaries with unrealized conversion casts.">
  ];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerBarePredicates
    : Pass<"hc-lower-bare-predicates"> {
  let summary = "Lower bare HC predicate containers to upstream i1 masks.";
  let description = [{
    Converts concrete `!hc.pred` values to `i1` and static `!hc.bare_vector`
    values to upstream `vector` values. Predicate vectors become
    `vector<...xi1>` masks, and supported HC mask operations are lowered to
    `arith`/`vector` operations. Bare tensors remain HC values for the later
    memory/tensor lowering. Unsupported HC boundaries are preserved with
    `builtin.unrealized_conversion_cast` so this pass can be staged before the
    full HC-to-upstream lowering.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::hc::HCDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def HCMaterializeBoundExprs
    : Pass<"hc-materialize-bound-exprs"> {
  let summary = "Materialize bound symbolic HC expressions as SSA values.";
  let description = [{
    Replaces uses of values typed as pinned `!hc.idx<expr>` or
    `!hc.pred<pred>` with `hc.materialize_bound_expr` when every symbolic leaf
    is a bound symbol or the expression is constant-only. The original producer
    is left in place for canonicalization/DCE, which lets this pass break
    dependence on launch-context query chains before scope normalization.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCNormalizeScopeRegions
    : Pass<"hc-normalize-scope-regions"> {
  let summary = "Flatten supported collective scope regions into SPMD HC IR.";
  let description = [{
    Normalizes HC toward per-workitem SPMD form before upstream GPU lowering.
    The first supported slice flattens resultless `hc.workitem_region` bodies
    after bound expression materialization has removed live uses of the
    workitem token. Result-producing workitem regions and subgroup regions are
    diagnosed until distributed value projection and subgroup-local workitem
    semantics are handled explicitly.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

#endif // HC_TRANSFORMS_PASSES
