// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPSINTERFACES
#define HC_IR_HCOPSINTERFACES

include "mlir/IR/OpBase.td"

def NameStoreRegionOpInterface : OpInterface<"NameStoreRegionOpInterface"> {
  let description = [{
    Marker for region-carrying ops participating in name-store promotion.
    `-hc-promote-names` walks function region trees bottom-up and rebuilds
    each tagged op so `hc.assign` / `hc.name_load` inside collapse to
    structured SSA. Semantics split on the scope-barrier axis:

      - `hc.for_range`:       transparent. Carried names → iter_args +
                              `hc.yield`. A leading `hc.assign "<n>", %iv`
                              self-bind is folded to direct uses of the
                              block arg.
      - `hc.if`:              transparent. Carried names → op results +
                              yields; silent-branch fallback is the outer
                              snapshot.
      - `hc.workitem_region`: nested Python scope. Inner reads lazily
                              capture the outer binding; inner writes
                              shadow and don't leak. Outbound leak is
                              opt-in via `hc.region_return <names>` — the
                              promoter adds one result per name and emits
                              writebacks into the enclosing scope.
      - `hc.subgroup_region`: same rules as workitem.

    Marker-only — the pass dispatches on op kind internally. The opt-in
    lives in ODS so the participating set is visible alongside the ops,
    and a dialect-extension op can join with just a new switch case.
  }];
  let cppNamespace = "::mlir::hc";
}

def HCInferTypeOpInterface : OpInterface<"HCInferTypeOpInterface"> {
  let description = [{
    Op-local type inference hook for the HC dataflow pass.

    Narrower than MLIR's `InferTypeOpInterface`: the pass passes in
    operand types learned by dataflow (potentially more precise than IR
    types mid-iteration). Hook appends one `$resultTypes` entry per
    result; null = no fact.
  }];
  let cppNamespace = "::mlir::hc";
  let methods = [
    InterfaceMethod<
      "Infer result types from dataflow-refined operand types.",
      "::mlir::LogicalResult",
      "inferHCTypes",
      (ins "::llvm::ArrayRef<::mlir::Type>":$operandTypes,
           "::llvm::SmallVectorImpl<::mlir::Type> &":$resultTypes)
    >
  ];
}

def HCInferRegionArgTypeOpInterface :
    OpInterface<"HCInferRegionArgTypeOpInterface"> {
  let description = [{
    Type inference hook for non-CFG-forwarded region args.

    The sparse dataflow framework forwards values along
    `RegionBranchOpInterface` edges; some entry-block args (e.g. a loop
    IV) aren't forwarded from operands or terminators. This hook lets the
    parent op describe them without baking semantics into the pass.
  }];
  let cppNamespace = "::mlir::hc";
  let methods = [
    InterfaceMethod<
      "Infer types for non-forwarded region successor inputs.",
      "::mlir::LogicalResult",
      "inferHCRegionArgTypes",
      (ins "::mlir::RegionSuccessor":$successor,
           "::mlir::ValueRange":$nonSuccessorInputs,
           "::llvm::SmallVectorImpl<::mlir::Type> &":$regionArgTypes)
    >
  ];
}

def HCYieldedResultsOpInterface : OpInterface<"HCYieldedResultsOpInterface"> {
  let description = [{
    Contract for single-region ops whose `hc.yield` operands define the
    parent results.

    Narrower than `RegionBranchOpInterface`: covers ops where the
    dataflow solver doesn't otherwise propagate terminator facts back to
    parent results, but the region's sole terminator is still the
    semantic source.
  }];
  let cppNamespace = "::mlir::hc";
  let methods = [
    InterfaceMethod<
      "Return the values yielded to the parent op results, or empty if absent.",
      "::mlir::ValueRange",
      "getYieldedResultValues"
    >
  ];
}

def HCStaticShapeOpInterface : OpInterface<"HCStaticShapeOpInterface"> {
  let description = [{
    Contract for ops whose result shape rides an SSA shape carrier.

    Verifier owns the policy: carrier must resolve to a tuple of pinned
    `!hc.idx<...>` dims, shaped result must carry the same static shape,
    addressed-source ops expose the source + index/slice operands for
    rank diagnostics. Ops only describe where those pieces live.
  }];
  let cppNamespace = "::mlir::hc";
  let methods = [
    InterfaceMethod<
      "Return the SSA value carrying the requested static shape.",
      "::mlir::Value",
      "getStaticShapeOperand"
    >,
    InterfaceMethod<
      "Return the tensor/vector result type checked against the shape carrier.",
      "::mlir::Type",
      "getStaticShapedResultType"
    >,
    InterfaceMethod<
      "Return true when the shaped result is expected to be `!hc.vector`.",
      "bool",
      "hasStaticVectorResult"
    >,
    InterfaceMethod<
      "Return the addressed source operand, or null for allocator-like ops.",
      "::mlir::Value",
      "getStaticShapeSourceOperand"
    >,
    InterfaceMethod<
      "Return index/slice operands addressing the source, if any.",
      "::mlir::ValueRange",
      "getStaticShapeIndexOperands"
    >
  ];
}

#endif // HC_IR_HCOPSINTERFACES
