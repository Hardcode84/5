// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCOPSINTERFACES
#define HC_IR_HCOPSINTERFACES

include "mlir/IR/OpBase.td"

def NameStoreRegionOpInterface : OpInterface<"NameStoreRegionOpInterface"> {
  let description = [{
    Marker for region-carrying ops that participate in Python name-store
    promotion. `-hc-promote-names` walks a function region tree bottom-up
    and rebuilds every op with this interface so that `hc.assign` /
    `hc.name_load` inside its region(s) get replaced by a structured SSA
    carrier, with semantics split along the scope-barrier axis:

      - `hc.for_range`:       transparent. Carried names become
                              iter_args + `hc.yield` values, threaded
                              across iterations. A leading
                              `hc.assign "<n>", %iv` self-bind is
                              recognized as the IV-publishing
                              placeholder and folded to direct uses
                              of the block arg — see `hc.assign` in
                              HCOps.td for the pattern rationale.
      - `hc.if`:              transparent. Carried names become op
                              results + yields in both branches;
                              silent-branch fallback = outer snapshot.
      - `hc.workitem_region`: Python-style nested scope. Reads inside
                              capture the outer binding (lazy
                              `hc.name_load` at the op's insertion
                              point); writes inside shadow and don't
                              leak out. Out-bound propagation is
                              opt-in via an explicit
                              `hc.region_return <names>` terminator:
                              promotion rebuilds the op with one
                              result per returned name and emits
                              writebacks into the enclosing scope's
                              name store.
      - `hc.subgroup_region`: same rules as `hc.workitem_region` —
                              isolated scope with lazy capture,
                              opt-in leak via `hc.region_return`.

    The interface is a marker only — the pass dispatches on op kind
    internally. Keeping the opt-in in ODS (rather than a hard-coded
    `isa<...>` list inside the pass) documents the set of participating
    ops next to the ops themselves, and lets a future dialect-extension
    op join the promotion without touching pass code beyond a new case
    in its switch.
  }];
  let cppNamespace = "::mlir::hc";
}

def HCInferTypeOpInterface : OpInterface<"HCInferTypeOpInterface"> {
  let description = [{
    Op-local type inference hook for the HC dataflow type inference pass.

    This is deliberately narrower than MLIR's generic `InferTypeOpInterface`:
    the pass supplies the operand types learned by dataflow, which may be more
    precise than the operands' current IR types during a solver iteration. The
    hook appends one entry per result to `$resultTypes`; a null `Type` entry
    means "no fact for this result".
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

#endif // HC_IR_HCOPSINTERFACES
