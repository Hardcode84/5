// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES
#define HC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCPromoteNames : Pass<"hc-promote-names", "::mlir::ModuleOp"> {
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

#endif // HC_TRANSFORMS_PASSES
