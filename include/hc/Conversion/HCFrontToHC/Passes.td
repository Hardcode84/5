// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_CONVERSION_HCFRONTTOHC_PASSES
#define HC_CONVERSION_HCFRONTTOHC_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHCFrontToHC : Pass<"convert-hc-front-to-hc", "::mlir::ModuleOp"> {
  let summary = "Mechanical hc_front -> hc lowering driven by `ref` metadata.";
  let description = [{
    Walks every `hc_front.kernel` / `hc_front.func` / `hc_front.intrinsic`
    top-level op and emits the parallel `hc` callable, consuming the `ref`
    DictAttr stamped by the Python driver on each `hc_front.name` /
    `hc_front.attr`. Every produced value has `!hc.undef`; a later
    inference pass pins concrete types.

    ## Pipeline contract (canonical)

    Python-level name bindings are materialized as `hc.assign` /
    `hc.name_load` placeholder ops. This pass does not resolve reaching
    definitions — it emits the placeholders, nothing more. The
    immediately-following pass `-hc-promote-names` folds the pairs into
    real SSA edges (iter_args for `hc.for_range`, results for `hc.if`,
    block args / region results for `hc.workitem_region` /
    `hc.subgroup_region`).

    The supported invocation is the two-pass pipeline:

        hc-opt --convert-hc-front-to-hc --hc-promote-names

    Modules that carry undecorated inline helpers (`hc_front.func` ops
    tagged `ref = {kind = "inline"}`, produced by the Python resolver
    when it BFS-discovers plain-Python helpers) must first run
    `-hc-front-inline`, which expands every call site into an
    `hc_front.inlined_region` and erases the marker funcs. This pass
    flattens those regions but refuses to lower a surviving
    `ref.kind = "inline"` `hc_front.call` — the diagnostic tells the
    operator the pipeline ordering is wrong. The full pipeline is
    therefore:

        hc-opt --hc-front-inline --convert-hc-front-to-hc --hc-promote-names

    `--hc-front-inline` is a no-op on modules the Python driver never
    stamped inline helpers on, so it is safe to include
    unconditionally.

    Placeholder IR (the output of this pass on its own) is verifier-legal
    but round-trip-only: no consumer other than `-hc-promote-names`
    should inspect or transform it. Diagnostics for driver-side bugs
    (unbound names, missing assigns) fire in the promotion pass, not
    here — see `test/HC/promote-names-invalid.mlir` and
    `test/HCFrontToHC/pipeline-invalid.mlir` for the user-visible
    wording. Every other doc site in the tree (`HCPromoteNames` pass
    description, the C++ banners, the LIT headers) points back to this
    paragraph; if the contract changes, change it here first.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::hc::front::HCFrontDialect",
  ];
}

#endif // HC_CONVERSION_HCFRONTTOHC_PASSES
