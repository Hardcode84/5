// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_CONVERSION_HCFRONTTOHC_PASSES
#define HC_CONVERSION_HCFRONTTOHC_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHCFrontToHC : Pass<"convert-hc-front-to-hc"> {
  let summary = "Mechanical hc_front -> hc lowering driven by frontend metadata.";
  let description = [{
    Walks every `hc_front.kernel` / `hc_front.func` / `hc_front.intrinsic`
    top-level op and emits the parallel `hc` callable, consuming the `ref`
    DictAttr stamped by the Python driver on each `hc_front.name` /
    `hc_front.attr` plus the callable `parameters` metadata. Most produced
    values start as `!hc.undef`; kernel `group` parameters are seeded as
    `!hc.group<...>`, and buffer parameters are seeded as `!hc.buffer<...>`
    when their dict carries `kind = "buffer"` and shape metadata. A later
    inference pass pins the remaining concrete types.

    ## Pipeline contract (canonical)

    Python-level name bindings are materialized as `hc.assign` /
    `hc.name_load` placeholder ops. This pass does not resolve reaching
    definitions — it emits the placeholders, nothing more. The
    immediately-following pass `-hc-promote-names` folds the pairs into
    real SSA edges (iter_args for `hc.for_range`, results for `hc.if`,
    block args / region results for `hc.workitem_region` /
    `hc.subgroup_region`).

    The canonical `hc-opt` line is:

        hc-opt --hc-front-fold-region-defs --hc-front-inline \
               --convert-hc-front-to-hc --hc-promote-names

    `--convert-hc-front-to-hc --hc-promote-names` is the irreducible
    core. The two front passes ahead of it are frontend ghost-trail
    scrubbers and are no-ops on modules the Python driver never
    stamped the relevant markers on, so both are safe to include
    unconditionally:

    - `-hc-front-inline` expands every `hc_front.call` targeting a
      `ref = {kind = "inline"}` marker func (emitted by the Python
      resolver for undecorated plain-Python helpers) into an
      `hc_front.inlined_region`, then erases the marker funcs. Skip
      it and a surviving `ref.kind = "inline"` call trips this pass's
      `run -hc-front-inline before -convert-hc-front-to-hc`
      diagnostic.
    - `-hc-front-fold-region-defs` erases the ghost
      `hc_front.name {ref.kind = "local"} + hc_front.call`
      (+ optional `hc_front.return`) trail the Python frontend emits
      next to a nested `@group.workitems` / `@group.subgroups` def
      with an immediate call (`inner()` / `return inner()`). The
      region op itself is already the lowering. Skip it and the
      `ref.kind = "local"` callee reaches this pass and trips the
      `unsupported callee ref.kind 'local'` diagnostic.

    In both cases the diagnostic at the conversion boundary is the
    operator's signal that the pipeline was ordered wrong upstream.

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
