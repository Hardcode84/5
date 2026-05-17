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
    and emits the parallel `hc` callable. Reads the `ref` DictAttr on each
    `hc_front.name` / `hc_front.attr` plus callable `parameters`. Most
    produced values start as `!hc.undef`; kernel `group` parameters seed as
    `!hc.group<...>`, buffer parameters as `!hc.buffer<...>` when the dict
    carries `kind = "buffer"` plus shape metadata. Later inference pins
    the rest.

    ## Pipeline contract (canonical)

    Python name bindings materialize as `hc.assign` / `hc.name_load`
    placeholders. This pass emits them; `-hc-promote-names` folds them
    into iter_args (`hc.for_range`), results (`hc.if`), and block / region
    args (`hc.workitem_region`, `hc.subgroup_region`).

    Canonical `hc-opt`:

        hc-opt --hc-front-fold-region-defs --hc-front-inline \
               --convert-hc-front-to-hc --hc-promote-names

    `--convert-hc-front-to-hc --hc-promote-names` is the irreducible core.
    The two front passes are ghost-trail scrubbers — no-ops when the
    relevant markers were never stamped, safe to include unconditionally:

    - `-hc-front-inline` expands every `hc_front.call` targeting a
      `ref = {kind = "inline"}` marker func into an
      `hc_front.inlined_region` and erases the marker. Skip it and a
      surviving `ref.kind = "inline"` call trips
      `run -hc-front-inline before -convert-hc-front-to-hc`.
    - `-hc-front-fold-region-defs` erases the
      `hc_front.name {ref.kind = "local"} + hc_front.call`
      (+ optional `hc_front.return`) trail emitted next to a nested
      `@group.workitems` / `@group.subgroups` def with an immediate
      call. The region op is already the lowering. Skip it and the
      surviving `ref.kind = "local"` callee trips
      `unsupported callee ref.kind 'local'`.

    Placeholder IR (this pass on its own) is verifier-legal but
    round-trip-only: only `-hc-promote-names` should consume it.
    Driver-bug diagnostics (unbound names, missing assigns) fire in
    promotion, not here. This is the canonical statement of the
    contract.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::hc::front::HCFrontDialect",
  ];
}

#endif // HC_CONVERSION_HCFRONTTOHC_PASSES
