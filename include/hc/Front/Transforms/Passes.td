// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_TRANSFORMS_PASSES
#define HC_FRONT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCFrontFoldRegionDefs : Pass<"hc-front-fold-region-defs"> {
  let summary = "Fold `@group.workitems`/`@group.subgroups` def + immediate call.";
  let description = [{
    Python materializes a workitem/subgroup region as

        @group.workitems
        def inner(wi):
            ...
        inner()            # or `return inner()`

    The frontend emits the region op followed by ghost ops: a
    `hc_front.name "inner" {ref.kind = "local"}`, an `hc_front.call`
    consuming it, and (for the `return` flavour) a trailing
    `hc_front.return %call`. `inner` never becomes a real callable —
    the region op is the lowering, and an inner `hc_front.return %v`
    falls through to terminate the enclosing func.

    Pass matches the ghost triad by the region op's `name` discardable
    attribute and erases it. After this, the converter sees only real
    `hc_front.call` ops; a surviving `ref.kind = "local"` call still
    errors — the operator just knows the folder didn't run.

    Match shapes: unused `hc_front.call` (the `inner()` line) and call
    whose only same-block use is `hc_front.return` (the `return
    inner()` shape). Any other use (`x = inner()`, etc.) is left
    alone — fold is structural, not semantic.
  }];
  let dependentDialects = [
    "::mlir::hc::front::HCFrontDialect",
  ];
}

def HCFrontInline : Pass<"hc-front-inline"> {
  let summary = "Inline `hc_front.func`s tagged `ref.kind = \"inline\"` at every call site.";
  let description = [{
    Rewrites each `hc_front.call` whose callee's `hc_front.name` carries
    `ref.kind = "inline"` into an `hc_front.inlined_region` wrapping a
    clone of the callee body, binds args to the callee's `parameters`
    entries, and maps `hc_front.return` values to region results. After
    every call is consumed the callee `hc_front.func` is erased, so no
    `kind = "inline"` symbol reaches `-convert-hc-front-to-hc`.

    `hc_front.inlined_region` is a name-scope boundary only — no
    execution semantics — so this pass stays oblivious to caller-local
    names. Per-site alpha-renaming happens at conversion time when the
    boundary flattens.

    Recursion is a hard error: inline helpers must not recurse. A
    `kind = "inline"` call whose callee is missing from the module is
    also a hard error.
  }];
  let dependentDialects = [
    "::mlir::hc::front::HCFrontDialect",
  ];
}

#endif // HC_FRONT_TRANSFORMS_PASSES
