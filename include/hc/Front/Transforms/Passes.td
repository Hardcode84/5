// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_TRANSFORMS_PASSES
#define HC_FRONT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCFrontFoldRegionDefs : Pass<"hc-front-fold-region-defs", "::mlir::ModuleOp"> {
  let summary = "Fold `@group.workitems`/`@group.subgroups` def + immediate call.";
  let description = [{
    Python's only way to materialize an `hc_front.workitem_region` or
    `hc_front.subgroup_region` is

        @group.workitems
        def inner(wi):
            ...
        inner()            # or `return inner()`

    The frontend emits that as the region op followed by three ghost
    ops — a `hc_front.name "inner" {ref.kind = "local"}`, an
    `hc_front.call` whose only operand is that name, and (for the
    `return` flavour) a trailing `hc_front.return %call`. The call has
    no meaning at `hc` level because `inner` never becomes a real
    callable — the region op itself is the lowering, and any
    `hc_front.return %v` inside it falls through the region to
    terminate the enclosing func.

    This pass matches the ghost triad by the region op's `name`
    discardable attribute and erases it. After this pass the converter
    sees only real `hc_front.call` ops (driver callees, DSL methods,
    built-ins); a surviving `ref.kind = "local"` call is therefore
    still an error — the operator just knows the folder didn't run.

    A `hc_front.call` site whose only caller-visible use is a
    same-block `hc_front.return` is the common `return inner()`
    shape; an unused call is `inner()` on a line by itself. Any other
    use of the call's result (binding through `x = inner()`,
    threading through subscripts, etc.) is left alone — the inliner
    contract is structural, not semantic, and a surviving
    `ref.kind = "local"` callee is the converter's error to raise.
  }];
  let dependentDialects = [
    "::mlir::hc::front::HCFrontDialect",
  ];
}

def HCFrontInline : Pass<"hc-front-inline", "::mlir::ModuleOp"> {
  let summary = "Inline `hc_front.func`s tagged `ref.kind = \"inline\"` at every call site.";
  let description = [{
    Rewrites each `hc_front.call` whose callee's `hc_front.name` carries
    `ref.kind = "inline"` into an `hc_front.inlined_region` op that
    wraps a clone of the callee's body, binds the call's arguments to
    the callee's ordered `parameters` attribute entries, and maps the
    body's `hc_front.return` values onto the region's results. Once
    every call has been consumed the callee `hc_front.func` itself is
    erased, so no `kind = "inline"` symbol survives to the
    `-convert-hc-front-to-hc` boundary.

    The `hc_front.inlined_region` op is purely a name-scope boundary
    — it has no execution semantics — which keeps this pass oblivious
    to the caller's local names. Per-site alpha-renaming happens at
    conversion time when the boundary is flattened into the caller.

    Cycles are a hard error (inline helpers must not recurse); a
    `kind = "inline"` call whose callee is absent from the module is
    also a hard error because the Python driver is expected to emit
    the marker func alongside every call it stamps.
  }];
  let dependentDialects = [
    "::mlir::hc::front::HCFrontDialect",
  ];
}

#endif // HC_FRONT_TRANSFORMS_PASSES
