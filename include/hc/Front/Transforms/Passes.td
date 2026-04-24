// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_TRANSFORMS_PASSES
#define HC_FRONT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

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
