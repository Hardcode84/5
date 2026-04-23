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
                              across iterations.
      - `hc.if`:              transparent. Carried names become op
                              results + yields in both branches;
                              silent-branch fallback = outer snapshot.
      - `hc.workitem_region`: Python-style nested scope. Reads inside
                              capture the outer binding (lazy
                              `hc.name_load` at the op's insertion
                              point); writes inside shadow and don't
                              leak out. Out-bound propagation needs
                              an ODS result extension (separate
                              change).
      - `hc.subgroup_region`: ditto.

    The interface is a marker only — the pass dispatches on op kind
    internally. Keeping the opt-in in ODS (rather than a hard-coded
    `isa<...>` list inside the pass) documents the set of participating
    ops next to the ops themselves, and lets a future dialect-extension
    op join the promotion without touching pass code beyond a new case
    in its switch.
  }];
  let cppNamespace = "::mlir::hc";
}

#endif // HC_IR_HCOPSINTERFACES
