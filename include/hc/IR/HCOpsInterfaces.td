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
    carrier:

      - `hc.for_range`:       iter_args + `hc.yield`
      - `hc.if`:              results + yields in both branches
      - `hc.workitem_region`: captures + results (via ODS extension, TBD)
      - `hc.subgroup_region`: captures + results (via ODS extension, TBD)

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
