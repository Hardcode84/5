// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-infer-types -split-input-file -verify-diagnostics %s

hc.func @const_result_conflict {
  // expected-error @+1 {{has conflicting HC type facts for result #0}}
  %c = hc.const<1 : i64> : i64
  hc.return
}

// -----

hc.func @tuple_getitem_oob(%tuple: tuple<!hc.idx<"A">, !hc.idx<"B">>,
                           %idx: !hc.idx<"2">) {
  // expected-error @+1 {{tuple index 2 out of bounds for tuple of size 2}}
  %item = hc.getitem %tuple[%idx]
      : (tuple<!hc.idx<"A">, !hc.idx<"B">>, !hc.idx<"2">) -> !hc.undef
  hc.return
}
