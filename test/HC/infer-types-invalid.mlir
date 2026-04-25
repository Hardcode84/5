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
