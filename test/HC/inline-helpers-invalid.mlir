// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt --hc-inline-helpers -split-input-file -verify-diagnostics %s

hc.func @recursive(%x: i32) -> i32 {
  // expected-error @+1 {{recursive hc.func inlining is not supported}}
  %r = hc.call @recursive(%x) : (i32) -> i32
  hc.return %r : i32
}
