// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative tests for `#hc.layout<...>` verifier diagnostics.
//
// RUN: hc-opt %s -verify-diagnostics -split-input-file

func.func @rank_mismatch_rejected() attributes {
  // expected-error@+1 {{shape_syms and index_syms must have the same length}}
  test.layout = #hc.layout<shape_syms = ["M", "K"],
                           index_syms = ["i", "j", "lane"],
                           params = {},
                           storage_size = #hc.expr<"K">,
                           offset = #hc.expr<"j">>
} {
  return
}
