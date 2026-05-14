// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt -hc-specialize-literals -split-input-file -verify-diagnostics %s

// A binding for a name the kernel hasn't declared in `literals` is a
// hard error — matches the Python-side `hc.compile(symbols=...)`
// rejection but covers IR-only stamping paths too.

// expected-error@+1 {{literal_bindings key 'NOT_DECLARED' is not declared in `literals`}}
hc.kernel @undeclared(%buf: !hc.buffer<f32, ["K"]>)
    attributes {literals = ["K"], literal_bindings = {NOT_DECLARED = 4 : i64},
                work_shape = #hc.shape<["1"]>,
                group_shape = #hc.shape<["1"]>} {
  %k = hc.const<0 : i64> : !hc.idx<"K">
  hc.return
}

// -----

// Non-integer binding value is a hard error.

// expected-error@+1 {{literal_bindings['K'] must be an IntegerAttr}}
hc.kernel @nonint(%buf: !hc.buffer<f32, ["K"]>)
    attributes {literals = ["K"], literal_bindings = {K = "four"},
                work_shape = #hc.shape<["1"]>,
                group_shape = #hc.shape<["1"]>} {
  %k = hc.const<0 : i64> : !hc.idx<"K">
  hc.return
}
