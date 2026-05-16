// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Verifies the front-to-hc handshake seeds launch-context literal
// bindings (`$WGS<axis>`, `$WS<axis>`, `$WV0`, `$GSZ0`) from any
// integer-literal `group_shape` / `work_shape` / `subgroup_size`
// metadata. `hc.compile(symbols={...})` only sees the user's name
// space, so these `$`-prefixed system symbols couldn't otherwise be
// folded by `hc-specialize-literals` — leaving downstream passes
// (`hc-lower-launch-body`'s static-shape check, `hc.zeros` /
// `hc.bare_tensor` carriers built off `group.shape`) tripping on
// unsubstituted launch-context dims.
// RUN: hc-opt --convert-hc-front-to-hc -split-input-file %s | FileCheck %s

// Both `work_shape` and `group_shape` are integer-literal arrays, so
// every axis contributes a binding. `subgroup_size` adds `$WV0`, and
// the static product of `group_shape` adds `$GSZ0` (8 * 4 = 32).
// CHECK-LABEL: hc.kernel @all_literal
// CHECK-SAME:    literal_bindings = {"$GSZ0" = 32 : i64, "$WGS0" = 8 : i64, "$WGS1" = 4 : i64, "$WS0" = 16 : i64, "$WS1" = 32 : i64, "$WV0" = 32 : i64}
hc_front.kernel "all_literal" attributes {
  decorators = ["kernel"],
  group_shape = ["8", "4"],
  parameters = [{name = "group"}],
  returns = "None",
  subgroup_size = 32 : i32,
  work_shape = ["16", "32"]
} {
  hc_front.return
}

// -----

// Symbolic axes don't contribute literal bindings — `$WGS1`
// references symbolic `BLOCK` and only resolves once the user pins
// `BLOCK` in `hc.compile(symbols=...)`. The literal `8` axis still
// contributes `$WGS0 = 8`. Static `$GSZ0` is skipped because not all
// dims are literal.
// CHECK-LABEL: hc.kernel @mixed_literal
// CHECK-SAME:    literal_bindings = {"$WGS0" = 8 : i64, "$WS0" = 16 : i64}
hc_front.kernel "mixed_literal" attributes {
  decorators = ["kernel"],
  group_shape = ["8", "BLOCK"],
  parameters = [{name = "group"}],
  returns = "None",
  work_shape = ["16", "N"]
} {
  hc_front.return
}

// -----

// User-supplied `literal_bindings` carry through; launch-geo entries
// are appended without clobbering existing keys (dict prints
// alphabetically so `$`-prefixed names lead).
// CHECK-LABEL: hc.kernel @user_plus_geo
// CHECK-SAME:    literal_bindings = {"$GSZ0" = 8 : i64, "$WGS0" = 8 : i64, M = 4 : i64}
hc_front.kernel "user_plus_geo" attributes {
  decorators = ["kernel"],
  group_shape = ["8"],
  literal_bindings = {M = 4 : i64},
  parameters = [{name = "group"}],
  returns = "None",
  work_shape = ["M"]
} {
  hc_front.return
}

// -----

// No integer-literal launch-geo dims, no user bindings -> no
// `literal_bindings` attr. Purely symbolic kernels remain
// byte-identical post-handshake.
// CHECK-LABEL: hc.kernel @no_literals
// CHECK-NOT:    literal_bindings
hc_front.kernel "no_literals" attributes {
  decorators = ["kernel"],
  group_shape = ["B"],
  parameters = [{name = "group"}],
  returns = "None",
  work_shape = ["N"]
} {
  hc_front.return
}
