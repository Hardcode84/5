// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Default hc_front -> hc schedule.
//
// Mirrors the pass order `hc-opt` and `doc/lowering.md` document for
// the frontend stage: fold/erase region scaffolding, inline undecorated
// helpers, convert to `hc`, then promote `hc.name_load` / `hc.assign`
// into SSA. `hc.compile` loads this via `-transform-preload-library`
// and runs it with `-transform-interpreter`; callers wanting a
// different order can pass `schedule=<path-or-text>` to override.
module attributes {transform.with_named_sequence} {
  // `%m` is consumed by each `apply_registered_pass`; the verifier
  // requires the entry block argument to reflect that by omitting the
  // `{transform.readonly}` attribute. Each pass produces a fresh handle
  // that threads into the next.
  transform.named_sequence @__transform_main(%m: !transform.any_op) {
    %m1 = transform.apply_registered_pass "hc-front-fold-region-defs" to %m
        : (!transform.any_op) -> !transform.any_op
    %m2 = transform.apply_registered_pass "hc-front-inline" to %m1
        : (!transform.any_op) -> !transform.any_op
    %m3 = transform.apply_registered_pass "convert-hc-front-to-hc" to %m2
        : (!transform.any_op) -> !transform.any_op
    %m4 = transform.apply_registered_pass "hc-promote-names" to %m3
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
