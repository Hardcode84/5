// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative coverage for `-hc-lower-pow`. Each chunk pins one of the
// per-case diagnostics the pass emits when an `hc.pow` op carries an
// unsupported rhs. The catch-all message family is uniform; only the
// trailing reason changes.
//
// RUN: hc-opt --hc-lower-pow --split-input-file --verify-diagnostics %s

// Non-constant rhs: a runtime exponent would need a `math.pow`-style
// lowering that the dialect doesn't have yet. Diagnose at the
// `hc.pow` op so the failure fingers the original `**`.
func.func @pow_runtime_exponent(%x: !hc.undef, %y: !hc.undef)
    -> !hc.undef {
  // expected-error@+1 {{unsupported `hc.pow` rhs: only positive integer-literal exponents are supported today; got a non-constant rhs}}
  %r = hc.pow %x, %y : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// Float-literal rhs: same family — fractional exponents need the
// runtime path the dialect doesn't have.
func.func @pow_float_exponent(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<2.5 : f64> : !hc.undef
  // expected-error@+1 {{unsupported `hc.pow` rhs: only positive integer-literal exponents are supported today; got rhs constant 2.500000e+00 : f64}}
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// Zero exponent: `x**0` is `1`, but we can't synthesize a well-typed
// `1` without inference (the lhs's element type may be anything). Tell
// the user to write the constant directly.
func.func @pow_zero_exponent(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<0 : i64> : !hc.undef
  // expected-error@+1 {{unsupported `hc.pow` rhs: only positive integer-literal exponents are supported today; got 0}}
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}

// -----

// Negative exponent: Python `**-N` on ints returns a float (1/x**N).
// Reciprocal codegen needs a runtime division shape we'd rather not
// bake into the unfold today.
func.func @pow_negative_exponent(%x: !hc.undef) -> !hc.undef {
  %e = hc.const<-3 : i64> : !hc.undef
  // expected-error@+1 {{unsupported `hc.pow` rhs: only positive integer-literal exponents are supported today; got -3}}
  %r = hc.pow %x, %e : (!hc.undef, !hc.undef) -> !hc.undef
  return %r : !hc.undef
}
