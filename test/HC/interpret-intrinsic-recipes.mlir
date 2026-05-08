// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-interpret-intrinsic-recipes`. The pass walks a sibling
// `__hc_intrinsic_lowerings__` module, applies the named sequences whose
// `hc.target` matches the requested target against the surrounding payload
// module, then erases the lowerings module so downstream passes don't see
// stray transform IR. Anything left as a bare `hc.call_intrinsic` is a
// hard error: we'd rather fail here with the callee + target named than
// punt the gap to whichever pass runs next.

// RUN: hc-opt --hc-interpret-intrinsic-recipes='target=test' --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s

// A matching recipe rewrites the `hc.call_intrinsic` and the lowerings
// module is erased on the way out. The intrinsic decl is also swept once
// its last call site goes away — keeping the post-interpretation IR free
// of stray HC ops without forcing every caller to add a separate
// symbol-DCE pass. The positive case is what every later target lowering
// looks like in miniature.
// CHECK-LABEL: func.func @user
// CHECK: %[[K:.*]] = arith.constant 42 : i32
// CHECK: return %[[K]] : i32
// CHECK-NOT: hc.call_intrinsic
// CHECK-NOT: hc.intrinsic
// CHECK-NOT: __hc_intrinsic_lowerings__
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @user(%a: i32) -> i32 {
    %r = hc.call_intrinsic @sized(%a) : (i32) -> i32
    return %r : i32
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    transform.named_sequence @__hc_lower_sized_test(%root: !transform.any_op) attributes {hc.target = "test"} {
      %call = transform.hc.match_intrinsic_call %root @sized target = "test"
          : (!transform.any_op) -> !transform.any_op
      %ty = transform.hc.get_intrinsic_result_type %call {index = 0 : i64}
          : (!transform.any_op) -> !transform.type
      %created = transform.hc.create_op "arith.constant" at %call ()
          result_types(%ty)
          dynamic_attrs []()
          static_attrs = {value = 42 : i32}
          : (!transform.any_op, !transform.type) -> (!transform.any_value)
      transform.hc.replace_intrinsic_call %call with %created
          : (!transform.any_op, !transform.any_value) -> ()
      transform.yield
    }
  }
}

// -----

// Mismatched-target recipes are skipped, so the call survives and the
// pass reports the gap with the callee + target so the caller can tell
// whether the recipe registry or the target string is wrong.
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @user(%a: i32) -> i32 {
    // expected-error@+1 {{no intrinsic lowering recipe matched @sized for target 'test'}}
    %r = hc.call_intrinsic @sized(%a) : (i32) -> i32
    return %r : i32
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    transform.named_sequence @__hc_lower_sized_other(%root: !transform.any_op) attributes {hc.target = "other"} {
      %call = transform.hc.match_intrinsic_call %root @sized target = "other"
          : (!transform.any_op) -> !transform.any_op
      %ty = transform.hc.get_intrinsic_result_type %call {index = 0 : i64}
          : (!transform.any_op) -> !transform.type
      %created = transform.hc.create_op "arith.constant" at %call ()
          result_types(%ty)
          dynamic_attrs []()
          static_attrs = {value = 99 : i32}
          : (!transform.any_op, !transform.type) -> (!transform.any_value)
      transform.hc.replace_intrinsic_call %call with %created
          : (!transform.any_op, !transform.any_value) -> ()
      transform.yield
    }
  }
}

// -----

// No lowerings module + a remaining call hits the same uncovered-call
// failure mode; the pass still surfaces the gap rather than silently
// passing through.
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @user(%a: i32) -> i32 {
    // expected-error@+1 {{no intrinsic lowering recipe matched @sized for target 'test'}}
    %r = hc.call_intrinsic @sized(%a) : (i32) -> i32
    return %r : i32
  }
}

// -----

// A payload with no calls at all is a no-op the pass tolerates without
// noise — unrelated kernels in the same compilation should not pay for
// the recipe-applying machinery.
// CHECK-LABEL: func.func @no_calls
// CHECK: %[[Z:.*]] = arith.constant 0 : i32
// CHECK: return %[[Z]] : i32
module {
  func.func @no_calls() -> i32 {
    %z = arith.constant 0 : i32
    return %z : i32
  }
}

// -----

// `transform.hc.require_intrinsic_attr` is the recipe author's pre-rewrite
// assertion. When the call's named attribute matches the literal, the
// rewrite proceeds; this is the path every well-formed call site takes.
// CHECK-LABEL: func.func @user
// CHECK: %[[K:.*]] = arith.constant 7 : i32
// CHECK: return %[[K]] : i32
// CHECK-NOT: hc.call_intrinsic
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @user(%a: i32) -> i32 {
    %r = hc.call_intrinsic @sized(%a) {arch = "gfx11", wave_size = 32 : i64}
        : (i32) -> i32
    return %r : i32
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    transform.named_sequence @__hc_lower_sized_test(%root: !transform.any_op) attributes {hc.target = "test"} {
      %call = transform.hc.match_intrinsic_call %root @sized target = "test"
          : (!transform.any_op) -> !transform.any_op
      transform.hc.require_intrinsic_attr %call {expected = "gfx11", name = "arch"}
          : !transform.any_op
      transform.hc.require_intrinsic_attr %call {expected = 32 : i64, name = "wave_size"}
          : !transform.any_op
      %ty = transform.hc.get_intrinsic_result_type %call {index = 0 : i64}
          : (!transform.any_op) -> !transform.type
      %created = transform.hc.create_op "arith.constant" at %call ()
          result_types(%ty)
          dynamic_attrs []()
          static_attrs = {value = 7 : i32}
          : (!transform.any_op, !transform.type) -> (!transform.any_value)
      transform.hc.replace_intrinsic_call %call with %created
          : (!transform.any_op, !transform.any_value) -> ()
      transform.yield
    }
  }
}

// -----

// A wrong attribute value short-circuits the rewrite with a definite
// failure. Definite (not silenceable) is intentional — silenceable would
// degrade to the generic "no recipe matched" message and the user would
// have to guess which assumption the recipe encoded was violated.
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @user(%a: i32) -> i32 {
    // expected-note@+1 {{call site}}
    %r = hc.call_intrinsic @sized(%a) {arch = "gfx12"} : (i32) -> i32
    return %r : i32
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    // expected-error@+1 {{failed to apply intrinsic lowering recipe '__hc_lower_sized_test' for target 'test'}}
    transform.named_sequence @__hc_lower_sized_test(%root: !transform.any_op) attributes {hc.target = "test"} {
      %call = transform.hc.match_intrinsic_call %root @sized target = "test"
          : (!transform.any_op) -> !transform.any_op
      // expected-error@+1 {{intrinsic call @sized has arch = "gfx12", expected "gfx11"}}
      transform.hc.require_intrinsic_attr %call {expected = "gfx11", name = "arch"}
          : !transform.any_op
      transform.yield
    }
  }
}

// -----

// Missing attribute is a distinct failure: the call doesn't even carry
// the name, so the recipe can't make a value-equality decision and the
// pass surfaces that explicitly.
module {
  hc.intrinsic @sized(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @user(%a: i32) -> i32 {
    // expected-note@+1 {{call site}}
    %r = hc.call_intrinsic @sized(%a) : (i32) -> i32
    return %r : i32
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    // expected-error@+1 {{failed to apply intrinsic lowering recipe '__hc_lower_sized_test' for target 'test'}}
    transform.named_sequence @__hc_lower_sized_test(%root: !transform.any_op) attributes {hc.target = "test"} {
      %call = transform.hc.match_intrinsic_call %root @sized target = "test"
          : (!transform.any_op) -> !transform.any_op
      // expected-error@+1 {{intrinsic call @sized missing required attribute 'arch'}}
      transform.hc.require_intrinsic_attr %call {expected = "gfx11", name = "arch"}
          : !transform.any_op
      transform.yield
    }
  }
}

// -----

// `transform.hc.cast_value` + `transform.hc.constant_type` bridge an
// intrinsic call's bare-typed operand through `unrealized_conversion_cast`
// to the upstream type the lowered op expects, then back to the call's
// bare result type for the replacement. The recipe-inserted casts pair
// with the existing UCCs the surrounding code planted on either side of
// the call boundary, so the post-rewrite `--canonicalize` collapses the
// upstream → bare → upstream chain to identity and the `math.absf` ends
// up surrounded by plain upstream values.
// CHECK-LABEL: func.func @bridged
// CHECK-SAME: %[[ARG:[a-z0-9_]+]]: vector<16xf16>
// CHECK: %[[ABS:.*]] = math.absf %[[ARG]] : vector<16xf16>
// CHECK: return %[[ABS]] : vector<16xf16>
// CHECK-NOT: hc.call_intrinsic
// CHECK-NOT: unrealized_conversion_cast
module {
  hc.intrinsic @bare_absf(%a: !hc.bare_vector<f16, ["16"]>) -> !hc.bare_vector<f16, ["16"]>
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @bridged(%upstream: vector<16xf16>) -> vector<16xf16> {
    %a = builtin.unrealized_conversion_cast %upstream
        : vector<16xf16> to !hc.bare_vector<f16, ["16"]>
    %r = hc.call_intrinsic @bare_absf(%a)
        : (!hc.bare_vector<f16, ["16"]>) -> !hc.bare_vector<f16, ["16"]>
    %back = builtin.unrealized_conversion_cast %r
        : !hc.bare_vector<f16, ["16"]> to vector<16xf16>
    return %back : vector<16xf16>
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    transform.named_sequence @__hc_lower_bare_absf_test(%root: !transform.any_op)
        attributes {hc.target = "test"} {
      %call = transform.hc.match_intrinsic_call %root @bare_absf target = "test"
          : (!transform.any_op) -> !transform.any_op
      %upstream = transform.hc.constant_type vector<16xf16> : !transform.type
      %a_raw = transform.hc.get_intrinsic_operand %call {index = 0 : i64}
          : (!transform.any_op) -> !transform.any_value
      %a_up = transform.hc.cast_value %a_raw to %upstream
          : (!transform.any_value, !transform.type) -> !transform.any_value
      %abs = transform.hc.create_op "math.absf" at %call (%a_up)
          result_types(%upstream)
          dynamic_attrs []()
          : (!transform.any_op, !transform.any_value, !transform.type)
          -> (!transform.any_value)
      %bare = transform.hc.get_intrinsic_result_type %call {index = 0 : i64}
          : (!transform.any_op) -> !transform.type
      %abs_bare = transform.hc.cast_value %abs to %bare
          : (!transform.any_value, !transform.type) -> !transform.any_value
      transform.hc.replace_intrinsic_call %call with %abs_bare
          : (!transform.any_op, !transform.any_value) -> ()
      transform.yield
    }
  }
}

// -----

// `transform.hc.cast_value` is a no-op when the source value already has
// the target type — the source forwards through unchanged and no UCC is
// inserted. This keeps recipes that opportunistically cast to the
// upstream type cheap when the surrounding pipeline already arranged for
// upstream-typed operands.
// CHECK-LABEL: func.func @noop_cast
// CHECK-SAME: %[[ARG:[a-z0-9_]+]]: i32
// CHECK: return %[[ARG]] : i32
// CHECK-NOT: hc.call_intrinsic
// CHECK-NOT: unrealized_conversion_cast
module {
  hc.intrinsic @passthrough(%a: i32) -> i32
      scope = #hc.scope<"WorkItem"> parameters = ["a"] {}
  func.func @noop_cast(%a: i32) -> i32 {
    %r = hc.call_intrinsic @passthrough(%a) : (i32) -> i32
    return %r : i32
  }
  module @__hc_intrinsic_lowerings__ attributes {transform.with_named_sequence} {
    transform.named_sequence @__hc_lower_passthrough_test(%root: !transform.any_op)
        attributes {hc.target = "test"} {
      %call = transform.hc.match_intrinsic_call %root @passthrough target = "test"
          : (!transform.any_op) -> !transform.any_op
      %i32_ty = transform.hc.constant_type i32 : !transform.type
      %a_raw = transform.hc.get_intrinsic_operand %call {index = 0 : i64}
          : (!transform.any_op) -> !transform.any_value
      // Source type already matches target; cast_value forwards %a_raw.
      %a_passthrough = transform.hc.cast_value %a_raw to %i32_ty
          : (!transform.any_value, !transform.type) -> !transform.any_value
      transform.hc.replace_intrinsic_call %call with %a_passthrough
          : (!transform.any_op, !transform.any_value) -> ()
      transform.yield
    }
  }
}
