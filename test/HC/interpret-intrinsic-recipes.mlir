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

// RUN: hc-opt --hc-interpret-intrinsic-recipes='target=test' --split-input-file --verify-diagnostics %s | FileCheck %s

// A matching recipe rewrites the `hc.call_intrinsic` and the lowerings
// module is erased on the way out. The positive case is what every later
// target lowering looks like in miniature.
// CHECK-LABEL: func.func @user
// CHECK: %[[K:.*]] = arith.constant 42 : i32
// CHECK: return %[[K]] : i32
// CHECK-NOT: hc.call_intrinsic
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
