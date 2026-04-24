// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s --hc-front-fold-region-defs | FileCheck %s

// Python pattern:
//   @kernel.func(scope=WorkGroup)
//   def init(group):
//       @group.workitems
//       def inner(wi):
//           _ = wi
//           return group.vzeros(...)
//       return inner()
//
// The frontend emits the region op with `name = "inner"`, followed by
// the ghost `hc_front.name "inner" {ref.kind = "local"} + hc_front.call
// + hc_front.return`. Fold erases all three — the region's inner
// `hc_front.return %v` falls through on conversion and terminates the
// func, so the trailing ops were always dead.
// CHECK-LABEL: hc_front.func "return_shape"
// CHECK: hc_front.workitem_region
// CHECK-SAME: name = "inner"
// CHECK: hc_front.call
// CHECK: hc_front.return
// CHECK-NOT: hc_front.name "inner"
// CHECK-NOT: ref = {kind = "local"}
hc_front.func "return_shape" attributes {parameters = [{name = "group"}], scope = "WorkGroup"} {
  hc_front.workitem_region captures = ["group"] attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
    %g = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %m = hc_front.attr %g, "vzeros" {ref = {kind = "dsl_method", method = "vzeros"}}
    %c = hc_front.call %m()
    hc_front.return %c
  }
  %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
  %1 = hc_front.call %0()
  hc_front.return %1
}

// Bare-call shape: Python `inner()` on a line by itself (no outer
// return). The region op may even have no inner `hc_front.return` (a
// side-effects-only region). Fold still erases the trailing
// name+call; no trailing return to worry about.
// CHECK-LABEL: hc_front.func "bare_shape"
// CHECK: hc_front.subgroup_region
// CHECK-SAME: name = "wave"
// CHECK-NOT: hc_front.name "wave"
// CHECK-NOT: ref = {kind = "local"}
hc_front.func "bare_shape" attributes {parameters = [{name = "group"}], scope = "WorkGroup"} {
  hc_front.subgroup_region captures = ["group"] attributes {decorators = ["group.subgroups"], name = "wave", parameters = [{name = "wi"}]} {
    %g = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %m = hc_front.attr %g, "vzeros" {ref = {kind = "dsl_method", method = "vzeros"}}
    %c = hc_front.call %m()
  }
  %0 = hc_front.name "wave" {ctx = "load", ref = {kind = "local"}}
  %1 = hc_front.call %0()
}

// Negative: the trailing call is to a real callee (`ref.kind =
// "callee"`), not a ghost local, so nothing matches and the call
// survives verbatim.
// CHECK-LABEL: hc_front.func "nonlocal_callee"
// CHECK: hc_front.workitem_region
// CHECK: hc_front.name "f"
// CHECK-SAME: kind = "callee"
// CHECK: hc_front.call
// CHECK: hc_front.return
hc_front.func "nonlocal_callee" attributes {parameters = [], scope = "WorkItem"} {
  hc_front.workitem_region attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
  }
  %0 = hc_front.name "f" {ctx = "load", ref = {callee = "@f", kind = "callee", scope = "WorkItem"}}
  %1 = hc_front.call %0()
  hc_front.return %1
}

// Negative: region has no `name` attr. The ghost trail uses a local
// callee, but without a stamped region name we refuse to fold — the
// Python driver never emits a name-less region in this pattern, and
// a driver that stops stamping is better diagnosed downstream than
// silently matched.
// CHECK-LABEL: hc_front.func "unnamed_region"
// CHECK: hc_front.workitem_region
// CHECK: hc_front.name "inner"
// CHECK-SAME: kind = "local"
// CHECK: hc_front.call
hc_front.func "unnamed_region" attributes {parameters = [], scope = "WorkItem"} {
  hc_front.workitem_region attributes {decorators = ["group.workitems"], parameters = [{name = "wi"}]} {
  }
  %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
  %1 = hc_front.call %0()
}

// Negative: the ghost name flows somewhere other than a call's
// callee slot — here it is passed as an argument. Folding would
// silently drop the argument; the match rejects the case so the
// converter's `ref.kind = "local"` diagnostic still fires.
// CHECK-LABEL: hc_front.func "local_as_arg"
// CHECK: hc_front.workitem_region
// CHECK-SAME: name = "inner"
// CHECK: hc_front.name "inner"
// CHECK-SAME: kind = "local"
// CHECK: hc_front.call
hc_front.func "local_as_arg" attributes {parameters = [], scope = "WorkItem"} {
  hc_front.workitem_region attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
  }
  %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
  %f = hc_front.name "consume" {ctx = "load", ref = {callee = "@consume", kind = "callee", scope = "WorkItem"}}
  %1 = hc_front.call %f(%0)
}

// Negative: the call's result is bound to another local (`x =
// inner()`) rather than being returned or discarded. The folder is
// structural and only matches the "tail return" / "no use" shapes;
// a bound result would leave a dangling SSA dep. The converter's
// `ref.kind = "local"` diagnostic is the right signal here — the
// pattern is not supported at all by the fold.
// CHECK-LABEL: hc_front.func "bound_result"
// CHECK: hc_front.workitem_region
// CHECK-SAME: name = "inner"
// CHECK: hc_front.name "inner"
// CHECK-SAME: kind = "local"
// CHECK: hc_front.call
// CHECK: hc_front.target_name "x"
// CHECK: hc_front.assign
hc_front.func "bound_result" attributes {parameters = [], scope = "WorkItem"} {
  hc_front.workitem_region attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
  }
  %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
  %1 = hc_front.call %0()
  %t = hc_front.target_name "x"
  hc_front.assign %t = %1
}

// Mixed shape: a local-as-arg rejection must not abort the forward
// scan; a second same-named ghost name that *is* a valid callee
// further down still folds. This pins `continue` over `return` in the
// arg-position guard so pathological or future IR shapes don't leak
// ghost triads past the folder.
// CHECK-LABEL: hc_front.func "mixed_same_name"
// CHECK: hc_front.workitem_region
// CHECK-SAME: name = "inner"
// First "inner" is used as a call argument — stays verbatim.
// CHECK: hc_front.name "inner"
// CHECK-SAME: kind = "local"
// CHECK: hc_front.call %{{.+}}(%{{.+}})
// Second "inner" was the real ghost callee — must be erased.
// CHECK-NOT: hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
// CHECK-NOT: hc_front.call %{{.+}}()
hc_front.func "mixed_same_name" attributes {parameters = [], scope = "WorkItem"} {
  hc_front.workitem_region attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
  }
  %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
  %f = hc_front.name "consume" {ctx = "load", ref = {callee = "@consume", kind = "callee", scope = "WorkItem"}}
  %1 = hc_front.call %f(%0)
  %2 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
  %3 = hc_front.call %2()
  hc_front.return %3
}
