// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s

// CHECK-LABEL: func.func @geometry_undef
// CHECK: %{{.*}}:2 = hc.group_id %{{.*}} : (!hc.undef) -> (!hc.undef, !hc.undef)
// CHECK: %{{.*}}:2 = hc.local_id %{{.*}} : (!hc.undef) -> (!hc.undef, !hc.undef)
// CHECK: %{{.*}} = hc.subgroup_id %{{.*}} : (!hc.undef) -> !hc.undef
// CHECK: %{{.*}}:2 = hc.group_shape %{{.*}} : (!hc.undef) -> (!hc.undef, !hc.undef)
// CHECK: %{{.*}} = hc.group_size %{{.*}} : (!hc.undef) -> !hc.undef
// CHECK: %{{.*}}:2 = hc.work_offset %{{.*}} : (!hc.undef) -> (!hc.undef, !hc.undef)
// CHECK: %{{.*}}:2 = hc.work_shape %{{.*}} : (!hc.undef) -> (!hc.undef, !hc.undef)
// CHECK: %{{.*}} = hc.wave_size %{{.*}} : (!hc.undef) -> !hc.undef
func.func @geometry_undef(%g: !hc.undef) {
  %gid:2    = hc.group_id    %g : (!hc.undef) -> (!hc.undef, !hc.undef)
  %lid:2    = hc.local_id    %g : (!hc.undef) -> (!hc.undef, !hc.undef)
  %sgid     = hc.subgroup_id %g : (!hc.undef) -> !hc.undef
  %gs:2     = hc.group_shape %g : (!hc.undef) -> (!hc.undef, !hc.undef)
  %gsize    = hc.group_size  %g : (!hc.undef) -> !hc.undef
  %woff:2   = hc.work_offset %g : (!hc.undef) -> (!hc.undef, !hc.undef)
  %wshape:2 = hc.work_shape  %g : (!hc.undef) -> (!hc.undef, !hc.undef)
  %ws       = hc.wave_size   %g : (!hc.undef) -> !hc.undef
  return
}

// CHECK-LABEL: func.func @geometry_refined
// CHECK: %{{.*}}:2 = hc.group_id %{{.*}} : (!hc.undef) -> (!hc.idx<"_gid0">, !hc.idx<"_gid1">)
// CHECK: %{{.*}} = hc.wave_size %{{.*}} : (!hc.undef) -> !hc.idx<"32">
func.func @geometry_refined(%g: !hc.undef) {
  %gid:2 = hc.group_id %g
      : (!hc.undef) -> (!hc.idx<"_gid0">, !hc.idx<"_gid1">)
  %ws    = hc.wave_size %g : (!hc.undef) -> !hc.idx<"32">
  return
}
