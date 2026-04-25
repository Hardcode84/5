// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: hc-opt %s | hc-opt | FileCheck %s
// RUN: hc-opt --hc-infer-types %s | FileCheck %s --check-prefix=INFER

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

// CHECK-LABEL: func.func @geometry_group
// CHECK: !hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>
func.func @geometry_group(%g: !hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>) {
  %gid:2 = hc.group_id %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> (!hc.undef, !hc.undef)
  return
}

// INFER-LABEL: hc.func @geometry_group_infer
// INFER: hc.group_id {{.*}} -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
// INFER: hc.group_shape {{.*}} -> (!hc.idx<"16">, !hc.idx<"8">)
// INFER: hc.work_shape {{.*}} -> (!hc.idx<"M">, !hc.idx<"N">)
// INFER: hc.group_size {{.*}} -> !hc.idx<"128">
// INFER: hc.wave_size {{.*}} -> !hc.idx<"32">
hc.func @geometry_group_infer(%g: !hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>) {
  %gid:2 = hc.group_id %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> (!hc.undef, !hc.undef)
  %gs:2 = hc.group_shape %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> (!hc.undef, !hc.undef)
  %ws:2 = hc.work_shape %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> (!hc.undef, !hc.undef)
  %size = hc.group_size %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> !hc.undef
  %wave = hc.wave_size %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> !hc.undef
  hc.return
}

// INFER-LABEL: hc.func @geometry_tuple_getitem_infer
// INFER: hc.group_id {{.*}} -> (!hc.idx<"$WG0">, !hc.idx<"$WG1">)
// INFER: hc.tuple({{.*}}) : (!hc.idx<"$WG0">, !hc.idx<"$WG1">) -> tuple<!hc.idx<"$WG0">, !hc.idx<"$WG1">>
// INFER: hc.getitem {{.*}} -> !hc.idx<"$WG1">
hc.func @geometry_tuple_getitem_infer(%g: !hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>) {
  %one = hc.const<1 : i64> : !hc.undef
  %gid:2 = hc.group_id %g
      : (!hc.group<work_shape = #hc.shape<["M", "N"]>, group_shape = #hc.shape<["16", "8"]>, subgroup_size = 32 : i32>)
        -> (!hc.undef, !hc.undef)
  %tuple = hc.tuple(%gid#0, %gid#1)
      : (!hc.undef, !hc.undef) -> tuple<!hc.undef, !hc.undef>
  %item = hc.getitem %tuple[%one]
      : (tuple<!hc.undef, !hc.undef>, !hc.undef) -> !hc.undef
  hc.return
}
