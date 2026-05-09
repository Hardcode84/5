// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Failure paths for `-hc-lower-launch-func-to-runtime`. Sibling to
// lower-launch-func-to-runtime.mlir; pins the diagnostics so a future
// rewrite can't silently regress them into a confusing crash.
//
// We don't pin the "missing kernel container" or "empty objects" cases
// here — the upstream verifiers on `gpu.launch_func` (kernel-container
// symbol must resolve) and `gpu.binary` (objects array must have at
// least one element) reject those at parse time, so our pass never
// sees them. The single-object/multi-object choice we DO own.

// RUN: hc-opt --hc-lower-launch-func-to-runtime --split-input-file --verify-diagnostics %s

// Binary carries multiple objects (e.g. would happen if a multi-target
// rocdl-attach-target ever lands). v0 refuses to pick — pin the message
// so the follow-up that adds target selection has a verifiable starting
// point. The diagnostic attaches to the binary op (not the launch),
// because the malformed thing is the binary's object array.
module attributes {gpu.container_module} {
  llvm.func @host_multi() {
    %c1 = llvm.mlir.constant(1 : index) : i64
    gpu.launch_func @multi::@entry blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) : i64
    llvm.return
  }
  // expected-error @+1 {{gpu.binary must carry exactly one object (got 2)}}
  gpu.binary @multi [
    #gpu.object<#rocdl.target<chip = "gfx1100">, bin = "blob-a">,
    #gpu.object<#rocdl.target<chip = "gfx1101">, bin = "blob-b">
  ]
}
