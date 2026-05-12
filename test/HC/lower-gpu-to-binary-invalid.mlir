// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative coverage for `-hc-lower-gpu-to-binary`. Each chunk pins a
// hard-error path so we fail loudly at the bead boundary instead of
// silently producing wrong-target HSACO or punting confusing
// TargetRegistry diagnostics to the user.
//
// We deliberately do not exercise the "no lld anywhere" path here:
// most CI environments have *some* `ld.lld` on PATH, so a "fail when
// no lld" test would be falsey. The bogus-lld-path path (option set
// to a path that doesn't exist) is independent of $PATH state and
// lives in `lower-gpu-to-binary-missing-lld.mlir` so the
// `expected-error` markers can be paired with their own RUN line.

// RUN: hc-opt --hc-lower-gpu-to-binary='lld-path=%hc_lld' --split-input-file --verify-diagnostics %s

module attributes {gpu.container_module} {
  // expected-error @below {{must carry exactly one target attribute}}
  gpu.module @no_target {
    llvm.func @c() attributes {gpu.kernel} { llvm.return }
  }
}

// -----

// More than one target is also rejected for now. The gpu dialect
// allows it — the convention is "compile each, pick at runtime" —
// but until we have a story for which blob ends up in `gpu.binary`
// (multi-object? pick first? error?) we just bail.
module attributes {gpu.container_module} {
  // expected-error @below {{must carry exactly one target attribute}}
  gpu.module @two_targets [#rocdl.target<chip = "gfx1100">,
                           #rocdl.target<chip = "gfx1101">] {
    llvm.func @d() attributes {gpu.kernel} { llvm.return }
  }
}

// -----

// Non-rocdl targets (NVVM, SPIRV, future ones) get a clean diagnostic
// rather than a confusing TargetRegistry lookup failure further down.
module attributes {gpu.container_module} {
  // expected-error @below {{only #rocdl.target<...> is supported}}
  gpu.module @nvvm [#nvvm.target<chip = "sm_80">] {
    llvm.func @e() attributes {gpu.kernel} { llvm.return }
  }
}
