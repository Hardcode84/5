// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pins the diagnostic surfaced when `--lld-path=` points at a path
// that doesn't exist. The MLIR `linkObjectCode` wrapper turns an
// `execve` failure into the same unhelpful "lld invocation failed"
// diagnostic it uses for a real linker error, so the pass
// pre-validates the path. The most common cause in source-tree
// development is `hc/_native/bin/ld.lld` not being staged (only
// `pip install -e .` populates that; `python -m
// build_tools.hc_native_tools` does not), and the diagnostic note
// points at that remediation directly.
//
// We thread an explicit bogus path through `--lld-path=` so this
// test is independent of `$PATH` / `HC_LLD` state on the host.

// RUN: hc-opt --hc-lower-gpu-to-binary='lld-path=/this/path/does/not/exist/ld.lld' --verify-diagnostics %s

module attributes {gpu.container_module} {
  // expected-error @below {{hc-lower-gpu-to-binary: ld.lld not executable at /this/path/does/not/exist/ld.lld (via --lld-path=)}}
  gpu.module @kernel [#rocdl.target<chip = "gfx1100">] {
    llvm.func @entry() attributes {gpu.kernel} {
      llvm.return
    }
  }
}
