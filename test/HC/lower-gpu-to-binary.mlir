// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Coverage for `-hc-lower-gpu-to-binary`. The pass takes each
// `gpu.module` carrying a `#rocdl.target<...>` attribute, runs it
// through translate→optimize→ISA→MC assemble→ld.lld link, and emits
// a sibling `gpu.binary` with the resulting HSACO blob attached as a
// `#gpu.object`. The original `gpu.module` is erased on success.
//
// Tests pin the pass surface — gpu.binary appears, gpu.module is gone,
// the rocdl target attr survives the round trip — without trying to
// FileCheck the actual byte sequence of the HSACO (binary, opaque,
// changes with every llvm/lld bump). Failure paths live in
// lower-gpu-to-binary-invalid.mlir.

// RUN: hc-opt --hc-lower-gpu-to-binary='lld-path=%hc_lld' --split-input-file %s | FileCheck %s
// RUN: rm -rf %t.dump && mkdir -p %t.dump \
// RUN:   && hc-opt --hc-lower-gpu-to-binary='lld-path=%hc_lld dump-intermediates=%t.dump' \
// RUN:        --split-input-file %s -o /dev/null \
// RUN:   && ls %t.dump | sort | FileCheck --check-prefix=DUMP %s
//
// Per gpu.module the pass writes four artifacts: pre-opt LLVM IR,
// post-opt LLVM IR, ISA assembly, and the linked HSACO blob. The
// names embed an order prefix so `ls | sort` matches the pipeline
// order. With --split-input-file three modules go through the pass
// (@kernel from the first split, @first / @second from the second),
// so we expect 12 files — DUMP-NEXT pins them in lexical order.
// DUMP: first.0-pre-opt.ll
// DUMP-NEXT: first.1-post-opt.ll
// DUMP-NEXT: first.2-isa.s
// DUMP-NEXT: first.3-binary.hsaco
// DUMP-NEXT: kernel.0-pre-opt.ll
// DUMP-NEXT: kernel.1-post-opt.ll
// DUMP-NEXT: kernel.2-isa.s
// DUMP-NEXT: kernel.3-binary.hsaco
// DUMP-NEXT: second.0-pre-opt.ll
// DUMP-NEXT: second.1-post-opt.ll
// DUMP-NEXT: second.2-isa.s
// DUMP-NEXT: second.3-binary.hsaco

// Trivial empty kernel: validates the entire pipeline runs and emits
// a non-empty HSACO. The kernel has no body beyond `llvm.return` —
// enough for the AMDGPU codegen path to lay down a real object,
// nothing fancy enough to need device-libs bitcode.
//
// `bin = ` is the gpu.object printer's marker for CompilationTarget
// Binary (other targets would print `assembly = `, `bitcode = `, etc.),
// so matching it is a stable proxy for "the blob is attached and the
// format is what we asked for".
// CHECK-LABEL: gpu.binary @kernel
// CHECK-SAME:    <#rocdl.target<chip = "gfx1100">
// CHECK-SAME:    bin = "
// CHECK-NOT:   gpu.module
module attributes {gpu.container_module} {
  gpu.module @kernel [#rocdl.target<chip = "gfx1100">] {
    llvm.func @entry() attributes {gpu.kernel} {
      llvm.return
    }
  }
}

// -----

// Multiple gpu.modules in one payload: each gets its own gpu.binary,
// neither source survives. The block walk uses early_inc-style
// snapshotting so erasing the first module mid-iteration doesn't
// skip the second.
// CHECK-LABEL: gpu.binary @first
// CHECK-SAME:    <#rocdl.target<chip = "gfx1100">
// CHECK-LABEL: gpu.binary @second
// CHECK-SAME:    <#rocdl.target<chip = "gfx1100">
// CHECK-NOT:   gpu.module
module attributes {gpu.container_module} {
  gpu.module @first [#rocdl.target<chip = "gfx1100">] {
    llvm.func @a() attributes {gpu.kernel} { llvm.return }
  }
  gpu.module @second [#rocdl.target<chip = "gfx1100">] {
    llvm.func @b() attributes {gpu.kernel} { llvm.return }
  }
}
