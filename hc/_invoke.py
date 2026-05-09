# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ctypes-driven invoke surface for `CompiledKernel`.

The host wrapper that `hc-lower-kernels-to-gpu-launch` lays down takes
one `PyObject *` per non-`!hc.group` kernel argument and unpacks each via
`_mlir_ciface_hc_get_*` calls (provided by `libhc_rt_helpers.so`), then
dispatches the actual GPU launch through the `hc_rt_load_kernel` /
`hc_rt_launch_kernel` shim from `libhc_hip_runtime.so`. To make that
runnable from Python we need to:

* JIT-compile the post-pipeline MLIR module with the LLVM-dialect → LLVM
  IR translator wired up — `hc.mlir.execution_engine.ExecutionEngine`
  does the translate-then-LLJIT dance for us in a single C API call.
* Load both runtime shared libraries into the host process so the JIT's
  process-symbol-search resolver finds `_mlir_ciface_hc_get_*` and
  `hc_rt_*` by name (no need to enumerate). The MLIR engine's
  `shared_libs=` parameter does this via LLVM's `LoadLibraryPermanently`.
* Look up the packed-args wrapper that `ExecutionEngine` synthesizes
  around our host wrapper — we have to go through it because the
  Python binding only exposes the packed-lookup C API entry point.
* Build a `void**` array per call where each slot points to a
  `PyObject *` storage cell, then call the packed wrapper which loads
  each slot and forwards to the real `@<kernel_name>` implementation.

Engine + cfunc creation is cached on the `CompiledKernel` via a tiny
mutable side-channel; the dataclass itself stays frozen so unrelated
fields are still safe to use as dict keys when their values permit.
The cache is process-local — no global engine pool — so two compiled
kernels keep their JIT'd code separate and a refcount drop releases
the engine immediately.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ._native_paths import hip_runtime_lib_path, runtime_helpers_lib_path

__all__ = ["InvokerCache", "make_invoker", "runtime_shared_libs"]


def runtime_shared_libs() -> list[str]:
    """Paths to the runtime shared libraries the host wrapper links against.

    Order mirrors call-graph height — helpers first because the host
    wrapper calls them on every arg, HIP runtime second because it only
    fires when there is an actual `gpu.launch_func` to dispatch — but
    LLVM's `LoadLibraryPermanently` does not care about order, so this
    is purely for human-readability when the list shows up in error
    messages.
    """
    return [str(runtime_helpers_lib_path()), str(hip_runtime_lib_path())]


def _missing_libs() -> list[str]:
    return [path for path in runtime_shared_libs() if not _path_exists(path)]


def _path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()


def _kernel_arg_count(module: Any, kernel_name: str) -> int:
    """Read the host wrapper's PyObject* arity straight from the IR.

    The `llvm.func @<name>(...)` op in the post-pipeline module is the
    source of truth — every operand is a `!llvm.ptr` (the lowering pass
    only emits PyObject* host slots). Inspecting `kernel_fn.__hc_kernel__`
    metadata + Python signature would re-derive the same number, but
    it would have to keep its arg-skipping rules (`CurrentGroup` → drop)
    in lockstep with the C++ pass; reading the IR keeps the two
    decoupled.
    """
    body = module.body if hasattr(module, "body") else module
    target = "@" + kernel_name + "("
    for op in body.operations:
        # `op.OPERATION_NAME` is a class attr on every dialect op binding;
        # we cross-check via the textual op name instead of importing the
        # `llvm` dialect class to keep the import surface narrow.
        if op.operation.name != "llvm.func":
            continue
        op_text = str(op.operation)
        # `llvm.func` headers always single-line in the printer (the body
        # is in a region after `{`). Split on `(` to read the arg list.
        header_end = op_text.find("{")
        header = op_text if header_end == -1 else op_text[:header_end]
        if target in header:
            args_open = header.index("(")
            args_close = header.rindex(")")
            args_text = header[args_open + 1 : args_close].strip()
            if not args_text:
                return 0
            # Split on top-level commas (no nested parens at this level —
            # arg types are all `!llvm.ptr`, attribute lists never use
            # `,` outside a `{...}` group which is also flat here).
            return args_text.count(",") + 1
    raise RuntimeError(
        f"_invoke: host wrapper '@{kernel_name}' not found in compiled module"
    )


@dataclass
class InvokerCache:
    """Side-channel mutable state for the otherwise-frozen `CompiledKernel`.

    Lazy-initialized on the first `invoke()` call so a `CompiledKernel`
    that's only inspected (e.g. tests asserting on `hc_ir_text`) never
    loads the JIT or the runtime libraries. Holding the engine here
    keeps the JIT'd code alive as long as the cache (and therefore the
    `CompiledKernel`) is alive; releasing the kernel handle drops the
    engine and reclaims the JIT memory.
    """

    invoker: Callable[..., Any] | None = field(default=None)


def make_invoker(
    module: Any,
    kernel_name: str,
) -> Callable[..., Any]:
    """Build a callable that ctypes-dispatches into the host wrapper.

    Raises `RuntimeError` (not `ImportError`) when either runtime
    shared library is missing — the package install is broken in a way
    that's not the caller's fault, and downgrading to a plain import
    error would invite "wrap with try/ImportError" patterns that mask
    the real cause.
    """
    missing = _missing_libs()
    if missing:
        raise RuntimeError(
            "hc.invoke: runtime shared libraries are missing from the "
            "package install:\n  "
            + "\n  ".join(missing)
            + "\nReinstall hc (the build copies the .so files under "
            "hc/_native/lib/) or set HC_RT_HELPERS_PATH / "
            "HC_RT_HIP_RUNTIME_PATH for source-tree development."
        )

    num_args = _kernel_arg_count(module, kernel_name)

    from .mlir.execution_engine import ExecutionEngine

    engine = ExecutionEngine(
        module,
        opt_level=2,
        shared_libs=runtime_shared_libs(),
    )
    # `raw_lookup(kernel_name)` resolves to `_mlir_<kernel_name>` — the
    # packed-args wrapper that `ExecutionEngine` synthesizes around every
    # public function in the JIT'd module. We deliberately call into the
    # packed wrapper instead of the original `@<kernel_name>` because the
    # MLIR ExecutionEngine Python binding only exposes the packed
    # lookup; the unpacked `mlirExecutionEngineLookup` is in the C API
    # but not in the nanobind module. Going through the wrapper costs
    # one extra load per arg and a fixed per-call setup, neither of
    # which matters at the granularity of a GPU launch.
    packed_ptr = engine.raw_lookup(kernel_name)
    if not packed_ptr:
        raise RuntimeError(
            f"hc.invoke: lookup of packed wrapper '_mlir_{kernel_name}' "
            "returned a null pointer (the JIT loaded the module but the "
            "symbol is not visible — check that the lowering pipeline "
            "did not rename or DCE the host wrapper)"
        )

    # Packed wrapper signature is `void(void**)` regardless of the original
    # function's arity — the wrapper itself loads each arg from the
    # `void**` array and forwards to the real implementation.
    packed_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    packed_call = packed_type(packed_ptr)

    def invoke(*args: Any) -> None:
        # Touching `engine` keeps it pinned in the closure so the JIT'd
        # code (and therefore `packed_ptr`) stays valid for the lifetime
        # of this callable. Without the explicit reference the engine
        # would be reachable only via the cfunc, which ctypes does not
        # treat as a strong reference.
        _ = engine
        if len(args) != num_args:
            raise TypeError(
                f"hc.invoke: kernel '{kernel_name}' takes {num_args} "
                f"argument(s), got {len(args)}"
            )
        # Per-arg storage cells holding each `PyObject *`, plus a
        # `void**` array of pointers into those cells. The packed
        # wrapper does `load PyObject*, argList[i]` so each cell must
        # outlive the call (all stack-local here, fine) and `argList[i]`
        # must point to it. Using `py_object` (rather than raw
        # `c_void_p(id(obj))`) keeps a strong reference to each Python
        # object for the duration of the call so the underlying object
        # cannot be reclaimed mid-launch.
        storages = [ctypes.py_object(arg) for arg in args]
        packed = (ctypes.c_void_p * num_args)()
        for index, storage in enumerate(storages):
            packed[index] = ctypes.cast(ctypes.byref(storage), ctypes.c_void_p)
        packed_call(packed)

    invoke.__name__ = f"invoke_{kernel_name}"
    invoke.__qualname__ = invoke.__name__
    return invoke
