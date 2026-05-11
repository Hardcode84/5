# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ctypes-driven invoke surface for `CompiledKernel`.

The host wrapper that `hc-lower-kernels-to-gpu-launch` lays down takes
one `PyObject *` per non-`!hc.group` kernel argument and unpacks each via
`_mlir_ciface_hc_get_*` calls (provided by `libhc_rt_helpers.so`), then
dispatches the actual GPU launch through the `hc_rt_load_kernel` /
`hc_rt_launch_kernel` shim from `libhc_hip_runtime.so`. To make that
runnable from Python:

* Spin up our `hc_execution_engine` extension, register the runtime
  helper + HIP shim symbols by ctypes-resolving them out of the bundled
  `.so` files and handing them to `set_symbol_map` — explicit and
  process-local, no `LoadLibraryPermanently` side effects.
* `engine.load_mlir(text)` parses the post-pipeline LLVM-dialect text,
  translates to LLVM IR, and JITs it. The engine returns the raw
  unpacked address of `@<kernel_name>` from `lookup`, so we can call it
  through a vanilla `ctypes.CFUNCTYPE(None, py_object * N)` thunk
  without the upstream MLIR engine's packed-args wrapper indirection.

Engine + cfunc creation is cached on the `CompiledKernel` via a tiny
mutable side-channel; the dataclass itself stays frozen. The cache is
process-local — no global engine pool — so two compiled kernels keep
their JIT'd code separate and a refcount drop releases the engine
immediately.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._native_paths import hip_runtime_lib_path, runtime_helpers_lib_path

__all__ = ["InvokerCache", "make_invoker", "runtime_symbol_map"]

# Symbols the host wrapper calls. The lists are short enough to enumerate
# explicitly — wave does the same — and listing them here doubles as
# documentation of the runtime ABI surface.
_RUNTIME_HELPER_SYMBOLS: tuple[str, ...] = (
    "_mlir_ciface_hc_get_buffer",
    "_mlir_ciface_hc_get_ptr",
    "_mlir_ciface_hc_get_int64",
    "_mlir_ciface_hc_get_float64",
    "_mlir_ciface_hc_get_dim",
    "_mlir_ciface_hc_get_stride",
)

_HIP_RUNTIME_SYMBOLS: tuple[str, ...] = (
    "hc_rt_init",
    "hc_rt_load_kernel",
    "hc_rt_launch_kernel",
)


def runtime_symbol_map() -> dict[str, int]:
    """Resolve all runtime helper + HIP shim symbols to their host addresses.

    The returned `{name: address}` dict is what
    `ExecutionEngineOptions.set_symbol_map` expects: each address is
    plumbed into the JIT's symbol table so unresolved externals in the
    host wrapper (`@_mlir_ciface_hc_get_*`, `@hc_rt_*`) bind to the
    real implementations. Loading the libraries via `ctypes.CDLL` keeps
    them alive for the lifetime of the process — `RTLD_LOCAL` is fine
    because the JIT only needs the addresses, not visibility against
    `dlsym(NULL, ...)`.
    """
    helpers = ctypes.CDLL(str(runtime_helpers_lib_path()))
    hip = ctypes.CDLL(str(hip_runtime_lib_path()))
    # `hc_rt_load_kernel` (called from JIT'd host wrappers) needs the
    # `g_hipModuleLoadData` / `g_hipModuleLaunchKernel` function pointers
    # bound, otherwise the first launch dereferences a null function
    # pointer and the process segfaults with no useful diagnostic. The
    # shim's `hc_rt_init` is mutex-serialized and double-checked, so
    # calling it eagerly here is safe and idempotent — once per process
    # is enough, but once per invoker construction is also harmless and
    # keeps the bootstrap close to the symbol resolution that needs it.
    hip.hc_rt_init.argtypes = []
    hip.hc_rt_init.restype = None
    hip.hc_rt_init()
    symbols: dict[str, int] = {}
    for name in _RUNTIME_HELPER_SYMBOLS:
        symbols[name] = _symbol_address(helpers, name)
    for name in _HIP_RUNTIME_SYMBOLS:
        symbols[name] = _symbol_address(hip, name)
    return symbols


def _symbol_address(lib: ctypes.CDLL, name: str) -> int:
    raw = ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
    if raw is None:
        raise RuntimeError(
            f"hc.invoke: symbol '{name}' resolved to a null address inside "
            f"{lib._name!r}; the shared library is built but the symbol is "
            "missing — rebuild the runtime libs."
        )
    return int(raw)


def _missing_libs() -> list[str]:
    paths = [str(runtime_helpers_lib_path()), str(hip_runtime_lib_path())]
    return [path for path in paths if not Path(path).exists()]


def _kernel_arg_count(module: Any, kernel_name: str) -> int:
    """Read the host wrapper's user-visible PyObject* arity from the IR.

    The `llvm.func @<name>(...)` op in the post-pipeline module has a
    leading `!llvm.ptr` stream slot followed by one `!llvm.ptr` per
    user kernel argument. We return the user-visible count (total - 1)
    so callers don't need to know about the stream slot at the
    arg-validation level. Reading the IR (rather than re-deriving from
    `kernel_fn.__hc_kernel__` + Python signature) keeps Python and C++
    decoupled — the lowering pass owns the ABI, this just observes it.
    """
    body = module.body if hasattr(module, "body") else module
    target = "@" + kernel_name + "("
    for op in body.operations:
        if op.operation.name != "llvm.func":
            continue
        op_text = str(op.operation)
        # `llvm.func` headers always single-line in the printer (the body
        # is in a region after `{`). Split on `{` to read the arg list.
        header_end = op_text.find("{")
        header = op_text if header_end == -1 else op_text[:header_end]
        if target in header:
            args_open = header.index("(")
            args_close = header.rindex(")")
            args_text = header[args_open + 1 : args_close].strip()
            if not args_text:
                # Should be impossible — every host wrapper has at least
                # the stream slot — but treat it as "0 user args" rather
                # than crashing here; the call will surface the mismatch.
                return 0
            # Split on top-level commas (no nested parens at this level —
            # arg types are all `!llvm.ptr`, attribute lists never use
            # `,` outside a `{...}` group which is also flat here).
            total = args_text.count(",") + 1
            return max(total - 1, 0)
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
            "hc/_native/lib/) or run `python -m build_tools.hc_native_tools` "
            "from a checkout."
        )

    num_args = _kernel_arg_count(module, kernel_name)

    from .execution_engine import ExecutionEngine, ExecutionEngineOptions

    options = ExecutionEngineOptions()
    options.set_symbol_map(runtime_symbol_map())
    engine = ExecutionEngine(options)

    # `str(module)` is the post-pipeline LLVM-dialect IR; the engine
    # parses + translates + JITs in one C++ trip. Going through MLIR
    # (rather than asking the caller to translate first) keeps the
    # invoke surface a single function call and matches what the rest
    # of the pipeline emits.
    handle = engine.load_mlir(str(module))
    func_ptr = engine.lookup(handle, kernel_name)
    if not func_ptr:
        raise RuntimeError(
            f"hc.invoke: lookup of host wrapper '@{kernel_name}' returned "
            "a null address (the JIT loaded the module but the symbol is "
            "not visible — check that the lowering pipeline did not "
            "rename or DCE the wrapper)"
        )

    # `void (void* stream, PyObject* arg0, PyObject* arg1, ...)` — the
    # leading `c_void_p` is the HIP stream pointer (null = default
    # stream); the rest are per-arg PyObject* slots, each passed by
    # reference (a `py_object` _is_ a borrowed `PyObject *`).
    func_type = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, *([ctypes.py_object] * num_args)
    )
    cfunc = func_type(func_ptr)

    def invoke(*args: Any, stream: int | None = None) -> None:
        # Touching `engine` keeps it pinned in the closure so the JIT'd
        # code (and therefore `func_ptr`) stays valid for the lifetime
        # of this callable. Without the explicit reference the engine
        # would be reachable only via the cfunc, which ctypes does not
        # treat as a strong reference.
        _ = engine
        if len(args) != num_args:
            raise TypeError(
                f"hc.invoke: kernel '{kernel_name}' takes {num_args} "
                f"argument(s), got {len(args)}"
            )
        # `None` → null pointer → HIP default stream. An explicit `int`
        # is the raw stream-handle address (for PyTorch users this is
        # `torch.cuda.current_stream().cuda_stream`).
        cfunc(stream, *(ctypes.py_object(arg) for arg in args))

    invoke.__name__ = f"invoke_{kernel_name}"
    invoke.__qualname__ = invoke.__name__
    return invoke
