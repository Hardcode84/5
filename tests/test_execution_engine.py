# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# JIT smoke for `hc.execution_engine`: parse, compile, lookup, call.
# Covers the per-process symbol generator and explicit symbol-map
# injection.

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

import pytest

from build_tools import hc_native_tools, llvm_toolchain
from hc._native_paths import package_native_root


def _engine_module_paths() -> list[Path]:
    paths: list[Path] = []
    pkg_dir = package_native_root() / "python_packages" / "hc_runtime"
    paths.extend(pkg_dir.glob("hc_execution_engine*.so"))
    install_root = hc_native_tools.hc_native_tools_layout(
        llvm_toolchain.llvm_toolchain_layout(
            llvm_toolchain.load_llvm_lock()
        ).install_root
    ).install_root
    paths.extend(
        (install_root / "python_packages" / "hc_runtime").glob(
            "hc_execution_engine*.so"
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    cached = sorted(
        (project_root / ".hc" / "native" / "install").glob(
            "*/python_packages/hc_runtime/hc_execution_engine*.so"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    paths.extend(cached)
    return paths


def _ensure_engine_on_path() -> None:
    """Prepend the engine `.so`'s parent so
    `hc_runtime.hc_execution_engine` imports before `pip install`
    has populated `hc/_native`."""
    for so in _engine_module_paths():
        if so.exists():
            entry = str(so.parent.parent)
            if entry not in sys.path:
                sys.path.insert(0, entry)
            return


@pytest.fixture(scope="module")
def engine_module() -> object:
    _ensure_engine_on_path()
    try:
        import hc_runtime.hc_execution_engine as ext
    except ImportError:
        pytest.skip("hc_execution_engine extension not built")
    return ext


@pytest.fixture(scope="module")
def engine(engine_module: object) -> object:
    options = engine_module.ExecutionEngineOptions()
    return engine_module.ExecutionEngine(options)


def test_engine_loads_llvm_ir_and_returns_42(engine: object) -> None:
    """Canonical ORC smoke: parse, JIT, lookup, call."""
    ir = """
define i64 @return42() {
entry:
  ret i64 42
}
"""
    handle = engine.load_llvm_ir(ir)
    addr = engine.lookup(handle, "return42")
    fn = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
    assert fn() == 42
    engine.release_module(handle)


def test_engine_resolves_libc_symbol_via_process_generator(
    engine: object,
) -> None:
    """`DynamicLibrarySearchGenerator` reaches any symbol loaded in
    the host process. `abs` from libc is the lowest-common probe."""
    ir = """
declare i32 @abs(i32)

define i32 @call_abs(i32 %x) {
entry:
  %r = call i32 @abs(i32 %x)
  ret i32 %r
}
"""
    handle = engine.load_llvm_ir(ir)
    addr = engine.lookup(handle, "call_abs")
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(addr)
    assert fn(-7) == 7
    assert fn(13) == 13
    engine.release_module(handle)


def test_engine_resolves_explicit_symbol_map(
    engine_module: object,
) -> None:
    """`set_symbol_map` surfaces to JIT'd IR. Route a libc symbol
    under a different name so process-level resolution can't shadow
    the injection path."""
    libc_path = ctypes.util.find_library("c")
    assert libc_path, "libc not found"
    libc = ctypes.CDLL(libc_path)
    abs_addr = ctypes.cast(libc.abs, ctypes.c_void_p).value
    assert abs_addr is not None

    options = engine_module.ExecutionEngineOptions()
    options.set_symbol_map({"hc_test_renamed_abs": abs_addr})
    isolated_engine = engine_module.ExecutionEngine(options)

    ir = """
declare i32 @hc_test_renamed_abs(i32)

define i32 @call_renamed(i32 %x) {
entry:
  %r = call i32 @hc_test_renamed_abs(i32 %x)
  ret i32 %r
}
"""
    handle = isolated_engine.load_llvm_ir(ir)
    addr = isolated_engine.lookup(handle, "call_renamed")
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(addr)
    assert fn(-99) == 99
    isolated_engine.release_module(handle)


def test_engine_lookup_missing_symbol_raises(engine: object) -> None:
    ir = """
define i64 @present() {
entry:
  ret i64 0
}
"""
    handle = engine.load_llvm_ir(ir)
    with pytest.raises(RuntimeError, match="lookup"):
        engine.lookup(handle, "definitely_not_defined")
    engine.release_module(handle)


def test_engine_release_invalidates_handle(engine: object) -> None:
    """Post-release lookups raise Python-level errors, not crash."""
    ir = """
define i64 @before_release() {
entry:
  ret i64 1
}
"""
    handle = engine.load_llvm_ir(ir)
    engine.release_module(handle)
    with pytest.raises(Exception):  # noqa: B017 - any error is acceptable
        engine.lookup(handle, "before_release")


def test_engine_module_attributes(engine_module: object) -> None:
    assert hasattr(engine_module, "ExecutionEngine")
    assert hasattr(engine_module, "ExecutionEngineOptions")
    assert hasattr(engine_module, "CodeGenOptLevel")
    levels = engine_module.CodeGenOptLevel
    for name in ("O0", "O1", "O2", "O3"):
        assert hasattr(levels, name), name


def test_engine_options_jit_opt_level_round_trips(engine_module: object) -> None:
    options = engine_module.ExecutionEngineOptions()
    assert options.jit_code_gen_opt_level is None
    options.jit_code_gen_opt_level = engine_module.CodeGenOptLevel.O2
    assert options.jit_code_gen_opt_level == engine_module.CodeGenOptLevel.O2


def test_engine_facade_module_re_exports() -> None:
    """`hc.execution_engine` re-exports match the extension."""
    if not any(p.exists() for p in _engine_module_paths()):
        pytest.skip("hc_execution_engine extension not built")
    # Clear stub left by earlier test.
    os.environ.pop("HC_EXECUTION_ENGINE_DISABLED", None)
    import hc.execution_engine as facade

    assert facade.ExecutionEngine is not None
    assert facade.ExecutionEngineOptions is not None
    options = facade.ExecutionEngineOptions()
    engine = facade.ExecutionEngine(options)
    handle = engine.load_llvm_ir("define i64 @facade_check() { entry: ret i64 7 }")
    addr = engine.lookup(handle, "facade_check")
    assert ctypes.CFUNCTYPE(ctypes.c_int64)(addr)() == 7
    engine.release_module(handle)
