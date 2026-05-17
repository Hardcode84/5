# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""hc_front -> hc pipeline driver (transform-dialect schedule based).

`hc.compile` calls `run_front_to_hc` to lower a resolved `hc_front`
module to `hc`. Schedule is MLIR: a `transform.named_sequence
@__transform_main` invoking passes via
`transform.apply_registered_pass`, loaded with
`-transform-preload-library` and run through `-transform-interpreter`.
Pass order lives in IR, swappable without Python.

After the schedule, a fixed `_GPU_LOWERING_PIPELINE` (raw pass string,
not transform ops — `convert-gpu-to-rocdl` is anchored on `GPUModuleOp`
and needs `gpu.module(...)` nesting) takes `#rocdl.target`-stamped
`gpu.module` ops to `gpu.binary` HSACO blobs via
`hc-lower-gpu-to-binary`. Custom schedules get the chain appended too.

`ld.lld` path is resolved Python-side from `_native_paths.lld_path`
and threaded through `--lld-path=`. No env dependency at run time.

Pipeline failure is non-fatal: result carries `module=None` + captured
diagnostics. Inspect, don't `try`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ._native_paths import lld_path

_ENTRY_POINT = "__transform_main"
_DEFAULT_SCHEDULE_PACKAGE = "hc.schedules"
_DEFAULT_SCHEDULE_NAME = "front_to_hc.mlir"
# Substituted with `hc.compile(target=...)` (empty for `None`, which
# tells `hc-interpret-intrinsic-recipes` to apply every recipe).
_TARGET_PLACEHOLDER = "__HC_TARGET__"
# Chars that would escape the MLIR string literal or break the
# transform option parser.
_TARGET_FORBIDDEN_CHARS = frozenset('"\\\n\r')

# Chip name (e.g. "gfx1100") for `rocdl-attach-target`,
# `convert-amdgpu-to-rocdl`, etc. Schedule without the placeholder
# silently ignores it.
_CHIP_PLACEHOLDER = "__HC_CHIP__"
_CHIP_FORBIDDEN_CHARS = _TARGET_FORBIDDEN_CHARS

# Used when caller passes no `target=`. Only matters for kernels that
# reach `gpu.module`; trivial kernels never use it.
_DEFAULT_CHIP = "gfx1100"
# Only `amdgpu-gfx11` is aliased today; bare `gfx<chip>` strings pass
# through verbatim.

# `ld.lld` path threaded through `hc-lower-gpu-to-binary` so the pass
# doesn't consult `HC_LLD`/`$PATH` at run time.
_LLD_PLACEHOLDER = "__HC_LLD__"
_LLD_FORBIDDEN_CHARS = _TARGET_FORBIDDEN_CHARS

# `--dump-intermediates=` for `hc-lower-gpu-to-binary`; empty disables.
_DUMP_DIR_PLACEHOLDER = "__HC_DUMP_DIR__"
_DUMP_DIR_FORBIDDEN_CHARS = _LLD_FORBIDDEN_CHARS
_DUMP_DIR_ENV = "HC_DUMP_DIR"

# AMDGPU LLVM `target-features`. gfx10+ MUST set `+wavefrontsize32`
# for WMMA correctness — LLVM defaults to wave64, which silently
# produces garbage WMMA results. gfx9 and below: empty, only ran wave64.
_FEATURES_PLACEHOLDER = "__HC_FEATURES__"
_FEATURES_FORBIDDEN_CHARS = _TARGET_FORBIDDEN_CHARS

# Optional `hc-emit-bench-wrapper` slot. Empty for `bench=False` so the
# IR stays byte-identical; `bench=True` expands to pass + trailing comma.
_BENCH_PLACEHOLDER = "__HC_BENCH__"
_BENCH_PASS_FRAGMENT = "hc-emit-bench-wrapper,"

# Device-side lowering chain appended after the user's schedule.
#
# `convert-gpu-to-rocdl` is anchored on `gpu::GPUModuleOp`, so
# `gpu.module(...)` nesting is mandatory; vector/arith/index conversions
# live inside the nested manager so they see LLVM types. Outer
# `gpu-to-llvm` handles host-side launch boundary.
#
# Raw pipeline string (not transform-dialect ops) because
# `transform.apply_registered_pass` can't express nested pass managers.
_GPU_LOWERING_PIPELINE = (
    # `!hc.ptr` family must be lowered before `convert-scf-to-cf` — the
    # convert-*-to-llvm passes nested below don't know `hc.alloc`/`hc.ptr_*`.
    "hc-lower-to-llvm,"
    "convert-scf-to-cf,"
    f"convert-amdgpu-to-rocdl{{chipset={_CHIP_PLACEHOLDER}}},"
    "gpu.module("
    f"convert-gpu-to-rocdl{{chipset={_CHIP_PLACEHOLDER}}},"
    "convert-arith-to-llvm,"
    "convert-vector-to-llvm,"
    "convert-index-to-llvm,"
    "reconcile-unrealized-casts"
    "),"
    "gpu-to-llvm,"
    "convert-vector-to-llvm,"
    "convert-index-to-llvm,"
    "reconcile-unrealized-casts,"
    "canonicalize,cse,"
    "hc-lower-gpu-to-binary{"
    f"lld-path={_LLD_PLACEHOLDER} "
    f"dump-intermediates={_DUMP_DIR_PLACEHOLDER}"
    "},"
    # `hc-lower-launch-func-to-runtime` replaces `gpu.launch_func`
    # with `hc_rt_load_kernel`/`launch_kernel` calls and embeds each
    # HSACO blob as an LLVM global; needs `gpu.binary` from
    # `hc-lower-gpu-to-binary` above. `hc-emit-bench-wrapper` (opt-in
    # via `__HC_BENCH__`) mints `<wrapper>_bench` before `symbol-dce`
    # so JIT can resolve it. `symbol-dce` drops unused runtime decls
    # (`hc_get_int64`, `hc_get_float64`).
    "hc-lower-launch-func-to-runtime,"
    f"{_BENCH_PLACEHOLDER}"
    "symbol-dce"
)

ScheduleSource = Path | str | None

# Once-per-process guard for hc's pass registration. CAPI is idempotent;
# this just skips a per-compile Python->C call.
_passes_registered = False


@dataclass(frozen=True)
class PipelineResult:
    """Transform-schedule run outcome.

    `module`/`module_text` are `None` on failure; `diagnostics`
    captures every severity (notes/warnings/errors) including parse-
    time. Success may still produce non-empty diagnostics.
    """

    module: Any | None
    module_text: str | None
    diagnostics: tuple[str, ...]


def run_front_to_hc(
    front_module: Any,
    *,
    schedule: ScheduleSource = None,
    target: str | None = None,
    bench: bool = False,
) -> PipelineResult:
    """Run the hc_front -> hc transform schedule on a parsed front module.

    `front_module` must come from `prepared_context()` — context with
    `hc_front` + `hc` dialects and hc's passes registered. Module is
    mutated in place on success. Clone first to preserve the original.

    `target`: substituted into `__HC_TARGET__`; the default schedule
    wires it into `hc-interpret-intrinsic-recipes`'s `target=`. `None`
    -> empty -> apply every recipe. Schedules without the placeholder
    ignore `target`.

    `__HC_CHIP__`: filled by `_resolve_chip(target)`. Used by
    `rocdl-attach-target` (in the schedule) and
    `convert-{amdgpu,gpu}-to-rocdl` (in the appended chain).
    `_DEFAULT_CHIP` covers `target=None` and trivial kernels that
    never grow `gpu.module`.

    `__HC_LLD__`: `hc._native_paths.lld_path` — bundled binary or
    `$HC_LLD` for source-tree work. Pipeline is self-contained.

    `bench=True`: splice `-hc-emit-bench-wrapper`. `False` leaves the
    IR byte-identical.
    """

    from .mlir import ir

    context = front_module.context
    diagnostics: list[str] = []

    def capture(diagnostic: Any) -> bool:
        # `True` suppresses MLIR's stderr print. Capture every severity:
        # actionable context often lives in notes attached to an error.
        diagnostics.append(str(diagnostic))
        return True

    with (
        context,
        _schedule_file(schedule, target=target, context=context) as schedule_path,
        context.attach_diagnostic_handler(capture),
    ):
        pipeline = _pipeline_string(schedule_path, target=target, bench=bench)
        try:
            pm = _build_pass_manager(pipeline, context)
            pm.run(front_module.operation)
        except ir.MLIRError as exc:
            # Narrow: `MLIRError` is the only contracted raise from
            # `PassManager.parse`/`.run`. Anything else propagates —
            # swallowing would turn real bugs into "hc_ir came back None".
            _capture_exception(diagnostics, exc)
            return PipelineResult(None, None, tuple(diagnostics))
        return PipelineResult(
            front_module,
            str(front_module),
            tuple(diagnostics),
        )


def prepared_context() -> Any:
    """MLIR context: hc + hc_front registered, passes loaded. Caller owns lifetime."""

    _ensure_passes_registered()
    from .mlir import ir
    from .mlir.dialects import hc as _hc
    from .mlir.dialects import hc_front as _hc_front

    ctx = ir.Context()
    _hc_front.register_dialects(ctx)
    _hc.register_dialects(ctx)
    return ctx


def _ensure_passes_registered() -> None:
    global _passes_registered
    if _passes_registered:
        return
    from .mlir.dialects import hc as _hc

    _hc.register_passes()
    _passes_registered = True


def _pipeline_string(
    schedule_path: Path, *, target: str | None, bench: bool = False
) -> str:
    # Two stages: 1) transform-preload-library + transform-interpreter
    # runs the schedule, 2) `_GPU_LOWERING_PIPELINE` takes
    # `#rocdl.target`-stamped `gpu.module` to `gpu.binary`. Stage 2 is
    # raw passes (transform-dialect can't express nested pass managers).
    # `entry-point=` is explicit so secondary entry points stay easy.
    gpu_lowering = _substitute_chip(_GPU_LOWERING_PIPELINE, target)
    gpu_lowering = _substitute_lld(gpu_lowering)
    gpu_lowering = _substitute_dump_dir(gpu_lowering)
    gpu_lowering = _substitute_bench(gpu_lowering, bench=bench)
    return (
        "builtin.module("
        f"transform-preload-library{{transform-library-paths={schedule_path}}},"
        f"transform-interpreter{{entry-point={_ENTRY_POINT}}},"
        f"{gpu_lowering}"
        ")"
    )


def _build_pass_manager(pipeline: str, context: Any) -> Any:
    from ._dump import dump_passes_enabled
    from .mlir import passmanager

    pm = passmanager.PassManager.parse(pipeline, context=context)
    if dump_passes_enabled():
        # IR printer can't install on a multi-threaded PM (LLVM ERRORs).
        # Opt-in debug, throughput loss is fine.
        context.enable_multithreading(False)
        # `--mlir-print-ir-after-all` equivalent for the appended GPU
        # chain and transform-interpreter. Per-pass output -> stderr so
        # `--dump-hc-ir > /tmp/hc.mlir` stays clean. Passes inside
        # `transform.apply_registered_pass` need `splice_dump_passes`
        # in `_dump`.
        pm.enable_ir_printing(
            print_before_all=False,
            print_after_all=True,
            print_module_scope=True,
            print_after_change=True,
        )
    return pm


def _capture_exception(diagnostics: list[str], exc: Exception) -> None:
    # `MLIRError`'s repr is already meaningful; surface alongside handler
    # diagnostics.
    text = f"{type(exc).__name__}: {exc}"
    if text not in diagnostics:
        diagnostics.append(text)


@contextmanager
def _schedule_file(
    schedule: ScheduleSource,
    *,
    target: str | None = None,
    context: Any = None,
) -> Iterator[Path]:
    """Yield a filesystem path to the schedule (always materialized).

    `transform-preload-library` takes file paths. Always tempfile so
    the driver controls the path — sidesteps option-parser-delimiter
    chars in user-supplied paths and gives us a substitution slot for
    `__HC_TARGET__`.
    """

    text = _resolve_schedule_text(schedule)
    text = _substitute_target(text, target)
    text = _substitute_chip(text, target)
    text = _substitute_features(text, target)
    text = _maybe_splice_dump_passes(text, context=context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=True, encoding="utf-8"
    ) as f:
        f.write(text)
        f.flush()
        yield Path(f.name)


def _maybe_splice_dump_passes(text: str, *, context: Any) -> str:
    # Only parse/walk/restringify when per-pass dumps are enabled. See
    # `_dump.splice_dump_passes` for the splicer rules. Borrows caller's
    # context if given (transform dialect must be loaded); else fresh
    # `prepared_context` for standalone use (tests, `--dump-hc-ir`).
    from ._dump import dump_passes_enabled, splice_dump_passes
    from .mlir import ir

    if not dump_passes_enabled():
        return text
    ctx = context if context is not None else prepared_context()
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(text)
        splice_dump_passes(module)
        return str(module)


def _resolve_schedule_text(schedule: ScheduleSource) -> str:
    if schedule is None:
        return _default_schedule_text()
    if isinstance(schedule, Path):
        resolved = schedule.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"schedule file not found: {schedule} (resolved to {resolved})"
            )
        return resolved.read_text(encoding="utf-8")
    if isinstance(schedule, str):
        return schedule
    raise TypeError(
        f"schedule must be Path | str | None, got {type(schedule).__name__}"
    )


def _substitute_target(text: str, target: str | None) -> str:
    # `None` -> empty -> recipe interpreter "run all".
    value = "" if target is None else _validate_target(target)
    return text.replace(_TARGET_PLACEHOLDER, value)


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError(f"target must be str | None, got {type(target).__name__}")
    bad = sorted({c for c in target if c in _TARGET_FORBIDDEN_CHARS})
    if bad:
        # Reject Python-side so callers see a clean ValueError, not a
        # "MLIR refused to parse this schedule" 200 lines later.
        raise ValueError(f"target contains forbidden characters {bad}: {target!r}")
    return target


def _substitute_chip(text: str, target: str | None) -> str:
    chip = _resolve_chip(target)
    return text.replace(_CHIP_PLACEHOLDER, chip)


def _substitute_lld(text: str) -> str:
    # Eager resolve: pipeline is self-contained. Existence not validated
    # here — C++ pass surfaces a clear "ld.lld not found" naming the
    # missing path.
    return text.replace(_LLD_PLACEHOLDER, _resolve_lld())


def _substitute_bench(text: str, *, bench: bool) -> str:
    # `bench=False`: collapse to empty so `prev,placeholder,next`
    # flattens to `prev,next`. `True`: pass + trailing comma.
    return text.replace(_BENCH_PLACEHOLDER, _BENCH_PASS_FRAGMENT if bench else "")


def _substitute_dump_dir(text: str) -> str:
    # `HC_DUMP_DIR` -> `--dump-intermediates=`. Empty -> pass disables.
    return text.replace(_DUMP_DIR_PLACEHOLDER, _resolve_dump_dir())


def _resolve_dump_dir() -> str:
    raw = os.environ.get(_DUMP_DIR_ENV, "")
    if not raw:
        return ""
    bad = sorted({c for c in raw if c in _DUMP_DIR_FORBIDDEN_CHARS})
    if bad:
        raise ValueError(
            f"{_DUMP_DIR_ENV} contains forbidden characters {bad}: {raw!r}"
        )
    return raw


def _resolve_lld() -> str:
    raw = str(lld_path())
    bad = sorted({c for c in raw if c in _LLD_FORBIDDEN_CHARS})
    if bad:
        raise ValueError(
            f"resolved ld.lld path contains forbidden characters {bad}: {raw!r}"
        )
    return raw


def _resolve_chip(target: str | None) -> str:
    """`target=` -> AMDGPU chip name.

    `None` -> `_DEFAULT_CHIP`; `amdgpu-gfx11` -> `gfx1100`;
    `gfx<chip>` -> verbatim; else -> default (recipe interpretation
    surfaces its own diagnostic).
    """

    if target is None:
        return _DEFAULT_CHIP
    if not isinstance(target, str):
        raise TypeError(f"target must be str | None, got {type(target).__name__}")
    bad = sorted({c for c in target if c in _CHIP_FORBIDDEN_CHARS})
    if bad:
        raise ValueError(f"target contains forbidden characters {bad}: {target!r}")
    if target == "amdgpu-gfx11":
        return "gfx1100"
    if target.startswith("gfx"):
        return target
    return _DEFAULT_CHIP


def _substitute_features(text: str, target: str | None) -> str:
    return text.replace(_FEATURES_PLACEHOLDER, _resolve_features(target))


def _resolve_features(target: str | None) -> str:
    """LLVM AMDGPU `target-features` for `target`.

    gfx10+ must set `+wavefrontsize32` for WMMA correctness (LLVM
    default is wave64 -> silent garbage results). gfx9 and below run
    wave64 only; empty.
    """

    chip = _resolve_chip(target)
    family = _gfx_family(chip)
    if family is not None and family >= 10:
        return "+wavefrontsize32"
    return ""


def _gfx_family(chip: str) -> int | None:
    # Major version from `gfx<major><minor><stepping>`. Tail width
    # varies (gfx900, gfx1100, gfx12_50). Non-`gfx<digits>` -> `None`.
    if not chip.startswith("gfx"):
        return None
    rest = chip[3:]
    digits: list[str] = []
    for char in rest:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    if len(digits) <= 2:
        return int(digits[0])
    return int("".join(digits[:-2]))


def _default_schedule_text() -> str:
    return (
        resources.files(_DEFAULT_SCHEDULE_PACKAGE)
        .joinpath(_DEFAULT_SCHEDULE_NAME)
        .read_text(encoding="utf-8")
    )
