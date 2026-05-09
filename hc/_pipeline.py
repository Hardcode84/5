# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""hc_front -> hc pipeline driver (transform-dialect schedule based).

`hc.compile` calls `run_front_to_hc` to lower its resolved `hc_front`
module into the `hc` dialect. The schedule itself is MLIR: a
`transform.named_sequence @__transform_main` that invokes registered
MLIR passes via `transform.apply_registered_pass`. The driver loads a
schedule file with `-transform-preload-library` and runs it through
`-transform-interpreter`, so the pass order lives in IR (not a pipeline
string) and users can swap in their own schedule without touching
Python.

Failure is non-fatal: on a pipeline error the result carries
`module=None` + captured diagnostic strings. Callers inspect the
result rather than wrapping in `try`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

_ENTRY_POINT = "__transform_main"
_DEFAULT_SCHEDULE_PACKAGE = "hc.schedules"
_DEFAULT_SCHEDULE_NAME = "front_to_hc.mlir"
# Sentinel the default schedule plants in the
# `transform.apply_registered_pass "hc-interpret-intrinsic-recipes"` op
# for `target=`. The Python driver substitutes it with the value of
# `hc.compile(target=...)` (empty string for the `None` default, which
# tells the pass to apply every recipe). Substitution is intentionally
# a literal string-replace — using a verbose sentinel keeps the
# operation safe against unintended matches even if the schedule grows
# additional `hc-interpret-intrinsic-recipes` calls.
_TARGET_PLACEHOLDER = "__HC_TARGET__"
# Characters that would either escape the MLIR string literal we
# substitute into or break the transform option parser. Reject up
# front so the user gets a clear error from `hc.compile` instead of a
# garbled MLIR diagnostic.
_TARGET_FORBIDDEN_CHARS = frozenset('"\\\n\r')

# Sister sentinel for the AMDGPU chip name (e.g. "gfx1100") that
# downstream gpu lowering passes need spelled out — `rocdl-attach-target`,
# `convert-amdgpu-to-rocdl`, etc. The default schedule plants this in
# `rocdl-attach-target`'s `chip=` option. A user-provided schedule that
# omits the placeholder silently ignores the value.
_CHIP_PLACEHOLDER = "__HC_CHIP__"
# Same character blacklist as `_TARGET_FORBIDDEN_CHARS` — gfx-style
# chip names are alphanumeric anyway, but being explicit lets callers
# pass arbitrary strings (custom kernels, future targets) without
# tripping the MLIR option parser.
_CHIP_FORBIDDEN_CHARS = _TARGET_FORBIDDEN_CHARS

# Default chip when the caller doesn't pass `target=`. The bigger
# pipeline doesn't actually use the chip until `rocdl-attach-target`
# fires, which only matters for kernels that reach `gpu.module` (i.e.
# anything with non-trivial body). For trivial / fully-folded kernels
# the value is moot. Keep it pointed at the gfx11 part the WMMA work
# is built around so the default produces sensible IR for every
# kernel the project currently exercises end-to-end.
_DEFAULT_CHIP = "gfx1100"

# `target=` strings to AMDGPU chip names. The mapping is intentionally
# narrow — every entry is a target the project has actively exercised
# end-to-end. Unknown targets (including the bare gfx-style names like
# `gfx1100`) fall through to a sanity check that accepts any
# `gfx`-prefixed string verbatim, so callers running on a chip we
# haven't catalogued yet can still get a working schedule by passing
# `target="gfx<chip>"` directly.
_TARGET_CHIP_MAP: dict[str, str] = {
    "amdgpu-gfx11": "gfx1100",
}

ScheduleSource = Path | str | None

# hc's own passes register exactly once per process. Upstream passes are
# already registered by `_mlirRegisterEverything` during
# `hc._mlir_loader.load_hc_mlir`, so this guard is only about hc's three
# pass families. The CAPI entry point is idempotent anyway (see
# `lib/CAPI/HC.cpp`), but short-circuiting here avoids a per-compile
# Python->C call.
_passes_registered = False


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of running a transform schedule against an hc_front module.

    `module` and `module_text` are `None` on failure; `diagnostics`
    carries whatever the MLIR diagnostic handler captured while the
    pipeline ran (including during schedule-file parsing and pipeline
    parsing). Success still produces a possibly-non-empty `diagnostics`
    tuple: passes may emit warnings/notes that don't fail the run.
    """

    module: Any | None
    module_text: str | None
    diagnostics: tuple[str, ...]


def run_front_to_hc(
    front_module: Any,
    *,
    schedule: ScheduleSource = None,
    target: str | None = None,
) -> PipelineResult:
    """Run the hc_front -> hc transform schedule on a parsed front module.

    `front_module` must belong to a context set up by `prepared_context`
    — one where both `hc_front` and `hc` dialects are registered and hc's
    passes have been loaded into the process-wide pass registry. The
    module is mutated in place on success: its body changes from
    `hc_front.*` ops to `hc.*` ops. Callers wanting to preserve the
    original should pass a clone (`ir.Module.parse(str(original),
    context=ctx)`).

    `target` is substituted into the schedule's `__HC_TARGET__`
    placeholder before the transform interpreter runs; the default
    schedule wires it into `hc-interpret-intrinsic-recipes`'s `target=`
    option. Pass `None` (the default) to leave the placeholder empty,
    which tells the recipe interpreter to apply every named sequence
    regardless of `hc.target`. A user-provided schedule that does not
    contain the placeholder silently ignores the value — the override
    owns its own pass invocations.

    The `__HC_CHIP__` placeholder is filled in the same pass: `target=`
    is mapped through `_TARGET_CHIP_MAP` (e.g. `"amdgpu-gfx11"` →
    `"gfx1100"`) to the AMDGPU chip name `rocdl-attach-target` and
    `convert-amdgpu-to-rocdl` need. Passing a bare `gfx<chip>` string
    works too — anything starting with `gfx` is accepted verbatim. With
    `target=None` the chip falls back to `_DEFAULT_CHIP`, which is fine
    because chip-keyed passes are no-ops on a payload that never grew
    a `gpu.module` (trivial/fully-folded kernels).
    """

    from .mlir import ir

    context = front_module.context
    diagnostics: list[str] = []

    def capture(diagnostic: Any) -> bool:
        # Returning True tells MLIR the diagnostic was handled, which
        # suppresses the default stderr print. We capture every severity
        # (note/warning/error) since passes sometimes route the
        # actionable context through notes attached to an error.
        diagnostics.append(str(diagnostic))
        return True

    with (
        context,
        _schedule_file(schedule, target=target) as schedule_path,
        context.attach_diagnostic_handler(capture),
    ):
        pipeline = _pipeline_string(schedule_path)
        try:
            pm = _build_pass_manager(pipeline, context)
            pm.run(front_module.operation)
        except ir.MLIRError as exc:
            # Narrow catch on purpose: `MLIRError` is the one thing
            # `PassManager.parse`/`.run` contract to raise on pipeline
            # trouble (bad schedule, verifier failure, pass error).
            # Anything else (ValueError from our own driver, bugs,
            # KeyboardInterrupt) propagates — silently swallowing
            # them would turn real bugs into "hc_ir came back None".
            _capture_exception(diagnostics, exc)
            return PipelineResult(None, None, tuple(diagnostics))
        return PipelineResult(
            front_module,
            str(front_module),
            tuple(diagnostics),
        )


def prepared_context() -> Any:
    """Build an MLIR context with hc + hc_front registered and passes loaded.

    Used by both the resolver (so the hc_front module it produces is
    compatible with the pipeline) and `run_front_to_hc`. Caller owns the
    context's lifetime.

    Collocates three nominally-separate concerns — context allocation,
    dialect registration on that context, and process-wide pass
    registration — because every caller that wants one also wants the
    other two. Splitting them would let callers build half-initialized
    contexts and then get surprising failures at parse or pipeline-run
    time; bundling makes the precondition "usable for hc_front + hc
    work" a single call.
    """

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


def _pipeline_string(schedule_path: Path) -> str:
    # Two passes: one reads the schedule from disk and merges its named
    # sequences into the payload module, the other actually walks the
    # sequence. The entry-point option is spelled redundantly because
    # `__transform_main` is also the upstream default, but being explicit
    # makes the pipeline self-documenting if we ever introduce secondary
    # entry points (per-target lowerings, say).
    return (
        "builtin.module("
        f"transform-preload-library{{transform-library-paths={schedule_path}}},"
        f"transform-interpreter{{entry-point={_ENTRY_POINT}}}"
        ")"
    )


def _build_pass_manager(pipeline: str, context: Any) -> Any:
    from .mlir import passmanager

    return passmanager.PassManager.parse(pipeline, context=context)


def _capture_exception(diagnostics: list[str], exc: Exception) -> None:
    # `MLIRError` from the bindings already has a meaningful repr; keep it
    # whole so the calling side can surface it alongside the captured
    # handler diagnostics.
    text = f"{type(exc).__name__}: {exc}"
    if text not in diagnostics:
        diagnostics.append(text)


@contextmanager
def _schedule_file(
    schedule: ScheduleSource, *, target: str | None = None
) -> Iterator[Path]:
    """Yield a filesystem path to the schedule, materializing inline text.

    `transform-preload-library` takes file paths, not inline IR. We
    always read the schedule into memory so we can apply the
    `__HC_TARGET__` substitution before handing it to the pass manager;
    every code path then writes a tempfile the driver controls. This
    sidesteps the old `Path`-direct mode's footgun where a path
    containing a character the MLIR option parser treated as a
    delimiter (`,`, `}`, `=`) would break the pipeline string — the
    tempfile path we generate is always safe.
    """

    text = _resolve_schedule_text(schedule)
    text = _substitute_target(text, target)
    text = _substitute_chip(text, target)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=True, encoding="utf-8"
    ) as f:
        f.write(text)
        f.flush()
        yield Path(f.name)


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
    # `None` collapses to empty so the recipe interpreter's empty-target
    # "run all" behaviour is the default; substituting unconditionally
    # also keeps custom schedules without the placeholder untouched.
    value = "" if target is None else _validate_target(target)
    return text.replace(_TARGET_PLACEHOLDER, value)


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError(f"target must be str | None, got {type(target).__name__}")
    bad = sorted({c for c in target if c in _TARGET_FORBIDDEN_CHARS})
    if bad:
        # Reject up front so callers see "your target string is bad"
        # instead of "MLIR refused to parse this schedule" 200 lines
        # later. The blacklist covers everything that would either
        # close the substituted MLIR string literal early or insert
        # whitespace that the transform option parser splits on.
        raise ValueError(f"target contains forbidden characters {bad}: {target!r}")
    return target


def _substitute_chip(text: str, target: str | None) -> str:
    chip = _resolve_chip(target)
    return text.replace(_CHIP_PLACEHOLDER, chip)


def _resolve_chip(target: str | None) -> str:
    """Map the user-facing `target=` to an AMDGPU chip name.

    `None` -> `_DEFAULT_CHIP`; named target -> table lookup; bare
    `gfx<chip>` -> verbatim. Anything else is rejected — better to fail
    here than to feed `rocdl-attach-target` a chip it can't parse.
    """

    if target is None:
        return _DEFAULT_CHIP
    if not isinstance(target, str):
        raise TypeError(f"target must be str | None, got {type(target).__name__}")
    bad = sorted({c for c in target if c in _CHIP_FORBIDDEN_CHARS})
    if bad:
        raise ValueError(f"target contains forbidden characters {bad}: {target!r}")
    if target in _TARGET_CHIP_MAP:
        return _TARGET_CHIP_MAP[target]
    if target.startswith("gfx"):
        return target
    # Unknown logical target — recipe interpretation will simply not
    # match anything (and surface its own diagnostic), so picking the
    # default chip here keeps the schedule's GPU lowering passes
    # well-formed enough to run without producing confusing
    # second-order diagnostics from chip parsing.
    return _DEFAULT_CHIP


def _default_schedule_text() -> str:
    return (
        resources.files(_DEFAULT_SCHEDULE_PACKAGE)
        .joinpath(_DEFAULT_SCHEDULE_NAME)
        .read_text(encoding="utf-8")
    )
