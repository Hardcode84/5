# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Host-side simulator for `hc` kernels.

This module executes kernels directly in Python against masked host runtime
objects, including collective workgroup, subgroup, and workitem regions. Catch
`SimulatorError` for any simulator failure, or `LaunchError` specifically for
pre-execution launch and binding failures.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, cast, get_args, get_origin

import numpy as np

from ._sim_types import (
    LaunchError,
    Poison,
    PoisonError,
    ScopeError,
    SimTensor,
    SimulatorError,
    SimVector,
    poison,
    resolve_layout,
)
from .core import BufferSpec, CurrentGroup, KernelMetadata, SubGroup, WorkItem
from .symbols import Bindings, Env, Expr, Symbol, SymbolError, sym

type FrozenEnv = Env
type BufferValue = np.ndarray[Any, np.dtype[Any]]

_SCOPE_WORKGROUP = "workgroup"
_SCOPE_SUBGROUP = "subgroup"
_SCOPE_WORKITEM = "workitem"


@dataclass(frozen=True)
class SimulatorTarget:
    """Launch-policy knobs for simulator validation and default group selection."""

    max_group_size: int | None = None

    def choose_group_shape(
        self, work_shape: tuple[int, ...], subgroup_size: int | None
    ) -> tuple[int, ...]:
        base = tuple(max(1, dim) for dim in work_shape)
        if not base or subgroup_size is None:
            return base
        return (_align_up(base[0], subgroup_size), *base[1:])


DEFAULT_TARGET = SimulatorTarget()


def launch(
    fn: Callable[..., Any],
    *args: Any,
    target: SimulatorTarget | None = None,
    **kwargs: Any,
) -> None:
    bound, env, literal_names, work_shape, group_shape, subgroup_size = _prepare_launch(
        fn, args, kwargs, target
    )
    if any(dim == 0 for dim in work_shape):
        return
    _run_workgroups(
        fn,
        bound,
        env,
        literal_names,
        work_shape,
        group_shape,
        subgroup_size,
    )


def _prepare_launch(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: SimulatorTarget | None,
) -> tuple[
    inspect.BoundArguments,
    FrozenEnv,
    frozenset[str],
    tuple[int, ...],
    tuple[int, ...],
    int | None,
]:
    try:
        runtime_target = DEFAULT_TARGET if target is None else target
        metadata = _kernel_metadata(fn)
        bound = _bind_launch_arguments(fn, args, kwargs)
        bindings = _collect_bindings(fn, bound)
        env = bindings.freeze()
        literal_names = _literal_names(metadata.literals)
        work_shape = _resolve_shape(
            metadata.work_shape, env, literal_names, static=False
        )
        subgroup_size = _resolve_optional_dim(
            metadata.subgroup_size,
            env,
            literal_names,
            static=True,
            name="subgroup_size",
        )
        group_shape = _resolve_group_shape(
            metadata.group_shape,
            runtime_target,
            work_shape,
            subgroup_size,
            env,
            literal_names,
        )
        _validate_launch(work_shape, group_shape, subgroup_size, runtime_target)
    except SymbolError as exc:
        raise LaunchError(str(exc)) from exc
    return bound, env, literal_names, work_shape, group_shape, subgroup_size


run = launch


class SimCurrentGroup(CurrentGroup):
    """Runtime `CurrentGroup` object passed into simulated kernels."""

    def __init__(
        self,
        *,
        group_id: tuple[int, ...],
        shape: tuple[int, ...],
        work_shape: tuple[int, ...],
        work_offset: tuple[int, ...],
        subgroup_size: int | None,
        env: FrozenEnv,
        literal_names: frozenset[str],
    ) -> None:
        super().__init__(
            group_id=group_id,
            shape=shape,
            work_shape=work_shape,
            work_offset=work_offset,
        )
        self._subgroup_size = subgroup_size
        self._env = env
        self._literal_names = literal_names
        self._scope = _SCOPE_WORKGROUP
        self._workitem_scheduler: _WorkItemScheduler | None = None

    def _set_runtime_scope(
        self, scope: str, scheduler: _WorkItemScheduler | None = None
    ) -> None:
        self._scope = scope
        self._workitem_scheduler = scheduler

    def _require_workgroup_scope(self, name: str) -> None:
        if self._scope != _SCOPE_WORKGROUP:
            raise ScopeError(f"{name} is only supported in WorkGroup scope")

    def _region_runner(
        self, fn: Callable[..., Any], *, scope: str
    ) -> Callable[..., Any]:
        @wraps(fn)
        def invoke(*args: Any, **kwargs: Any) -> None:
            if args or kwargs:
                raise SimulatorError(
                    "collective region runners do not accept call-time arguments"
                )
            self._require_workgroup_scope(
                "@group.subgroups" if scope == _SCOPE_SUBGROUP else "@group.workitems"
            )
            if scope == _SCOPE_SUBGROUP:
                self._run_subgroups(fn)
                return
            self._run_workitems(fn)

        return invoke

    def subgroups(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return self._region_runner(fn, scope=_SCOPE_SUBGROUP)

    def workitems(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return self._region_runner(fn, scope=_SCOPE_WORKITEM)

    def barrier(self) -> None:
        if self._scope != _SCOPE_WORKITEM or self._workitem_scheduler is None:
            raise ScopeError(
                "group.barrier() is only supported inside @group.workitems"
            )
        self._workitem_scheduler.barrier()

    def load(
        self,
        source: Any,
        *,
        shape: Sequence[Any] | None = None,
        mask: SimTensor | SimVector | None = None,
        layout: Any = None,
    ) -> SimTensor:
        """Load a logical tensor tile.

        `shape=` requests an explicit logical tile. The simulator materializes
        that tile in dense NumPy order and carries any `layout=` as validated
        metadata. If the source slice is larger than that tile, the extra
        source region is ignored.
        """
        self._require_workgroup_scope("group.load()")
        return cast(
            SimTensor,
            _load_value(
                SimTensor,
                source,
                shape=shape,
                mask=mask,
                env=self._env,
                literal_names=self._literal_names,
                static=False,
                layout=layout,
            ),
        )

    def vload(
        self,
        source: Any,
        *,
        shape: Sequence[Any] | None = None,
        mask: SimTensor | SimVector | None = None,
        layout: Any = None,
    ) -> SimVector:
        """Load a logical vector tile using the same rules as `load`."""
        return cast(
            SimVector,
            _load_value(
                SimVector,
                source,
                shape=shape,
                mask=mask,
                env=self._env,
                literal_names=self._literal_names,
                static=True,
                layout=layout,
            ),
        )

    def store(self, target: Any, value: Any) -> None:
        if isinstance(value, SimTensor):
            self._require_workgroup_scope("group.store() with a tensor source")
        _store_value(target, value)

    def empty(
        self, *, shape: Sequence[Any], dtype: Any, layout: Any = None
    ) -> SimTensor:
        self._require_workgroup_scope("group.empty()")
        return cast(
            SimTensor,
            _allocate_value(
                SimTensor,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=False,
                layout=layout,
                active=False,
                fill_value=0,
            ),
        )

    def zeros(
        self, *, shape: Sequence[Any], dtype: Any, layout: Any = None
    ) -> SimTensor:
        self._require_workgroup_scope("group.zeros()")
        return cast(
            SimTensor,
            _allocate_value(
                SimTensor,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=False,
                layout=layout,
                active=True,
                fill_value=0,
            ),
        )

    def ones(
        self, *, shape: Sequence[Any], dtype: Any, layout: Any = None
    ) -> SimTensor:
        self._require_workgroup_scope("group.ones()")
        return cast(
            SimTensor,
            _allocate_value(
                SimTensor,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=False,
                layout=layout,
                active=True,
                fill_value=1,
            ),
        )

    def full(
        self, *, shape: Sequence[Any], dtype: Any, fill_value: Any, layout: Any = None
    ) -> SimTensor:
        self._require_workgroup_scope("group.full()")
        return cast(
            SimTensor,
            _allocate_value(
                SimTensor,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=False,
                layout=layout,
                active=True,
                fill_value=fill_value,
            ),
        )

    def vzeros(
        self, *, shape: Sequence[Any], dtype: Any, layout: Any = None
    ) -> SimVector:
        return cast(
            SimVector,
            _allocate_value(
                SimVector,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=True,
                layout=layout,
                active=True,
                fill_value=0,
            ),
        )

    def vones(
        self, *, shape: Sequence[Any], dtype: Any, layout: Any = None
    ) -> SimVector:
        return cast(
            SimVector,
            _allocate_value(
                SimVector,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=True,
                layout=layout,
                active=True,
                fill_value=1,
            ),
        )

    def vfull(
        self, *, shape: Sequence[Any], dtype: Any, fill_value: Any, layout: Any = None
    ) -> SimVector:
        return cast(
            SimVector,
            _allocate_value(
                SimVector,
                shape,
                dtype,
                env=self._env,
                literal_names=self._literal_names,
                static=True,
                layout=layout,
                active=True,
                fill_value=fill_value,
            ),
        )

    def _run_subgroups(self, fn: Callable[..., Any]) -> None:
        subgroup_size = self._subgroup_size
        if subgroup_size is None:
            raise ScopeError("@group.subgroups requires subgroup_size")
        subgroup_grid = (self.shape[0] // subgroup_size, *self.shape[1:])
        for subgroup_id, _ in enumerate(_iterate_indices(subgroup_grid)):
            subgroup = SimSubGroup(subgroup_id=subgroup_id, size=subgroup_size)
            self._set_runtime_scope(_SCOPE_SUBGROUP)
            try:
                fn(subgroup)
            finally:
                self._set_runtime_scope(_SCOPE_WORKGROUP)

    def _run_workitems(self, fn: Callable[..., Any]) -> None:
        scheduler = _WorkItemScheduler(self, fn)
        self._set_runtime_scope(_SCOPE_WORKGROUP)
        scheduler.run()


class SimSubGroup(SubGroup):
    """Runtime `SubGroup` object used by collective subgroup regions."""

    def __init__(self, *, subgroup_id: int, size: int) -> None:
        self._subgroup_id = subgroup_id
        self._size = size

    def subgroup_id(self) -> int:
        return self._subgroup_id

    def size(self) -> int:
        return self._size


class SimWorkItem(WorkItem):
    """Runtime `WorkItem` object used by collective workitem regions."""

    def __init__(
        self, *, local_id: tuple[int, ...], global_id: tuple[int, ...]
    ) -> None:
        self._local_id = local_id
        self._global_id = global_id

    def global_id(self) -> tuple[int, ...]:
        return self._global_id

    def local_id(self) -> tuple[int, ...]:
        return self._local_id


@dataclass(frozen=True)
class _BarrierWait:
    generation: int


@dataclass
class _WorkItemState:
    workitem: SimWorkItem
    runner: Any
    started: bool = False
    done: bool = False
    barrier_generation: int = 0
    waiting_generation: int | None = None


class _WorkItemScheduler:
    def __init__(self, group: SimCurrentGroup, fn: Callable[..., Any]) -> None:
        self._group = group
        self._fn = fn
        self._greenlet = _greenlet_runtime()
        self._scheduler_greenlet = self._greenlet.getcurrent()
        self._states = [
            self._make_state(local_id) for local_id in _iterate_indices(group.shape)
        ]
        self._current_state: _WorkItemState | None = None

    def run(self) -> None:
        while True:
            live = [state for state in self._states if not state.done]
            if not live:
                self._group._set_runtime_scope(_SCOPE_WORKGROUP)
                return
            for state in live:
                if state.waiting_generation is not None:
                    continue
                self._resume_state(state)
            self._release_barrier(live)

    def barrier(self) -> None:
        state = self._current_state
        if state is None:
            raise SimulatorError(
                "group.barrier() is only valid during workitem execution"
            )
        generation = state.barrier_generation + 1
        self._scheduler_greenlet.switch(_BarrierWait(generation))

    def _make_state(self, local_id: tuple[int, ...]) -> _WorkItemState:
        global_id = tuple(
            self._group.work_offset[idx] + local_id[idx] for idx in range(len(local_id))
        )
        workitem = SimWorkItem(local_id=local_id, global_id=global_id)
        runner = self._greenlet.greenlet(self._run_workitem)
        return _WorkItemState(workitem=workitem, runner=runner)

    def _run_workitem(self, workitem: SimWorkItem) -> None:
        self._fn(workitem)

    def _resume_state(self, state: _WorkItemState) -> None:
        self._current_state = state
        self._group._set_runtime_scope(_SCOPE_WORKITEM, scheduler=self)
        try:
            if state.started:
                outcome = state.runner.switch()
            else:
                state.started = True
                outcome = state.runner.switch(state.workitem)
        finally:
            self._group._set_runtime_scope(_SCOPE_WORKGROUP)
            self._current_state = None
        if state.runner.dead:
            state.done = True
            return
        if not isinstance(outcome, _BarrierWait):
            raise SimulatorError("unexpected workitem scheduler yield")
        state.waiting_generation = outcome.generation

    def _release_barrier(self, live: list[_WorkItemState]) -> None:
        waiting = [
            state
            for state in live
            if not state.done and state.waiting_generation is not None
        ]
        if not waiting:
            return
        if len(waiting) != len(live):
            raise SimulatorError("divergent barrier use in workitem region")
        generation = waiting[0].waiting_generation
        if generation is None:
            raise SimulatorError("divergent barrier use in workitem region")
        if any(state.waiting_generation != generation for state in waiting[1:]):
            raise SimulatorError("divergent barrier use in workitem region")
        for state in waiting:
            state.barrier_generation = generation
            state.waiting_generation = None


def _greenlet_runtime() -> Any:
    try:
        return importlib.import_module("greenlet")
    except ModuleNotFoundError as exc:
        raise SimulatorError(
            "@group.workitems requires the optional greenlet dependency"
        ) from exc


def _kernel_metadata(fn: Callable[..., Any]) -> KernelMetadata:
    metadata = getattr(fn, "__hc_kernel__", None)
    if metadata is None:
        raise LaunchError("launch() expects a function decorated with @kernel")
    return cast(KernelMetadata, metadata)


def _bind_launch_arguments(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> inspect.BoundArguments:
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    if not params:
        raise LaunchError("kernel must accept a CurrentGroup parameter")
    launch_sig = inspect.Signature(params[1:])
    return launch_sig.bind(*args, **kwargs)


def _collect_bindings(
    fn: Callable[..., Any], bound: inspect.BoundArguments
) -> Bindings:
    bindings = Bindings()
    annotations = _resolved_annotations(fn)
    for name, value in bound.arguments.items():
        annotation = annotations.get(name, inspect.Signature.empty)
        _bind_annotation(bindings, annotation, value, source=name)
    return bindings


def _resolved_annotations(fn: Callable[..., Any]) -> dict[str, Any]:
    raw = inspect.get_annotations(fn, eval_str=False)
    if not any(isinstance(annotation, str) for annotation in raw.values()):
        return raw
    closure = inspect.getclosurevars(fn)
    localns = _AnnotationLocals(fn.__globals__)
    localns.update(closure.nonlocals)
    localns.update(_annotation_symbol_locals(_kernel_metadata(fn)))
    resolved: dict[str, Any] = {}
    for name, annotation in raw.items():
        if not isinstance(annotation, str):
            resolved[name] = annotation
            continue
        try:
            resolved[name] = eval(annotation, fn.__globals__, localns)
        except (NameError, TypeError) as exc:
            raise LaunchError(
                f"failed to resolve annotations for {fn.__name__}"
            ) from exc
    return resolved


class _AnnotationLocals(dict[str, Any]):
    def __missing__(self, name: str) -> Any:
        if name[:1].isupper():
            return getattr(sym, name)
        raise KeyError(name)


def _annotation_symbol_locals(metadata: KernelMetadata) -> dict[str, Symbol]:
    result: dict[str, Symbol] = {}
    for value in (
        metadata.work_shape,
        metadata.group_shape,
        metadata.subgroup_size,
        metadata.literals,
    ):
        _collect_annotation_symbols(value, result)
    return result


def _collect_annotation_symbols(value: Any, result: dict[str, Symbol]) -> None:
    if isinstance(value, Symbol):
        result.setdefault(value.name, value)
        return
    if isinstance(value, Expr):
        for symbol in value.free_symbols():
            result.setdefault(symbol.name, symbol)
        return
    if isinstance(value, tuple | list | frozenset):
        for item in value:
            _collect_annotation_symbols(item, result)


def _bind_annotation(
    bindings: Bindings, annotation: Any, value: Any, *, source: str
) -> None:
    if annotation is inspect.Signature.empty:
        return
    if isinstance(annotation, BufferSpec):
        _bind_buffer(bindings, annotation, value, source=source)
        return
    if get_origin(annotation) is tuple:
        _bind_tuple(bindings, get_args(annotation), value, source=source)


def _bind_buffer(
    bindings: Bindings, spec: BufferSpec, value: Any, *, source: str
) -> None:
    if not isinstance(value, np.ndarray):
        raise LaunchError(f"{source} must be a numpy.ndarray buffer")
    if value.ndim != len(spec.dimensions):
        raise LaunchError(f"{source} rank does not match its Buffer annotation")
    for index, dim in enumerate(spec.dimensions):
        _bind_dimension(
            bindings, dim, int(value.shape[index]), f"{source}.shape[{index}]"
        )


def _bind_tuple(
    bindings: Bindings, dims: tuple[Any, ...], value: Any, *, source: str
) -> None:
    if not isinstance(value, tuple):
        raise LaunchError(f"{source} must be a tuple")
    if len(value) != len(dims):
        raise LaunchError(f"{source} length does not match its tuple annotation")
    for index, (dim, item) in enumerate(zip(dims, value, strict=True)):
        _bind_dimension(bindings, dim, int(item), f"{source}[{index}]")


def _bind_dimension(bindings: Bindings, dim: Any, value: int, source: str) -> None:
    if isinstance(dim, Symbol):
        bindings.bind(dim, value, source=source)
        return
    if isinstance(dim, Expr):
        bindings.require(dim, value, source=source)
        return
    if isinstance(dim, int) and dim != value:
        raise LaunchError(f"{source} expected {dim}, got {value}")


def _literal_names(literals: frozenset[Any]) -> frozenset[str]:
    return frozenset(item.name for item in literals if isinstance(item, Symbol))


def _resolve_shape(
    spec: Any, env: FrozenEnv, literal_names: frozenset[str], *, static: bool
) -> tuple[int, ...]:
    if spec is None:
        raise LaunchError("kernel metadata is missing required work_shape")
    values = spec if isinstance(spec, tuple) else (spec,)
    return tuple(
        _resolve_dim(value, env, literal_names, static=static, name="shape")
        for value in values
    )


def _resolve_optional_dim(
    spec: Any, env: FrozenEnv, literal_names: frozenset[str], *, static: bool, name: str
) -> int | None:
    if spec is None:
        return None
    return _resolve_dim(spec, env, literal_names, static=static, name=name)


def _resolve_dim(
    value: Any,
    env: FrozenEnv,
    literal_names: frozenset[str],
    *,
    static: bool,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise LaunchError(f"{name} must be an integer")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Expr):
        _require_static_expr(value, literal_names, name=name, static=static)
        return int(value.eval(env))
    raise LaunchError(f"{name} must be an integer expression")


def _require_static_expr(
    expr: Expr, literal_names: frozenset[str], *, name: str, static: bool
) -> None:
    if not static:
        return
    free = {symbol.name for symbol in expr.free_symbols()}
    if not free.issubset(literal_names):
        raise LaunchError(f"{name} must be constant or use only literal symbols")


def _resolve_group_shape(
    spec: Any,
    target: SimulatorTarget,
    work_shape: tuple[int, ...],
    subgroup_size: int | None,
    env: FrozenEnv,
    literal_names: frozenset[str],
) -> tuple[int, ...]:
    if spec is None:
        return target.choose_group_shape(work_shape, subgroup_size)
    return _resolve_shape(spec, env, literal_names, static=False)


def _validate_launch(
    work_shape: tuple[int, ...],
    group_shape: tuple[int, ...],
    subgroup_size: int | None,
    target: SimulatorTarget,
) -> None:
    _validate_rank(work_shape, group_shape)
    _validate_nonnegative(work_shape, "work_shape")
    _validate_positive(group_shape, "group_shape")
    if subgroup_size is not None:
        _validate_subgroup(group_shape, subgroup_size)
    if target.max_group_size is not None:
        _validate_group_size(group_shape, target.max_group_size)


def _validate_rank(work_shape: tuple[int, ...], group_shape: tuple[int, ...]) -> None:
    if len(work_shape) != len(group_shape):
        raise LaunchError("work_shape and group_shape must have the same rank")


def _validate_nonnegative(shape: tuple[int, ...], name: str) -> None:
    if any(dim < 0 for dim in shape):
        raise LaunchError(f"{name} values must be non-negative")


def _validate_positive(shape: tuple[int, ...], name: str) -> None:
    if any(dim <= 0 for dim in shape):
        raise LaunchError(f"{name} values must be positive")


def _validate_subgroup(group_shape: tuple[int, ...], subgroup_size: int) -> None:
    if subgroup_size <= 0:
        raise LaunchError("subgroup_size must be positive")
    if group_shape and group_shape[0] % subgroup_size != 0:
        raise LaunchError("group_shape[0] must be divisible by subgroup_size")


def _validate_group_size(group_shape: tuple[int, ...], limit: int) -> None:
    size = 1
    for dim in group_shape:
        size *= dim
    if size > limit:
        raise LaunchError("group size exceeds the simulator target limit")


def _run_workgroups(
    fn: Callable[..., Any],
    bound: inspect.BoundArguments,
    env: FrozenEnv,
    literal_names: frozenset[str],
    work_shape: tuple[int, ...],
    group_shape: tuple[int, ...],
    subgroup_size: int | None,
) -> None:
    group_counts = tuple(
        _ceil_div(work, local)
        for work, local in zip(work_shape, group_shape, strict=True)
    )
    for group_id in _iterate_indices(group_counts):
        work_offset = tuple(
            group_id[idx] * group_shape[idx] for idx in range(len(group_shape))
        )
        group = SimCurrentGroup(
            group_id=group_id,
            shape=group_shape,
            work_shape=work_shape,
            work_offset=work_offset,
            subgroup_size=subgroup_size,
            env=env,
            literal_names=literal_names,
        )
        fn(group, *bound.args, **bound.kwargs)


def _iterate_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if not shape:
        yield ()
        return
    yield from _iter_indices(shape)


def _iter_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    # Dimension 0 varies fastest, matching the simulator's documented launch order.
    for reversed_index in np.ndindex(tuple(reversed(shape))):
        yield tuple(reversed(reversed_index))


def _load_value(
    kind: type[SimTensor] | type[SimVector],
    source: Any,
    *,
    shape: Sequence[Any] | None,
    mask: SimTensor | SimVector | None,
    env: FrozenEnv,
    literal_names: frozenset[str],
    static: bool,
    layout: Any,
) -> SimTensor | SimVector:
    if (shape is None) == (mask is None):
        raise SimulatorError("load requires exactly one of shape= or mask=")
    if mask is not None:
        return _load_masked(kind, source, mask, layout=layout)
    resolved = _resolve_runtime_shape(shape, env, literal_names, static=static)
    resolved_layout = resolve_layout(layout, resolved)
    return _copy_loaded_value(kind, source, resolved, layout=resolved_layout)


def _load_masked(
    kind: type[SimTensor] | type[SimVector],
    source: Any,
    mask: SimTensor | SimVector,
    *,
    layout: Any,
) -> SimTensor | SimVector:
    _require_bool_mask(mask)
    resolved_layout = resolve_layout(layout, mask.shape)
    copied = _copy_loaded_value(kind, source, mask.shape, layout=resolved_layout)
    active = np.logical_and(copied._mask, np.logical_and(mask._mask, mask._data))
    return kind(copied._data, active, layout=copied.layout)


def _copy_loaded_value(
    kind: type[SimTensor] | type[SimVector],
    source: Any,
    shape: tuple[int, ...],
    *,
    layout: Any,
) -> SimTensor | SimVector:
    source_data, source_mask = _source_arrays(source)
    if source_data.ndim != len(shape):
        raise SimulatorError("load rank does not match the requested shape")
    # Layout metadata is tracked separately; simulator payloads stay dense in
    # logical shape for now.
    result_data = np.zeros(shape, dtype=source_data.dtype)
    result_mask = np.zeros(shape, dtype=bool)
    overlap = tuple(
        slice(0, min(shape[idx], source_data.shape[idx])) for idx in range(len(shape))
    )
    result_data[overlap] = source_data[overlap]
    result_mask[overlap] = source_mask[overlap]
    return kind(result_data, result_mask, layout=layout)


def _resolve_runtime_shape(
    shape: Sequence[Any] | None,
    env: FrozenEnv,
    literal_names: frozenset[str],
    *,
    static: bool,
) -> tuple[int, ...]:
    if shape is None:
        raise SimulatorError("shape= is required")
    return tuple(
        _resolve_dim(value, env, literal_names, static=static, name="shape")
        for value in shape
    )


def _source_arrays(
    source: Any,
) -> tuple[BufferValue, np.ndarray[Any, np.dtype[np.bool_]]]:
    if isinstance(source, Poison):
        raise SimulatorError("cannot load from a poison scalar")
    if isinstance(source, SimTensor | SimVector):
        return source._data, source._mask
    if isinstance(source, np.ndarray):
        return source, np.ones(source.shape, dtype=bool)
    array = np.asarray(source)
    if array.ndim == 0:
        return array, np.ones((), dtype=bool)
    raise SimulatorError("load source must be a numpy array or simulator value")


def _allocate_value(
    kind: type[SimTensor] | type[SimVector],
    shape: Sequence[Any],
    dtype: Any,
    *,
    env: FrozenEnv,
    literal_names: frozenset[str],
    static: bool,
    layout: Any,
    active: bool,
    fill_value: Any,
) -> SimTensor | SimVector:
    resolved = _resolve_runtime_shape(shape, env, literal_names, static=static)
    resolved_layout = resolve_layout(layout, resolved)
    # Allocations use dense logical storage and carry layout metadata alongside
    # the host payload.
    data = np.full(resolved, fill_value, dtype=np.dtype(dtype))
    mask = np.full(resolved, active, dtype=bool)
    return kind(data, mask, layout=resolved_layout)


def _require_bool_mask(mask: SimTensor | SimVector) -> None:
    if mask.dtype != np.dtype(bool):
        raise SimulatorError("mask= expects a bool tensor or vector")


def _store_value(target: Any, value: Any) -> None:
    if isinstance(target, SimTensor):
        if target._read_only:
            raise SimulatorError("store target is read-only")
        _store_masked(target._data, target._mask, value)
        return
    if isinstance(target, np.ndarray):
        _store_masked(target, None, value)
        return
    raise SimulatorError("store target must be a numpy buffer or tensor view")


def _store_masked(
    target_data: np.ndarray[Any, np.dtype[Any]], target_mask: Any, value: Any
) -> None:
    if isinstance(value, Poison):
        raise SimulatorError("cannot store a poison scalar")
    if isinstance(value, SimTensor | SimVector):
        _store_from_value(target_data, target_mask, value)
        return
    target_data[...] = value
    if target_mask is not None:
        target_mask[...] = True


def _store_from_value(
    target_data: np.ndarray[Any, np.dtype[Any]],
    target_mask: Any,
    value: SimTensor | SimVector,
) -> None:
    if target_data.ndim != value.ndim:
        raise SimulatorError("store source and target ranks must match")
    overlap = tuple(
        slice(0, min(target_data.shape[idx], value.shape[idx]))
        for idx in range(target_data.ndim)
    )
    source_data = value._data[overlap]
    source_mask = value._mask[overlap]
    target_region = target_data[overlap]
    target_region[source_mask] = source_data[source_mask]
    if target_mask is not None:
        target_mask_region = target_mask[overlap]
        target_mask_region[source_mask] = True


def _align_up(value: int, factor: int) -> int:
    return ((value + factor - 1) // factor) * factor


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


__all__ = [
    "DEFAULT_TARGET",
    "LaunchError",
    "Poison",
    "PoisonError",
    "ScopeError",
    "SimSubGroup",
    "SimTensor",
    "SimWorkItem",
    "SimVector",
    "SimulatorError",
    "SimulatorTarget",
    "launch",
    "poison",
    "run",
]
