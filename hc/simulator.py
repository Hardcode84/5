# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Milestone 0 host-side simulator for `hc` kernels.

This module executes WorkGroup-only kernels directly in Python against masked
host runtime objects. Catch `SimulatorError` for any simulator failure, or
`LaunchError` specifically for pre-execution launch and binding failures.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
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
)
from .core import BufferSpec, CurrentGroup, KernelMetadata
from .symbols import Bindings, Env, Expr, Symbol, SymbolError, sym

type FrozenEnv = Env
type BufferValue = np.ndarray[Any, np.dtype[Any]]


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
    bound, env, literal_names, work_shape, group_shape = _prepare_launch(
        fn, args, kwargs, target
    )
    if any(dim == 0 for dim in work_shape):
        return
    _run_workgroups(fn, bound, env, literal_names, work_shape, group_shape)


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
    return bound, env, literal_names, work_shape, group_shape


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
        env: FrozenEnv,
        literal_names: frozenset[str],
    ) -> None:
        super().__init__(
            group_id=group_id,
            shape=shape,
            work_shape=work_shape,
            work_offset=work_offset,
        )
        self._env = env
        self._literal_names = literal_names

    def subgroups(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        raise ScopeError("Milestone 0 simulator does not support @group.subgroups")

    def workitems(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        raise ScopeError("Milestone 0 simulator does not support @group.workitems")

    def barrier(self) -> None:
        raise ScopeError("Milestone 0 simulator does not support group.barrier()")

    def load(
        self,
        source: Any,
        *,
        shape: Sequence[Any] | None = None,
        mask: SimTensor | SimVector | None = None,
        layout: Any = None,
    ) -> SimTensor:
        """Load a dense tensor tile.

        `shape=` requests an explicit logical tile. If the source slice is
        larger than that tile, the extra source region is ignored.
        """
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
        """Load a dense vector tile using the same truncation rules as `load`."""
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
        _store_value(target, value)

    def empty(
        self, *, shape: Sequence[Any], dtype: Any, layout: Any = None
    ) -> SimTensor:
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
    _require_dense_layout(layout)
    if (shape is None) == (mask is None):
        raise SimulatorError("load requires exactly one of shape= or mask=")
    if mask is not None:
        return _load_masked(kind, source, mask)
    resolved = _resolve_runtime_shape(shape, env, literal_names, static=static)
    return _copy_loaded_value(kind, source, resolved)


def _load_masked(
    kind: type[SimTensor] | type[SimVector], source: Any, mask: SimTensor | SimVector
) -> SimTensor | SimVector:
    _require_bool_mask(mask)
    copied = _copy_loaded_value(kind, source, mask.shape)
    active = np.logical_and(copied._mask, np.logical_and(mask._mask, mask._data))
    return kind(copied._data, active)


def _copy_loaded_value(
    kind: type[SimTensor] | type[SimVector], source: Any, shape: tuple[int, ...]
) -> SimTensor | SimVector:
    source_data, source_mask = _source_arrays(source)
    if source_data.ndim != len(shape):
        raise SimulatorError("load rank does not match the requested shape")
    result_data = np.zeros(shape, dtype=source_data.dtype)
    result_mask = np.zeros(shape, dtype=bool)
    overlap = tuple(
        slice(0, min(shape[idx], source_data.shape[idx])) for idx in range(len(shape))
    )
    result_data[overlap] = source_data[overlap]
    result_mask[overlap] = source_mask[overlap]
    return kind(result_data, result_mask)


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
    _require_dense_layout(layout)
    resolved = _resolve_runtime_shape(shape, env, literal_names, static=static)
    data = np.full(resolved, fill_value, dtype=np.dtype(dtype))
    mask = np.full(resolved, active, dtype=bool)
    return kind(data, mask)


def _require_dense_layout(layout: Any) -> None:
    if layout is not None:
        raise SimulatorError("non-trivial layouts are not supported in Milestone 0")


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
    "SimTensor",
    "SimVector",
    "SimulatorError",
    "SimulatorTarget",
    "launch",
    "poison",
    "run",
]
