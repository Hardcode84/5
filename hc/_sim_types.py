# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Internal masked-value runtime primitives for `hc.simulator`.

This module implements the simulator's tensor/vector value model: masked
payloads, poison-aware scalar reads, a deliberately small NumPy-facing
operator surface, and layout metadata over dense host storage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Never

import numpy as np
from numpy.typing import NDArray

from .core import IndexMap

type Array = NDArray[Any]
type ReducerIndex = tuple[int | slice, ...]

_KEEP_LAYOUT = object()
_KEEP_COLLECTIVE_SUFFIX = object()
# Keep host-side validation bounded for large shapes while still checking the
# layout's declared shape and storage size.
_LAYOUT_VALIDATION_LIMIT = 4096

_SUPPORTED_UFUNCS = frozenset(
    {
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
        np.negative,
        np.power,
        np.exp,
        np.sqrt,
        np.maximum,
        np.minimum,
        np.less,
        np.less_equal,
        np.greater,
        np.greater_equal,
        np.equal,
        np.not_equal,
    }
)


@dataclass(frozen=True)
class ResolvedLayout:
    spec: IndexMap
    shape: tuple[int, ...]
    params: dict[str, Any]
    storage_size: int


class SimulatorError(RuntimeError):
    pass


class LaunchError(SimulatorError):
    pass


class ScopeError(SimulatorError):
    pass


class PoisonError(SimulatorError):
    pass


class Poison:
    """Sentinel for observing an inactive scalar value."""

    def __repr__(self) -> str:
        return "Poison"

    def __bool__(self) -> bool:
        _raise_poison()

    def __int__(self) -> int:
        _raise_poison()

    def __float__(self) -> float:
        _raise_poison()

    def __index__(self) -> int:
        _raise_poison()

    def __add__(self, other: object) -> object:
        _raise_poison()

    def __sub__(self, other: object) -> object:
        _raise_poison()

    def __mul__(self, other: object) -> object:
        _raise_poison()

    def __truediv__(self, other: object) -> object:
        _raise_poison()

    def __lt__(self, other: object) -> bool:
        _raise_poison()

    def __le__(self, other: object) -> bool:
        _raise_poison()

    def __gt__(self, other: object) -> bool:
        _raise_poison()

    def __ge__(self, other: object) -> bool:
        _raise_poison()

    def __hash__(self) -> int:
        _raise_poison()

    def __format__(self, format_spec: str) -> str:
        _raise_poison()

    def __array__(self) -> Array:
        _raise_poison()


poison = Poison()


def _raise_poison() -> Never:
    raise PoisonError("attempted to observe an inactive scalar value")


def resolve_layout(layout: Any, shape: Sequence[int]) -> ResolvedLayout | None:
    """Resolve validated layout metadata for a concrete logical shape.

    The simulator keeps dense NumPy payloads and uses the resolved layout as
    logical-placement metadata for values that still preserve that mapping.
    """
    resolved_shape = tuple(int(dim) for dim in shape)
    if layout is None:
        return None
    if isinstance(layout, ResolvedLayout):
        if layout.shape != resolved_shape:
            raise SimulatorError("layout shape does not match the requested shape")
        return layout
    if not isinstance(layout, IndexMap):
        raise SimulatorError("layout must be an IndexMap or None")
    params = _layout_params(layout, resolved_shape)
    storage_size = _layout_storage_size(layout, resolved_shape, params)
    _validate_layout_offsets(layout, resolved_shape, params, storage_size)
    return ResolvedLayout(
        spec=layout,
        shape=resolved_shape,
        params=params,
        storage_size=storage_size,
    )


def _layout_params(layout: IndexMap, shape: tuple[int, ...]) -> dict[str, Any]:
    if layout.params is None:
        return {}
    raw = layout.params(*shape)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SimulatorError("layout params() must return a mapping or None")
    return dict(raw)


def _layout_storage_size(
    layout: IndexMap, shape: tuple[int, ...], params: dict[str, Any]
) -> int:
    raw = (
        layout.storage_size(*shape)
        if layout.params is None
        else layout.storage_size(*shape, params)
    )
    size = _layout_int(raw, what="layout storage size")
    if size < 0:
        raise SimulatorError("layout storage size must be non-negative")
    return size


def _validate_layout_offsets(
    layout: IndexMap,
    shape: tuple[int, ...],
    params: dict[str, Any],
    storage_size: int,
) -> None:
    logical_size = int(np.prod(shape))
    if storage_size < logical_size:
        raise SimulatorError("layout storage size is smaller than the logical shape")
    if logical_size == 0:
        return
    if logical_size > _LAYOUT_VALIDATION_LIMIT:
        return
    seen: set[int] = set()
    for index in np.ndindex(shape):
        raw = (
            layout.offset(*index, *shape)
            if layout.params is None
            else layout.offset(*index, *shape, params)
        )
        offset = _layout_int(raw, what="layout offset")
        if offset < 0 or offset >= storage_size:
            raise SimulatorError("layout offset is out of bounds")
        if offset in seen:
            raise SimulatorError(
                "layout offset must be injective over the logical shape"
            )
        seen.add(offset)


def _layout_int(value: Any, *, what: str) -> int:
    if isinstance(value, bool):
        raise SimulatorError(f"{what} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SimulatorError(f"{what} must be an integer") from exc


class _MaskedValue:
    __array_priority__ = 1000

    def __init__(
        self,
        data: Array,
        mask: Array,
        *,
        layout: Any = None,
        collective_suffix: Sequence[int] = (),
        read_only: bool = False,
    ) -> None:
        self._data = np.asarray(data)
        self._mask = np.asarray(mask, dtype=bool)
        if self._data.shape != self._mask.shape:
            raise SimulatorError("payload and mask shapes must match")
        self.layout = layout
        self._collective_suffix = _validated_collective_suffix(
            self.shape, collective_suffix
        )
        self._read_only = read_only

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._data.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return int(self._data.ndim)

    @property
    def collective_suffix(self) -> tuple[int, ...]:
        return self._collective_suffix

    @property
    def mask(self) -> _MaskedValue:
        return self._new_like(
            self._mask,
            np.ones(self.shape, dtype=bool),
            read_only=True,
        )

    @property
    def T(self) -> _MaskedValue:
        return self.transpose()

    def __array__(self) -> Array:
        raise TypeError("simulator values do not implicitly convert to numpy arrays")

    def __bool__(self) -> bool:
        raise TypeError("simulator values do not define truthiness")

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, index: Any) -> Any:
        _ensure_supported_index(index)
        data = self._data[index]
        mask = self._mask[index]
        if np.isscalar(mask):
            return _scalar_or_poison(data, bool(mask))
        collective_suffix = _indexed_collective_suffix(
            self.shape,
            self._collective_suffix,
            index,
            np.shape(data),
        )
        return self._new_like(
            np.asarray(data),
            np.asarray(mask, dtype=bool),
            layout=None,
            collective_suffix=collective_suffix,
        )

    def astype(self, dtype: Any) -> _MaskedValue:
        return self._new_like(self._data.astype(dtype), self._mask.copy())

    def reshape(self, *shape: int | Sequence[int]) -> _MaskedValue:
        resolved = _reshape_args(shape)
        data = self._data.reshape(resolved)
        mask = self._mask.reshape(resolved)
        layout = self.layout if resolved == self.shape else None
        collective_suffix = _reshaped_collective_suffix(
            self._collective_suffix,
            resolved,
        )
        return self._new_like(
            data, mask, layout=layout, collective_suffix=collective_suffix
        )

    def transpose(self, *axes: int) -> _MaskedValue:
        data = self._data.transpose(*axes) if axes else self._data.transpose()
        mask = self._mask.transpose(*axes) if axes else self._mask.transpose()
        collective_suffix = _transposed_collective_suffix(
            self.shape,
            self._collective_suffix,
            axes,
            np.shape(data),
        )
        return self._new_like(
            data,
            mask,
            layout=None,
            collective_suffix=collective_suffix,
        )

    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Any:
        return self._reduce("sum", axis=axis, keepdims=keepdims)

    def max(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Any:
        return self._reduce("max", axis=axis, keepdims=keepdims)

    def min(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Any:
        return self._reduce("min", axis=axis, keepdims=keepdims)

    def with_inactive(self, *, value: Any) -> _MaskedValue:
        fill = np.asarray(value, dtype=self.dtype)
        data = np.where(self._mask, self._data, fill)
        mask = np.ones(self.shape, dtype=bool)
        return self._new_like(data, mask)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if method != "__call__":
            raise SimulatorError(
                f"ufunc method {method!r} is not supported in the simulator"
            )
        if ufunc not in _SUPPORTED_UFUNCS:
            raise SimulatorError(
                f"ufunc {ufunc.__name__!r} is not supported in Milestone 0"
            )
        out = kwargs.pop("out", None)
        where = kwargs.pop("where", True)
        if where is not True:
            raise SimulatorError("ufunc where= is not supported in the simulator")
        result = _call_ufunc(ufunc, inputs, kwargs, out)
        return result

    def __neg__(self) -> _MaskedValue:
        return self._unary(np.negative)

    def __add__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.add)

    def __radd__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.add, reverse=True)

    def __sub__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.subtract)

    def __rsub__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.subtract, reverse=True)

    def __mul__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.multiply)

    def __rmul__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.multiply, reverse=True)

    def __truediv__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.divide)

    def __rtruediv__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.divide, reverse=True)

    def __pow__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.power)

    def __rpow__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.power, reverse=True)

    def __lt__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.less)

    def __le__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.less_equal)

    def __gt__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.greater)

    def __ge__(self, other: Any) -> _MaskedValue:
        return self._binary(other, np.greater_equal)

    def __eq__(self, other: object) -> _MaskedValue:  # type: ignore[override]
        return self._binary(other, np.equal)

    def __ne__(self, other: object) -> _MaskedValue:  # type: ignore[override]
        return self._binary(other, np.not_equal)

    def __matmul__(self, other: Any) -> Any:
        other_value = _require_same_kind(self, other)
        data = np.matmul(self._data, other_value._data)
        mask = _matmul_mask(self._mask, other_value._mask)
        return _wrap_like(
            self,
            data,
            mask,
            layout=self._matmul_layout(other_value, np.shape(data)),
            collective_suffix=self._matmul_collective_suffix(
                other_value, tuple(int(dim) for dim in np.shape(data))
            ),
        )

    def _new_like(
        self,
        data: Array,
        mask: Array,
        *,
        layout: Any = _KEEP_LAYOUT,
        collective_suffix: Any = _KEEP_COLLECTIVE_SUFFIX,
        read_only: bool | None = None,
    ) -> _MaskedValue:
        return type(self)(
            data,
            mask,
            layout=self.layout if layout is _KEEP_LAYOUT else layout,
            collective_suffix=(
                self._collective_suffix
                if collective_suffix is _KEEP_COLLECTIVE_SUFFIX
                else collective_suffix
            ),
            read_only=self._read_only if read_only is None else read_only,
        )

    def _binary(
        self, other: Any, op: Callable[[Any, Any], Array], *, reverse: bool = False
    ) -> _MaskedValue:
        other_data, other_mask = _coerce_operand(self, other)
        lhs_data, rhs_data = (
            (other_data, self._data) if reverse else (self._data, other_data)
        )
        data = op(lhs_data, rhs_data)
        if other_mask is None and np.shape(data) != self.shape:
            raise SimulatorError(
                "tensor/vector operations with non-simulator values "
                "only support scalars"
            )
        mask = (
            self._mask.copy()
            if other_mask is None
            else np.logical_and(self._mask, other_mask)
        )
        other_value = other if isinstance(other, _MaskedValue) else None
        return self._new_like(
            data,
            mask,
            layout=self._binary_layout(other_value, np.shape(data)),
            collective_suffix=self._binary_collective_suffix(
                other_value, tuple(int(dim) for dim in np.shape(data))
            ),
        )

    def _unary(self, op: Callable[[Any], Array]) -> _MaskedValue:
        data = op(self._data)
        return self._new_like(
            data,
            self._mask.copy(),
            layout=self._unary_layout(np.shape(data)),
            collective_suffix=self._unary_collective_suffix(
                tuple(int(dim) for dim in np.shape(data))
            ),
        )

    def _reduce(
        self, name: str, *, axis: int | tuple[int, ...] | None, keepdims: bool
    ) -> Any:
        if axis is None and self.ndim == 0:
            return _scalar_or_poison(self._data[()], bool(self._mask[()]))
        axes = _normalize_axes(axis, self.ndim)
        if not axes:
            return self
        out_shape = _reduced_shape(self.shape, axes, keepdims)
        data, mask = _reduce_arrays(self._data, self._mask, name, axes, out_shape)
        if data.shape == ():
            return _scalar_or_poison(data, bool(mask[()]))
        return self._new_like(
            data,
            mask,
            layout=self._reduction_layout(data.shape),
            collective_suffix=self._reduction_collective_suffix(data.shape),
        )

    def _binary_layout(
        self, other: _MaskedValue | None, result_shape: tuple[int, ...]
    ) -> Any:
        return _common_result_layout(
            (self,) if other is None else (self, other), result_shape
        )

    def _binary_collective_suffix(
        self, other: _MaskedValue | None, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        _ = (other, result_shape)
        return ()

    def _unary_layout(self, result_shape: tuple[int, ...]) -> Any:
        return self.layout if result_shape == self.shape else None

    def _unary_collective_suffix(
        self, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        _ = result_shape
        return ()

    def _reduction_layout(self, result_shape: tuple[int, ...]) -> Any:
        return self.layout if result_shape == self.shape else None

    def _reduction_collective_suffix(
        self, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        _ = result_shape
        return ()

    def _matmul_layout(self, other: _MaskedValue, result_shape: tuple[int, ...]) -> Any:
        return _common_result_layout((self, other), result_shape)

    def _matmul_collective_suffix(
        self, other: _MaskedValue, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        _ = (other, result_shape)
        return ()


class SimTensor(_MaskedValue):
    """Workgroup-local masked tensor value for `hc.simulator`."""

    def __setitem__(self, index: Any, value: Any) -> None:
        _ensure_supported_index(index)
        _require_writable(self)
        if np.isscalar(self._mask[index]):
            _assign_scalar_index(self._data, self._mask, index, value)
            return
        target_data = self._data[index]
        target_mask = self._mask[index]
        _assign_masked(target_data, target_mask, value)

    def vec(self, shape: Sequence[int] | None = None) -> SimVector:
        resolved = self.shape if shape is None else tuple(int(dim) for dim in shape)
        if int(np.prod(resolved)) != int(np.prod(self.shape)):
            raise SimulatorError("vec() shape must preserve the element count")
        data = self._data.reshape(resolved)
        mask = self._mask.reshape(resolved)
        layout = self.layout if resolved == self.shape else None
        return SimVector(
            data,
            mask,
            layout=layout,
            collective_suffix=(),
            read_only=self._read_only,
        )


class SimVector(_MaskedValue):
    """Immutable masked vector value for `hc.simulator`."""

    def as_layout(self, layout: Any = None) -> SimVector:
        if self.collective_suffix and layout is not None:
            raise SimulatorError(
                "collective return vectors do not support explicit as_layout()"
            )
        resolved = resolve_layout(layout, self.shape)
        return SimVector(
            self._data.copy(),
            self._mask.copy(),
            layout=resolved,
            collective_suffix=self.collective_suffix,
            read_only=self._read_only,
        )

    def _binary_layout(
        self, other: _MaskedValue | None, result_shape: tuple[int, ...]
    ) -> Any:
        values = (self,) if other is None else (self, other)
        return _vector_result_layout(values, result_shape)

    def _binary_collective_suffix(
        self, other: _MaskedValue | None, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        values = (self,) if other is None else (self, other)
        return _vector_collective_suffix(values, result_shape)

    def _unary_layout(self, result_shape: tuple[int, ...]) -> Any:
        return _vector_result_layout((self,), result_shape)

    def _unary_collective_suffix(
        self, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return _vector_collective_suffix((self,), result_shape)

    def _reduction_layout(self, result_shape: tuple[int, ...]) -> Any:
        return _vector_result_layout((self,), result_shape)

    def _reduction_collective_suffix(
        self, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return _vector_collective_suffix((self,), result_shape)

    def _matmul_layout(self, other: _MaskedValue, result_shape: tuple[int, ...]) -> Any:
        return _vector_result_layout((self, other), result_shape)

    def _matmul_collective_suffix(
        self, other: _MaskedValue, result_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return _vector_collective_suffix((self, other), result_shape)


def _reshape_args(shape: tuple[Any, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        return tuple(int(dim) for dim in shape[0])
    return tuple(int(dim) for dim in shape)


def _validated_collective_suffix(
    shape: tuple[int, ...], collective_suffix: Sequence[int]
) -> tuple[int, ...]:
    suffix = tuple(int(dim) for dim in collective_suffix)
    if len(suffix) > len(shape):
        raise SimulatorError("collective suffix rank exceeds value rank")
    if suffix and tuple(shape[-len(suffix) :]) != suffix:
        raise SimulatorError("collective suffix must match trailing value dimensions")
    return suffix


def _reshaped_collective_suffix(
    collective_suffix: tuple[int, ...], result_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if not collective_suffix:
        return ()
    if len(result_shape) < len(collective_suffix):
        raise SimulatorError(
            "reshape must keep collective return dimensions as a trailing suffix"
        )
    if tuple(result_shape[-len(collective_suffix) :]) != collective_suffix:
        raise SimulatorError(
            "reshape must keep collective return dimensions as a trailing suffix"
        )
    return collective_suffix


def _indexed_collective_suffix(
    shape: tuple[int, ...],
    collective_suffix: tuple[int, ...],
    index: Any,
    result_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if not collective_suffix:
        return ()
    axes = _collective_axes_after_index(shape, collective_suffix, index)
    if not axes:
        return ()
    expected = list(range(len(result_shape) - len(axes), len(result_shape)))
    if axes != expected:
        raise SimulatorError(
            "indexing must keep collective return dimensions as a trailing suffix"
        )
    return tuple(int(dim) for dim in result_shape[-len(axes) :])


def _collective_axes_after_index(
    shape: tuple[int, ...], collective_suffix: tuple[int, ...], index: Any
) -> list[int]:
    expanded = _expanded_basic_index(index, len(shape))
    collective_start = len(shape) - len(collective_suffix)
    source_axis = 0
    result_axis = 0
    result: list[int] = []
    for item in expanded:
        if item is None:
            result_axis += 1
            continue
        is_collective = source_axis >= collective_start
        if isinstance(item, (int, np.integer)):
            source_axis += 1
            continue
        if is_collective:
            result.append(result_axis)
        source_axis += 1
        result_axis += 1
    return result


def _expanded_basic_index(index: Any, ndim: int) -> tuple[Any, ...]:
    items = index if isinstance(index, tuple) else (index,)
    ellipses = _count_ellipses(items)
    if ellipses > 1:
        raise SimulatorError("only one ellipsis is supported in simulator indexing")
    missing = _missing_basic_index_axes(items, ndim)
    expanded = _expand_ellipsis_items(items, missing)
    if ellipses == 0:
        return expanded + (slice(None),) * missing
    return expanded


def _count_ellipses(items: tuple[Any, ...]) -> int:
    return sum(item is Ellipsis for item in items)


def _missing_basic_index_axes(items: tuple[Any, ...], ndim: int) -> int:
    consumed = sum(1 for item in items if item is not None and item is not Ellipsis)
    if consumed > ndim:
        raise SimulatorError("too many indices for simulator value")
    return ndim - consumed


def _expand_ellipsis_items(items: tuple[Any, ...], missing: int) -> tuple[Any, ...]:
    expanded: list[Any] = []
    for item in items:
        if item is Ellipsis:
            expanded.extend(slice(None) for _ in range(missing))
            continue
        expanded.append(item)
    return tuple(expanded)


def _transposed_collective_suffix(
    shape: tuple[int, ...],
    collective_suffix: tuple[int, ...],
    axes: tuple[int, ...],
    result_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if not collective_suffix:
        return ()
    resolved_axes = _normalized_transpose_axes(axes, len(shape))
    collective_start = len(shape) - len(collective_suffix)
    positions = [
        result_axis
        for result_axis, source_axis in enumerate(resolved_axes)
        if source_axis >= collective_start
    ]
    expected = list(range(len(result_shape) - len(positions), len(result_shape)))
    if positions != expected:
        raise SimulatorError(
            "transpose must keep collective return dimensions as a trailing suffix"
        )
    return tuple(int(dim) for dim in result_shape[-len(positions) :])


def _normalized_transpose_axes(axes: tuple[Any, ...], ndim: int) -> tuple[int, ...]:
    resolved = _resolved_transpose_axes(axes, ndim)
    if len(resolved) != ndim:
        raise SimulatorError("transpose axes must match value rank")
    wrapped = _wrapped_transpose_axes(resolved, ndim)
    _validate_transpose_axes(wrapped, ndim)
    return wrapped


def _resolved_transpose_axes(axes: tuple[Any, ...], ndim: int) -> tuple[int, ...]:
    if not axes:
        return tuple(reversed(range(ndim)))
    raw = axes[0] if len(axes) == 1 and isinstance(axes[0], Sequence) else axes
    return tuple(int(axis) for axis in raw)


def _wrapped_transpose_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(axis + ndim if axis < 0 else axis for axis in axes)


def _validate_transpose_axes(axes: tuple[int, ...], ndim: int) -> None:
    if len(set(axes)) != ndim:
        raise SimulatorError("transpose axes are out of range")
    if any(axis < 0 or axis >= ndim for axis in axes):
        raise SimulatorError("transpose axes are out of range")


def _scalar_or_poison(data: Any, active: bool) -> Any:
    if not active:
        return poison
    if isinstance(data, np.ndarray):
        return data.item()
    return data.item() if isinstance(data, np.generic) else data


def _coerce_operand(value: _MaskedValue, other: Any) -> tuple[Any, Array | None]:
    if isinstance(other, Poison):
        _raise_poison()
    if isinstance(other, _MaskedValue):
        _require_same_kind(value, other)
        return other._data, other._mask
    if isinstance(other, np.ndarray):
        if other.ndim == 0:
            return other.item(), None
        raise SimulatorError("tensor/vector operations require explicit loads first")
    if np.isscalar(other):
        return other, None
    maybe_scalar = np.asarray(other)
    if maybe_scalar.ndim == 0:
        return maybe_scalar.item(), None
    raise SimulatorError(
        "tensor/vector operations only support simulator values or scalars"
    )


def _require_same_kind(value: _MaskedValue, other: Any) -> _MaskedValue:
    if not isinstance(other, _MaskedValue) or type(value) is not type(other):
        raise SimulatorError("mixed tensor/vector operations are not allowed")
    return other


def _require_writable(value: _MaskedValue) -> None:
    if value._read_only:
        raise SimulatorError("simulator value is read-only")


def _ensure_supported_index(index: Any) -> None:
    indices = index if isinstance(index, tuple) else (index,)
    for item in indices:
        if item is Ellipsis or item is None:
            continue
        if isinstance(item, (bool, np.bool_)):
            raise SimulatorError("only basic indexing is supported in Milestone 0")
        if isinstance(item, (int, np.integer, slice)):
            _validate_slice(item)
            continue
        raise SimulatorError("only basic indexing is supported in Milestone 0")


def _validate_slice(item: int | np.integer[Any] | slice) -> None:
    if not isinstance(item, slice):
        return
    for bound in (item.start, item.stop, item.step):
        if isinstance(bound, (bool, np.bool_)):
            raise SimulatorError("slice bounds must be integers or None")
        if bound is not None and not isinstance(bound, (int, np.integer)):
            raise SimulatorError("slice bounds must be integers or None")


def _call_ufunc(
    ufunc: np.ufunc, inputs: tuple[Any, ...], kwargs: dict[str, Any], out: Any
) -> Any:
    kind = _ufunc_kind(inputs)
    data_inputs = [
        item._data if isinstance(item, _MaskedValue) else item for item in inputs
    ]
    data = ufunc(*data_inputs, **kwargs)
    mask = _combine_masks(inputs, np.shape(data))
    layout = _ufunc_layout(kind, inputs, np.shape(data))
    collective_suffix = _ufunc_collective_suffix(kind, inputs, np.shape(data))
    if out is None:
        return kind(data, mask, layout=layout, collective_suffix=collective_suffix)
    out_value = _validate_ufunc_out(kind, out)
    out_value._data[...] = data
    out_value._mask[...] = mask
    out_value.layout = layout
    out_value._collective_suffix = collective_suffix
    return out_value


def _ufunc_kind(inputs: tuple[Any, ...]) -> type[_MaskedValue]:
    kinds = [type(item) for item in inputs if isinstance(item, _MaskedValue)]
    if not kinds:
        raise SimulatorError("ufunc dispatch requires a simulator value")
    if any(kind is not kinds[0] for kind in kinds[1:]):
        raise SimulatorError("mixed tensor/vector operations are not allowed")
    return kinds[0]


def _validate_ufunc_out(kind: type[_MaskedValue], out: Any) -> _MaskedValue:
    if not isinstance(out, tuple) or len(out) != 1 or out[0] is None:
        raise SimulatorError("ufunc out= must be a one-element tuple")
    target = out[0]
    if not isinstance(target, SimTensor):
        raise SimulatorError("ufunc out= is only supported for tensors")
    if kind is not SimTensor:
        raise SimulatorError("vector out= is not supported")
    _require_writable(target)
    return target


def _ufunc_layout(
    kind: type[_MaskedValue], inputs: tuple[Any, ...], result_shape: tuple[int, ...]
) -> Any:
    values = tuple(item for item in inputs if isinstance(item, _MaskedValue))
    if not values:
        return None
    if kind is SimVector:
        return _vector_result_layout(values, result_shape)
    return _common_result_layout(values, result_shape)


def _ufunc_collective_suffix(
    kind: type[_MaskedValue], inputs: tuple[Any, ...], result_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if kind is not SimVector:
        return ()
    values = tuple(item for item in inputs if isinstance(item, _MaskedValue))
    return _vector_collective_suffix(values, result_shape)


def _common_result_layout(
    values: Sequence[_MaskedValue], result_shape: tuple[int, ...]
) -> Any:
    if not values or result_shape == ():
        return None
    first = values[0]
    if first.layout is None:
        return None
    if any(
        value.layout != first.layout or value.shape != result_shape for value in values
    ):
        return None
    return first.layout


def _vector_result_layout(
    values: Sequence[_MaskedValue], result_shape: tuple[int, ...]
) -> Any:
    vectors = _vector_inputs(values)
    if not vectors or result_shape == ():
        return None
    if _all_vector_layouts_none(vectors):
        return None
    _require_vector_shape_preservation(vectors, result_shape)
    return _require_uniform_vector_layout(vectors)


def _vector_collective_suffix(
    values: Sequence[_MaskedValue], result_shape: tuple[int, ...]
) -> tuple[int, ...]:
    vectors = _vector_inputs(values)
    if not vectors or result_shape == ():
        return ()
    if _all_vector_collective_suffixes_empty(vectors):
        return ()
    _require_vector_shape_preservation(vectors, result_shape)
    return _require_uniform_collective_suffix(vectors)


def _vector_inputs(values: Sequence[_MaskedValue]) -> list[SimVector]:
    return [value for value in values if isinstance(value, SimVector)]


def _all_vector_layouts_none(vectors: Sequence[SimVector]) -> bool:
    return all(vector.layout is None for vector in vectors)


def _all_vector_collective_suffixes_empty(vectors: Sequence[SimVector]) -> bool:
    return all(not vector.collective_suffix for vector in vectors)


def _require_vector_shape_preservation(
    vectors: Sequence[SimVector], result_shape: tuple[int, ...]
) -> None:
    if any(vector.shape != result_shape for vector in vectors):
        raise SimulatorError(
            "vector operations that change shape require explicit as_layout()"
        )


def _require_uniform_vector_layout(vectors: Sequence[SimVector]) -> Any:
    if any(vector.layout is None for vector in vectors):
        raise SimulatorError(
            "vector operations with mixed default and explicit layouts require "
            "explicit as_layout()"
        )
    first = vectors[0].layout
    if any(vector.layout != first for vector in vectors[1:]):
        raise SimulatorError(
            "vector operations with mismatched layouts require explicit as_layout()"
        )
    return first


def _require_uniform_collective_suffix(
    vectors: Sequence[SimVector],
) -> tuple[int, ...]:
    first = vectors[0].collective_suffix
    if any(vector.collective_suffix != first for vector in vectors[1:]):
        raise SimulatorError(
            "vector operations with collective return dimensions require "
            "matching collective suffixes"
        )
    return first


def _combine_masks(inputs: tuple[Any, ...], shape: tuple[int, ...]) -> Array:
    masks = [
        np.asarray(item._mask, dtype=bool)
        for item in inputs
        if isinstance(item, _MaskedValue)
    ]
    result = np.ones(shape, dtype=bool)
    for mask in masks:
        result = np.logical_and(result, mask)
    return result


def _matmul_mask(lhs: Array, rhs: Array) -> Array:
    counts = np.matmul(lhs.astype(np.int64), rhs.astype(np.int64))
    width = int(lhs.shape[-1])
    return np.asarray(counts == width, dtype=bool)


def _normalize_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw = (axis,) if isinstance(axis, int) else axis
    resolved = tuple(dim + ndim if dim < 0 else dim for dim in raw)
    if len(set(resolved)) != len(resolved):
        raise SimulatorError("reduction axes must be unique")
    if any(dim < 0 or dim >= ndim for dim in resolved):
        raise SimulatorError("reduction axis is out of range")
    return tuple(sorted(resolved))


def _reduced_shape(
    shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool
) -> tuple[int, ...]:
    if keepdims:
        return tuple(1 if idx in axes else dim for idx, dim in enumerate(shape))
    return tuple(dim for idx, dim in enumerate(shape) if idx not in axes)


def _reduce_arrays(
    data: Array,
    mask: Array,
    name: str,
    axes: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> tuple[Array, Array]:
    out_dtype = _reduce_dtype(data.dtype, name)
    out_data = np.zeros(out_shape, dtype=out_dtype)
    out_mask = np.zeros(out_shape, dtype=bool)
    for out_index, reducer_index in _iter_reduction_indices(out_shape, data.ndim, axes):
        payload = data[reducer_index]
        active = mask[reducer_index]
        if np.any(active):
            out_mask[out_index] = True
            out_data[out_index] = _reduce_active(payload, active, name)
    return out_data, out_mask


def _reduce_dtype(dtype: np.dtype[Any], name: str) -> np.dtype[Any]:
    sample = np.zeros((1,), dtype=dtype)
    reduced = getattr(np, name)(sample)
    return np.asarray(reduced).dtype


def _iter_reduction_indices(
    out_shape: tuple[int, ...], ndim: int, axes: tuple[int, ...]
) -> list[tuple[tuple[int, ...], ReducerIndex]]:
    if out_shape == ():
        return [((), tuple(slice(None) if idx in axes else 0 for idx in range(ndim)))]
    result: list[tuple[tuple[int, ...], ReducerIndex]] = []
    for out_index in np.ndindex(out_shape):
        result.append((out_index, _reducer_index(out_index, ndim, axes)))
    return result


def _reducer_index(
    out_index: tuple[int, ...], ndim: int, axes: tuple[int, ...]
) -> ReducerIndex:
    data_index: list[int | slice] = []
    out_pos = 0
    for dim in range(ndim):
        if dim in axes:
            data_index.append(slice(None))
            continue
        data_index.append(out_index[out_pos])
        out_pos += 1
    return tuple(data_index)


def _reduce_active(payload: Array, active: Array, name: str) -> Any:
    values = payload[active]
    return getattr(np, name)(values)


def _wrap_like(
    value: _MaskedValue,
    data: Any,
    mask: Any,
    *,
    layout: Any,
    collective_suffix: tuple[int, ...],
) -> Any:
    if np.isscalar(mask):
        return _scalar_or_poison(data, bool(mask))
    return value._new_like(
        np.asarray(data),
        np.asarray(mask, dtype=bool),
        layout=layout,
        collective_suffix=collective_suffix,
    )


def _assign_masked(target_data: Array, target_mask: Array, value: Any) -> None:
    if isinstance(value, Poison):
        _raise_poison()
    if isinstance(value, _MaskedValue):
        _assign_from_value(target_data, target_mask, value)
        return
    target_data[...] = value
    target_mask[...] = True


def _assign_scalar_index(
    target_data: Array, target_mask: Array, index: Any, value: Any
) -> None:
    if isinstance(value, Poison):
        _raise_poison()
    if isinstance(value, _MaskedValue):
        if value.shape != ():
            raise SimulatorError("assignment source and destination shapes must match")
        if bool(value._mask[()]):
            target_data[index] = value._data[()]
            target_mask[index] = True
        return
    target_data[index] = value
    target_mask[index] = True


def _assign_from_value(
    target_data: Array, target_mask: Array, value: _MaskedValue
) -> None:
    if target_data.shape != value.shape:
        raise SimulatorError("assignment source and destination shapes must match")
    active = value._mask
    target_data[active] = value._data[active]
    target_mask[active] = True
