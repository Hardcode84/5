# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Internal masked-value runtime primitives for `hc.simulator`.

This module implements the simulator's tensor/vector value model: masked
payloads, poison-aware scalar reads, and a deliberately small NumPy-facing
operator surface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Never

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[Any]
type ReducerIndex = tuple[int | slice, ...]

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


class _MaskedValue:
    __array_priority__ = 1000

    def __init__(
        self,
        data: Array,
        mask: Array,
        *,
        layout: Any = None,
        read_only: bool = False,
    ) -> None:
        self._data = np.asarray(data)
        self._mask = np.asarray(mask, dtype=bool)
        if self._data.shape != self._mask.shape:
            raise SimulatorError("payload and mask shapes must match")
        self.layout = layout
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
        return self._new_like(np.asarray(data), np.asarray(mask, dtype=bool))

    def astype(self, dtype: Any) -> _MaskedValue:
        return self._new_like(self._data.astype(dtype), self._mask.copy())

    def reshape(self, *shape: int | Sequence[int]) -> _MaskedValue:
        resolved = _reshape_args(shape)
        data = self._data.reshape(resolved)
        mask = self._mask.reshape(resolved)
        return self._new_like(data, mask)

    def transpose(self, *axes: int) -> _MaskedValue:
        data = self._data.transpose(*axes) if axes else self._data.transpose()
        mask = self._mask.transpose(*axes) if axes else self._mask.transpose()
        return self._new_like(data, mask)

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
        return _wrap_like(self, data, mask)

    def _new_like(
        self,
        data: Array,
        mask: Array,
        *,
        read_only: bool | None = None,
    ) -> _MaskedValue:
        return type(self)(
            data,
            mask,
            layout=self.layout,
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
        return self._new_like(data, mask)

    def _unary(self, op: Callable[[Any], Array]) -> _MaskedValue:
        return self._new_like(op(self._data), self._mask.copy())

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
        return self._new_like(data, mask)


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
        return SimVector(data, mask, layout=self.layout, read_only=self._read_only)


class SimVector(_MaskedValue):
    """Immutable masked vector value for `hc.simulator`."""

    pass


def _reshape_args(shape: tuple[Any, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        return tuple(int(dim) for dim in shape[0])
    return tuple(int(dim) for dim in shape)


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
    if out is None:
        return kind(data, mask)
    out_value = _validate_ufunc_out(kind, out)
    out_value._data[...] = data
    out_value._mask[...] = mask
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


def _wrap_like(value: _MaskedValue, data: Any, mask: Any) -> Any:
    if np.isscalar(mask):
        return _scalar_or_poison(data, bool(mask))
    return value._new_like(np.asarray(data), np.asarray(mask, dtype=bool))


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
