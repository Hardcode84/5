# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Mapping, Sequence

_KIND_KEY = "kind"
_SHAPE_KEY = "shape"
_DTYPE_KEY = "dtype"
_EXPR_KEY = "expr"

_UNDEF_KIND = "undef"
_IDX_KIND = "idx"
_TENSOR_KIND = "tensor"
_VECTOR_KIND = "vector"

_SHAPED_KINDS = frozenset({_TENSOR_KIND, _VECTOR_KIND})
_KNOWN_KINDS = frozenset({_UNDEF_KIND, _IDX_KIND, *_SHAPED_KINDS})
_ALLOWED_KEYS_BY_KIND = {
    _UNDEF_KIND: frozenset({_KIND_KEY}),
    _IDX_KIND: frozenset({_KIND_KEY, _EXPR_KEY}),
    _TENSOR_KIND: frozenset({_KIND_KEY, _SHAPE_KEY, _DTYPE_KEY}),
    _VECTOR_KIND: frozenset({_KIND_KEY, _SHAPE_KEY, _DTYPE_KEY}),
}


TypeContractRecord = dict[str, object]


def serialize_intrinsic_type_spec(value: object) -> TypeContractRecord:
    from .core import IdxTypeSpec, TensorTypeSpec, UndefTypeSpec, VectorTypeSpec

    if isinstance(value, TensorTypeSpec):
        return validate_intrinsic_type_contract_record(
            {
                _KIND_KEY: _TENSOR_KIND,
                _SHAPE_KEY: tuple(str(dim) for dim in value.dimensions),
                _DTYPE_KEY: value.dtype,
            }
        )
    if isinstance(value, VectorTypeSpec):
        return validate_intrinsic_type_contract_record(
            {
                _KIND_KEY: _VECTOR_KIND,
                _SHAPE_KEY: tuple(str(dim) for dim in value.dimensions),
                _DTYPE_KEY: value.dtype,
            }
        )
    if isinstance(value, IdxTypeSpec):
        record: TypeContractRecord = {_KIND_KEY: _IDX_KIND}
        if value.expr is not None:
            record[_EXPR_KEY] = str(value.expr)
        return validate_intrinsic_type_contract_record(record)
    if isinstance(value, UndefTypeSpec) or value is None:
        return {_KIND_KEY: _UNDEF_KIND}
    raise RuntimeError(f"unsupported HC intrinsic type spec: {value!r}")


def validate_intrinsic_type_contract_record(
    value: Mapping[str, object],
) -> TypeContractRecord:
    kind = value.get(_KIND_KEY)
    if not isinstance(kind, str):
        raise RuntimeError(f"type contract record missing string kind: {value!r}")
    if kind not in _KNOWN_KINDS:
        raise RuntimeError(f"type contract record has unsupported kind {kind!r}")

    extras = set(value) - _ALLOWED_KEYS_BY_KIND[kind]
    if extras:
        formatted = ", ".join(sorted(repr(key) for key in extras))
        raise RuntimeError(
            f"type contract record kind {kind!r} has unsupported key(s): {formatted}"
        )

    if kind in _SHAPED_KINDS:
        shape = _string_sequence_field(value, _SHAPE_KEY, kind)
        dtype = value.get(_DTYPE_KEY)
        if not isinstance(dtype, str):
            raise RuntimeError(
                f"type contract record kind {kind!r} missing string dtype"
            )
        return {_KIND_KEY: kind, _SHAPE_KEY: shape, _DTYPE_KEY: dtype}

    if kind == _IDX_KIND:
        expr = value.get(_EXPR_KEY)
        if expr is None:
            return {_KIND_KEY: kind}
        if not isinstance(expr, str):
            raise RuntimeError("type contract record kind 'idx' has non-string expr")
        return {_KIND_KEY: kind, _EXPR_KEY: expr}

    return {_KIND_KEY: kind}


def _string_sequence_field(
    value: Mapping[str, object],
    key: str,
    kind: str,
) -> tuple[str, ...]:
    items = value.get(key)
    if isinstance(items, str | bytes) or not isinstance(items, Sequence):
        raise RuntimeError(
            f"type contract record kind {kind!r} missing string sequence {key}"
        )
    result = []
    for item in items:
        if not isinstance(item, str):
            raise RuntimeError(
                f"type contract record kind {kind!r} has non-string {key} entry"
            )
        result.append(item)
    return tuple(result)
