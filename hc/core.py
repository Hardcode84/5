# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import dis
import inspect
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, Protocol, cast

from ._intrinsic_recipes import (
    IntrinsicTransformRecipe,
    build_intrinsic_transform_recipe,
)


@dataclass(frozen=True)
class BufferSpec:
    dimensions: tuple[Any, ...]
    dtype: str | None = None
    # Optional captured `IndexMap` overriding the boundary's default
    # fully-strided np/torch layout. The frontend lowering pass reads
    # this off the resolved parameter annotation and stamps a structured
    # `#hc.layout<...>` attribute on the matching parameter dict in
    # `hc_front.kernel.parameters`, which `-convert-hc-front-to-hc`
    # then uses verbatim instead of the default builder. `None` means
    # "no override" — the C++ side falls back to its
    # `$STRIDE_<i>_<argname>` layout, which is what every existing call
    # site continues to get.
    layout: IndexMap | None = None

    def __repr__(self) -> str:
        parts = [str(dim) for dim in self.dimensions]
        if self.dtype is not None:
            parts.append(self.dtype)
        if self.layout is not None:
            parts.append(repr(self.layout))
        body = ", ".join(parts)
        return f"Buffer[{body}]"


@dataclass(frozen=True)
class TensorTypeSpec:
    dimensions: tuple[Any, ...]
    dtype: str


@dataclass(frozen=True)
class VectorTypeSpec:
    dimensions: tuple[Any, ...]
    dtype: str


@dataclass(frozen=True)
class IdxTypeSpec:
    expr: Any = None


@dataclass(frozen=True)
class UndefTypeSpec:
    pass


def tensor_type(shape: Sequence[Any], dtype: Any) -> TensorTypeSpec:
    return TensorTypeSpec(tuple(shape), _dtype_annotation_name(dtype))


def vector_type(shape: Sequence[Any], dtype: Any) -> VectorTypeSpec:
    return VectorTypeSpec(tuple(shape), _dtype_annotation_name(dtype))


def idx_type(expr: Any = None) -> IdxTypeSpec:
    return IdxTypeSpec(expr)


def undef_type() -> UndefTypeSpec:
    return UndefTypeSpec()


class Buffer:
    def __class_getitem__(cls, item: Any) -> BufferSpec:
        if not isinstance(item, tuple):
            item = (item,)
        # Trailing `IndexMap` (`Buffer[d1, d2, dtype, A_LAYOUT]`) is read
        # as a layout override. `[]` syntax can't carry real kwargs, so
        # the positional-by-type rule is the only way to attach a layout
        # at the type-annotation surface — `IndexMap` is unambiguous
        # against dims (`Symbol`/`int`/`Expr`) and dtypes (numpy types).
        layout: IndexMap | None = None
        if item and isinstance(item[-1], IndexMap):
            layout = item[-1]
            item = item[:-1]
        dtype = _buffer_dtype_annotation_name(item[-1]) if len(item) >= 2 else None
        if dtype is not None:
            dims = item[:-1]
            item = tuple(dims)
        return BufferSpec(item, dtype=dtype, layout=layout)


def _buffer_dtype_annotation_name(value: Any) -> str | None:
    import numpy as np

    if isinstance(value, np.dtype):
        return str(value.name)
    if isinstance(value, type) and issubclass(value, np.generic):
        return _dtype_annotation_name(value)
    return None


def _dtype_annotation_name(value: Any) -> str:
    import numpy as np

    return str(np.dtype(value).name)


@dataclass(frozen=True)
class Scope:
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


WorkGroup = Scope("WorkGroup")


@dataclass(frozen=True)
class Result:
    type: Any
    mask: Any = None
    layout: Any = None


@dataclass(frozen=True)
class IndexMap:
    params: Callable[..., Any] | None
    storage_size: Callable[..., Any]
    offset: Callable[..., Any]
    # Names the layout's `offset` / `storage_size` may reference
    # without declaring them as shape / index / params syms. Resolved
    # at access time from the surrounding kernel scope (kernel-arg
    # aux, ancestor block argument, ambient launch geometry); see
    # `doc/layouts.md` "Free symbols in layout offsets". Each name
    # arrives in the lambdas as a keyword argument with a
    # `hc.symbols.Symbol` value — `offset=lambda i, j, M, N, *, row0:
    # ...` is the canonical signature shape.
    free_syms: tuple[str, ...] = ()


class SupportsAsLayout(Protocol):
    def as_layout(self, layout: Any = None) -> Any: ...


def index_map(
    *,
    params: Callable[..., Any] | None = None,
    storage_size: Callable[..., Any],
    offset: Callable[..., Any],
    free_syms: tuple[str, ...] = (),
) -> IndexMap:
    return IndexMap(
        params=params,
        storage_size=storage_size,
        offset=offset,
        free_syms=tuple(free_syms),
    )


def as_layout(value: SupportsAsLayout, layout: Any = None) -> Any:
    """Request an explicit layout on a layout-aware value."""
    method = getattr(value, "as_layout", None)
    if method is None:
        raise TypeError("as_layout() expects a layout-aware value")
    return method(layout)


@dataclass(frozen=True)
class KernelMetadata:
    work_shape: Any = None
    group_shape: Any = None
    subgroup_size: Any = None
    literals: frozenset[Any] = field(default_factory=frozenset)


@dataclass(frozen=True)
class FuncMetadata:
    scope: Any = None


@dataclass(frozen=True)
class IntrinsicMetadata:
    scope: Any = None
    effects: Any = None
    const_attrs: frozenset[str] = field(default_factory=frozenset)
    operand_types: tuple[Any, ...] | None = None
    result_types: tuple[Any, ...] = ()


class _KernelFunction(Protocol):
    __hc_kernel__: KernelMetadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _HelperFunction(Protocol):
    __name__: str
    __hc_func__: FuncMetadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _IntrinsicFunction(Protocol):
    __name__: str
    __hc_intrinsic__: IntrinsicMetadata
    __hc_lowerings__: dict[str, IntrinsicTransformRecipe]
    __hc_verify__: Callable[..., Any] | None
    __hc_infer__: Callable[..., Any] | None
    __hc_has_fallback__: bool
    lower: Any
    verify: Any
    infer: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


HelperFunction = _HelperFunction
IntrinsicFunction = _IntrinsicFunction

_SIM_CALLABLES_BY_CODE: dict[CodeType, Callable[..., Any]] = {}


def _sim_callable_from_code(code: CodeType) -> Callable[..., Any] | None:
    return _SIM_CALLABLES_BY_CODE.get(code)


def sim_callable_from_code(code: CodeType) -> Callable[..., Any] | None:
    return _sim_callable_from_code(code)


def _register_sim_callable(fn: Callable[..., Any]) -> None:
    _SIM_CALLABLES_BY_CODE[fn.__code__] = fn


def _has_intrinsic_fallback_body(fn: Callable[..., Any]) -> bool:
    body = _function_body(fn)
    if body is None:
        # Be conservative when source is unavailable: only obvious non-trivial
        # bytecode counts as a simulator fallback body.
        return _has_nontrivial_fallback_bytecode(fn)
    return not _is_empty_fallback_body(body)


def _function_body(fn: Callable[..., Any]) -> list[ast.stmt] | None:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    if not module.body:
        return None
    node = module.body[0]
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    body = list(node.body)
    if _starts_with_docstring(body):
        body = body[1:]
    return body


def _starts_with_docstring(body: list[ast.stmt]) -> bool:
    if not body or not isinstance(body[0], ast.Expr):
        return False
    value = body[0].value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _is_empty_fallback_body(body: list[ast.stmt]) -> bool:
    if not body:
        return True
    return len(body) == 1 and _is_empty_fallback_stmt(body[0])


def _is_empty_fallback_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    return isinstance(value, ast.Constant) and value.value is Ellipsis


def _has_nontrivial_fallback_bytecode(fn: Callable[..., Any]) -> bool:
    instructions = [
        ins
        for ins in dis.get_instructions(fn)
        if ins.opname not in {"RESUME", "CACHE", "EXTENDED_ARG", "NOP"}
    ]
    if not instructions:
        return False
    if len(instructions) == 1:
        ins = instructions[0]
        return not (ins.opname == "RETURN_CONST" and ins.argval is None)
    if len(instructions) == 2:
        return not (
            instructions[0].opname == "LOAD_CONST"
            and instructions[0].argval is None
            and instructions[1].opname == "RETURN_VALUE"
        )
    return True


def _attach_intrinsic_hooks(fn: Callable[..., Any]) -> Callable[..., Any]:
    intrinsic_fn = cast(_IntrinsicFunction, fn)
    intrinsic_fn.__hc_lowerings__ = {}
    intrinsic_fn.__hc_verify__ = None
    intrinsic_fn.__hc_infer__ = None
    intrinsic_fn.__hc_has_fallback__ = _has_intrinsic_fallback_body(fn)

    def lower(*, target: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(cb: Callable[..., Any]) -> Callable[..., Any]:
            metadata = intrinsic_fn.__hc_intrinsic__
            intrinsic_fn.__hc_lowerings__[target] = build_intrinsic_transform_recipe(
                cb,
                intrinsic_name=intrinsic_fn.__name__,
                target=target,
                operand_names=_intrinsic_operand_names(intrinsic_fn, metadata),
                attr_names=metadata.const_attrs,
                # Recipes run after `hc-decompose-shaped-values`, which splits
                # every shaped result into a `.data` + `.mask` pair (matching
                # the operand-side split). Mirror that here so the recipe sees
                # the same number of result handles the actual call site
                # exposes once it reaches the interpreter pass.
                result_count=_intrinsic_result_count(metadata),
            )
            return cb

        return register

    def verify(cb: Callable[..., Any]) -> Callable[..., Any]:
        intrinsic_fn.__hc_verify__ = cb
        return cb

    def infer(cb: Callable[..., Any]) -> Callable[..., Any]:
        intrinsic_fn.__hc_infer__ = cb
        return cb

    intrinsic_fn.lower = lower
    intrinsic_fn.verify = verify
    intrinsic_fn.infer = infer
    return intrinsic_fn


def _intrinsic_operand_names(
    fn: Callable[..., Any],
    metadata: IntrinsicMetadata,
) -> tuple[str, ...]:
    operand_types = metadata.operand_types
    result: list[str] = []
    type_index = 0
    for param in inspect.signature(fn).parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if (
            param.kind is inspect.Parameter.KEYWORD_ONLY
            and param.name in metadata.const_attrs
        ):
            continue
        # Mirror the parameter-name split that `hc-decompose-shaped-values`
        # performs on shaped operand types: a single `a_frag` declaration
        # becomes `a_frag.data` + `a_frag.mask` at the call site once the
        # decomposition pass runs. Recipes addressing operands by name need
        # the same dotted form to land on the right index post-decompose;
        # operands that aren't shaped (idx, undef, scalar) keep their name.
        kind = (
            None
            if operand_types is None or type_index >= len(operand_types)
            else operand_types[type_index]
        )
        if isinstance(kind, TensorTypeSpec | VectorTypeSpec):
            result.append(f"{param.name}.data")
            result.append(f"{param.name}.mask")
        else:
            result.append(param.name)
        type_index += 1
    return tuple(result)


def _intrinsic_result_count(metadata: IntrinsicMetadata) -> int:
    # Same `.data`/`.mask` split that the decomposition pass applies to
    # shaped result types, counted at recipe-build time so the recipe records
    # one handle per post-decomposition result.
    count = 0
    for kind in metadata.result_types:
        if isinstance(kind, TensorTypeSpec | VectorTypeSpec):
            count += 2
        else:
            count += 1
    return count


class _KernelNamespace:
    def __call__(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        work_shape: Any = None,
        group_shape: Any = None,
        subgroup_size: Any = None,
        literals: set[Any] | frozenset[Any] | None = None,
    ) -> Callable[..., Any]:
        metadata = KernelMetadata(
            work_shape=work_shape,
            group_shape=group_shape,
            subgroup_size=subgroup_size,
            literals=frozenset() if literals is None else frozenset(literals),
        )

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            kernel_target = cast(_KernelFunction, target)
            kernel_target.__hc_kernel__ = metadata
            return kernel_target

        if fn is None:
            return decorate
        return decorate(fn)

    def func(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        scope: Any = None,
    ) -> Callable[..., Any]:
        metadata = FuncMetadata(scope=scope)

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            helper_target = cast(_HelperFunction, target)
            helper_target.__hc_func__ = metadata
            _register_sim_callable(helper_target)
            return helper_target

        if fn is None:
            return decorate
        return decorate(fn)

    def intrinsic(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        scope: Any = None,
        effects: Any = None,
        const_attrs: set[str] | frozenset[str] | None = None,
        operand_types: Sequence[Any] | None = None,
        result_types: Sequence[Any] | Any = (),
    ) -> Callable[..., Any]:
        if result_types is None:
            normalized_results: tuple[Any, ...] = ()
        elif isinstance(result_types, Sequence) and not isinstance(
            result_types, str | bytes
        ):
            normalized_results = tuple(result_types)
        else:
            normalized_results = (result_types,)
        metadata = IntrinsicMetadata(
            scope=scope,
            effects=effects,
            const_attrs=frozenset() if const_attrs is None else frozenset(const_attrs),
            operand_types=None if operand_types is None else tuple(operand_types),
            result_types=normalized_results,
        )

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            intrinsic_target = cast(_IntrinsicFunction, target)
            intrinsic_target.__hc_intrinsic__ = metadata
            configured = _attach_intrinsic_hooks(intrinsic_target)
            _register_sim_callable(configured)
            return configured

        if fn is None:
            return decorate
        return decorate(fn)


kernel = _KernelNamespace()


class CurrentGroup:
    def __init__(
        self,
        *,
        group_id: tuple[int, ...] = (),
        shape: tuple[int, ...] = (),
        work_shape: tuple[int, ...] = (),
        work_offset: tuple[int, ...] = (),
    ) -> None:
        self.group_id = group_id
        self.shape = shape
        self.work_shape = work_shape
        self.work_offset = work_offset

    @property
    def size(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def subgroups(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    def workitems(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    def barrier(self) -> None:
        raise NotImplementedError("barrier is implemented by runtime backends")

    def load(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("load is implemented by runtime backends")

    def vload(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("vload is implemented by runtime backends")

    def store(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("store is implemented by runtime backends")

    def empty(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("empty is implemented by runtime backends")

    def zeros(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("zeros is implemented by runtime backends")

    def ones(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ones is implemented by runtime backends")

    def full(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("full is implemented by runtime backends")

    def vzeros(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("vzeros is implemented by runtime backends")

    def vones(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("vones is implemented by runtime backends")

    def vfull(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("vfull is implemented by runtime backends")


class SubGroup:
    def subgroup_id(self) -> int:
        raise NotImplementedError("subgroup_id is implemented by runtime backends")

    def size(self) -> int:
        raise NotImplementedError("size is implemented by runtime backends")


class WorkItem:
    def global_id(self) -> tuple[int, ...]:
        raise NotImplementedError("global_id is implemented by runtime backends")

    def local_id(self) -> tuple[int, ...]:
        raise NotImplementedError("local_id is implemented by runtime backends")
