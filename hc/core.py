# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast


@dataclass(frozen=True)
class BufferSpec:
    dimensions: tuple[Any, ...]

    def __repr__(self) -> str:
        body = ", ".join(str(dim) for dim in self.dimensions)
        return f"Buffer[{body}]"


class Buffer:
    def __class_getitem__(cls, item: Any) -> BufferSpec:
        if not isinstance(item, tuple):
            item = (item,)
        return BufferSpec(item)


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


def index_map(
    *,
    params: Callable[..., Any] | None = None,
    storage_size: Callable[..., Any],
    offset: Callable[..., Any],
) -> IndexMap:
    return IndexMap(params=params, storage_size=storage_size, offset=offset)


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


class _KernelFunction(Protocol):
    __hc_kernel__: KernelMetadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _HelperFunction(Protocol):
    __hc_func__: FuncMetadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _IntrinsicFunction(Protocol):
    __hc_intrinsic__: IntrinsicMetadata
    __hc_lowerings__: dict[str, Callable[..., Any]]
    __hc_verify__: Callable[..., Any] | None
    __hc_infer__: Callable[..., Any] | None
    lower: Any
    verify: Any
    infer: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _attach_intrinsic_hooks(fn: Callable[..., Any]) -> Callable[..., Any]:
    intrinsic_fn = cast(_IntrinsicFunction, fn)
    intrinsic_fn.__hc_lowerings__ = {}
    intrinsic_fn.__hc_verify__ = None
    intrinsic_fn.__hc_infer__ = None

    def lower(*, target: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(cb: Callable[..., Any]) -> Callable[..., Any]:
            intrinsic_fn.__hc_lowerings__[target] = cb
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
    ) -> Callable[..., Any]:
        metadata = IntrinsicMetadata(
            scope=scope,
            effects=effects,
            const_attrs=frozenset() if const_attrs is None else frozenset(const_attrs),
        )

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            intrinsic_target = cast(_IntrinsicFunction, target)
            intrinsic_target.__hc_intrinsic__ = metadata
            return _attach_intrinsic_hooks(intrinsic_target)

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
