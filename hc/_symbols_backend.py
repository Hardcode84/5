# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any, Protocol, cast

from ._ixsimpl_loader import load_ixsimpl


class RawIxSimplContext(Protocol):
    errors: list[str]

    def int_(self, value: int) -> Any: ...

    def rat(self, num: int, den: int) -> Any: ...

    def true_(self) -> Any: ...

    def false_(self) -> Any: ...

    def parse(self, text: str) -> Any: ...

    def eq(self, lhs: Any, rhs: Any) -> Any: ...

    def ne(self, lhs: Any, rhs: Any) -> Any: ...

    def check(self, expr: Any, assumptions: list[Any] | None = None) -> bool | None: ...


class IxSimplModule(Protocol):
    INT: int
    RAT: int
    SYM: int
    ADD: int
    MUL: int
    FLOOR: int
    CEIL: int
    MOD: int
    PIECEWISE: int
    MAX: int
    MIN: int
    XOR: int
    CMP: int
    AND: int
    OR: int
    NOT: int
    TRUE: int
    FALSE: int
    Context: type[RawIxSimplContext]

    def same_node(self, lhs: Any, rhs: Any) -> bool: ...

    def and_(self, lhs: Any, rhs: Any) -> Any: ...

    def or_(self, lhs: Any, rhs: Any) -> Any: ...

    def not_(self, value: Any) -> Any: ...

    def floor(self, value: Any) -> Any: ...

    def ceil(self, value: Any) -> Any: ...

    def mod(self, lhs: Any, rhs: Any) -> Any: ...

    def min_(self, lhs: Any, rhs: Any) -> Any: ...

    def max_(self, lhs: Any, rhs: Any) -> Any: ...

    def xor_(self, lhs: Any, rhs: Any) -> Any: ...

    def pw(self, *branches: tuple[Any, Any]) -> Any: ...


def backend() -> IxSimplModule:
    return cast(IxSimplModule, load_ixsimpl())


class BackendProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(backend(), name)
