# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Symbolic expressions backed by the vendored `ixsimpl` engine.

The module-level `sym` object is a lazy default namespace. Use `Context.sym(...)`
when an isolated symbolic context is required.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Literal

from ._symbols_backend import BackendProxy

_ixs = BackendProxy()


class SymbolError(RuntimeError):
    pass


class SymbolConflictError(SymbolError):
    pass


class UnboundSymbolError(SymbolError):
    pass


class ContextMismatchError(SymbolError):
    pass


class NonConstantError(SymbolError):
    pass


class Truth(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def from_optional_bool(cls, value: bool | None) -> Truth:
        if value is True:
            return cls.TRUE
        if value is False:
            return cls.FALSE
        return cls.UNKNOWN


@lru_cache(maxsize=1)
def _predicate_tags() -> frozenset[int]:
    return frozenset(
        {
            _ixs.CMP,
            _ixs.AND,
            _ixs.OR,
            _ixs.NOT,
            _ixs.TRUE,
            _ixs.FALSE,
        }
    )


@lru_cache(maxsize=1)
def _op_names() -> dict[int, str]:
    return {
        _ixs.INT: "int",
        _ixs.RAT: "rat",
        _ixs.SYM: "sym",
        _ixs.ADD: "add",
        _ixs.MUL: "mul",
        _ixs.FLOOR: "floor",
        _ixs.CEIL: "ceil",
        _ixs.MOD: "mod",
        _ixs.PIECEWISE: "piecewise",
        _ixs.MAX: "max",
        _ixs.MIN: "min",
        _ixs.XOR: "xor",
        _ixs.CMP: "cmp",
        _ixs.AND: "and",
        _ixs.OR: "or",
        _ixs.NOT: "not",
        _ixs.TRUE: "true",
        _ixs.FALSE: "false",
    }


class Context:
    def __init__(self) -> None:
        self._raw = _ixs.Context()
        self._namespace = SymbolNamespace(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def namespace(self) -> SymbolNamespace:
        return self._namespace

    def sym(self, name: str) -> Symbol:
        return self._namespace[name]

    def const(self, value: int) -> Expr:
        return _wrap_expr(self._raw.int_(value), self)

    def int_(self, value: int) -> Expr:
        return self.const(value)

    def rat(self, num: int, den: int) -> Expr:
        if den == 0:
            raise ValueError("rational denominator must be non-zero")
        return _wrap_expr(self._raw.rat(num, den), self)

    def true_(self) -> Pred:
        return _wrap_pred(self._raw.true_(), self)

    def false_(self) -> Pred:
        return _wrap_pred(self._raw.false_(), self)

    def parse(self, text: str) -> Node:
        return _wrap_nonerror(self._raw.parse(text), self)

    def eq(self, a: ExprLike, b: ExprLike) -> Pred:
        lhs, rhs = _coerce_binary_exprs(a, b, self)
        return _wrap_pred(self._raw.eq(lhs._raw, rhs._raw), self)

    def ne(self, a: ExprLike, b: ExprLike) -> Pred:
        lhs, rhs = _coerce_binary_exprs(a, b, self)
        return _wrap_pred(self._raw.ne(lhs._raw, rhs._raw), self)

    def check(
        self,
        pred: PredLike,
        *,
        assumptions: Sequence[PredLike] = (),
        env: Env | Mapping[Symbol | str, int] | None = None,
    ) -> Truth:
        return check(pred, assumptions=assumptions, env=env, ctx=self)


class Node:
    def __init__(self, ctx: Context, raw: Any) -> None:
        self._ctx = ctx
        self._raw = raw

    @property
    def ctx(self) -> Context:
        return self._ctx

    @property
    def op(self) -> str:
        return _op_names().get(self._raw.tag, "unknown")

    @property
    def children(self) -> tuple[Node, ...]:
        return tuple(_wrap_nonerror(child, self._ctx) for child in self._raw.children)

    def format(self, dialect: Literal["sympy", "c"] = "sympy") -> str:
        if dialect == "sympy":
            return str(self)
        if dialect == "c":
            return str(self._raw.to_c())
        raise ValueError(f"unsupported symbol format dialect: {dialect}")

    def walk_pre(self) -> Iterator[Node]:
        stack: list[Node] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def walk_post(self) -> Iterator[Node]:
        stack: list[tuple[Node, bool]] = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                yield node
                continue
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.children))

    def simplify(self, *, assumptions: Sequence[PredLike] = ()) -> Node:
        raw_assumptions = _raw_assumptions(self._ctx, assumptions)
        raw = self._raw.simplify(assumptions=raw_assumptions or None)
        return _wrap_nonerror(raw, self._ctx)

    def __bool__(self) -> bool:
        raise TypeError(
            "symbolic nodes do not define truthiness; use check() or eval()"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Node)
            and self._ctx is other._ctx
            and bool(self._raw == other._raw)
        )

    def __hash__(self) -> int:
        return hash((id(self._ctx), hash(self._raw)))

    def __str__(self) -> str:
        return str(self._raw)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


class Expr(Node):
    def free_symbols(self) -> frozenset[Symbol]:
        return frozenset(
            _wrap_symbol(raw_sym, self._ctx) for raw_sym in self._raw.free_symbols
        )

    def subs(self, mapping: Mapping[Symbol | str, ExprLike]) -> Expr:
        raw = self._raw.subs(_raw_subs_mapping(self._ctx, mapping))
        return _wrap_expr(raw, self._ctx)

    def eval(self, env: Env | Mapping[Symbol | str, int]) -> int:
        name_map = _name_map_for(self._ctx, env)
        _ensure_bound(self, name_map)
        raw = self._raw.subs(name_map)
        return _int_from_raw(raw, self._ctx)

    def try_eval(self, env: Env | Mapping[Symbol | str, int]) -> int | None:
        try:
            return self.eval(env)
        except (NonConstantError, SymbolError, UnboundSymbolError):
            return None

    def is_constant(self) -> bool:
        return self.try_eval({}) is not None

    def constant_value(self) -> int:
        return self.eval({})

    def eval_int(self, env: Env | Mapping[Symbol | str, int]) -> int:
        return self.eval(env)

    def same_as(self, other: Expr) -> bool:
        _ensure_same_context(other, self._ctx)
        return bool(_ixs.same_node(self._raw, other._raw))

    def __add__(self, other: ExprLike) -> Expr:
        return _wrap_expr(self._raw + _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __radd__(self, other: ExprLike) -> Expr:
        return _coerce_expr(other, self._ctx).__add__(self)

    def __sub__(self, other: ExprLike) -> Expr:
        return _wrap_expr(self._raw - _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __rsub__(self, other: ExprLike) -> Expr:
        return _coerce_expr(other, self._ctx).__sub__(self)

    def __mul__(self, other: ExprLike) -> Expr:
        return _wrap_expr(self._raw * _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __rmul__(self, other: ExprLike) -> Expr:
        return _coerce_expr(other, self._ctx).__mul__(self)

    def __truediv__(self, other: ExprLike) -> Expr:
        return _wrap_expr(self._raw / _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __rtruediv__(self, other: ExprLike) -> Expr:
        return _coerce_expr(other, self._ctx).__truediv__(self)

    def __floordiv__(self, other: ExprLike) -> Expr:
        return floor_div(self, other)

    def __rfloordiv__(self, other: ExprLike) -> Expr:
        return floor_div(other, self)

    def __mod__(self, other: ExprLike) -> Expr:
        return mod(self, other)

    def __rmod__(self, other: ExprLike) -> Expr:
        return mod(other, self)

    def __neg__(self) -> Expr:
        return _wrap_expr(-self._raw, self._ctx)

    def __lt__(self, other: ExprLike) -> Pred:
        return _wrap_pred(self._raw < _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __le__(self, other: ExprLike) -> Pred:
        return _wrap_pred(self._raw <= _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __gt__(self, other: ExprLike) -> Pred:
        return _wrap_pred(self._raw > _coerce_expr(other, self._ctx)._raw, self._ctx)

    def __ge__(self, other: ExprLike) -> Pred:
        return _wrap_pred(self._raw >= _coerce_expr(other, self._ctx)._raw, self._ctx)


class Pred(Expr):
    def eval(self, env: Env | Mapping[Symbol | str, int]) -> bool:
        truth = check(self, env=env, ctx=self._ctx)
        if truth is Truth.TRUE:
            return True
        if truth is Truth.FALSE:
            return False
        raise NonConstantError(
            f"predicate is not constant under the provided env: {self}"
        )

    def eval_bool(self, env: Env | Mapping[Symbol | str, int]) -> bool:
        return self.eval(env)

    def prove(self, *, assumptions: Sequence[PredLike] = ()) -> Truth:
        return check(self, assumptions=assumptions, ctx=self._ctx)

    def __and__(self, other: PredLike) -> Pred:
        return _wrap_pred(
            _ixs.and_(self._raw, _coerce_pred(other, self._ctx)._raw),
            self._ctx,
        )

    def __rand__(self, other: PredLike) -> Pred:
        return _coerce_pred(other, self._ctx).__and__(self)

    def __or__(self, other: PredLike) -> Pred:
        return _wrap_pred(
            _ixs.or_(self._raw, _coerce_pred(other, self._ctx)._raw),
            self._ctx,
        )

    def __ror__(self, other: PredLike) -> Pred:
        return _coerce_pred(other, self._ctx).__or__(self)

    def __invert__(self) -> Pred:
        return _wrap_pred(_ixs.not_(self._raw), self._ctx)


class Symbol(Expr):
    @property
    def name(self) -> str:
        return str(self._raw.sym_name)


class SymbolNamespace:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._cache: dict[str, Symbol] = {}

    def __getattr__(self, name: str) -> Symbol:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def __getitem__(self, name: str) -> Symbol:
        symbol = self._cache.get(name)
        if symbol is not None:
            return symbol
        return self._intern(self._ctx._raw.sym(name))

    def _intern(self, raw: Any) -> Symbol:
        name = str(raw.sym_name)
        symbol = self._cache.get(name)
        if symbol is not None:
            return symbol
        symbol = Symbol(self._ctx, raw)
        self._cache[name] = symbol
        return symbol


@dataclass(frozen=True)
class Env(Mapping[Symbol, int]):
    _ctx: Context | None
    _bindings: dict[Symbol, int] = field(default_factory=dict)

    @property
    def ctx(self) -> Context | None:
        return self._ctx

    def __getitem__(self, key: Symbol) -> int:
        return self._bindings[key]

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)

    def eval(self, expr: ExprLike) -> int:
        return _coerce_expr(expr, _context_from_env_or_value(self, expr)).eval(self)

    def holds(self, pred: PredLike) -> bool:
        return _coerce_pred(pred, _context_from_env_or_value(self, pred)).eval(self)

    def _name_map(self) -> dict[str, int]:
        return {symbol.name: value for symbol, value in self._bindings.items()}


class Bindings:
    def __init__(self) -> None:
        self._ctx: Context | None = None
        self._bindings: dict[Symbol, int] = {}
        self._sources: dict[Symbol, str | None] = {}

    def bind(self, sym: Symbol, value: int, *, source: str | None = None) -> None:
        self._ctx = _merge_context(self._ctx, sym.ctx)
        existing = self._bindings.get(sym)
        if existing is not None and existing != value:
            prev = self._sources.get(sym)
            raise SymbolConflictError(
                f"conflicting binding for {sym}: {existing} from {prev!r} vs "
                f"{value} from {source!r}"
            )
        self._bindings[sym] = value
        self._sources[sym] = source

    def bind_many(
        self,
        items: Mapping[Symbol, int],
        *,
        source: str | None = None,
    ) -> None:
        for sym, value in items.items():
            self.bind(sym, value, source=source)

    def require(
        self,
        expr_or_pred: NodeLike,
        value: int | bool = True,
        *,
        source: str | None = None,
    ) -> None:
        env = self.freeze()
        if isinstance(expr_or_pred, bool):
            raise TypeError("require() does not accept bare bool values")
        if isinstance(expr_or_pred, Pred):
            self._require_predicate(expr_or_pred, env, bool(value), source)
            return

        expr = _coerce_expr(expr_or_pred, _context_from_env_or_value(env, expr_or_pred))
        actual = expr.eval(env)
        if actual != int(value):
            raise SymbolConflictError(
                f"requirement failed at {source!r}: "
                f"expected {expr} == {value}, got {actual}"
            )

    def _require_predicate(
        self, pred: Pred, env: Env, value: bool, source: str | None
    ) -> None:
        truth = check(pred, env=env)
        expected = Truth.TRUE if value else Truth.FALSE
        if truth is Truth.UNKNOWN:
            raise NonConstantError(f"require could not prove predicate: {pred}")
        if truth is not expected:
            raise SymbolConflictError(
                f"requirement failed at {source!r}: expected {expected.value}, "
                f"got {truth.value} for {pred}"
            )

    def get(self, sym: Symbol) -> int | None:
        return self._bindings.get(sym)

    def freeze(self) -> Env:
        return Env(self._ctx, dict(self._bindings))


type ExprLike = Expr | int
type PredLike = Pred | bool
type NodeLike = Expr | Pred | int | bool


def const(value: int) -> Expr:
    return _default_context().const(value)


def rat(num: int, den: int) -> Expr:
    return _default_context().rat(num, den)


def floor(expr: ExprLike) -> Expr:
    coerced = _coerce_expr(expr)
    return _wrap_expr(_ixs.floor(coerced._raw), coerced.ctx)


def ceil(expr: ExprLike) -> Expr:
    coerced = _coerce_expr(expr)
    return _wrap_expr(_ixs.ceil(coerced._raw), coerced.ctx)


def floor_div(a: ExprLike, b: ExprLike) -> Expr:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return floor(lhs / rhs)


def ceil_div(a: ExprLike, b: ExprLike) -> Expr:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return ceil(lhs / rhs)


def mod(a: ExprLike, b: ExprLike) -> Expr:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return _wrap_expr(_ixs.mod(lhs._raw, rhs._raw), lhs.ctx)


def min_(a: ExprLike, b: ExprLike) -> Expr:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return _wrap_expr(_ixs.min_(lhs._raw, rhs._raw), lhs.ctx)


def max_(a: ExprLike, b: ExprLike) -> Expr:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return _wrap_expr(_ixs.max_(lhs._raw, rhs._raw), lhs.ctx)


def xor(a: ExprLike, b: ExprLike) -> Expr:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return _wrap_expr(_ixs.xor_(lhs._raw, rhs._raw), lhs.ctx)


def eq(a: ExprLike, b: ExprLike) -> Pred:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return lhs.ctx.eq(lhs, rhs)


def ne(a: ExprLike, b: ExprLike) -> Pred:
    lhs, rhs = _coerce_binary_exprs(a, b)
    return lhs.ctx.ne(lhs, rhs)


def all_of(*preds: PredLike) -> Pred:
    ctx = _resolve_context(*preds)
    result = _coerce_pred(True, ctx)
    for pred in preds:
        result = result & pred
    return result


def any_of(*preds: PredLike) -> Pred:
    ctx = _resolve_context(*preds)
    result = _coerce_pred(False, ctx)
    for pred in preds:
        result = result | pred
    return result


def not_(pred: PredLike) -> Pred:
    return ~_coerce_pred(pred)


def ite(pred: PredLike, then: ExprLike, else_: ExprLike) -> Expr:
    return piecewise((then, pred), (else_, True))


def piecewise(*branches: tuple[ExprLike, PredLike]) -> Expr:
    if not branches:
        raise ValueError("piecewise requires at least one branch")
    ctx = _resolve_context(
        *(value for value, _ in branches), *(cond for _, cond in branches)
    )
    raw_branches = [
        (_coerce_expr(value, ctx)._raw, _coerce_pred(cond, ctx)._raw)
        for value, cond in branches
    ]
    return _wrap_expr(_ixs.pw(*raw_branches), ctx)


def simplify(node: NodeLike, *, assumptions: Sequence[PredLike] = ()) -> Node:
    return _coerce_node(node).simplify(assumptions=assumptions)


def check(
    pred: PredLike,
    *,
    assumptions: Sequence[PredLike] = (),
    env: Env | Mapping[Symbol | str, int] | None = None,
    ctx: Context | None = None,
) -> Truth:
    resolved_ctx = _resolve_context(pred, *assumptions, ctx=ctx, env=env)
    predicate = _coerce_pred(pred, resolved_ctx)
    name_map = _name_map_for(resolved_ctx, env)
    raw_pred = predicate._raw.subs(name_map) if name_map else predicate._raw
    raw_assumptions = _raw_assumptions(resolved_ctx, assumptions, name_map=name_map)
    return _truth_from_raw(raw_pred, resolved_ctx, raw_assumptions)


def parse(text: str, *, ctx: Context | None = None) -> Node:
    resolved_ctx = _default_context() if ctx is None else ctx
    return resolved_ctx.parse(text)


def _wrap_nonerror(raw: Any, ctx: Context) -> Node:
    _raise_if_error(raw, ctx)
    if raw.tag == _ixs.SYM:
        return ctx._namespace._intern(raw)
    if raw.tag in _predicate_tags():
        return Pred(ctx, raw)
    return Expr(ctx, raw)


def _wrap_expr(raw: Any, ctx: Context) -> Expr:
    wrapped = _wrap_nonerror(raw, ctx)
    if not isinstance(wrapped, Expr):
        raise TypeError("expected expression node")
    return wrapped


def _wrap_pred(raw: Any, ctx: Context) -> Pred:
    wrapped = _wrap_nonerror(raw, ctx)
    if not isinstance(wrapped, Pred):
        raise TypeError("expected predicate node")
    return wrapped


def _wrap_symbol(raw: Any, ctx: Context) -> Symbol:
    wrapped = _wrap_nonerror(raw, ctx)
    if not isinstance(wrapped, Symbol):
        raise TypeError("expected symbol node")
    return wrapped


def _raise_if_error(raw: Any, ctx: Context) -> None:
    if not raw.is_error:
        return
    errors = list(ctx._raw.errors)
    if raw.is_parse_error:
        kind = "parse error"
    elif raw.is_domain_error:
        kind = "domain error"
    else:
        kind = "symbol backend error"
    detail = errors[-1] if errors else kind
    raise SymbolError(detail)


def _merge_context(current: Context | None, incoming: Context) -> Context:
    if current is None:
        return incoming
    if current is not incoming:
        raise ContextMismatchError(
            "symbol operation mixed values from different contexts"
        )
    return current


def _resolve_context(
    *values: object,
    ctx: Context | None = None,
    env: Env | Mapping[Symbol | str, int] | None = None,
) -> Context:
    resolved = ctx
    if isinstance(env, Env) and env.ctx is not None:
        resolved = _merge_context(resolved, env.ctx)
    for value in values:
        if isinstance(value, Node):
            resolved = _merge_context(resolved, value.ctx)
    return _default_context() if resolved is None else resolved


def _context_from_env_or_value(
    env: Env | Mapping[Symbol | str, int] | None,
    value: object,
) -> Context:
    if isinstance(value, Node):
        return value.ctx
    if isinstance(env, Env) and env.ctx is not None:
        return env.ctx
    return _default_context()


def _coerce_node(value: NodeLike, ctx: Context | None = None) -> Node:
    resolved_ctx = _default_context() if ctx is None else ctx
    if isinstance(value, Node):
        _ensure_same_context(value, resolved_ctx)
        return value
    if isinstance(value, bool):
        return _coerce_pred(value, resolved_ctx)
    return _coerce_expr(value, resolved_ctx)


def _coerce_expr(value: ExprLike, ctx: Context | None = None) -> Expr:
    resolved_ctx = _default_context() if ctx is None else ctx
    if isinstance(value, Expr):
        _ensure_same_context(value, resolved_ctx)
        return value
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid symbolic expressions")
    return resolved_ctx.const(int(value))


def _coerce_pred(value: PredLike, ctx: Context | None = None) -> Pred:
    resolved_ctx = _default_context() if ctx is None else ctx
    if isinstance(value, Pred):
        _ensure_same_context(value, resolved_ctx)
        return value
    if isinstance(value, bool):
        return resolved_ctx.true_() if value else resolved_ctx.false_()
    if isinstance(value, Node):
        _ensure_same_context(value, resolved_ctx)
        raise TypeError(f"expected predicate, got {type(value).__name__}")
    raise TypeError(f"expected predicate or bool, got {type(value).__name__}")


def _coerce_binary_exprs(
    left: ExprLike,
    right: ExprLike,
    ctx: Context | None = None,
) -> tuple[Expr, Expr]:
    resolved_ctx = _resolve_context(left, right, ctx=ctx)
    return _coerce_expr(left, resolved_ctx), _coerce_expr(right, resolved_ctx)


def _ensure_same_context(node_or_ctx: Node | Context, ctx: Context) -> None:
    incoming = node_or_ctx.ctx if isinstance(node_or_ctx, Node) else node_or_ctx
    if incoming is not ctx:
        raise ContextMismatchError(
            "symbol operation mixed values from different contexts"
        )


def _name_map_for(
    ctx: Context,
    env: Env | Mapping[Symbol | str, int] | None,
) -> dict[str, int]:
    if env is None:
        return {}
    if isinstance(env, Env):
        if env.ctx is not None:
            _ensure_same_context(env.ctx, ctx)
        return env._name_map()
    name_map: dict[str, int] = {}
    for key, value in env.items():
        if isinstance(key, Symbol):
            _ensure_same_context(key, ctx)
            name_map[key.name] = int(value)
        else:
            name_map[str(key)] = int(value)
    return name_map


def _raw_subs_mapping(
    ctx: Context,
    mapping: Mapping[Symbol | str, ExprLike],
) -> dict[str | Any, Any]:
    raw_mapping: dict[str | Any, Any] = {}
    for key, value in mapping.items():
        raw_key: str | Any
        if isinstance(key, Symbol):
            _ensure_same_context(key, ctx)
            raw_key = key._raw
        else:
            raw_key = str(key)
        raw_mapping[raw_key] = _coerce_expr(value, ctx)._raw
    return raw_mapping


def _raw_assumptions(
    ctx: Context,
    assumptions: Sequence[PredLike],
    *,
    name_map: Mapping[str, int] | None = None,
) -> list[Any]:
    raw_assumptions: list[Any] = []
    for assumption in assumptions:
        raw = _coerce_pred(assumption, ctx)._raw
        raw_assumptions.append(raw.subs(dict(name_map)) if name_map else raw)
    return raw_assumptions


def _truth_from_raw(raw: Any, ctx: Context, raw_assumptions: Sequence[Any]) -> Truth:
    simplified = _wrap_nonerror(raw.simplify(assumptions=raw_assumptions or None), ctx)
    if isinstance(simplified, Pred) and simplified._raw.tag == _ixs.TRUE:
        return Truth.TRUE
    if isinstance(simplified, Pred) and simplified._raw.tag == _ixs.FALSE:
        return Truth.FALSE
    checked = ctx._raw.check(simplified._raw, assumptions=list(raw_assumptions) or None)
    return Truth.from_optional_bool(checked)


def _ensure_bound(expr: Expr, name_map: Mapping[str, int]) -> None:
    missing = sorted(
        symbol.name for symbol in expr.free_symbols() if symbol.name not in name_map
    )
    if missing:
        joined = ", ".join(missing)
        raise UnboundSymbolError(f"missing bindings for symbols: {joined}")


def _int_from_raw(raw: Any, ctx: Context) -> int:
    wrapped = _wrap_nonerror(raw, ctx)
    try:
        return int(wrapped._raw)
    except TypeError as exc:
        raise NonConstantError(f"expression is not constant: {wrapped}") from exc


@lru_cache(maxsize=1)
def _default_context() -> Context:
    return Context()


class _DefaultSymbolNamespace:
    def __getattr__(self, name: str) -> Symbol:
        return _default_context().namespace()[name]

    def __getitem__(self, name: str) -> Symbol:
        return _default_context().namespace()[name]

    def __repr__(self) -> str:
        return "sym"


sym = _DefaultSymbolNamespace()

__all__ = [
    "Bindings",
    "Context",
    "ContextMismatchError",
    "Env",
    "Expr",
    "Node",
    "NonConstantError",
    "Pred",
    "Symbol",
    "SymbolConflictError",
    "SymbolError",
    "Truth",
    "UnboundSymbolError",
    "all_of",
    "any_of",
    "ceil",
    "ceil_div",
    "check",
    "const",
    "eq",
    "floor",
    "floor_div",
    "ite",
    "max_",
    "min_",
    "mod",
    "ne",
    "not_",
    "parse",
    "piecewise",
    "rat",
    "simplify",
    "sym",
    "xor",
]
