# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Restricted AST lowering for the `hc_front` frontend boundary.

Source recovery, AST validation, shared visitor/emitter protocol.
`RecordingEmitter` is a test harness; `lower_*_to_front_ir()` emits real
`hc_front` MLIR via the managed Python bindings. Unsupported syntax raises
`FrontendError`. Collective regions track outer-binding captures only.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any, Protocol

from ._intrinsic_contracts import serialize_intrinsic_type_spec

__all__ = [
    "FrontendEmitError",
    "FrontendEmitter",
    "FrontendError",
    "RecordedEvent",
    "RecordingEmitter",
    "lower_function",
    "lower_function_to_front_ir",
    "lower_functions_to_front_ir",
    "lower_module",
    "lower_module_to_front_ir",
    "lower_source",
    "lower_source_to_front_ir",
]

_TOPLEVEL_KINDS = {
    "kernel": "kernel",
    "kernel.func": "func",
    "kernel.intrinsic": "intrinsic",
}
_REGION_KINDS = {
    "group.subgroups": "subgroup_region",
    "group.workitems": "workitem_region",
}
_PASSING_POSITIONAL = "positional"
_PASSING_KEYWORD_ONLY = "keyword_only"
_SCALAR_ANNOTATION_NAMES: dict[type, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    bytes: "bytes",
}


class FrontendError(SyntaxError):
    def __init__(
        self,
        message: str,
        *,
        filename: str | None = None,
        lineno: int | None = None,
        offset: int | None = None,
        text: str | None = None,
        end_lineno: int | None = None,
        end_offset: int | None = None,
    ) -> None:
        self.end_lineno = end_lineno
        self.end_offset = end_offset
        if filename is None or lineno is None or offset is None:
            super().__init__(message)
            return
        super().__init__(message, (filename, lineno, offset, text))


class FrontendEmitError(Exception):
    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column


_FrontendEmitError = FrontendEmitError


@dataclass(frozen=True)
class RecordedEvent:
    kind: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _SourceBuffer:
    filename: str
    parsed_source: str
    display_lines: tuple[str, ...]
    line_base: int = 0
    column_bases: tuple[int, ...] = ()

    def display_line(self, line: int | None) -> str | None:
        if line is None:
            return None
        index = line - self.line_base - 1
        if not 0 <= index < len(self.display_lines):
            return None
        return self.display_lines[index]

    def file_lineno(self, parsed_lineno: int | None) -> int | None:
        if parsed_lineno is None:
            return None
        return parsed_lineno + self.line_base

    def file_col_offset(
        self,
        parsed_lineno: int | None,
        parsed_col_offset: int | None,
    ) -> int | None:
        if parsed_lineno is None or parsed_col_offset is None:
            return None
        return _column_base(self.column_bases, parsed_lineno) + parsed_col_offset

    def file_offset(
        self,
        parsed_lineno: int | None,
        parsed_offset: int | None,
    ) -> int | None:
        if parsed_lineno is None or parsed_offset is None:
            return None
        return _column_base(self.column_bases, parsed_lineno) + parsed_offset


class FrontendEmitter(Protocol):
    def begin_module(self, *, filename: str) -> None: ...

    def end_module(self) -> None: ...

    def begin_op(self, kind: str, **payload: object) -> None: ...

    def end_op(self, kind: str, **payload: object) -> None: ...

    def emit_op(self, kind: str, **payload: object) -> None: ...


class RecordingEmitter:
    """Frontend-shaped event trace for visitor unit tests."""

    def __init__(self) -> None:
        self.events: list[RecordedEvent] = []

    def begin_module(self, *, filename: str) -> None:
        self._record("module_begin", filename=filename)

    def end_module(self) -> None:
        self._record("module_end")

    def begin_op(self, kind: str, **payload: object) -> None:
        self._record(f"{kind}_begin", **payload)

    def end_op(self, kind: str, **payload: object) -> None:
        self._record(f"{kind}_end", **payload)

    def emit_op(self, kind: str, **payload: object) -> None:
        self._record(kind, **payload)

    def _record(self, kind: str, **payload: object) -> None:
        self.events.append(RecordedEvent(kind=kind, payload=dict(payload)))


def lower_function(fn: Any, emitter: FrontendEmitter) -> None:
    """Recover `fn`'s source and lower the supported subset."""

    source = _source_buffer_from_function(fn)
    module = _parse_source(source)
    _lower_parsed_module(
        module, emitter, source, toplevel_overrides=_function_overrides(fn)
    )


def lower_source(
    source: str,
    emitter: FrontendEmitter,
    *,
    filename: str = "<memory>",
) -> None:
    """Lower source text (dedented) as a standalone module buffer."""

    source_buffer = _source_buffer_from_text(source, filename=filename)
    module = _parse_source(source_buffer)
    _lower_parsed_module(module, emitter, source_buffer)


def lower_module(
    module: ast.Module,
    emitter: FrontendEmitter,
    *,
    filename: str = "<memory>",
) -> None:
    """Lower a pre-parsed AST module in place; no source-snippet diagnostics."""

    _lower_parsed_module(module, emitter, _opaque_source_buffer(filename))


def lower_function_to_front_ir(
    fn: Any,
    *,
    context: Any | None = None,
) -> Any:
    """Lower `fn` into an `hc_front` MLIR module.

    Bad frontend input raises `FrontendError`; emitter invariant failures
    raise `RuntimeError`.
    """

    emitter = _new_hc_front_emitter(context=context)
    lower_function(fn, emitter)
    return emitter.module


def lower_source_to_front_ir(
    source: str,
    *,
    filename: str = "<memory>",
    context: Any | None = None,
) -> Any:
    """Lower source text into an `hc_front` MLIR module.

    Bad input -> `FrontendError`; emitter invariant failures -> `RuntimeError`.
    """

    emitter = _new_hc_front_emitter(context=context)
    lower_source(source, emitter, filename=filename)
    return emitter.module


def lower_module_to_front_ir(
    module: ast.Module,
    *,
    filename: str = "<memory>",
    context: Any | None = None,
) -> Any:
    """Lower a pre-parsed AST module into an `hc_front` MLIR module.

    Bad input -> `FrontendError`; emitter invariant failures -> `RuntimeError`.
    """

    emitter = _new_hc_front_emitter(context=context)
    lower_module(module, emitter, filename=filename)
    return emitter.module


def lower_functions_to_front_ir(
    fns: Sequence[Any],
    *,
    context: Any | None = None,
    filename: str | None = None,
    per_function_overrides: Mapping[int, Mapping[str, object]] | None = None,
) -> Any:
    """Lower several functions into one shared `hc_front` module.

    Each `fn` becomes one top-level op (`hc_front.kernel`/`.func`/`.intrinsic`).
    Driver uses this to pack a kernel with transitively-invoked helpers and
    resolver-discovered inline helpers.

    `per_function_overrides` (keyed by `id(fn)`): extra per-top-level metadata
    spliced into the emitted op. Resolver uses it to force
    `force_kind="func"` + `ref.kind="inline"` on undecorated helpers so
    `-hc-front-inline` can target them.
    """

    if not fns:
        raise ValueError("lower_functions_to_front_ir requires at least one function")
    emitter = _new_hc_front_emitter(context=context)
    module_filename = filename
    if module_filename is None:
        module_filename = _source_buffer_from_function(fns[0]).filename
    emitter.begin_module(filename=module_filename)
    overrides_by_id = dict(per_function_overrides or {})
    for fn in fns:
        source = _source_buffer_from_function(fn)
        module = _parse_source(source)
        combined = _function_overrides(fn)
        extra = overrides_by_id.get(id(fn))
        if extra:
            # `_lower_toplevel` keys the override bag on the AST
            # `FunctionDef.name`. Reject non-string `__name__` loudly:
            # silent drop would leave `force_kind` unapplied.
            name = getattr(fn, "__name__", None)
            if not isinstance(name, str):
                raise RuntimeError(
                    f"per_function_overrides supplied for {fn!r} but "
                    f"`__name__` is not a string (got {name!r}); override "
                    "payload cannot be keyed into the emitter bag"
                )
            slot = dict(combined.get(name, {}))
            slot.update(extra)
            combined = dict(combined)
            combined[name] = slot
        try:
            _FrontendLowerer(
                emitter=emitter,
                source=source,
                toplevel_overrides=combined,
            ).lower_module_body(module)
        except FrontendEmitError as exc:
            raise _frontend_emit_error(source, exc) from exc
    emitter.end_module()
    return emitter.module


def _new_hc_front_emitter(*, context: Any | None = None) -> Any:
    from ._frontend_mlir import HCFrontEmitter

    return HCFrontEmitter(context=context)


def _lower_parsed_module(
    module: ast.Module,
    emitter: FrontendEmitter,
    source: _SourceBuffer,
    *,
    toplevel_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    try:
        _FrontendLowerer(
            emitter=emitter,
            source=source,
            toplevel_overrides=toplevel_overrides,
        ).lower_module(module)
    except FrontendEmitError as exc:
        raise _frontend_emit_error(source, exc) from exc


class _FrontendLowerer:
    def __init__(
        self,
        *,
        emitter: FrontendEmitter,
        source: _SourceBuffer,
        toplevel_overrides: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self._emitter = emitter
        self._source = source
        self._binding_stack: list[frozenset[str]] = []
        self._scope_stack: list[_RegionScope] = []
        self._callable_kind_stack: list[str] = []
        # Decorator metadata + resolved annotations from `lower_function`;
        # source-only entrypoints pass `None`.
        self._toplevel_overrides: Mapping[str, Mapping[str, object]] = (
            toplevel_overrides or {}
        )

    def lower_module(self, module: ast.Module) -> None:
        self._emitter.begin_module(filename=self._source.filename)
        self.lower_module_body(module)
        self._emitter.end_module()

    def lower_module_body(self, module: ast.Module) -> None:
        # Split out so `lower_functions_to_front_ir` can append several
        # fns into one open module without re-bracketing begin/end.
        for stmt in _strip_docstring(module.body):
            self._lower_toplevel(stmt)

    def _lower_toplevel(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.AsyncFunctionDef):
            raise self._error("async functions are not supported", stmt)
        if not isinstance(stmt, ast.FunctionDef):
            raise self._error("unsupported top-level statement", stmt)
        overrides = self._toplevel_overrides.get(stmt.name)
        forced = _peek_force_kind(overrides) if overrides else None
        # `force_kind` bypasses the decorator sniff for undecorated
        # inline helpers; decorated tops never set it.
        kind = forced if forced is not None else _toplevel_kind(stmt, self._source)
        payload = _function_payload(stmt, self._source)
        if overrides:
            payload.update(overrides)
        self._emitter.begin_op(kind, **payload)
        self._callable_kind_stack.append(kind)
        self._push_bindings(stmt)
        self._lower_statements(stmt.body)
        self._pop_bindings()
        self._callable_kind_stack.pop()
        self._emitter.end_op(kind, **payload)

    def _lower_statements(self, body: list[ast.stmt]) -> None:
        for stmt in _strip_docstring(body):
            self._lower_stmt(stmt)

    def _lower_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.FunctionDef):
            self._lower_nested_function(stmt)
            return
        if isinstance(stmt, ast.AsyncFunctionDef):
            raise self._error("async functions are not supported", stmt)
        if isinstance(stmt, ast.Assign):
            self._lower_assign(stmt)
            return
        if isinstance(stmt, ast.AugAssign):
            self._lower_aug_assign(stmt)
            return
        if isinstance(stmt, ast.Return):
            self._lower_return(stmt)
            return
        if isinstance(stmt, ast.Expr):
            self._lower_expr(stmt.value)
            return
        if isinstance(stmt, ast.If):
            self._lower_if(stmt)
            return
        if isinstance(stmt, ast.For):
            self._lower_for(stmt)
            return
        raise self._error("unsupported statement", stmt)

    def _lower_nested_function(self, stmt: ast.FunctionDef) -> None:
        kind = _region_kind(stmt, self._source)
        payload = _region_payload(stmt, self._visible_bindings(), self._source)
        self._emitter.begin_op(kind, **payload)
        self._push_bindings(stmt)
        self._lower_statements(stmt.body)
        self._pop_bindings()
        self._emitter.end_op(kind, **payload)

    def _lower_assign(self, stmt: ast.Assign) -> None:
        if len(stmt.targets) != 1:
            raise self._error("multiple assignment targets are not supported", stmt)
        payload = _node_payload(stmt)
        self._emitter.begin_op("assign", **payload)
        self._lower_target(stmt.targets[0])
        self._lower_expr(stmt.value)
        self._emitter.end_op("assign", **payload)

    def _lower_aug_assign(self, stmt: ast.AugAssign) -> None:
        payload = _node_payload(stmt, op=type(stmt.op).__name__)
        self._emitter.begin_op("aug_assign", **payload)
        self._lower_target(stmt.target)
        self._lower_expr(stmt.value)
        self._emitter.end_op("aug_assign", **payload)

    def _lower_return(self, stmt: ast.Return) -> None:
        # `has_value` = `hc_front.return` gets an SSA operand. Kernels
        # are void, so `return None` shapes like bare `return`;
        # funcs/intrinsics keep the value-returning contract.
        has_value = stmt.value is not None and not (
            self._is_kernel_body() and _is_none_literal(stmt.value)
        )
        payload = _node_payload(stmt, has_value=has_value)
        self._emitter.begin_op("return", **payload)
        if has_value:
            self._lower_expr(stmt.value)
        self._emitter.end_op("return", **payload)

    def _is_kernel_body(self) -> bool:
        return (
            bool(self._callable_kind_stack)
            and self._callable_kind_stack[-1] == "kernel"
        )

    def _lower_if(self, stmt: ast.If) -> None:
        payload = _node_payload(stmt, has_orelse=bool(stmt.orelse))
        self._emitter.begin_op("if", **payload)
        self._emit_condition(stmt)
        self._emit_then(stmt)
        if stmt.orelse:
            self._emit_else(stmt)
        self._emitter.end_op("if", **payload)

    def _lower_for(self, stmt: ast.For) -> None:
        if stmt.orelse:
            raise self._error("for-else is not supported", stmt)
        payload = _node_payload(stmt)
        self._emitter.begin_op("for", **payload)
        self._emit_target(stmt.target)
        self._emit_iter(stmt)
        self._emit_body(stmt)
        self._emitter.end_op("for", **payload)

    def _emit_target(self, target: ast.expr) -> None:
        self._emit_control_region(
            "target",
            target,
            lambda: self._lower_target(target),
        )

    def _emit_condition(self, stmt: ast.If) -> None:
        self._emit_control_region(
            "condition",
            stmt,
            lambda: self._lower_expr(stmt.test),
        )

    def _emit_then(self, stmt: ast.If) -> None:
        self._emit_control_region(
            "then",
            stmt,
            lambda: self._lower_statements(stmt.body),
        )

    def _emit_else(self, stmt: ast.If) -> None:
        self._emit_control_region(
            "else",
            stmt,
            lambda: self._lower_statements(stmt.orelse),
        )

    def _emit_iter(self, stmt: ast.For) -> None:
        self._emit_control_region(
            "iter",
            stmt.iter,
            lambda: self._lower_expr(stmt.iter),
        )

    def _emit_body(self, stmt: ast.For) -> None:
        self._emit_control_region(
            "body",
            stmt,
            lambda: self._lower_statements(stmt.body),
        )

    def _emit_control_region(
        self,
        kind: str,
        node: ast.AST,
        body: Callable[[], None],
    ) -> None:
        # Private region markers mirror MLIR-style nested regions.
        payload = _node_payload(node)
        self._emitter.begin_op(kind, **payload)
        body()
        self._emitter.end_op(kind, **payload)

    @singledispatchmethod
    def _lower_expr(self, expr: ast.expr) -> None:
        raise self._error("unsupported expression", expr)

    @_lower_expr.register
    def _lower_constant_expr(self, expr: ast.Constant) -> None:
        self._emitter.emit_op("constant", **_node_payload(expr, value=expr.value))

    @_lower_expr.register
    def _lower_name_expr(self, expr: ast.Name) -> None:
        payload = _node_payload(expr, id=expr.id, ctx=_expr_context_name(expr.ctx))
        # Only stamp refs on loads; stores/deletes become
        # `hc_front.target_name` and inherit classification.
        if isinstance(expr.ctx, ast.Load):
            ref = self._classify_name(expr.id)
            if ref is not None:
                payload["ref"] = ref
        self._emitter.emit_op("name", **payload)

    @_lower_expr.register
    def _lower_attribute_expr(self, expr: ast.Attribute) -> None:
        self._lower_attr(expr)

    @_lower_expr.register
    def _lower_subscript_expr(self, expr: ast.Subscript) -> None:
        self._lower_subscript(expr)

    @_lower_expr.register
    def _lower_call_expr(self, expr: ast.Call) -> None:
        self._lower_call(expr)

    @_lower_expr.register
    def _lower_tuple_expr(self, expr: ast.Tuple) -> None:
        self._lower_tuple(expr)

    @_lower_expr.register
    def _lower_list_expr(self, expr: ast.List) -> None:
        self._lower_list(expr)

    @_lower_expr.register
    def _lower_binop_expr(self, expr: ast.BinOp) -> None:
        self._lower_binop(expr)

    @_lower_expr.register
    def _lower_unaryop_expr(self, expr: ast.UnaryOp) -> None:
        self._lower_unaryop(expr)

    @_lower_expr.register
    def _lower_compare_expr(self, expr: ast.Compare) -> None:
        self._lower_compare(expr)

    def _lower_attr(self, expr: ast.Attribute) -> None:
        payload = _node_payload(expr, attr=expr.attr)
        self._emitter.begin_op("attr", **payload)
        self._lower_expr(expr.value)
        self._emitter.end_op("attr", **payload)

    def _lower_subscript(self, expr: ast.Subscript) -> None:
        self._lower_subscript_like("subscript", expr)

    def _lower_subscript_like(self, kind: str, node: ast.Subscript) -> None:
        payload = _node_payload(node)
        self._emitter.begin_op(kind, **payload)
        self._lower_expr(node.value)
        self._lower_subscript_item(node.slice)
        self._emitter.end_op(kind, **payload)

    def _lower_subscript_item(self, node: ast.AST) -> None:
        if isinstance(node, ast.Slice):
            self._lower_slice(node)
            return
        if isinstance(node, ast.Tuple):
            payload = _node_payload(node, length=len(node.elts))
            self._emitter.begin_op("tuple", **payload)
            for item in node.elts:
                self._lower_subscript_item(item)
            self._emitter.end_op("tuple", **payload)
            return
        if isinstance(node, ast.expr):
            self._lower_expr(node)
            return
        raise self._error("unsupported subscript item", node)

    def _lower_slice(self, node: ast.Slice) -> None:
        payload = _node_payload(
            node,
            has_lower=node.lower is not None,
            has_upper=node.upper is not None,
            has_step=node.step is not None,
        )
        self._emitter.begin_op("slice", **payload)
        self._lower_optional_expr(node.lower)
        self._lower_optional_expr(node.upper)
        self._lower_optional_expr(node.step)
        self._emitter.end_op("slice", **payload)

    def _lower_optional_expr(self, expr: ast.expr | None) -> None:
        if expr is not None:
            self._lower_expr(expr)

    def _lower_call(self, expr: ast.Call) -> None:
        payload = _node_payload(
            expr,
            argc=len(expr.args),
            callee=_node_text(expr.func),
            keywords=tuple(keyword.arg for keyword in expr.keywords),
        )
        self._emitter.begin_op("call", **payload)
        self._lower_expr(expr.func)
        for arg in expr.args:
            self._lower_expr(arg)
        for keyword in expr.keywords:
            self._lower_keyword(keyword)
        self._emitter.end_op("call", **payload)

    def _lower_keyword(self, keyword: ast.keyword) -> None:
        if keyword.arg is None:
            raise self._error("**kwargs are not supported", keyword)
        payload = _node_payload(keyword, name=keyword.arg)
        self._emitter.begin_op("keyword", **payload)
        self._lower_expr(keyword.value)
        self._emitter.end_op("keyword", **payload)

    def _lower_tuple(self, expr: ast.Tuple) -> None:
        payload = _node_payload(expr, length=len(expr.elts))
        self._emitter.begin_op("tuple", **payload)
        for elt in expr.elts:
            self._lower_expr(elt)
        self._emitter.end_op("tuple", **payload)

    def _lower_list(self, expr: ast.List) -> None:
        payload = _node_payload(expr, length=len(expr.elts))
        self._emitter.begin_op("list", **payload)
        for elt in expr.elts:
            self._lower_expr(elt)
        self._emitter.end_op("list", **payload)

    def _lower_binop(self, expr: ast.BinOp) -> None:
        payload = _node_payload(expr, op=type(expr.op).__name__)
        self._emitter.begin_op("binop", **payload)
        self._lower_expr(expr.left)
        self._lower_expr(expr.right)
        self._emitter.end_op("binop", **payload)

    def _lower_unaryop(self, expr: ast.UnaryOp) -> None:
        payload = _node_payload(expr, op=type(expr.op).__name__)
        self._emitter.begin_op("unaryop", **payload)
        self._lower_expr(expr.operand)
        self._emitter.end_op("unaryop", **payload)

    def _lower_compare(self, expr: ast.Compare) -> None:
        payload = _node_payload(expr, ops=tuple(type(op).__name__ for op in expr.ops))
        self._emitter.begin_op("compare", **payload)
        self._lower_expr(expr.left)
        for comparator in expr.comparators:
            self._lower_expr(comparator)
        self._emitter.end_op("compare", **payload)

    @singledispatchmethod
    def _lower_target(self, target: ast.expr) -> None:
        raise self._error("unsupported assignment target", target)

    @_lower_target.register
    def _lower_name_target(self, target: ast.Name) -> None:
        self._emitter.emit_op("target_name", **_node_payload(target, id=target.id))

    @_lower_target.register
    def _lower_tuple_target(self, target: ast.Tuple) -> None:
        payload = _node_payload(target, length=len(target.elts))
        self._emitter.begin_op("target_tuple", **payload)
        for element in target.elts:
            self._lower_target(element)
        self._emitter.end_op("target_tuple", **payload)

    @_lower_target.register
    def _lower_subscript_target(self, target: ast.Subscript) -> None:
        self._lower_subscript_like("target_subscript", target)

    def _push_bindings(self, fn: ast.FunctionDef) -> None:
        self._binding_stack.append(_function_bindings(fn, self._source))
        self._scope_stack.append(_region_scope(fn, self._source))

    def _pop_bindings(self) -> None:
        self._binding_stack.pop()
        self._scope_stack.pop()

    def _visible_bindings(self) -> frozenset[str]:
        # Nested collective regions capture every enclosing scope.
        bindings: set[str] = set()
        for frame in self._binding_stack:
            bindings.update(frame)
        return frozenset(bindings)

    def _classify_name(self, name: str) -> dict[str, object] | None:
        # Innermost -> outermost: diagnostics want the closest matching kind.
        for scope in reversed(self._scope_stack):
            if name in scope.params:
                return {"kind": "param"}
            if name in scope.for_ivs:
                return {"kind": "iv"}
            if name in scope.locals:
                return {"kind": "local"}
        return None

    def _error(self, message: str, node: ast.AST) -> FrontendError:
        return _frontend_error(self._source, message, node)


class _BindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self._seen: set[str] = set()

    def bind(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)

    def visit_block(self, body: Sequence[ast.stmt]) -> None:
        for stmt in _strip_docstring(list(body)):
            self.visit_stmt(stmt)

    def visit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            self.bind(stmt.name)
            return
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                self.visit_target(target)
            return
        if isinstance(stmt, ast.AugAssign):
            self.visit_target(stmt.target)
            return
        if isinstance(stmt, ast.For):
            self.visit_target(stmt.target)
            self.visit_block(stmt.body)
            self.visit_block(stmt.orelse)
            return
        if isinstance(stmt, ast.If):
            self.visit_block(stmt.body)
            self.visit_block(stmt.orelse)

    def visit_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bind(target.id)
            return
        if isinstance(target, ast.Subscript):
            return
        if isinstance(target, ast.Starred):
            self.visit_target(target.value)
            return
        if isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self.visit_target(element)


class _CaptureCollector(ast.NodeVisitor):
    """Names loaded from enclosing scopes within one region body."""

    def __init__(self, *, outer: frozenset[str], local: frozenset[str]) -> None:
        self._outer = outer
        self._local = local
        self._captures: list[str] = []
        self._seen: set[str] = set()

    @property
    def captures(self) -> tuple[str, ...]:
        return tuple(self._captures)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        if node.id not in self._outer or node.id in self._local:
            return
        if node.id in self._seen:
            return
        self._seen.add(node.id)
        self._captures.append(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return  # nested fn: own lexical scope.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return  # nested async fn: own lexical scope.


def _source_buffer_from_function(fn: Any) -> _SourceBuffer:
    try:
        lines, start_line = inspect.getsourcelines(fn)
    except (OSError, TypeError) as exc:
        raise FrontendError(
            "frontend source recovery requires an importable Python function"
        ) from exc
    filename = inspect.getsourcefile(fn) or inspect.getfile(fn) or "<unknown>"
    return _source_buffer_from_lines(lines, filename=filename, start_line=start_line)


def _source_buffer_from_text(source: str, *, filename: str) -> _SourceBuffer:
    parsed_source = textwrap.dedent(source)
    display_lines = _split_lines(parsed_source)
    return _SourceBuffer(
        filename=filename,
        parsed_source=parsed_source,
        display_lines=display_lines,
        column_bases=_zero_column_bases(display_lines),
    )


def _source_buffer_from_lines(
    lines: Sequence[str],
    *,
    filename: str,
    start_line: int,
) -> _SourceBuffer:
    raw_source = "".join(lines)
    parsed_source = textwrap.dedent(raw_source)
    display_lines = _split_lines(raw_source)
    return _SourceBuffer(
        filename=filename,
        parsed_source=parsed_source,
        display_lines=display_lines,
        line_base=start_line - 1,
        column_bases=_column_bases_from_lines(display_lines, parsed_source),
    )


def _opaque_source_buffer(filename: str) -> _SourceBuffer:
    return _SourceBuffer(filename=filename, parsed_source="", display_lines=())


def _parse_source(source: _SourceBuffer) -> ast.Module:
    try:
        module = ast.parse(source.parsed_source, filename=source.filename)
    except SyntaxError as exc:
        raise _frontend_parse_error(source, exc) from exc
    _rebase_module_locations(module, source)
    return module


def _frontend_parse_error(
    source: _SourceBuffer,
    exc: SyntaxError,
) -> FrontendError:
    lineno = source.file_lineno(exc.lineno)
    return FrontendError(
        f"failed to parse source: {exc.msg}",
        filename=source.filename,
        lineno=lineno,
        offset=source.file_offset(exc.lineno, exc.offset),
        text=source.display_line(lineno),
        end_lineno=source.file_lineno(exc.end_lineno),
        end_offset=source.file_offset(exc.end_lineno, exc.end_offset),
    )


def _frontend_emit_error(
    source: _SourceBuffer,
    exc: FrontendEmitError,
) -> FrontendError:
    return FrontendError(
        exc.message,
        filename=source.filename,
        lineno=exc.line,
        offset=_syntax_offset(exc.column),
        text=source.display_line(exc.line),
    )


def _frontend_error(
    source: _SourceBuffer,
    message: str,
    node: ast.AST,
) -> FrontendError:
    lineno = _line(node)
    return FrontendError(
        message,
        filename=source.filename,
        lineno=lineno,
        offset=_syntax_offset(_column(node)),
        text=source.display_line(lineno),
        end_lineno=_end_line(node),
        end_offset=_syntax_offset(_end_column(node)),
    )


def _rebase_module_locations(module: ast.Module, source: _SourceBuffer) -> None:
    for node in ast.walk(module):
        _rebase_node_location(node, source)


def _rebase_node_location(node: ast.AST, source: _SourceBuffer) -> None:
    parsed_line = _line(node)
    parsed_end_line = _end_line(node)
    if parsed_line is not None:
        file_line = source.file_lineno(parsed_line)
        if file_line is not None:
            _set_node_lineno(node, file_line)
        file_col = source.file_col_offset(parsed_line, _column(node))
        if file_col is not None:
            _set_node_col_offset(node, file_col)
    if parsed_end_line is not None:
        end_line = source.file_lineno(parsed_end_line)
        if end_line is not None:
            _set_node_end_lineno(node, end_line)
        end_col = source.file_col_offset(
            parsed_end_line,
            _end_column(node),
        )
        if end_col is not None:
            _set_node_end_col_offset(node, end_col)


def _set_node_lineno(node: Any, lineno: int) -> None:
    node.lineno = lineno


def _set_node_col_offset(node: Any, col_offset: int) -> None:
    node.col_offset = col_offset


def _set_node_end_lineno(node: Any, end_lineno: int) -> None:
    node.end_lineno = end_lineno


def _set_node_end_col_offset(node: Any, end_col_offset: int) -> None:
    node.end_col_offset = end_col_offset


def _column_bases_from_lines(
    display_lines: Sequence[str],
    parsed_source: str,
) -> tuple[int, ...]:
    parsed_lines = _split_lines(parsed_source)
    if len(display_lines) != len(parsed_lines):
        raise FrontendError(
            "frontend source recovery produced inconsistent line mapping"
        )
    bases = [
        _removed_prefix_width(original, parsed)
        for original, parsed in zip(display_lines, parsed_lines, strict=True)
    ]
    return tuple(bases)


def _removed_prefix_width(original: str, parsed: str) -> int:
    if not parsed:
        return 0
    if original.endswith(parsed):
        return len(original) - len(parsed)
    return max(
        0,
        _leading_whitespace_width(original) - _leading_whitespace_width(parsed),
    )


def _leading_whitespace_width(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))


def _zero_column_bases(lines: Sequence[str]) -> tuple[int, ...]:
    return tuple(0 for _ in lines)


def _column_base(column_bases: Sequence[int], parsed_lineno: int) -> int:
    index = parsed_lineno - 1
    if not 0 <= index < len(column_bases):
        return 0
    return column_bases[index]


def _split_lines(text: str) -> tuple[str, ...]:
    return tuple(text.splitlines())


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if not body or not _is_docstring_expr(body[0]):
        return body
    return body[1:]


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _is_none_literal(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and expr.value is None


def _toplevel_kind(fn: ast.FunctionDef, source: _SourceBuffer) -> str:
    for decorator in _decorator_names(fn):
        kind = _TOPLEVEL_KINDS.get(decorator)
        if kind is not None:
            return kind
    raise _frontend_error(
        source,
        "top-level functions must use @kernel, @kernel.func, or @kernel.intrinsic",
        fn,
    )


def _region_kind(fn: ast.FunctionDef, source: _SourceBuffer) -> str:
    for decorator in _decorator_names(fn):
        kind = _REGION_KINDS.get(decorator)
        if kind is not None:
            return kind
    raise _frontend_error(
        source,
        "nested functions must use @group.subgroups or @group.workitems",
        fn,
    )


def _decorator_names(fn: ast.FunctionDef) -> tuple[str, ...]:
    names = [_decorator_name(decorator) for decorator in fn.decorator_list]
    return tuple(name for name in names if name is not None)


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _function_payload(
    fn: ast.FunctionDef,
    source: _SourceBuffer,
) -> dict[str, object]:
    return _node_payload(
        fn,
        decorators=_decorator_names(fn),
        name=fn.name,
        parameters=_parameter_records(fn.args, source),
        returns=_annotation_text(fn.returns),
    )


def _peek_force_kind(overrides: Mapping[str, object]) -> str | None:
    """Return normalized `force_kind` (or `None`); leave mapping intact.

    Emitter ignores unknown kwargs so `force_kind` is harmless if it
    flows through.
    """

    value = overrides.get("force_kind")
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"`force_kind` override must be a string, got {value!r}")
    if value not in {"kernel", "func", "intrinsic"}:
        raise RuntimeError(
            f"`force_kind` override {value!r} is not one of the known top-level "
            f"kinds (kernel/func/intrinsic)"
        )
    return value


def _function_overrides(fn: Any) -> dict[str, dict[str, object]]:
    """Decorator metadata + resolved annotations, keyed by `__name__`.

    Empty mapping when nothing's worth overriding so callers can
    forward unconditionally.
    """

    name = getattr(fn, "__name__", None)
    if not isinstance(name, str):
        return {}
    extras: dict[str, object] = {}
    metadata = _collect_toplevel_metadata(fn)
    if metadata:
        extras["metadata"] = metadata
    annotations = _collect_parameter_annotations(fn)
    if annotations:
        extras["parameter_annotations"] = annotations
    if not extras:
        return {}
    return {name: extras}


def _collect_toplevel_metadata(fn: Any) -> dict[str, object]:
    kernel_meta = getattr(fn, "__hc_kernel__", None)
    if kernel_meta is not None:
        return _serialize_kernel_metadata(kernel_meta)
    func_meta = getattr(fn, "__hc_func__", None)
    if func_meta is not None:
        return _serialize_func_metadata(func_meta)
    intrinsic_meta = getattr(fn, "__hc_intrinsic__", None)
    if intrinsic_meta is not None:
        return _serialize_intrinsic_metadata(intrinsic_meta, fn)
    return {}


def _serialize_kernel_metadata(meta: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    if meta.work_shape is not None:
        result["work_shape"] = _shape_tuple(meta.work_shape)
    if meta.group_shape is not None:
        result["group_shape"] = _shape_tuple(meta.group_shape)
    if meta.subgroup_size is not None:
        result["subgroup_size"] = int(meta.subgroup_size)
    if meta.literals:
        # `frozenset` iteration order is nondeterministic; sort.
        result["literals"] = tuple(
            sorted(_literal_name(item) for item in meta.literals)
        )
    return result


def _serialize_func_metadata(meta: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    scope = _scope_text(meta.scope)
    if scope is not None:
        result["scope"] = scope
    return result


def _serialize_intrinsic_metadata(meta: Any, fn: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    scope = _scope_text(meta.scope)
    if scope is not None:
        result["scope"] = scope
    if meta.effects is not None:
        result["effects"] = str(meta.effects)
    if meta.const_attrs:
        result["const_kwargs"] = tuple(sorted(str(name) for name in meta.const_attrs))
    if meta.operand_types is not None:
        result["operand_types"] = tuple(
            serialize_intrinsic_type_spec(item) for item in meta.operand_types
        )
    if meta.result_types:
        result["result_types"] = tuple(
            serialize_intrinsic_type_spec(item) for item in meta.result_types
        )
    recipes = _intrinsic_lowering_recipes(fn)
    if recipes:
        result["lowering_recipes"] = recipes
    return result


def _intrinsic_lowering_recipes(fn: Any) -> tuple[Any, ...]:
    lowerings = getattr(fn, "__hc_lowerings__", None)
    if not lowerings:
        return ()
    return tuple(recipe for _target, recipe in sorted(lowerings.items()))


def _shape_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _literal_name(value: Any) -> str:
    # `literals=` accepts `Symbol` (`.name`) or `str`; normalize to text.
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _scope_text(scope: Any) -> str | None:
    if scope is None:
        return None
    from .core import Scope

    if isinstance(scope, Scope):
        return scope.name
    # `WorkItem` / `SubGroup` are bare classes; key on `__name__` so
    # users can keep writing `scope=WorkItem`.
    name = getattr(scope, "__name__", None)
    if isinstance(name, str):
        return name
    return str(scope)


def _collect_parameter_annotations(fn: Any) -> dict[str, dict[str, object]]:
    raw = getattr(fn, "__annotations__", None) or {}
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except (NameError, AttributeError, SyntaxError, TypeError):
        # PEP 563 strings can reference out-of-scope names. Raw fallback;
        # string entries fail the isinstance check and drop.
        hints = raw
    result: dict[str, dict[str, object]] = {}
    for name, value in hints.items():
        if name == "return":
            continue
        record = _serialize_parameter_annotation(value)
        if record is not None:
            result[name] = record
    return result


def _serialize_parameter_annotation(value: Any) -> dict[str, object] | None:
    from .core import BufferSpec, CurrentGroup, SubGroup, WorkItem
    from .symbols import Symbol

    if isinstance(value, BufferSpec):
        return _serialize_buffer_spec(value)
    if value is CurrentGroup:
        return {"kind": "launch_context", "launch_context": "group"}
    if value is WorkItem:
        return {"kind": "launch_context", "launch_context": "workitem"}
    if value is SubGroup:
        return {"kind": "launch_context", "launch_context": "subgroup"}
    if isinstance(value, Symbol):
        return {"kind": "symbol"}
    if isinstance(value, type):
        dtype = _SCALAR_ANNOTATION_NAMES.get(value)
        if dtype is not None:
            return {"kind": "scalar", "dtype": dtype}
    return None


def _serialize_buffer_spec(value: Any) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": "buffer",
        "shape": tuple(str(dim) for dim in value.dimensions),
    }
    if value.dtype is not None:
        record["dtype"] = value.dtype
    if value.layout is not None:
        # Raw `IndexMap`; emitter does the `IndexMap -> #hc.layout`
        # conversion via the same helper as body-level `layout=`.
        record["layout"] = value.layout
    return record


def _region_payload(
    fn: ast.FunctionDef,
    outer_bindings: frozenset[str],
    source: _SourceBuffer,
) -> dict[str, object]:
    parameters = _parameter_records(fn.args, source)
    if len(parameters) != 1:
        raise _frontend_error(
            source,
            "collective regions must take exactly one parameter",
            fn,
        )
    return _node_payload(
        fn,
        captures=_region_captures(fn, outer_bindings, source),
        decorators=_decorator_names(fn),
        name=fn.name,
        parameters=parameters,
    )


def _parameter_records(
    args: ast.arguments,
    source: _SourceBuffer,
) -> tuple[tuple[str, str | None, str], ...]:
    _validate_arguments(args, source)
    # Tag positional vs kw-only so callers don't infer from list index.
    positional = tuple(
        (arg.arg, _annotation_text(arg.annotation), _PASSING_POSITIONAL)
        for arg in args.args
    )
    keyword_only = tuple(
        (arg.arg, _annotation_text(arg.annotation), _PASSING_KEYWORD_ONLY)
        for arg in args.kwonlyargs
    )
    return (*positional, *keyword_only)


def _validate_arguments(args: ast.arguments, source: _SourceBuffer) -> None:
    if args.vararg is not None or args.kwarg is not None:
        raise _frontend_error(source, "varargs are not supported", args)
    if args.posonlyargs:
        raise _frontend_error(
            source,
            "positional-only parameters are not supported",
            args.posonlyargs[0],
        )
    if args.defaults:
        raise _frontend_error(
            source,
            "parameter defaults are not supported",
            args.defaults[0],
        )
    kw_default = _first_present_node(args.kw_defaults)
    if kw_default is not None:
        raise _frontend_error(
            source,
            "parameter defaults are not supported",
            kw_default,
        )


def _annotation_text(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None
    return _node_text(annotation)


def _function_bindings(
    fn: ast.FunctionDef,
    source: _SourceBuffer,
) -> frozenset[str]:
    collector = _BindingCollector()
    for name, *_ in _parameter_records(fn.args, source):
        collector.bind(name)
    collector.visit_block(fn.body)
    return frozenset(collector.names)


@dataclass(frozen=True)
class _RegionScope:
    """Region's own bindings for name-ref stamping; no outer captures."""

    params: frozenset[str]
    for_ivs: frozenset[str]
    locals: frozenset[str]


def _region_scope(fn: ast.FunctionDef, source: _SourceBuffer) -> _RegionScope:
    params = frozenset(name for name, *_ in _parameter_records(fn.args, source))
    iv_collector = _ForIVCollector()
    iv_collector.visit_block(fn.body)
    for_ivs = frozenset(iv_collector.names) - params
    body = _BindingCollector()
    body.visit_block(fn.body)
    locals_ = frozenset(body.names) - params - for_ivs
    return _RegionScope(params=params, for_ivs=for_ivs, locals=locals_)


class _ForIVCollector(ast.NodeVisitor):
    """Names bound by `for` targets; skip nested function bodies."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._seen: set[str] = set()

    def bind(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)

    def visit_block(self, body: Sequence[ast.stmt]) -> None:
        for stmt in _strip_docstring(list(body)):
            self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        self._visit_target(node.target)
        self.visit_block(node.body)
        self.visit_block(node.orelse)

    def visit_If(self, node: ast.If) -> None:
        self.visit_block(node.body)
        self.visit_block(node.orelse)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return  # nested fn: separate region.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def _visit_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bind(target.id)
            return
        if isinstance(target, ast.Starred):
            self._visit_target(target.value)
            return
        if isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self._visit_target(element)


def _region_captures(
    fn: ast.FunctionDef,
    outer_bindings: frozenset[str],
    source: _SourceBuffer,
) -> tuple[str, ...]:
    collector = _CaptureCollector(
        outer=outer_bindings,
        local=_function_bindings(fn, source),
    )
    for stmt in _strip_docstring(fn.body):
        collector.visit(stmt)
    return collector.captures


def _node_payload(node: ast.AST, **payload: object) -> dict[str, object]:
    return dict(line=_line(node), column=_column(node), **payload)


def _line(node: ast.AST) -> int | None:
    return getattr(node, "lineno", None)


def _column(node: ast.AST) -> int | None:
    return getattr(node, "col_offset", None)


def _end_line(node: ast.AST) -> int | None:
    return getattr(node, "end_lineno", None)


def _end_column(node: ast.AST) -> int | None:
    return getattr(node, "end_col_offset", None)


def _syntax_offset(column: int | None) -> int | None:
    if column is None:
        return None
    # AST: 0-based; SyntaxError.offset: 1-based.
    return column + 1


def _node_text(node: ast.AST) -> str:
    return ast.unparse(node)


def _expr_context_name(ctx: ast.expr_context) -> str:
    return type(ctx).__name__.lower()


def _first_present_node(
    nodes: Sequence[ast.expr | None],
) -> ast.expr | None:
    for node in nodes:
        if node is not None:
            return node
    return None
