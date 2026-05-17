# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python-side name resolution for `hc_front`.

Walks every `@kernel.func` / `@kernel.intrinsic` reachable from the compile
target, lowers them into one shared `hc_front` module, stamps a `ref` dict
attr on each `hc_front.name` / `hc_front.attr`. The `hc_front -> hc` pass
dispatches on the refs, never on Python state.

`hc_front.name` kinds:
    param / iv / local   -- stamped by the frontend from scope state.
    constant             -- captured int/float/bool/str.
    symbol               -- captured `hc.symbols.Symbol`.
    callee               -- `@kernel.func` helper.
    intrinsic            -- `@kernel.intrinsic` helper.
    inline               -- undecorated Python helper.
    builtin              -- `range`, `len`, etc.
    module               -- whole-module alias (numpy only).

`hc_front.attr` kinds:
    dsl_method           -- attr on a param/iv/local-rooted value.
    numpy_dtype_type     -- `np.<scalar>`; pass decides call-vs-descriptor per
                           use.
    numpy_attr           -- other `np.*`; opaque, treated as inline call.

Other bases (constant, callee, dtype chains, ...) stay unstamped -- the pass
uses the base's own ref. Unresolvable name loads raise `FrontendError`.
"""

from __future__ import annotations

import builtins
import inspect
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ._frontend import FrontendError, lower_functions_to_front_ir
from .core import FuncMetadata, IndexMap, IntrinsicMetadata, KernelMetadata
from .core import as_layout as _dsl_as_layout

__all__ = [
    "ResolvedFrontIR",
    "build_index_map_layout_dict_attr",
    "resolve_front_ir",
]


def _numpy_dtype_name(module: Any, attr: str) -> str | None:
    """Return `attr` iff `module.attr` is a numpy scalar dtype, else None.

    Resolves via live numpy so platform-specific aliases (`intp`,
    `float128`) and any future scalar work without edits. Typos fall
    through to `numpy_attr`.
    """

    value = getattr(module, attr, None)
    if value is None or not isinstance(value, type):
        return None
    try:
        import numpy as _np
    except ImportError:
        return None
    if issubclass(value, _np.generic):
        return attr
    return None


@dataclass(frozen=True)
class ResolvedFrontIR:
    """Resolver output: combined module + dep set.

    `functions` is BFS discovery order (decorated and inline
    interleaved). `inline_names` lists names emitted as
    `hc_front.func` with `ref.kind = "inline"`.
    """

    module: Any
    kernel_fn: Any
    functions: tuple[Any, ...]
    inline_names: frozenset[str]

    @property
    def symbol_names(self) -> tuple[str, ...]:
        return tuple(_fn_name(fn) for fn in self.functions)

    @property
    def exported_symbol_names(self) -> tuple[str, ...]:
        """User-visible top-level symbols: discovered minus inline helpers."""

        return tuple(
            _fn_name(fn)
            for fn in self.functions
            if _fn_name(fn) not in self.inline_names
        )


def resolve_front_ir(
    kernel_fn: Any,
    *,
    context: Any | None = None,
) -> ResolvedFrontIR:
    """Collect, lower, classify names reachable from `kernel_fn`.

    Module layout: kernel first, then decorated helpers/intrinsics, then
    undecorated inline helpers (discovery order). Every `hc_front.name`
    load carries a `ref` DictAttr; inline helpers carry
    `ref.kind = "inline"` for `-hc-front-inline`. Raises `FrontendError`
    on unclassifiable captures.
    """

    if not _is_kernel(kernel_fn):
        raise TypeError(
            f"resolve_front_ir expects a @kernel-decorated function, got "
            f"{kernel_fn!r}"
        )

    fns, inline_names = _walk_dep_set(kernel_fn)
    overrides = _inline_overrides(fns, inline_names)
    module = lower_functions_to_front_ir(
        fns, context=context, per_function_overrides=overrides
    )
    _classify_module(module, fns)
    return ResolvedFrontIR(
        module=module,
        kernel_fn=kernel_fn,
        functions=fns,
        inline_names=frozenset(inline_names),
    )


# --- Dependency walk ---------------------------------------------------------


def _walk_dep_set(kernel_fn: Any) -> tuple[tuple[Any, ...], set[str]]:
    """BFS reachable helpers; return `(ordered, inline_names)`.

    `ordered` starts with `kernel_fn`, then interleaves decorated
    callees and inline helpers in first-seen order. Dedup on `id(fn)`
    (handles aliases); inline `__name__` collisions are loud by
    design -- `hc_front -> hc` keys lookups on the name.
    """

    seen_ids: set[int] = set()
    ordered: list[Any] = []
    inline_names: set[str] = set()
    queue: deque[Any] = deque([kernel_fn])
    while queue:
        fn = queue.popleft()
        fn_id = id(fn)
        if fn_id in seen_ids:
            continue
        seen_ids.add(fn_id)
        ordered.append(fn)
        if fn is not kernel_fn and _is_inlinable_helper(fn):
            inline_names.add(_fn_name(fn))
        for dep in _reachable_references(fn):
            if id(dep) not in seen_ids:
                queue.append(dep)
    return tuple(ordered), inline_names


def _reachable_references(fn: Any) -> Iterable[Any]:
    """Yield decorated and inline-helper free references of `fn`.

    Inline helpers yield at the same level so BFS picks up
    helpers-of-helpers.
    """

    namespace = _FunctionNamespace(fn)
    yielded: set[int] = set()
    for name in _referenced_names(fn):
        value = namespace.lookup(name)
        if value is _UNDEFINED:
            continue
        if not (_is_decorated(value) or _is_inlinable_helper(value)):
            continue
        if id(value) in yielded:
            continue
        yielded.add(id(value))
        yield value


def _inline_overrides(
    fns: tuple[Any, ...], inline_names: set[str]
) -> dict[int, Mapping[str, object]]:
    """Per-fn overrides pinning inline helpers as `hc_front.func` +
    `ref.kind = "inline"`. Keyed by `id(fn)`; `force_kind` skips the
    decorator sniff.
    """

    overrides: dict[int, Mapping[str, object]] = {}
    for fn in fns:
        name = _fn_name(fn)
        if name not in inline_names:
            continue
        overrides[id(fn)] = {
            "force_kind": "func",
            "ref": {"kind": "inline", "qualified_name": _qualified_name(fn)},
        }
    return overrides


def _referenced_names(fn: Any) -> tuple[str, ...]:
    """All free names in `fn`, including nested `def` bodies.

    `inspect.getclosurevars` only sees the top-level code; walk
    `co_consts` transitively to catch nested `@group.workitems` etc.
    Conservative over-approximation -- false matches add harmless deps,
    missing a real dep would break compilation.
    """

    code = getattr(fn, "__code__", None)
    if code is None:
        return ()
    names: list[str] = []
    seen: set[str] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        for name in current.co_names:
            if name not in seen:
                seen.add(name)
                names.append(name)
        for name in current.co_freevars:
            if name not in seen:
                seen.add(name)
                names.append(name)
        for const in current.co_consts:
            if inspect.iscode(const):
                stack.append(const)
    return tuple(names)


_UNDEFINED = object()


def _is_decorated(value: Any) -> bool:
    return (
        isinstance(getattr(value, "__hc_func__", None), FuncMetadata)
        or isinstance(getattr(value, "__hc_intrinsic__", None), IntrinsicMetadata)
        or isinstance(getattr(value, "__hc_kernel__", None), KernelMetadata)
    )


def _is_kernel(value: Any) -> bool:
    return isinstance(getattr(value, "__hc_kernel__", None), KernelMetadata)


def _fn_name(fn: Any) -> str:
    name = getattr(fn, "__name__", None)
    return name if isinstance(name, str) else repr(fn)


# --- Classification walk -----------------------------------------------------


def _classify_module(module: Any, fns: tuple[Any, ...]) -> None:
    from .mlir import ir as _ir

    # `lower_functions_to_front_ir` emits one classifiable top-level
    # per `fn` plus optional support modules (e.g.
    # `__hc_intrinsic_lowerings__`) as siblings. Drop the latter -- no
    # Python fn behind them.
    ctx = module.context
    toplevels = [op for op in module.body.operations if _classifiable_toplevel(op)]
    if len(toplevels) != len(fns):
        raise FrontendError(
            f"hc_front module has {len(toplevels)} classifiable top-level ops "
            f"but the resolver collected {len(fns)} Python fns; emission order "
            "broke"
        )
    for toplevel, fn in zip(toplevels, fns, strict=True):
        _OpClassifier(fn=fn, ctx=ctx, ir=_ir).classify(toplevel)


def _classifiable_toplevel(op: Any) -> bool:
    name = str(op.operation.name)
    return name.startswith("hc_front.")


class _OpClassifier:
    """Walks one top-level region, stamps refs on captures + attrs."""

    def __init__(self, *, fn: Any, ctx: Any, ir: Any) -> None:
        self._fn = fn
        self._ctx = ctx
        self._ir = ir
        # ~200 names re-resolved across nested regions; cache once.
        self._namespace = _FunctionNamespace(fn)

    def classify(self, toplevel_op: Any) -> None:
        for region in toplevel_op.regions:
            for block in region.blocks:
                for op in list(block.operations):
                    self._classify_op(op)

    def _classify_op(self, op: Any) -> None:
        op_name = op.operation.name
        if op_name == "hc_front.name":
            self._classify_name(op)
        elif op_name == "hc_front.attr":
            self._classify_attr(op)
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    self._classify_op(inner)

    def _classify_name(self, op: Any) -> None:
        attrs = op.operation.attributes
        if "ref" in attrs:
            return  # frontend already classified.
        ctx_value = _str_attr_or_none(attrs, "ctx")
        if ctx_value != "load":
            return  # store targets never classified here.
        ident = _str_attr_or_none(attrs, "name")
        if ident is None:
            raise FrontendError(
                f"hc_front.name at {_loc_hint(op)} is missing its identifier"
            )
        ref = self._classify_captured(ident, op)
        attrs["ref"] = self._dict_attr(ref)

    def _classify_attr(self, op: Any) -> None:
        attrs = op.operation.attributes
        if "ref" in attrs:
            return
        base = op.operation.operands[0]
        base_ref = _read_ref(base.owner)
        if base_ref is None:
            return  # Unclassified base: pass uses the base op's own kind.
        method_name = _str_attr_or_none(op.operation.attributes, "name") or ""
        kind = base_ref.get("kind")
        if kind in {"param", "iv", "local"}:
            attrs["ref"] = self._dict_attr(
                {"kind": "dsl_method", "method": method_name}
            )
            return
        if kind == "module" and base_ref.get("module") == "numpy":
            # Dtypes get a dedicated kind so the pass can produce
            # element types without a numpy catalog clone.
            import numpy as _np

            dtype = _numpy_dtype_name(_np, method_name)
            if dtype is not None:
                attrs["ref"] = self._dict_attr(
                    {"kind": "numpy_dtype_type", "dtype": dtype}
                )
            else:
                attrs["ref"] = self._dict_attr(
                    {"kind": "numpy_attr", "attr": method_name}
                )
            return
        # Chained attrs on already-classified bases stay unstamped; the
        # lowering pass either uses the base's kind or fails at the use.

    def _classify_captured(self, name: str, op: Any) -> Mapping[str, object]:
        value = self._namespace.lookup(name)
        if value is _UNDEFINED:
            raise FrontendError(
                f"unresolved name {name!r} at {_loc_hint(op)}: not a parameter, "
                f"local, iv, capture, or builtin of "
                f"{_fn_name(self._fn)!r}"
            )
        try:
            return _classify_capture_value(name, value)
        except FrontendError as exc:
            raise FrontendError(
                f"{exc} at {_loc_hint(op)} in {_fn_name(self._fn)!r}"
            ) from None

    def _dict_attr(self, mapping: Mapping[str, object]) -> Any:
        ir = self._ir
        entries: dict[str, Any] = {}
        for key, val in mapping.items():
            entries[key] = self._to_attr(val)
        return ir.DictAttr.get(entries, context=self._ctx)

    def _to_attr(self, value: object) -> Any:
        return _encode_ref_payload(value, self._ctx, self._ir)


def _encode_ref_payload(value: object, ctx: Any, ir: Any) -> Any:
    """Python ref-payload value -> typed MLIR attribute.

    Fixed table:
        str          -> StringAttr
        bool / int   -> i64 IntegerAttr
        tuple        -> ArrayAttr (recursive)
        Mapping      -> DictionaryAttr (recursive, str keys)
        hc Expr      -> #hc.expr<...>

    One encoder for body-name captures and parameter-side layout stamps
    (`build_index_map_layout_dict_attr`); C++ readers see one shape.
    """
    # Lazy: `hc.symbols` only needed for `Expr` carriers.
    from .symbols import Expr

    if isinstance(value, str):
        return ir.StringAttr.get(value, context=ctx)
    if isinstance(value, bool | int):
        return ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, context=ctx),
            int(value),
        )
    if isinstance(value, tuple):
        return ir.ArrayAttr.get(
            [_encode_ref_payload(item, ctx, ir) for item in value],
            context=ctx,
        )
    if isinstance(value, Mapping):
        return ir.DictAttr.get(
            {str(k): _encode_ref_payload(v, ctx, ir) for k, v in value.items()},
            context=ctx,
        )
    if isinstance(value, Expr):
        # One `sym::parseExpr` at the frontend boundary; downstream C++
        # sees typed `ExprAttr`. See AGENTS.md.
        return ir.Attribute.parse(f'#hc.expr<"{value}">', context=ctx)
    raise FrontendError(f"cannot encode ref payload value {value!r}")


def build_index_map_layout_dict_attr(
    layout: IndexMap,
    ctx: Any,
    ir: Any,
) -> Any:
    """Symbolic-eval `layout`; return a `DictionaryAttr` matching the
    body-level `kind = "layout"` ref shape.

    Use when the layout lands on something other than `hc_front.name`
    (e.g. an `hc_front.kernel` parameter dict). C++ reads via
    `layoutAttrFromRef` with the same key set.
    """
    payload = _index_map_ref(layout)
    return _encode_ref_payload(payload, ctx, ir)


_CaptureClassifier = Callable[[str, Any], "Mapping[str, object] | None"]


def _classify_capture_value(name: str, value: Any) -> Mapping[str, object]:
    for classifier in _CAPTURE_CLASSIFIERS:
        ref = classifier(name, value)
        if ref is not None:
            return ref
    raise FrontendError(
        f"unclassifiable capture {name!r}: value of type "
        f"{type(value).__name__} is not supported in hc_front resolution; "
        f"captures must be builtins, numpy, hc.symbols.Symbol, literal "
        f"int/float/bool/str, or @kernel.func / @kernel.intrinsic helpers"
    )


def _classify_builtin(name: str, value: Any) -> Mapping[str, object] | None:
    if not _is_builtin(name, value):
        return None
    return {"kind": "builtin", "builtin": name}


def _classify_numpy_module(name: str, value: Any) -> Mapping[str, object] | None:
    del name
    if not _is_numpy_module(value):
        return None
    return {"kind": "module", "module": "numpy"}


def _classify_symbol(name: str, value: Any) -> Mapping[str, object] | None:
    del name
    if not _is_symbol(value):
        return None
    return {"kind": "symbol", "name": _symbol_name(value)}


def _classify_constant(name: str, value: Any) -> Mapping[str, object] | None:
    del name
    if not _is_constant(value):
        return None
    return {
        "kind": "constant",
        "python_kind": _python_kind(value),
        "value": _repr_constant(value),
    }


def _classify_callee(name: str, value: Any) -> Mapping[str, object] | None:
    del name
    if not _is_kernel_func(value):
        return None
    return {
        "kind": "callee",
        "callee": f"@{_fn_name(value)}",
        "scope": _scope_ref_text(value.__hc_func__.scope),
    }


def _classify_intrinsic(name: str, value: Any) -> Mapping[str, object] | None:
    del name
    if not _is_intrinsic(value):
        return None
    meta = value.__hc_intrinsic__
    payload: dict[str, object] = {
        "kind": "intrinsic",
        "callee": f"@{_fn_name(value)}",
        "scope": _scope_ref_text(meta.scope),
    }
    if meta.effects is not None:
        payload["effects"] = str(meta.effects)
    if meta.const_attrs:
        payload["const_kwargs"] = tuple(sorted(str(x) for x in meta.const_attrs))
    return payload


def _classify_inline(name: str, value: Any) -> Mapping[str, object] | None:
    del name
    if not _is_inlinable_helper(value):
        return None
    return {"kind": "inline", "qualified_name": _qualified_name(value)}


def _classify_as_layout(name: str, value: Any) -> Mapping[str, object] | None:
    """`hc.core.as_layout` recognized by identity -- DSL primitive, not inlinable.

    Catches imports / aliases before `_classify_inline` tries to re-parse
    the dispatcher's body. Lowering keys on `kind = "layout_op"`.
    """
    del name
    if value is not _dsl_as_layout:
        return None
    return {"kind": "layout_op", "op": "as_layout"}


def _classify_index_map(name: str, value: Any) -> Mapping[str, object] | None:
    """Module-level `IndexMap` captured as a layout descriptor.

    Lambdas eval symbolically; result lands as `#hc.expr` bodies feeding
    `LayoutAttr`. Eval failure becomes a `FrontendError` naming the capture.
    """
    if not isinstance(value, IndexMap):
        return None
    try:
        return _index_map_ref(value)
    except FrontendError as exc:
        raise FrontendError(f"layout descriptor {name!r}: {exc}") from None


def _index_map_ref(layout: IndexMap) -> Mapping[str, object]:
    """Symbolic eval of `layout` -> serializable ref payload.

    Discover shape syms from `params`/`storage_size` signatures, index
    syms from `offset`'s leading positionals. Bind through a fresh
    `SymbolNamespace`. Params feed back as symbol-valued (not
    expr-valued) so offset prints `i * row_stride + j` instead of
    fully-substituted -- matches `doc/layouts.md` and keeps diagnostics
    readable.

    Returns parallel arrays for the params table -- encoder only handles
    flat scalar/tuple values. Pairing: `params_names[i] -> params_exprs[i]`.
    """
    from .symbols import Context, SymbolNamespace

    shape_param_names = _layout_shape_param_names(layout)
    index_param_names = _layout_index_param_names(layout, shape_param_names)
    free_sym_names = tuple(layout.free_syms)
    _validate_free_syms(layout, shape_param_names, index_param_names)

    ctx = Context()
    syms = SymbolNamespace(ctx)
    shape_syms = tuple(syms[n] for n in shape_param_names)
    index_syms = tuple(syms[n] for n in index_param_names)
    free_syms = {name: syms[name] for name in free_sym_names}

    params_exprs, params_named = _layout_eval_params(layout, ctx, syms, shape_syms)

    storage_args = (
        (*shape_syms, params_named) if layout.params is not None else shape_syms
    )
    storage_call = _layout_invoke(
        layout.storage_size, storage_args, role="storage_size", kwargs=free_syms
    )
    storage_call = _coerce_layout_expr(storage_call, ctx)

    offset_args = (
        (*index_syms, *shape_syms, params_named)
        if layout.params is not None
        else (*index_syms, *shape_syms)
    )
    offset_call = _layout_invoke(
        layout.offset, offset_args, role="offset", kwargs=free_syms
    )
    offset_call = _coerce_layout_expr(offset_call, ctx)

    # Payload carries raw `Expr` carriers; encoder stamps typed
    # `#hc.expr` attrs via MLIR Python bindings. C++ never sees text.
    return {
        "kind": "layout",
        "shape_syms": tuple(shape_param_names),
        "index_syms": tuple(index_param_names),
        "params": params_exprs,
        "storage_size": storage_call,
        "offset": offset_call,
    }


def _validate_free_syms(
    layout: IndexMap,
    shape_param_names: tuple[str, ...],
    index_param_names: tuple[str, ...],
) -> None:
    """Reject duplicate / colliding `free_syms` before evaluation.

    `LayoutAttr::verify` enforces this on the post-lambda payload, but a
    Python-side conflict would surface as an opaque MLIR dup-name. Catch
    here so the error names the descriptor and the colliding token.
    """
    reserved = set(shape_param_names) | set(index_param_names)
    if layout.params is not None:
        reserved.update(_layout_positional_names(layout.params, role="params"))
    seen_free: set[str] = set()
    for name in layout.free_syms:
        if not name:
            raise FrontendError("free_syms entries must be non-empty strings")
        if name in reserved:
            raise FrontendError(
                f"free_syms entry {name!r} collides with a shape / index / "
                "params sym name"
            )
        if name in seen_free:
            raise FrontendError(f"duplicate free_syms entry {name!r}")
        seen_free.add(name)


def _layout_eval_params(
    layout: IndexMap,
    ctx: Any,
    syms: Any,
    shape_syms: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Eval `layout.params` symbolically; return `(params_exprs, params_named)`.

    `params_exprs[name]` -> `Expr` (stamps as `#hc.expr`).
    `params_named[name]` -> `Symbol`, fed into `storage_size` / `offset`
    so offset prints `i * row_stride + j`, not the substituted form.
    """
    params_exprs: dict[str, Any] = {}
    params_named: dict[str, Any] = {}
    if layout.params is None:
        return params_exprs, params_named
    raw = _layout_invoke(layout.params, shape_syms, role="params")
    if raw is None:
        return params_exprs, params_named
    if not isinstance(raw, Mapping):
        raise FrontendError("params(...) must return a mapping or None")
    for key, expr in raw.items():
        if not isinstance(key, str):
            raise FrontendError(f"params(...) key {key!r} is not a string")
        params_exprs[key] = _coerce_layout_expr(expr, ctx)
        params_named[key] = syms[key]
    return params_exprs, params_named


def _coerce_layout_expr(value: Any, ctx: Any) -> Any:
    """Wrap bare `int` in `ctx.const(...)` so encoders see `Expr`.

    `LayoutAttr` needs `ExprAttr` everywhere; lambdas that never touch a
    sym (e.g. `lambda lc, fc: WMMA_M*WMMA_N`) eval to bare `int`. Coerce
    here so layout authors don't have to spell `ctx.const(...)`.
    """
    if isinstance(value, bool):
        raise FrontendError("layout expression cannot be a boolean")
    if isinstance(value, int):
        return ctx.const(int(value))
    return value


def _layout_invoke(
    fn: Any,
    args: tuple[Any, ...],
    *,
    role: str,
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Call lambda; rewrap exceptions as `FrontendError` naming `role`.

    `kwargs` is the layout-wide `free_syms`; pass only the subset the
    lambda declares keyword-only so others aren't forced to swallow it.
    """
    try:
        if kwargs:
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError) as exc:
                raise FrontendError(f"cannot inspect {role}: {exc}") from None
            wanted = {
                param.name
                for param in sig.parameters.values()
                if param.kind == inspect.Parameter.KEYWORD_ONLY
            }
            subset = {name: value for name, value in kwargs.items() if name in wanted}
            return fn(*args, **subset)
        return fn(*args)
    except Exception as exc:
        raise FrontendError(f"{role}(...) raised {type(exc).__name__}: {exc}") from None


def _layout_positional_names(
    fn: Any,
    *,
    role: str,
    allowed_kwonly: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """`fn`'s positional names (no varargs, no defaults).

    Layout lambdas must be straight positional shape -> ... -> params.
    Keyword-only names in `allowed_kwonly` (the layout's `free_syms`)
    are permitted but not returned -- caller binds separately.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise FrontendError(f"cannot inspect {role}: {exc}") from None
    names: list[str] = []
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # Lambda kw-only outside `allowed_kwonly` is a typo.
            if param.name not in allowed_kwonly:
                raise FrontendError(
                    f"{role} keyword-only parameter {param.name!r} is not "
                    "declared in the layout's free_syms"
                )
            if param.default is not inspect.Parameter.empty:
                raise FrontendError(
                    f"{role} free-sym parameter {param.name!r} must not "
                    "have a default value"
                )
            continue
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise FrontendError(
                f"{role} must use positional parameters only, "
                f"got {param.name!r} ({param.kind.description})"
            )
        if param.default is not inspect.Parameter.empty:
            raise FrontendError(
                f"{role} parameter {param.name!r} must not have a default value"
            )
        names.append(param.name)
    return tuple(names)


def _layout_shape_param_names(layout: IndexMap) -> tuple[str, ...]:
    """Shape-sym names: from `params` if present, else `storage_size`.

    `params` takes shape syms only -- full signature is the shape list.
    Without it, `storage_size`'s signature is the list (offset must match).
    """
    free = frozenset(layout.free_syms)
    if layout.params is not None:
        return _layout_positional_names(
            layout.params, role="layout params", allowed_kwonly=free
        )
    return _layout_positional_names(
        layout.storage_size, role="layout storage_size", allowed_kwonly=free
    )


def _layout_index_param_names(
    layout: IndexMap, shape_param_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Index-sym names from `offset`'s leading positionals.

    Convention (`doc/layouts.md`):
        offset(i, j, ..., *shape_syms[, params], *, free_syms...)

    `LayoutAttr` requires `index_syms.size() == shape_syms.size()`.
    """
    free = frozenset(layout.free_syms)
    offset_names = _layout_positional_names(
        layout.offset, role="layout offset", allowed_kwonly=free
    )
    trailing = 1 if layout.params is not None else 0
    n_shape = len(shape_param_names)
    expected = 2 * n_shape + trailing
    if len(offset_names) != expected:
        raise FrontendError(
            f"offset(...) must have exactly {expected} positional "
            f"parameter(s) ({n_shape} index sym(s) + {n_shape} shape sym(s)"
            + (" + params dict" if trailing else "")
            + f"); got {offset_names!r}"
        )
    index_names = offset_names[:n_shape]
    shape_tail = offset_names[n_shape : 2 * n_shape]
    if shape_tail != shape_param_names:
        raise FrontendError(
            f"offset(...) shape parameters {shape_tail!r} disagree with "
            f"the layout's shape parameters {shape_param_names!r}"
        )
    return index_names


# Most-specific first: builtins beat constants (`True`, not `1`), numpy
# beats callables. `IndexMap` and `as_layout` ahead of `_classify_inline`
# so neither falls through to "re-parse as kernel".
_CAPTURE_CLASSIFIERS: tuple[_CaptureClassifier, ...] = (
    _classify_builtin,
    _classify_numpy_module,
    _classify_symbol,
    _classify_constant,
    _classify_callee,
    _classify_intrinsic,
    _classify_as_layout,
    _classify_index_map,
    _classify_inline,
)


def _is_builtin(name: str, value: Any) -> bool:
    return value is getattr(builtins, name, _UNDEFINED)


def _is_numpy_module(value: Any) -> bool:
    return getattr(value, "__name__", None) == "numpy" and hasattr(value, "ndarray")


def _is_symbol(value: Any) -> bool:
    # Lazy: `hc.symbols` not required on simulator path.
    from .symbols import Symbol

    return isinstance(value, Symbol)


def _symbol_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def _is_constant(value: Any) -> bool:
    # `bool` subclasses `int`; check first for accurate `python_kind`.
    return isinstance(value, bool | int | float | str)


def _python_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    raise FrontendError(f"internal: no python_kind for {value!r}")


def _repr_constant(value: Any) -> str:
    return repr(value)


def _is_kernel_func(value: Any) -> bool:
    return isinstance(getattr(value, "__hc_func__", None), FuncMetadata)


def _is_intrinsic(value: Any) -> bool:
    return isinstance(getattr(value, "__hc_intrinsic__", None), IntrinsicMetadata)


def _is_plain_callable(value: Any) -> bool:
    return callable(value) and not (
        _is_kernel_func(value) or _is_intrinsic(value) or _is_kernel(value)
    )


def _is_inlinable_helper(value: Any) -> bool:
    """True if `value` is a pure-Python helper re-parsable from source.

    `types.FunctionType` only. Decorated callables route via their own
    metadata; DSL primitives (`as_layout`) get a dedicated ref kind.
    """
    import types

    if not isinstance(value, types.FunctionType):
        return False
    if _is_kernel_func(value) or _is_intrinsic(value) or _is_kernel(value):
        return False
    if value is _dsl_as_layout:
        return False
    try:
        source_file = inspect.getsourcefile(value)
    except TypeError:
        return False
    return bool(source_file)


def _qualified_name(fn: Any) -> str:
    module = getattr(fn, "__module__", None) or "<unknown>"
    name = getattr(fn, "__qualname__", None) or _fn_name(fn)
    return f"{module}.{name}"


def _scope_ref_text(scope: Any) -> str:
    # Ref dict has a fixed string schema; spell "no scope" as "None".
    if scope is None:
        return "None"
    from .core import Scope

    if isinstance(scope, Scope):
        return scope.name
    name = getattr(scope, "__name__", None)
    if isinstance(name, str):
        return name
    return str(scope)


# --- MLIR helpers ------------------------------------------------------------


class _FunctionNamespace:
    """Lazy namespace snapshot; avoids recomputing closurevars per name."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self._nonlocals: Mapping[str, Any] | None = None
        self._globals: Mapping[str, Any] | None = None

    def _ensure(self) -> None:
        if self._nonlocals is not None and self._globals is not None:
            return
        try:
            cv = inspect.getclosurevars(self._fn)
            self._nonlocals = cv.nonlocals
        except TypeError:
            self._nonlocals = {}
        fn_globals = getattr(self._fn, "__globals__", None)
        self._globals = fn_globals if isinstance(fn_globals, Mapping) else {}

    def lookup(self, name: str) -> Any:
        self._ensure()
        assert self._nonlocals is not None
        assert self._globals is not None
        if name in self._nonlocals:
            return self._nonlocals[name]
        if name in self._globals:
            return self._globals[name]
        if name in vars(builtins):
            return vars(builtins)[name]
        return _UNDEFINED


def _read_ref(op: Any) -> Mapping[str, object] | None:
    if op is None or not hasattr(op, "attributes"):
        return None
    attrs = op.attributes
    if "ref" not in attrs:
        return None
    return _dict_attr_to_mapping(attrs["ref"])


def _dict_attr_to_mapping(attr: Any) -> dict[str, object]:
    out: dict[str, object] = {}
    for named in attr:
        key = named.name
        out[key] = _attr_to_python(named.attr)
    return out


def _attr_to_python(attr: Any) -> object:
    if hasattr(attr, "value") and isinstance(attr.value, str):
        return attr.value
    value = getattr(attr, "value", None)
    if isinstance(value, int | float):
        return value
    text = str(attr)
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def _str_attr_or_none(attrs: Any, key: str) -> str | None:
    if key not in attrs:
        return None
    value = attrs[key]
    inner = getattr(value, "value", None)
    if isinstance(inner, str):
        return inner
    text = str(value)
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def _loc_hint(op: Any) -> str:
    loc = getattr(op, "location", None)
    if loc is None:
        return "<unknown location>"
    return str(loc)
