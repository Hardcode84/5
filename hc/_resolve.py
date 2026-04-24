# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python-side name resolution for `hc_front`.

Walks every `@kernel.func` / `@kernel.intrinsic` reachable from the compile
target, lowers them into one shared `hc_front` module, and stamps each
`hc_front.name` / `hc_front.attr` with a `ref` dict attribute classifying the
identifier. The eventual `hc_front -> hc` MLIR pass dispatches mechanically on
these attrs instead of reaching back into Python state.

Kinds recognized on `hc_front.name`:
    param / iv / local   — stamped by the frontend from scope state.
    constant             — captured int/float/bool/str.
    symbol               — captured ``hc.symbols.Symbol``.
    callee               — `@kernel.func` helper.
    intrinsic            — `@kernel.intrinsic` helper.
    inline               — undecorated Python helper.
    builtin              — ``range``, ``len``, etc.
    module               — whole-module alias (only ``numpy`` today).

Kinds recognized on `hc_front.attr`:
    dsl_method           — attribute access on a param/iv/local-rooted value.
    numpy_dtype_type     — attribute access on the numpy module that resolves
                           to a scalar dtype (``np.float16``); the pass decides
                           per-use whether the attr is called or passed as a
                           type descriptor.
    numpy_attr           — other attribute access on the numpy module
                           (``np.empty``); opaque helper the pass treats as an
                           inline call.

Attribute access rooted at any other classified base (constant, callee, dtype
chain, ...) is left unstamped: the pass can still see the base's own ref and
decide what to do without the driver guessing. Unresolvable plain-name loads
surface as `FrontendError` with file + line.
"""

from __future__ import annotations

import builtins
import inspect
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ._frontend import FrontendError, lower_functions_to_front_ir
from .core import FuncMetadata, IntrinsicMetadata, KernelMetadata

__all__ = [
    "ResolvedFrontIR",
    "resolve_front_ir",
]


def _numpy_dtype_name(module: Any, attr: str) -> str | None:
    """Return ``attr`` iff ``module.attr`` is a numpy scalar dtype.

    We resolve against the live numpy module instead of a hardcoded list so
    platform-specific aliases (``intp``, ``float128`` on 64-bit linux, ...)
    and any future scalar are recognized without edits here. A typo like
    ``np.flaot32`` (intentional misspelling) still fails: ``getattr`` returns
    ``None`` and the branch falls through to ``numpy_attr``.
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
    """Output of the resolver: combined module + the dep set it describes.

    ``functions`` interleaves decorated and inline helpers in BFS
    discovery order; ``inline_names`` exposes the set of identifiers
    that were emitted as `hc_front.func` with `ref.kind = "inline"`
    so tests and callers don't have to re-walk the IR to find them.
    """

    module: Any
    kernel_fn: Any
    functions: tuple[Any, ...]
    inline_names: frozenset[str]

    @property
    def symbol_names(self) -> tuple[str, ...]:
        return tuple(_fn_name(fn) for fn in self.functions)

    @property
    def decorated_symbol_names(self) -> tuple[str, ...]:
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
    """Transitively collect + lower + resolve names from ``kernel_fn``.

    The returned module contains one top-level op per reachable function
    (kernel first, then decorated helpers/intrinsics and undecorated
    inline helpers in discovery order), with every `hc_front.name` load
    carrying a `ref` DictAttr. Undecorated helpers surface as
    `hc_front.func` ops tagged `ref.kind = "inline"` so the
    `-hc-front-inline` pass can consume them. Raises ``FrontendError``
    when a captured name cannot be classified.
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
    """BFS for every decorated or inline helper reachable from ``kernel_fn``.

    Returns ``(ordered, inline_names)``: ``ordered`` starts with
    ``kernel_fn`` and then interleaves `@kernel.func` /
    `@kernel.intrinsic` callees and undecorated Python helpers (plain
    callables with retrievable source) in first-seen order.
    ``inline_names`` is the set of ``__name__``s that were discovered
    as inline helpers — the caller uses it to tag those top-levels at
    emission time.

    Deduplicated on ``id(fn)`` because two bindings may point at the
    same underlying function (re-export, alias). Inline helpers with
    the same ``__name__`` but different identities collide at
    emission (one top-level op per ``__name__``); this is loud by
    design — two `_tile_origin`s in one module would be ambiguous
    anyway, and the hc_front -> hc side keys lookups on the name.
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
    """Yield every decorated or plain-callable free reference of ``fn``.

    Plain callables are what become `hc_front.func` + `ref.kind =
    "inline"` top-levels. We yield them at the same level as decorated
    deps so the BFS picks up helpers-of-helpers too (e.g.
    `_lane_output_row_slice_args` references `_lane_output_row_step`).
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
    """Per-fn overrides pinning inline helpers to `hc_front.func` +
    `ref.kind = "inline"`.

    Keyed by ``id(fn)`` so the emitter can thread the override through
    without relying on the helper's `__name__` (which it also uses as
    the op's sym_name). The emitter skips the decorator sniff for any
    fn carrying ``force_kind``.
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
    """All free names in ``fn`` (and any nested ``def`` bodies inside it).

    ``inspect.getclosurevars`` only looks at the top-level function's code,
    so nested ``@group.workitems`` / ``@group.subgroups`` bodies would hide
    their deps. Walk co_consts transitively to pick those up too.

    This is intentionally a conservative over-approximation: a name string
    that happens to appear in a nested code object's ``co_names`` resolves
    against the outer fn's namespace. False matches only produce extra
    (harmless) deps in the lowered module; missing a real dep would break
    compilation, so we err on the inclusive side.
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

    # ``lower_functions_to_front_ir`` emits one top-level op per ``fn`` in the
    # order of ``fns`` (see hc/_frontend.py). Pair by index rather than by IR
    # ``sym_name``: two helpers with the same ``__name__`` would collide in a
    # name-keyed dict and silently mis-resolve the wrong region.
    ctx = module.context
    toplevels = list(module.body.operations)
    if len(toplevels) != len(fns):
        raise FrontendError(
            f"hc_front module has {len(toplevels)} top-level ops but the "
            f"resolver collected {len(fns)} Python fns; emission order broke"
        )
    for toplevel, fn in zip(toplevels, fns, strict=True):
        _OpClassifier(fn=fn, ctx=ctx, ir=_ir).classify(toplevel)


class _OpClassifier:
    """Walks one top-level region, stamps refs on captures + attrs."""

    def __init__(self, *, fn: Any, ctx: Any, ir: Any) -> None:
        self._fn = fn
        self._ctx = ctx
        self._ir = ir
        # Cache the namespace: closurevars is cheap but not free, and the same
        # ~200 names get looked up again and again for deeply-nested regions.
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
            return  # frontend already classified (param / iv / local).
        ctx_value = _str_attr_or_none(attrs, "ctx")
        if ctx_value != "load":
            return  # targets are never classified here.
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
            return  # Unclassified base (e.g. subscript) — pass inspects the
            # base op's own kind instead of a synthesized attr ref.
        method_name = _str_attr_or_none(op.operation.attributes, "name") or ""
        kind = base_ref.get("kind")
        if kind in {"param", "iv", "local"}:
            attrs["ref"] = self._dict_attr(
                {"kind": "dsl_method", "method": method_name}
            )
            return
        if kind == "module" and base_ref.get("module") == "numpy":
            # Dtypes need a dedicated kind so the pass can produce element
            # types without reimplementing numpy's type catalog; everything
            # else in ``np.*`` is opaque, pass handles as an inline call.
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
        # Chained attrs rooted in already-classified bases (constant, dtype,
        # callee, ...) aren't meaningful enough for the driver to stamp a
        # useful ref — leave it unstamped so the lowering pass can decide
        # from the base's kind or fail with a precise diagnostic at the
        # site where the attr is actually consumed.

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
        ir = self._ir
        if isinstance(value, str):
            return ir.StringAttr.get(value, context=self._ctx)
        if isinstance(value, bool) or isinstance(value, int):
            return ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64, context=self._ctx),
                int(value),
            )
        if isinstance(value, tuple):
            return ir.ArrayAttr.get(
                [self._to_attr(item) for item in value],
                context=self._ctx,
            )
        raise FrontendError(f"cannot encode ref payload value {value!r}")


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


# Ordered most-specific-first: builtins win over constants (``True`` is a
# builtin name rather than a random `1`), numpy wins over generic callables.
_CAPTURE_CLASSIFIERS: tuple[_CaptureClassifier, ...] = (
    _classify_builtin,
    _classify_numpy_module,
    _classify_symbol,
    _classify_constant,
    _classify_callee,
    _classify_intrinsic,
    _classify_inline,
)


def _is_builtin(name: str, value: Any) -> bool:
    return value is getattr(builtins, name, _UNDEFINED)


def _is_numpy_module(value: Any) -> bool:
    return getattr(value, "__name__", None) == "numpy" and hasattr(value, "ndarray")


def _is_symbol(value: Any) -> bool:
    # Lazy import: hc.symbols is heavy and not required on the simulator path.
    from .symbols import Symbol

    return isinstance(value, Symbol)


def _symbol_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def _is_constant(value: Any) -> bool:
    # `bool` is a subclass of `int`; check it first for an accurate python_kind.
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
    """True if ``value`` is a pure-Python helper we can re-parse from source.

    Filters out the long tail of other callables (C builtins, classes,
    bound methods, partials, numpy ufuncs, ...): the inliner frontend
    needs a ``__code__`` object backed by real source, and only
    ``types.FunctionType`` gives us that reliably. Decorated callables
    are also excluded — those take the `@kernel.func` / `@kernel.intrinsic`
    path with their own metadata.
    """
    import types

    if not isinstance(value, types.FunctionType):
        return False
    if _is_kernel_func(value) or _is_intrinsic(value) or _is_kernel(value):
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
    # A ref dict carries a fixed schema per kind, so we always emit a string
    # ``scope`` — unlike ``hc/_frontend._scope_text`` which returns ``None``
    # for unscoped ops and causes the emitter to drop the key. "None" is the
    # chosen spelling for "no scope" in ref payloads.
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
    """Lazy namespace snapshot; avoids recomputing closurevars per-name."""

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
