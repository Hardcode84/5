# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypedIntAttr:
    """Width-annotated integer literal for recipe attributes.

    Plain Python `int`s default to `i64` because nothing in the recipe layer
    knows what the target op expects. Some upstream ops (notably
    `amdgpu.wmma`'s `m`/`n`/`k`) require `i32`; rather than guessing per-op
    in the lowering machinery, the recipe author opts in by wrapping the
    literal with `t.i32(...)` / `t.i64(...)` builder helpers.
    """

    value: int
    width: int = 64

    def to_record(self) -> dict[str, object]:
        return {"value": self.value, "width": self.width}


RecipeLiteral = str | int | float | bool | TypedIntAttr | None


@dataclass(frozen=True)
class RecipeValue:
    source: str
    key: str | int
    name: str

    def to_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "key": self.key,
            "name": self.name,
        }


NormalizedRecipeAttr = RecipeLiteral | RecipeValue
NormalizedRecipeType = str | RecipeValue


@dataclass(frozen=True)
class RecipeCreateStep:
    name: str
    op_name: str
    operands: tuple[RecipeValue, ...]
    result_types: tuple[NormalizedRecipeType, ...]
    attrs: tuple[tuple[str, NormalizedRecipeAttr], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "create",
            "name": self.name,
            "op_name": self.op_name,
            "operands": tuple(value.to_record() for value in self.operands),
            "result_types": tuple(_value_record(value) for value in self.result_types),
            "attrs": {name: _value_record(value) for name, value in self.attrs},
        }


@dataclass(frozen=True)
class RecipeRequireAttrStep:
    """Pre-rewrite assertion that a call's named attribute equals `expected`.

    Lowers to `transform.hc.require_intrinsic_attr` in the named_sequence body.
    Failure is definite (not silenceable): the apply aborts with a diagnostic
    pinpointing the call site, instead of silently degrading to the generic
    "no recipe matched" message that hides the underlying cause.
    """

    name: str
    expected: RecipeLiteral

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "require_attr",
            "name": self.name,
            "expected": _value_record(self.expected),
        }


@dataclass(frozen=True)
class RecipeConstantTypeStep:
    """Materializes a literal MLIR type as a recipe-side type handle.

    Lowers to `transform.hc.constant_type` in the named_sequence body. The
    builder dedupes by literal text so multiple `t.literal_type("vector<...>")`
    calls share a single payload op; recipe authors normally don't construct
    these directly — they pop out of `t.cast(value, to="vector<...>")` and
    `t.create(..., result_types=["vector<...>"])` callsites.
    """

    name: str
    type_literal: str

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "constant_type",
            "name": self.name,
            "type": self.type_literal,
        }


@dataclass(frozen=True)
class RecipeCastStep:
    """Bridges an operand value through `builtin.unrealized_conversion_cast`.

    Lowers to `transform.hc.cast_value`, which inserts a UCC at the source
    value's definition site (or skips when source/target types match). The
    motivating use is bare ↔ upstream bridging at intrinsic call boundaries:
    the recipe-inserted casts pair with the UCCs `hc-lower-launch-body`
    plants on either side of the call, so post-rewrite `--canonicalize`
    folds upstream → bare → upstream chains back to identity.
    """

    name: str
    source: RecipeValue
    target_type: RecipeValue

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "cast",
            "name": self.name,
            "source": self.source.to_record(),
            "target_type": self.target_type.to_record(),
        }


RecipeStep = (
    RecipeCreateStep | RecipeRequireAttrStep | RecipeConstantTypeStep | RecipeCastStep
)


@dataclass(frozen=True)
class CreatedOpHandle:
    step: RecipeCreateStep

    def result(self, index: int) -> RecipeValue:
        if index < 0 or index >= len(self.step.result_types):
            raise ValueError(
                f"created op {self.step.name!r} has no result at index {index}"
            )
        return RecipeValue(
            source="op_result",
            key=f"{self.step.name}:{index}",
            name=f"{self.step.name}_{index}",
        )


RecipeValueLike = RecipeValue | CreatedOpHandle
RecipeAttrValue = RecipeLiteral | RecipeValueLike
RecipeResultTypeValue = str | RecipeValueLike
# Recipe value sources whose payload SSA values are produced inside the
# named_sequence body (by create/constant_type/cast steps), as opposed to
# input handles derived from the matched call site. Kept as a frozenset so
# it can be reused by `_input_handles` without rebuilding per call.
_BODY_SOURCES: frozenset[str] = frozenset({"op_result", "cast", "const_type"})
_INPUT_SOURCES: frozenset[str] = frozenset({"operand", "result_type", "attr"})


def _input_only(values: Iterable[Any]) -> list[RecipeValue]:
    # Filter to call-derived handles. Non-`RecipeValue` entries (e.g. raw
    # `RecipeLiteral` attribute values) are silently dropped — they don't
    # carry an SSA handle the named-sequence prelude needs to materialize.
    return [
        value
        for value in values
        if isinstance(value, RecipeValue) and value.source in _INPUT_SOURCES
    ]


@dataclass(frozen=True)
class IntrinsicTransformRecipe:
    intrinsic_name: str
    target: str
    steps: tuple[RecipeStep, ...]
    replacement: tuple[RecipeValue, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "intrinsic": self.intrinsic_name,
            "target": self.target,
            "steps": tuple(step.to_record() for step in self.steps),
            "replacement": tuple(value.to_record() for value in self.replacement),
        }

    def to_module(self, *, context: Any | None = None) -> Any:
        # Standalone form: a fresh `transform.with_named_sequence` module
        # carrying just this recipe. Useful for tests, dumps, and any caller
        # that wants the recipe in isolation rather than packed alongside
        # peers in a shared lowerings module.
        return _TransformModuleBuilder(self, context=context).build_module()

    def to_mlir(self, *, context: Any | None = None) -> str:
        return str(self.to_module(context=context))

    def append_named_sequence(
        self, target_block: Any, *, context: Any | None = None
    ) -> Any:
        # Pack this recipe as a `transform.named_sequence` inside an existing
        # block — typically the body block of a shared `__hc_intrinsic_lowerings__`
        # module that aggregates every recipe from one frontend lowering pass.
        return _TransformModuleBuilder(self, context=context).append_named_sequence(
            target_block
        )

    def sequence_symbol_name(self) -> str:
        return _recipe_symbol_name(self)

    def _input_handles(self) -> tuple[RecipeValue, ...]:
        # Only call-derived handles (operand/result_type/attr) need their
        # `transform.hc.get_intrinsic_*` op emitted up front before the body
        # steps run. Body-produced values (op_result, cast, const_type) get
        # registered in the handle table when the corresponding step emits.
        result: list[RecipeValue] = []
        for step in self.steps:
            if isinstance(step, RecipeCreateStep):
                result.extend(_input_only(step.operands))
                result.extend(_input_only(step.result_types))
                result.extend(_input_only(value for _name, value in step.attrs))
            elif isinstance(step, RecipeCastStep):
                result.extend(_input_only((step.source, step.target_type)))
        result.extend(_input_only(self.replacement))
        return tuple(result)


@dataclass(frozen=True)
class IntrinsicRecipeCall:
    operand_names: tuple[str, ...]
    attr_names: frozenset[str]
    result_count: int
    # The owning builder, snapshot at recipe-build time. Optional because
    # tests can construct an `IntrinsicRecipeCall` standalone (e.g. for
    # operand-name validation) without involving a builder; only the
    # `expected_type` shortcut needs the builder reference.
    _builder: IntrinsicRecipeBuilder | None = None

    def operand(
        self,
        name_or_index: str | int,
        *,
        expected_type: str | RecipeValueLike | None = None,
    ) -> RecipeValue:
        if isinstance(name_or_index, int):
            index = name_or_index
            if index < 0 or index >= len(self.operand_names):
                raise ValueError(f"intrinsic operand index out of range: {index}")
            name = self.operand_names[index]
        else:
            name = name_or_index
            try:
                index = self.operand_names.index(name)
            except ValueError as exc:
                raise ValueError(f"unknown intrinsic operand {name!r}") from exc
        value = RecipeValue(
            source="operand",
            key=index,
            name=f"operand_{_symbol_part(name)}",
        )
        if expected_type is None:
            return value
        # Sugar around `t.cast(call.operand(name), to=expected_type)`. The
        # cast is a no-op at apply time when the actual operand type already
        # matches (the C++ side forwards the source value unchanged), so
        # passing `expected_type` is safe for recipes targeting both bridged
        # and not-yet-bridged pipeline configurations.
        if self._builder is None:
            raise RuntimeError(
                "call.operand expected_type requires a builder context; "
                "use IntrinsicRecipeBuilder via build_intrinsic_transform_recipe"
            )
        return self._builder.cast(value, to=expected_type)

    def result_type(self, index: int) -> RecipeValue:
        if index < 0 or index >= self.result_count:
            raise ValueError(f"intrinsic result type index out of range: {index}")
        return RecipeValue(
            source="result_type",
            key=index,
            name=f"result_type_{index}",
        )

    def attr(self, name: str) -> RecipeValue:
        if name not in self.attr_names:
            raise ValueError(f"unknown intrinsic constant attribute {name!r}")
        return RecipeValue(
            source="attr",
            key=name,
            name=f"attr_{_symbol_part(name)}",
        )


class IntrinsicRecipeBuilder:
    def __init__(self, *, attr_names: frozenset[str] = frozenset()) -> None:
        self._steps: list[RecipeStep] = []
        # Snapshot of the intrinsic's declared `const_attrs`, used by
        # `require_attr` to reject typos at recipe-build time. Empty for
        # builders constructed directly (e.g. tests); only the
        # `build_intrinsic_transform_recipe` entry point fills this in.
        self._attr_names = attr_names
        # Constant_type ops dedupe by literal text: every `t.literal_type`
        # call (including the implicit ones from `t.cast(..., to=str)` and
        # `t.create(result_types=[str])`) shares one payload op per type so
        # the recipe IR stays compact.
        self._constant_types: dict[str, RecipeConstantTypeStep] = {}

    @staticmethod
    def i32(value: int) -> TypedIntAttr:
        # Width-annotated literal; pair with a `static_attrs` slot whose
        # target op expects an `i32` attribute (e.g. `amdgpu.wmma`'s
        # `m`/`n`/`k`).
        return TypedIntAttr(value=int(value), width=32)

    @staticmethod
    def i64(value: int) -> TypedIntAttr:
        return TypedIntAttr(value=int(value), width=64)

    def literal_type(self, type_text: str) -> RecipeValue:
        """Materialize a literal MLIR type as a recipe-side type handle.

        Recipes use this to declare an upstream type they want to assert at
        create-op time (e.g. `vector<16xf16>` for `amdgpu.wmma`). The
        resulting handle is interchangeable with the type handle
        `call.result_type(N)` produces, so it can flow into either
        `t.create(result_types=[...])` or `t.cast(value, to=...)`.
        """
        if not type_text:
            raise ValueError("literal type text must be non-empty")
        existing = self._constant_types.get(type_text)
        if existing is not None:
            step = existing
        else:
            step = RecipeConstantTypeStep(
                name=f"const_type{len(self._constant_types)}",
                type_literal=type_text,
            )
            self._constant_types[type_text] = step
            self._steps.append(step)
        return RecipeValue(source="const_type", key=step.name, name=step.name)

    def cast(
        self,
        source: RecipeValueLike,
        *,
        to: str | RecipeValueLike,
    ) -> RecipeValue:
        """Bridge a value to a target type via `unrealized_conversion_cast`.

        `source` is any recipe value (call operand, prior create result,
        prior cast). `to` is either a literal type string (sugar that
        materializes a `constant_type` op behind the scenes) or another
        type handle (typically `call.result_type(N)` for casting back to a
        bare HC type). The C++ side skips the cast when source and target
        types already match, so writing `t.cast(...)` against a recipe that
        only sometimes needs the bridge stays cheap.
        """
        src = _coerce_value(source)
        target = self.literal_type(to) if isinstance(to, str) else _coerce_value(to)
        step = RecipeCastStep(
            name=f"cast{self._next_cast_index()}",
            source=src,
            target_type=target,
        )
        self._steps.append(step)
        return RecipeValue(source="cast", key=step.name, name=step.name)

    def create(
        self,
        op_name: str,
        *,
        operands: Sequence[RecipeValueLike] = (),
        result_types: Sequence[RecipeResultTypeValue] = (),
        attrs: Mapping[str, RecipeAttrValue] | None = None,
    ) -> CreatedOpHandle:
        if not op_name:
            raise ValueError("created op name must be non-empty")
        normalized_attrs = () if attrs is None else _normalize_attrs(attrs)
        step = RecipeCreateStep(
            name=f"created{self._next_create_index()}",
            op_name=op_name,
            operands=tuple(_coerce_value(value) for value in operands),
            result_types=tuple(self._coerce_type(value) for value in result_types),
            attrs=normalized_attrs,
        )
        self._steps.append(step)
        return CreatedOpHandle(step)

    def require_attr(
        self,
        call: IntrinsicRecipeCall,
        name: str,
        expected: RecipeLiteral,
    ) -> None:
        # The `call` parameter is the same view the recipe author already
        # uses for `call.operand(...)`/`call.attr(...)`; we accept it for API
        # symmetry and to validate the attribute exists on the intrinsic.
        if not isinstance(call, IntrinsicRecipeCall):
            raise TypeError(
                "require_attr expects the recipe call view as its first argument"
            )
        if self._attr_names and name not in self._attr_names:
            raise ValueError(f"unknown intrinsic constant attribute {name!r}")
        if isinstance(expected, CreatedOpHandle | RecipeValue):
            raise TypeError(
                "require_attr expected value must be a literal, not a handle"
            )
        coerced = _coerce_attr_value(expected)
        # Narrow back to literal: dynamic handles were rejected above.
        assert not isinstance(coerced, RecipeValue)
        self._steps.append(RecipeRequireAttrStep(name=name, expected=coerced))

    def finish(
        self,
        *,
        intrinsic_name: str,
        target: str,
        replacement: object,
    ) -> IntrinsicTransformRecipe:
        return IntrinsicTransformRecipe(
            intrinsic_name=intrinsic_name,
            target=target,
            steps=tuple(self._steps),
            replacement=_replacement_values(replacement),
        )

    def _next_create_index(self) -> int:
        return sum(1 for step in self._steps if isinstance(step, RecipeCreateStep))

    def _next_cast_index(self) -> int:
        return sum(1 for step in self._steps if isinstance(step, RecipeCastStep))

    def _coerce_type(self, value: RecipeResultTypeValue) -> RecipeValue:
        # `t.create(result_types=[...])` accepts literal type strings as
        # sugar for `t.literal_type(...)`. Coercing through the builder
        # keeps `_coerce_type` (the standalone helper) string-aware as well,
        # but here we route through `literal_type` so the dedupe table sees
        # every literal request.
        if isinstance(value, str):
            return self.literal_type(value)
        return _coerce_value(value)


def build_intrinsic_transform_recipe(
    callback: Callable[[IntrinsicRecipeBuilder, IntrinsicRecipeCall], object],
    *,
    intrinsic_name: str,
    target: str,
    operand_names: Sequence[str],
    attr_names: frozenset[str],
    result_count: int,
) -> IntrinsicTransformRecipe:
    builder = IntrinsicRecipeBuilder(attr_names=attr_names)
    call = IntrinsicRecipeCall(
        operand_names=tuple(operand_names),
        attr_names=attr_names,
        result_count=result_count,
        _builder=builder,
    )
    replacement = callback(builder, call)
    return builder.finish(
        intrinsic_name=intrinsic_name,
        target=target,
        replacement=replacement,
    )


def _replacement_values(value: object) -> tuple[RecipeValue, ...]:
    if value is None:
        return ()
    if isinstance(value, RecipeValue):
        return (value,)
    if isinstance(value, CreatedOpHandle):
        return tuple(
            value.result(index) for index in range(len(value.step.result_types))
        )
    if isinstance(value, tuple | list):
        return tuple(_coerce_value(item) for item in value)
    raise TypeError(f"unsupported intrinsic recipe replacement: {value!r}")


def _normalize_attrs(
    attrs: Mapping[str, RecipeAttrValue],
) -> tuple[tuple[str, NormalizedRecipeAttr], ...]:
    return tuple(
        (name, _coerce_attr_value(value)) for name, value in sorted(attrs.items())
    )


def _coerce_value(value: RecipeValueLike) -> RecipeValue:
    if isinstance(value, RecipeValue):
        return value
    if isinstance(value, CreatedOpHandle):
        if len(value.step.result_types) != 1:
            raise ValueError(
                f"created op {value.step.name!r} does not have exactly one result"
            )
        return value.result(0)
    raise TypeError(f"expected an intrinsic recipe value, got {value!r}")


def _coerce_attr_value(value: RecipeAttrValue) -> NormalizedRecipeAttr:
    if isinstance(value, CreatedOpHandle):
        return _coerce_value(value)
    if isinstance(value, RecipeValue):
        return value
    if isinstance(value, TypedIntAttr):
        return value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported intrinsic recipe attribute value: {value!r}")


def _value_record(value: NormalizedRecipeAttr | NormalizedRecipeType) -> object:
    if isinstance(value, RecipeValue):
        return value.to_record()
    if isinstance(value, TypedIntAttr):
        return value.to_record()
    return value


class _TransformModuleBuilder:
    def __init__(
        self, recipe: IntrinsicTransformRecipe, *, context: Any | None
    ) -> None:
        from .mlir import ir
        from .mlir.dialects import hc

        self.recipe = recipe
        self.ir = ir
        self.transform = importlib.import_module("hc_mlir.dialects.transform")
        self.context = ir.Context() if context is None else context
        hc.register_dialects(self.context)
        self._handles: dict[tuple[str, str | int], Any] = {}

    def build_module(self) -> Any:
        with self.context, self.ir.Location.unknown(self.context):
            module = self.ir.Module.create()
            module.operation.attributes["transform.with_named_sequence"] = (
                self.ir.UnitAttr.get(self.context)
            )
            with self.ir.InsertionPoint(module.body):
                self._build_named_sequence()
            module.operation.verify()
            return module

    def append_named_sequence(self, target_block: Any) -> Any:
        # The caller owns the parent module's `transform.with_named_sequence`
        # marker; we only emit the named_sequence op. Symbol uniqueness comes
        # from `_recipe_symbol_name(recipe)`, which encodes intrinsic + target.
        with (
            self.context,
            self.ir.Location.unknown(self.context),
            self.ir.InsertionPoint(target_block),
        ):
            return self._build_named_sequence()

    def _build_named_sequence(self) -> Any:
        any_op_type = self.transform.AnyOpType.get(self.context)
        sequence = self.transform.NamedSequenceOp(
            _recipe_symbol_name(self.recipe),
            [any_op_type],
            [],
        )
        # Outer `hc.target` lets the eventual interpreter pass index into the
        # lowerings module by target string without parsing each body.
        sequence.operation.attributes["hc.target"] = self.ir.StringAttr.get(
            self.recipe.target, self.context
        )
        with self.ir.InsertionPoint(sequence.body):
            call = self._create_match_op(sequence.bodyTarget)
            self._create_input_handles(call.result)
            for step in self.recipe.steps:
                self._emit_step(call.result, step)
            if self.recipe.replacement:
                self._create_replace_op(call.result)
            self.transform.YieldOp([])
        return sequence

    def _emit_step(self, call: Any, step: RecipeStep) -> None:
        if isinstance(step, RecipeCreateStep):
            self._create_payload_op(call, step)
            return
        if isinstance(step, RecipeRequireAttrStep):
            self._create_require_attr_op(call, step)
            return
        if isinstance(step, RecipeConstantTypeStep):
            self._create_constant_type_op(step)
            return
        if isinstance(step, RecipeCastStep):
            self._create_cast_value_op(step)
            return
        raise TypeError(f"unsupported intrinsic recipe step: {step!r}")

    def _create_constant_type_op(self, step: RecipeConstantTypeStep) -> None:
        type_attr = self.ir.TypeAttr.get(
            self.ir.Type.parse(step.type_literal, context=self.context)
        )
        op = self.ir.Operation.create(
            "transform.hc.constant_type",
            results=[self._transform_type_param_type()],
            attributes={"value": type_attr},
        )
        self._handles[("const_type", step.name)] = op.result

    def _create_cast_value_op(self, step: RecipeCastStep) -> None:
        source = self._handles[_value_key(step.source)]
        target = self._handles[_value_key(step.target_type)]
        op = self.ir.Operation.create(
            "transform.hc.cast_value",
            results=[self.transform.AnyValueType.get(self.context)],
            operands=[source, target],
        )
        self._handles[("cast", step.name)] = op.result

    def _create_require_attr_op(self, call: Any, step: RecipeRequireAttrStep) -> None:
        self.ir.Operation.create(
            "transform.hc.require_intrinsic_attr",
            operands=[call],
            attributes={
                "name": self.ir.StringAttr.get(step.name, self.context),
                "expected": self._literal_attr(step.expected),
            },
        )

    def _create_match_op(self, root: Any) -> Any:
        return self.ir.Operation.create(
            "transform.hc.match_intrinsic_call",
            results=[self.transform.AnyOpType.get(self.context)],
            operands=[root],
            attributes={
                "callee": self.ir.FlatSymbolRefAttr.get(
                    self.recipe.intrinsic_name,
                    self.context,
                ),
                "target": self.ir.StringAttr.get(self.recipe.target, self.context),
            },
        )

    def _create_input_handles(self, call: Any) -> None:
        for value in self.recipe._input_handles():
            key = _value_key(value)
            if key in self._handles:
                continue
            self._handles[key] = self._create_input_handle(call, value).result

    def _create_input_handle(self, call: Any, value: RecipeValue) -> Any:
        if value.source == "operand":
            return self._create_operand_handle(call, value)
        if value.source == "result_type":
            return self._create_result_type_handle(call, value)
        if value.source == "attr":
            return self._create_attr_handle(call, value)
        raise TypeError(f"unsupported intrinsic recipe input handle: {value!r}")

    def _create_operand_handle(self, call: Any, value: RecipeValue) -> Any:
        return self.ir.Operation.create(
            "transform.hc.get_intrinsic_operand",
            results=[self.transform.AnyValueType.get(self.context)],
            operands=[call],
            attributes={"index": self._i64_attr(int(value.key))},
        )

    def _create_result_type_handle(self, call: Any, value: RecipeValue) -> Any:
        return self.ir.Operation.create(
            "transform.hc.get_intrinsic_result_type",
            results=[self._transform_type_param_type()],
            operands=[call],
            attributes={"index": self._i64_attr(int(value.key))},
        )

    def _create_attr_handle(self, call: Any, value: RecipeValue) -> Any:
        return self.ir.Operation.create(
            "transform.hc.get_intrinsic_attr",
            results=[self.transform.AnyParamType.get(self.context)],
            operands=[call],
            attributes={"name": self.ir.StringAttr.get(str(value.key), self.context)},
        )

    def _create_payload_op(self, call: Any, step: RecipeCreateStep) -> None:
        result_types = _dynamic_result_types(step)
        dynamic_attrs, static_attrs = _split_attrs(step)
        operation = self.ir.Operation.create(
            "transform.hc.create_op",
            results=[
                self.transform.AnyValueType.get(self.context) for _value in result_types
            ],
            operands=[
                call,
                *self._lookup_all(step.operands),
                *self._lookup_all(result_types),
                *self._lookup_all(value for _name, value in dynamic_attrs),
            ],
            attributes=self._create_op_attrs(step, dynamic_attrs, static_attrs),
        )
        for index, result in enumerate(operation.results):
            self._handles[_value_key(step_result_value(step, index))] = result

    def _create_replace_op(self, call: Any) -> None:
        self.ir.Operation.create(
            "transform.hc.replace_intrinsic_call",
            operands=[call, *self._lookup_all(self.recipe.replacement)],
        )

    def _create_op_attrs(
        self,
        step: RecipeCreateStep,
        dynamic_attrs: tuple[tuple[str, RecipeValue], ...],
        static_attrs: tuple[tuple[str, RecipeLiteral], ...],
    ) -> dict[str, Any]:
        attrs = {
            "op_name": self.ir.StringAttr.get(step.op_name, self.context),
            "dynamic_attr_names": self.ir.ArrayAttr.get(
                [
                    self.ir.StringAttr.get(name, self.context)
                    for name, _value in dynamic_attrs
                ],
                self.context,
            ),
            "operandSegmentSizes": self.ir.DenseI32ArrayAttr.get(
                [1, len(step.operands), len(step.result_types), len(dynamic_attrs)],
                self.context,
            ),
        }
        if static_attrs:
            attrs["static_attrs"] = self.ir.DictAttr.get(
                {name: self._literal_attr(value) for name, value in static_attrs},
                self.context,
            )
        return attrs

    def _lookup_all(self, values: Iterable[RecipeValue]) -> tuple[Any, ...]:
        return tuple(self._handles[_value_key(value)] for value in values)

    def _literal_attr(self, value: RecipeLiteral) -> Any:
        if isinstance(value, TypedIntAttr):
            return self.ir.IntegerAttr.get(
                self.ir.IntegerType.get_signless(value.width, context=self.context),
                value.value,
            )
        # `bool` must precede `int`: Python booleans are an int subclass and
        # would otherwise serialize to a 64-bit integer attribute instead of
        # the intended `i1`.
        if isinstance(value, bool):
            return self.ir.BoolAttr.get(value, context=self.context)
        if isinstance(value, int):
            return self.ir.IntegerAttr.get(self._i64_type(), value)
        if isinstance(value, float):
            return self.ir.FloatAttr.get_f64(value, context=self.context)
        if isinstance(value, str):
            return self.ir.StringAttr.get(value, context=self.context)
        if value is None:
            return self.ir.UnitAttr.get(self.context)
        raise TypeError(f"unsupported intrinsic recipe static attribute: {value!r}")

    def _i64_attr(self, value: int) -> Any:
        return self.ir.IntegerAttr.get(self._i64_type(), value)

    def _i64_type(self) -> Any:
        return self.ir.IntegerType.get_signless(64, context=self.context)

    def _transform_type_param_type(self) -> Any:
        return self.ir.Type.parse("!transform.type", context=self.context)


def _dynamic_result_types(step: RecipeCreateStep) -> tuple[RecipeValue, ...]:
    result = []
    for value in step.result_types:
        if not isinstance(value, RecipeValue):
            raise TypeError("intrinsic transform recipes require symbolic result types")
        result.append(value)
    return tuple(result)


def _split_attrs(
    step: RecipeCreateStep,
) -> tuple[
    tuple[tuple[str, RecipeValue], ...],
    tuple[tuple[str, RecipeLiteral], ...],
]:
    dynamic_attrs = []
    static_attrs = []
    for name, value in step.attrs:
        if isinstance(value, RecipeValue):
            dynamic_attrs.append((name, value))
        else:
            static_attrs.append((name, value))
    return tuple(dynamic_attrs), tuple(static_attrs)


def step_result_value(step: RecipeCreateStep, index: int) -> RecipeValue:
    return RecipeValue(
        source="op_result",
        key=f"{step.name}:{index}",
        name=f"{step.name}_{index}",
    )


def _value_key(value: RecipeValue) -> tuple[str, str | int]:
    return value.source, value.key


def _recipe_symbol_name(recipe: IntrinsicTransformRecipe) -> str:
    return f"__hc_lower_{_symbol_part(f'{recipe.intrinsic_name}_{recipe.target}')}"


def _symbol_part(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    return "".join(chars).strip("_") or "value"
