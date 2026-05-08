# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

RecipeLiteral = str | int | float | bool | None


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

    def to_mlir_name(self) -> str:
        return f"%{self.name}"


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


@dataclass(frozen=True)
class IntrinsicTransformRecipe:
    intrinsic_name: str
    target: str
    steps: tuple[RecipeCreateStep, ...]
    replacement: tuple[RecipeValue, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "intrinsic": self.intrinsic_name,
            "target": self.target,
            "steps": tuple(step.to_record() for step in self.steps),
            "replacement": tuple(value.to_record() for value in self.replacement),
        }

    def to_mlir(self) -> str:
        symbol = _symbol_part(f"{self.intrinsic_name}_{self.target}")
        lines = [
            "module attributes {transform.with_named_sequence} {",
            "  transform.named_sequence "
            f"@__hc_lower_{symbol}(%root: !transform.any_op) {{",
            "    %call = transform.hc.match_intrinsic_call %root "
            f'@{self.intrinsic_name} target = "{self.target}" '
            ": (!transform.any_op) -> !transform.any_op",
        ]
        lines.extend(f"    {line}" for line in self._input_handle_mlir())
        lines.extend(f"    {self._create_step_mlir(step)}" for step in self.steps)
        if self.replacement:
            lines.append(f"    {self._replace_mlir()}")
        lines.append("    transform.yield")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    def _create_step_mlir(self, step: RecipeCreateStep) -> str:
        result_types = _dynamic_result_types(step)
        dynamic_attrs, static_attrs = _split_attrs(step)
        operands = ", ".join(value.to_mlir_name() for value in step.operands)
        result_type_names = ", ".join(value.to_mlir_name() for value in result_types)
        dynamic_attr_names = ", ".join(f'"{name}"' for name, _value in dynamic_attrs)
        dynamic_attr_values = ", ".join(
            value.to_mlir_name() for _name, value in dynamic_attrs
        )
        prefix = _create_step_result_prefix(step, result_types)
        return (
            f'{prefix}transform.hc.create_op "{step.op_name}" at %call '
            f"({operands}) result_types({result_type_names}) "
            f"dynamic_attrs [{dynamic_attr_names}]({dynamic_attr_values})"
            f"{_static_attrs_suffix(static_attrs)} : "
            f"{_create_step_functional_type(step, result_types, dynamic_attrs)}"
        )

    def _replace_mlir(self) -> str:
        results = ", ".join(value.to_mlir_name() for value in self.replacement)
        operand_types = ["!transform.any_op"]
        operand_types.extend("!transform.any_value" for _value in self.replacement)
        return (
            f"transform.hc.replace_intrinsic_call %call with {results} "
            f": ({', '.join(operand_types)}) -> ()"
        )

    def _input_handle_mlir(self) -> tuple[str, ...]:
        seen: set[tuple[str, str | int]] = set()
        lines: list[str] = []
        for value in self._input_handles():
            key = (value.source, value.key)
            if key in seen:
                continue
            seen.add(key)
            if value.source == "operand":
                lines.append(
                    f"{value.to_mlir_name()} = transform.hc.get_intrinsic_operand "
                    f'%call {{name = "{value.name.removeprefix("operand_")}"}} '
                    ": (!transform.any_op) -> !transform.any_value"
                )
            elif value.source == "result_type":
                lines.append(
                    f"{value.to_mlir_name()} = "
                    "transform.hc.get_intrinsic_result_type "
                    f"%call {{index = {value.key} : i64}} "
                    ": (!transform.any_op) -> !transform.type"
                )
            elif value.source == "attr":
                lines.append(
                    f"{value.to_mlir_name()} = transform.hc.get_intrinsic_attr "
                    f'%call {{name = "{value.key}"}} '
                    ": (!transform.any_op) -> !transform.any_param"
                )
        return tuple(lines)

    def _input_handles(self) -> tuple[RecipeValue, ...]:
        result: list[RecipeValue] = []
        for step in self.steps:
            result.extend(step.operands)
            result.extend(
                value for value in step.result_types if isinstance(value, RecipeValue)
            )
            result.extend(
                value for _name, value in step.attrs if isinstance(value, RecipeValue)
            )
        result.extend(
            value for value in self.replacement if value.source != "op_result"
        )
        return tuple(result)


@dataclass(frozen=True)
class IntrinsicRecipeCall:
    operand_names: tuple[str, ...]
    attr_names: frozenset[str]
    result_count: int

    def operand(self, name_or_index: str | int) -> RecipeValue:
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
        return RecipeValue(
            source="operand",
            key=index,
            name=f"operand_{_symbol_part(name)}",
        )

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
    def __init__(self) -> None:
        self._steps: list[RecipeCreateStep] = []

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
            name=f"created{len(self._steps)}",
            op_name=op_name,
            operands=tuple(_coerce_value(value) for value in operands),
            result_types=tuple(_coerce_type(value) for value in result_types),
            attrs=normalized_attrs,
        )
        self._steps.append(step)
        return CreatedOpHandle(step)

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


def build_intrinsic_transform_recipe(
    callback: Callable[[IntrinsicRecipeBuilder, IntrinsicRecipeCall], object],
    *,
    intrinsic_name: str,
    target: str,
    operand_names: Sequence[str],
    attr_names: frozenset[str],
    result_count: int,
) -> IntrinsicTransformRecipe:
    builder = IntrinsicRecipeBuilder()
    call = IntrinsicRecipeCall(
        operand_names=tuple(operand_names),
        attr_names=attr_names,
        result_count=result_count,
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
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported intrinsic recipe attribute value: {value!r}")


def _coerce_type(value: RecipeResultTypeValue) -> NormalizedRecipeType:
    if isinstance(value, str):
        return value
    return _coerce_value(value)


def _value_record(value: NormalizedRecipeAttr | NormalizedRecipeType) -> object:
    if isinstance(value, RecipeValue):
        return value.to_record()
    return value


def _dynamic_result_types(step: RecipeCreateStep) -> tuple[RecipeValue, ...]:
    result = []
    for value in step.result_types:
        if not isinstance(value, RecipeValue):
            raise TypeError(
                "textual intrinsic transform recipes require symbolic result types"
            )
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


def _create_step_result_prefix(
    step: RecipeCreateStep,
    result_types: tuple[RecipeValue, ...],
) -> str:
    result_names = ", ".join(
        step_result_name(step, index) for index in range(len(result_types))
    )
    return f"{result_names} = " if result_names else ""


def _static_attrs_suffix(static_attrs: tuple[tuple[str, RecipeLiteral], ...]) -> str:
    if not static_attrs:
        return ""
    attrs = ", ".join(
        f"{name} = {_format_mlir_static_attr(value)}" for name, value in static_attrs
    )
    return f" static_attrs = {{{attrs}}}"


def _create_step_functional_type(
    step: RecipeCreateStep,
    result_types: tuple[RecipeValue, ...],
    dynamic_attrs: tuple[tuple[str, RecipeValue], ...],
) -> str:
    operand_types = ["!transform.any_op"]
    operand_types.extend("!transform.any_value" for _value in step.operands)
    operand_types.extend("!transform.type" for _value in result_types)
    operand_types.extend("!transform.any_param" for _value in dynamic_attrs)
    result_transform_types = ", ".join(
        "!transform.any_value" for _value in result_types
    )
    return f"({', '.join(operand_types)}) -> ({result_transform_types})"


def _format_mlir_value(value: NormalizedRecipeAttr | NormalizedRecipeType) -> str:
    if isinstance(value, RecipeValue):
        return value.to_mlir_name()
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unit"
    return str(value)


def _format_mlir_static_attr(value: NormalizedRecipeAttr) -> str:
    if isinstance(value, RecipeValue):
        return value.to_mlir_name()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if value is None:
        return "unit"
    raise TypeError(f"unsupported intrinsic recipe static attribute: {value!r}")


def step_result_name(step: RecipeCreateStep, index: int) -> str:
    return RecipeValue(
        source="op_result",
        key=f"{step.name}:{index}",
        name=f"{step.name}_{index}",
    ).to_mlir_name()


def _symbol_part(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    return "".join(chars).strip("_") or "value"
