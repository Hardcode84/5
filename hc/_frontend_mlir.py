# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ._frontend import FrontendEmitError
from .mlir import ir
from .mlir.dialects import hc_front


@dataclass
class _ValueFrame:
    kind: str
    payload: dict[str, object]
    values: list[Any] = field(default_factory=list)


@dataclass
class _OwnerFrame:
    kind: str
    op: Any


@dataclass
class _SyntheticRegionFrame:
    kind: str


_Frame = _ValueFrame | _OwnerFrame | _SyntheticRegionFrame
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


class HCFrontEmitter:
    """Build an `hc_front` MLIR module from frontend emitter callbacks."""

    def __init__(self, *, context: Any | None = None) -> None:
        self._context = ir.Context() if context is None else context
        hc_front.register_dialects(self._context)
        self._filename = "<unknown>"
        self._module: Any | None = None
        self._blocks: list[Any] = []
        self._frames: list[_Frame] = []
        self._value_type = hc_front.ValueType.get(self._context)

    @property
    def module(self) -> Any:
        if self._module is None:
            raise RuntimeError("frontend lowering did not produce an MLIR module")
        return self._module

    def begin_module(self, *, filename: str) -> None:
        self._filename = filename
        self._frames = []
        with ir.Location.unknown(self._context):
            self._module = ir.Module.create()
        self._blocks = [self._module.body]

    def end_module(self) -> None:
        if self._frames:
            raise RuntimeError("frontend lowering left unclosed operation frames")
        self._blocks = []

    def begin_op(self, kind: str, **payload: object) -> None:
        if kind == "kernel":
            self._begin_named_region_op(kind, hc_front.KernelOp, payload)
            return
        if kind == "func":
            self._begin_named_region_op(kind, hc_front.FuncOp, payload)
            return
        if kind == "intrinsic":
            self._begin_named_region_op(kind, hc_front.IntrinsicOp, payload)
            return
        if kind == "workitem_region":
            self._begin_capture_region_op(kind, hc_front.WorkitemRegionOp, payload)
            return
        if kind == "subgroup_region":
            self._begin_capture_region_op(kind, hc_front.SubgroupRegionOp, payload)
            return
        if kind == "if":
            op = self._insert_op(
                lambda loc: hc_front.IfOp(loc=loc),
                payload,
            )
            self._set_optional_bool_attr(
                op,
                "has_orelse",
                payload.get("has_orelse"),
            )
            self._frames.append(_OwnerFrame(kind=kind, op=op))
            return
        if kind == "for":
            op = self._insert_op(
                lambda loc: hc_front.ForOp(loc=loc),
                payload,
            )
            self._frames.append(_OwnerFrame(kind=kind, op=op))
            return
        if kind in {"condition", "then", "else", "target", "iter", "body"}:
            self._begin_synthetic_region(kind)
            return
        self._frames.append(_ValueFrame(kind=kind, payload=dict(payload)))

    def end_op(self, kind: str, **payload: object) -> None:
        if kind in {
            "kernel",
            "func",
            "intrinsic",
            "workitem_region",
            "subgroup_region",
        }:
            self._pop_block()
            self._pop_owner(kind)
            return
        if kind == "if":
            frame = self._pop_owner(kind)
            self._ensure_region_block(frame.op.condition)
            self._ensure_region_block(frame.op.thenRegion)
            self._ensure_region_block(frame.op.elseRegion)
            return
        if kind == "for":
            frame = self._pop_owner(kind)
            self._ensure_region_block(frame.op.target)
            self._ensure_region_block(frame.op.iter)
            self._ensure_region_block(frame.op.body)
            return
        if kind in {"condition", "then", "else", "target", "iter", "body"}:
            self._pop_block()
            self._pop_synthetic_region(kind)
            return
        value_frame = self._pop_value_frame(kind)
        value = self._build_value_frame(value_frame)
        if value is not None:
            self._append_value(value)

    def emit_op(self, kind: str, **payload: object) -> None:
        if kind == "constant":
            op = self._insert_op(
                lambda loc: hc_front.ConstantOp(
                    self._value_type,
                    self._constant_attr(payload["value"], payload),
                    loc=loc,
                ),
                payload,
            )
            self._set_optional_string_attr(
                op,
                "python_kind",
                self._constant_kind(payload["value"]),
            )
            self._append_value(op.result)
            return
        if kind == "name":
            op = self._insert_op(
                lambda loc: hc_front.NameOp(
                    self._value_type,
                    self._required_str(payload, "id"),
                    loc=loc,
                ),
                payload,
            )
            self._set_optional_string_attr(op, "ctx", payload.get("ctx"))
            self._set_optional_ref_attr(op, payload.get("ref"))
            self._append_value(op.result)
            return
        if kind == "target_name":
            op = self._insert_op(
                lambda loc: hc_front.TargetNameOp(
                    self._value_type,
                    self._required_str(payload, "id"),
                    loc=loc,
                ),
                payload,
            )
            self._append_value(op.result)
            return
        raise RuntimeError(f"unsupported leaf frontend op {kind!r}")

    def _begin_named_region_op(
        self,
        kind: str,
        constructor: Any,
        payload: dict[str, object],
    ) -> None:
        op = self._insert_op(
            lambda loc: constructor(
                self._required_str(payload, "name"),
                loc=loc,
            ),
            payload,
        )
        self._set_function_metadata_attrs(op, payload)
        self._frames.append(_OwnerFrame(kind=kind, op=op))
        self._blocks.append(op.body.blocks.append())

    def _begin_capture_region_op(
        self,
        kind: str,
        constructor: Any,
        payload: dict[str, object],
    ) -> None:
        captures = tuple(self._string_sequence(payload.get("captures")))
        op = self._insert_op(
            lambda loc: constructor(captures=captures, loc=loc),
            payload,
        )
        self._set_region_metadata_attrs(op, payload)
        self._frames.append(_OwnerFrame(kind=kind, op=op))
        self._blocks.append(op.body.blocks.append())

    def _begin_synthetic_region(self, kind: str) -> None:
        owner = self._nearest_owner()
        region = self._synthetic_region(owner, kind)
        self._frames.append(_SyntheticRegionFrame(kind=kind))
        self._blocks.append(region.blocks.append())

    def _build_value_frame(self, frame: _ValueFrame) -> Any | None:
        builder = getattr(self, f"_build_{frame.kind}", None)
        if builder is None:
            raise RuntimeError(f"unsupported structured frontend op {frame.kind!r}")
        return builder(frame.payload, frame.values)

    def _build_assign(self, payload: dict[str, object], values: list[Any]) -> None:
        self._expect_value_count("assign", values, 2)
        self._insert_op(
            lambda loc: hc_front.AssignOp(values[0], values[1], loc=loc),
            payload,
        )
        return

    def _build_aug_assign(self, payload: dict[str, object], values: list[Any]) -> None:
        self._expect_value_count("aug_assign", values, 2)
        self._insert_op(
            lambda loc: hc_front.AugAssignOp(
                self._required_str(payload, "op"),
                values[0],
                values[1],
                loc=loc,
            ),
            payload,
        )
        return

    def _build_return(self, payload: dict[str, object], values: list[Any]) -> None:
        self._insert_op(
            lambda loc: hc_front.ReturnOp(values, loc=loc),
            payload,
        )
        return

    def _build_attr(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_value_count("attr", values, 1)
        op = self._insert_op(
            lambda loc: hc_front.AttrOp(
                self._value_type,
                values[0],
                self._required_str(payload, "attr"),
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_subscript(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_min_value_count("subscript", values, 1)
        op = self._insert_op(
            lambda loc: hc_front.SubscriptOp(
                self._value_type,
                values[0],
                values[1:],
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_target_subscript(
        self,
        payload: dict[str, object],
        values: list[Any],
    ) -> Any:
        self._expect_min_value_count("target_subscript", values, 1)
        op = self._insert_op(
            lambda loc: hc_front.TargetSubscriptOp(
                self._value_type,
                values[0],
                values[1:],
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_slice(self, payload: dict[str, object], values: list[Any]) -> Any:
        op = self._insert_op(
            lambda loc: hc_front.SliceOp(
                self._value_type,
                values,
                loc=loc,
            ),
            payload,
        )
        self._set_optional_bool_attr(op, "has_lower", payload.get("has_lower"))
        self._set_optional_bool_attr(op, "has_upper", payload.get("has_upper"))
        self._set_optional_bool_attr(op, "has_step", payload.get("has_step"))
        return op.result

    def _build_call(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_min_value_count("call", values, 1)
        op = self._insert_op(
            lambda loc: hc_front.CallOp(
                self._value_type,
                values[0],
                values[1:],
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_keyword(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_value_count("keyword", values, 1)
        op = self._insert_op(
            lambda loc: hc_front.KeywordOp(
                self._value_type,
                self._required_str(payload, "name"),
                values[0],
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_tuple(self, payload: dict[str, object], values: list[Any]) -> Any:
        op = self._insert_op(
            lambda loc: hc_front.TupleOp(
                self._value_type,
                values,
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_list(self, payload: dict[str, object], values: list[Any]) -> Any:
        op = self._insert_op(
            lambda loc: hc_front.ListOp(
                self._value_type,
                values,
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_binop(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_value_count("binop", values, 2)
        op = self._insert_op(
            lambda loc: hc_front.BinOp(
                self._value_type,
                self._required_str(payload, "op"),
                values[0],
                values[1],
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_unaryop(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_value_count("unaryop", values, 1)
        op = self._insert_op(
            lambda loc: hc_front.UnaryOp(
                self._value_type,
                self._required_str(payload, "op"),
                values[0],
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_compare(self, payload: dict[str, object], values: list[Any]) -> Any:
        self._expect_min_value_count("compare", values, 2)
        op = self._insert_op(
            lambda loc: hc_front.CompareOp(
                self._value_type,
                self._string_array_attr(self._string_sequence(payload.get("ops"))),
                values,
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _build_target_tuple(self, payload: dict[str, object], values: list[Any]) -> Any:
        op = self._insert_op(
            lambda loc: hc_front.TargetTupleOp(
                self._value_type,
                values,
                loc=loc,
            ),
            payload,
        )
        return op.result

    def _insert_op(self, builder: Any, payload: dict[str, object]) -> Any:
        with ir.InsertionPoint(self._current_block()):
            return builder(self._location(payload))

    def _append_value(self, value: Any) -> None:
        for frame in reversed(self._frames):
            if isinstance(frame, _ValueFrame):
                frame.values.append(value)
                return

    def _current_block(self) -> Any:
        if not self._blocks:
            raise RuntimeError("frontend lowering has no active insertion block")
        return self._blocks[-1]

    def _location(self, payload: dict[str, object]) -> Any:
        line = payload.get("line")
        column = payload.get("column")
        if isinstance(line, int) and isinstance(column, int):
            return ir.Location.file(
                self._filename,
                line,
                column + 1,
                context=self._context,
            )
        return ir.Location.unknown(self._context)

    def _nearest_owner(self) -> _OwnerFrame:
        for frame in reversed(self._frames):
            if isinstance(frame, _OwnerFrame):
                return frame
        raise RuntimeError("frontend lowering expected a surrounding region owner")

    def _synthetic_region(self, owner: _OwnerFrame, kind: str) -> Any:
        if owner.kind == "if":
            if kind == "condition":
                return owner.op.condition
            if kind == "then":
                return owner.op.thenRegion
            if kind == "else":
                return owner.op.elseRegion
        if owner.kind == "for":
            if kind == "target":
                return owner.op.target
            if kind == "iter":
                return owner.op.iter
            if kind == "body":
                return owner.op.body
        raise RuntimeError(f"unexpected synthetic region {kind!r} under {owner.kind!r}")

    def _pop_block(self) -> None:
        if not self._blocks:
            raise RuntimeError("frontend lowering block stack underflow")
        self._blocks.pop()

    def _pop_owner(self, kind: str) -> _OwnerFrame:
        if not self._frames or not isinstance(self._frames[-1], _OwnerFrame):
            raise RuntimeError(f"frontend lowering expected owner frame for {kind!r}")
        frame = self._frames.pop()
        frame = cast(_OwnerFrame, frame)
        if frame.kind != kind:
            raise RuntimeError(
                f"frontend lowering closed {kind!r} while {frame.kind!r} was active"
            )
        return frame

    def _pop_synthetic_region(self, kind: str) -> _SyntheticRegionFrame:
        if not self._frames or not isinstance(self._frames[-1], _SyntheticRegionFrame):
            raise RuntimeError(
                f"frontend lowering expected synthetic region frame for {kind!r}"
            )
        frame = self._frames.pop()
        frame = cast(_SyntheticRegionFrame, frame)
        if frame.kind != kind:
            raise RuntimeError(
                f"frontend lowering closed {kind!r} while {frame.kind!r} was active"
            )
        return frame

    def _pop_value_frame(self, kind: str) -> _ValueFrame:
        if not self._frames or not isinstance(self._frames[-1], _ValueFrame):
            raise RuntimeError(f"frontend lowering expected value frame for {kind!r}")
        frame = self._frames.pop()
        frame = cast(_ValueFrame, frame)
        if frame.kind != kind:
            raise RuntimeError(
                f"frontend lowering closed {kind!r} while {frame.kind!r} was active"
            )
        return frame

    def _ensure_region_block(self, region: Any) -> None:
        if len(region.blocks) == 0:
            region.blocks.append()

    def _constant_attr(self, value: object, payload: dict[str, object]) -> Any:
        if isinstance(value, bool):
            return ir.BoolAttr.get(value, context=self._context)
        if isinstance(value, int):
            if not _I64_MIN <= value <= _I64_MAX:
                raise self._frontend_error(
                    payload,
                    f"integer literal {value} is outside the supported signed "
                    "64-bit range",
                )
            i64 = ir.IntegerType.get_signless(64, context=self._context)
            return ir.IntegerAttr.get(i64, value)
        if isinstance(value, float):
            return ir.FloatAttr.get_f64(value, context=self._context)
        if isinstance(value, str):
            return ir.StringAttr.get(value, context=self._context)
        return ir.StringAttr.get(repr(value), context=self._context)

    def _frontend_error(
        self,
        payload: dict[str, object],
        message: str,
    ) -> FrontendEmitError:
        line = payload.get("line")
        column = payload.get("column")
        return FrontendEmitError(
            message,
            line=line if isinstance(line, int) else None,
            column=column if isinstance(column, int) else None,
        )

    def _constant_kind(self, value: object) -> str | None:
        if isinstance(value, bool | int | float | str):
            return None
        return type(value).__name__

    def _set_function_metadata_attrs(
        self,
        op: Any,
        payload: dict[str, object],
    ) -> None:
        self._set_optional_string_array_attr(
            op,
            "decorators",
            payload.get("decorators"),
        )
        self._set_optional_parameters_attr(
            op,
            payload.get("parameters"),
            payload.get("parameter_annotations"),
        )
        self._set_optional_string_attr(op, "returns", payload.get("returns"))
        self._set_optional_toplevel_metadata_attrs(op, payload.get("metadata"))
        # `ref` on a top-level op is used by `-hc-front-inline` to spot
        # undecorated helpers tagged `{kind = "inline"}` — same payload
        # shape as the per-site `ref` stamped by the resolver on name
        # ops, just rehomed on the function itself.
        self._set_optional_ref_attr(op, payload.get("ref"))

    def _set_region_metadata_attrs(
        self,
        op: Any,
        payload: dict[str, object],
    ) -> None:
        self._set_optional_string_array_attr(
            op,
            "decorators",
            payload.get("decorators"),
        )
        self._set_optional_parameters_attr(
            op,
            payload.get("parameters"),
            payload.get("parameter_annotations"),
        )
        # The region's source-level name (the inner `def`'s identifier).
        # `-hc-front-fold-region-defs` uses it to pair a region with the
        # ghost `hc_front.name {ref = {kind = "local"}} + hc_front.call`
        # trail emitted when Python writes `inner()` right after
        # `def inner(...)`.
        self._set_optional_string_attr(op, "name", payload.get("name"))

    def _set_optional_parameters_attr(
        self,
        op: Any,
        value: object,
        annotations: object = None,
    ) -> None:
        if value is None:
            return
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise RuntimeError(f"invalid frontend parameter records: {value!r}")
        annotation_records = self._coerce_parameter_annotations(annotations)
        parameters = []
        for item in value:
            if not isinstance(item, tuple) or len(item) != 2:
                raise RuntimeError(f"invalid frontend parameter record: {item!r}")
            name, annotation = item
            if not isinstance(name, str):
                raise RuntimeError(f"invalid frontend parameter name: {name!r}")
            parameter = {"name": self._string_attr(name)}
            if isinstance(annotation, str):
                parameter["annotation"] = self._string_attr(annotation)
            record = annotation_records.get(name)
            if record is not None:
                self._apply_structural_annotation(parameter, record)
            parameters.append(ir.DictAttr.get(parameter, context=self._context))
        op.operation.attributes["parameters"] = ir.ArrayAttr.get(
            parameters,
            context=self._context,
        )

    def _coerce_parameter_annotations(
        self,
        annotations: object,
    ) -> dict[str, dict[str, object]]:
        if annotations is None:
            return {}
        if not isinstance(annotations, Mapping):
            raise RuntimeError(
                f"invalid frontend parameter annotation mapping: {annotations!r}"
            )
        return {str(key): dict(value) for key, value in annotations.items()}

    def _apply_structural_annotation(
        self,
        parameter: dict[str, Any],
        record: Mapping[str, object],
    ) -> None:
        kind = record.get("kind")
        if isinstance(kind, str):
            parameter["kind"] = self._string_attr(kind)
        dtype = record.get("dtype")
        if isinstance(dtype, str):
            parameter["dtype"] = self._string_attr(dtype)
        shape = record.get("shape")
        if shape is not None:
            parameter["shape"] = self._string_array_attr(self._string_sequence(shape))

    def _set_optional_toplevel_metadata_attrs(
        self,
        op: Any,
        metadata: object,
    ) -> None:
        if metadata is None:
            return
        if not isinstance(metadata, Mapping):
            raise RuntimeError(
                f"invalid frontend toplevel metadata payload: {metadata!r}"
            )
        if "work_shape" in metadata:
            self._set_optional_string_array_attr(
                op, "work_shape", metadata["work_shape"]
            )
        if "group_shape" in metadata:
            self._set_optional_string_array_attr(
                op, "group_shape", metadata["group_shape"]
            )
        if "subgroup_size" in metadata:
            self._set_optional_i32_attr(op, "subgroup_size", metadata["subgroup_size"])
        if "literals" in metadata:
            self._set_optional_string_array_attr(op, "literals", metadata["literals"])
        if "scope" in metadata:
            self._set_optional_string_attr(op, "scope", metadata["scope"])
        if "effects" in metadata:
            self._set_optional_string_attr(op, "effects", metadata["effects"])
        if "const_kwargs" in metadata:
            self._set_optional_string_array_attr(
                op, "const_kwargs", metadata["const_kwargs"]
            )

    def _set_optional_i32_attr(self, op: Any, name: str, value: object) -> None:
        # Symmetrical with sibling _set_optional_* helpers: loud on bad
        # payload so a serializer regression is caught at emit time, not
        # in the eventual hc_front -> hc pass.
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"frontend metadata '{name}' must be a non-bool int, got {value!r}"
            )
        op.operation.attributes[name] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32, context=self._context),
            value,
        )

    def _set_optional_string_attr(self, op: Any, name: str, value: object) -> None:
        if isinstance(value, str):
            op.operation.attributes[name] = self._string_attr(value)

    def _set_optional_bool_attr(self, op: Any, name: str, value: object) -> None:
        if isinstance(value, bool):
            op.operation.attributes[name] = ir.BoolAttr.get(
                value,
                context=self._context,
            )

    def _set_optional_ref_attr(self, op: Any, value: object) -> None:
        # `ref` keys are always strings; values are the JSON-like leaves
        # accepted by ``_ref_value_attr`` (str / int / bool / sequence[str]).
        # Kinds are disambiguated by their payload shape during the
        # hc_front -> hc pass, not inside hc_front itself.
        if value is None:
            return
        if not isinstance(value, Mapping):
            raise RuntimeError(f"expected ref dict, got {value!r}")
        entries: list[tuple[str, Any]] = []
        for raw_key, raw_val in value.items():
            if not isinstance(raw_key, str):
                raise RuntimeError(f"ref keys must be strings, got {raw_key!r}")
            entries.append((raw_key, self._ref_value_attr(raw_val)))
        op.operation.attributes["ref"] = ir.DictAttr.get(
            dict(entries),
            context=self._context,
        )

    def _ref_value_attr(self, value: object) -> Any:
        if isinstance(value, str):
            return self._string_attr(value)
        if isinstance(value, bool | int):
            # Drop bools together with ints — both serialize via i64 so the
            # pass gets stable numeric payloads.
            return ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64, context=self._context),
                int(value),
            )
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return self._string_array_attr(self._string_sequence(value))
        raise RuntimeError(f"unsupported ref payload value {value!r}")

    def _set_optional_string_array_attr(
        self,
        op: Any,
        name: str,
        value: object,
    ) -> None:
        if value is None:
            return
        op.operation.attributes[name] = self._string_array_attr(
            self._string_sequence(value)
        )

    def _string_array_attr(self, values: Sequence[str]) -> Any:
        return ir.ArrayAttr.get(
            [self._string_attr(value) for value in values],
            context=self._context,
        )

    def _string_attr(self, value: str) -> Any:
        return ir.StringAttr.get(value, context=self._context)

    def _string_sequence(self, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise RuntimeError(f"expected string sequence, got {value!r}")
        result = []
        for item in value:
            if not isinstance(item, str):
                raise RuntimeError(f"expected string sequence, got {item!r}")
            result.append(item)
        return tuple(result)

    def _required_str(self, payload: dict[str, object], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str):
            raise RuntimeError(
                f"frontend payload {key!r} must be a string, got {value!r}"
            )
        return value

    def _expect_value_count(self, kind: str, values: Sequence[Any], count: int) -> None:
        if len(values) != count:
            raise RuntimeError(
                f"frontend {kind!r} expected {count} child values, got {len(values)}"
            )

    def _expect_min_value_count(
        self,
        kind: str,
        values: Sequence[Any],
        count: int,
    ) -> None:
        if len(values) < count:
            raise RuntimeError(
                f"frontend {kind!r} expected at least {count} child values, got "
                f"{len(values)}"
            )
