# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from functools import lru_cache
from typing import Any, cast

import pytest
from frontend_return_fixtures import KERNEL_RETURN_NONE_SOURCE

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.llvm_toolchain import ensure_llvm_toolchain
from hc import Buffer, CurrentGroup, WorkGroup, WorkItem, kernel, sym
from hc._frontend import (
    FrontendError,
    lower_function_to_front_ir,
    lower_module_to_front_ir,
    lower_source_to_front_ir,
)
from hc.symbols import ceil_div

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc_front dialect smoke tests disabled by env",
)

_CONTROL_FLOW_SOURCE = textwrap.dedent("""\
    @kernel.func(scope=WorkGroup)
    def helper(group, acc):
        if acc:
            acc = -acc
        for i in range(1):
            acc += [i]
        return acc
    """)

_IF_WITHOUT_ELSE_SOURCE = textwrap.dedent("""\
    @kernel(work_shape=(4,), group_shape=(4,))
    def demo(group, x):
        if x:
            return x
        return x
    """)


@kernel(work_shape=(4,), group_shape=(4,))
def _sample_kernel(group: CurrentGroup, x: int) -> int:
    tmp = x

    @group.workitems
    def lane(wi: WorkItem) -> int:
        return tmp

    return cast(int, lane())


_M = sym.M
_K = sym.K


# Fixture exercising every decorator kwarg surfaced to `hc_front`: symbolic
# `work_shape`, concrete `group_shape`, `subgroup_size`, and `literals`.
@kernel(
    work_shape=(ceil_div(_M, 16) * 32, ceil_div(_K, 16)),
    group_shape=(32, 1),
    subgroup_size=32,
    literals={sym.WMMA_M, sym.WMMA_K},
)
def _metadata_kernel(
    group: CurrentGroup,
    a: Buffer[_M, _K],
    b: int,
    n: _M,
) -> None: ...


@kernel.func(scope=WorkGroup)
def _metadata_helper(group: CurrentGroup, x: int) -> int:
    return x


@kernel.intrinsic(
    scope=WorkItem,
    effects="pure",
    const_attrs={"wave_size", "arch"},
)
def _metadata_intrinsic(group, *, wave_size, arch):  # pragma: no cover - metadata-only
    ...


@lru_cache(maxsize=1)
def _ensure_hc_front_bindings_available() -> None:
    llvm_install_root = ensure_llvm_toolchain()
    native_install_root = ensure_hc_native_tools_built(llvm_install_root)
    os.environ.update(
        export_hc_native_environment(native_install_root, dict(os.environ))
    )


def _string_array_values(attr: Any) -> list[str]:
    return [value.value for value in attr]


def _parameter_records(op: Any) -> list[tuple[str, str | None]]:
    return [
        (
            parameter["name"].value,
            parameter["annotation"].value if "annotation" in parameter else None,
        )
        for parameter in op.attributes["parameters"]
    ]


def _parameter_passing_records(op: Any) -> list[str]:
    return [parameter["passing"].value for parameter in op.attributes["parameters"]]


def _parameter_structural_records(op: Any) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for parameter in op.attributes["parameters"]:
        record: dict[str, object] = {}
        for key in ("kind", "dtype", "launch_context"):
            if key in parameter:
                record[key] = parameter[key].value
        if "shape" in parameter:
            record["shape"] = _string_array_values(parameter["shape"])
        records.append(record)
    return records


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_lower_function_to_front_ir_builds_kernel_module() -> None:
    _ensure_hc_front_bindings_available()

    from hc.mlir import ir
    from hc.mlir.dialects import hc_front

    with ir.Context() as context:
        module = lower_function_to_front_ir(_sample_kernel, context=context)

        kernel_op = module.body.operations[0]
        body_ops = list(kernel_op.body.blocks[0].operations)
        workitem_region = next(
            op for op in body_ops if isinstance(op, hc_front.WorkitemRegionOp)
        )
        source_lines, start_line = inspect.getsourcelines(_sample_kernel)
        def_line = start_line + next(
            index
            for index, line in enumerate(source_lines)
            if line.lstrip().startswith("def ")
        )

        assert isinstance(kernel_op, hc_front.KernelOp)
        assert _string_array_values(kernel_op.attributes["decorators"]) == ["kernel"]
        assert _parameter_records(kernel_op) == [
            ("group", "CurrentGroup"),
            ("x", "int"),
        ]
        assert _parameter_passing_records(kernel_op) == ["positional", "positional"]
        # Decorator kwargs arrive as builtin attrs on the op: shape axes as
        # string arrays for later `#hc.shape` assembly, and `subgroup_size`
        # absent because the sample kernel does not declare it.
        assert _string_array_values(kernel_op.attributes["work_shape"]) == ["4"]
        assert _string_array_values(kernel_op.attributes["group_shape"]) == ["4"]
        assert "subgroup_size" not in kernel_op.attributes
        assert "literals" not in kernel_op.attributes
        assert _parameter_structural_records(kernel_op) == [
            {"kind": "launch_context", "launch_context": "group"},
            {"kind": "scalar", "dtype": "int"},
        ]
        assert kernel_op.attributes["returns"].value == "int"
        assert _string_array_values(workitem_region.attributes["captures"]) == ["tmp"]
        # Pinned so `-hc-front-fold-region-defs` can pair the region op
        # with the ghost `name {ref.kind = "local"} + call` trail below.
        assert workitem_region.attributes["name"].value == "lane"
        assert str(kernel_op.location) == f'loc("{__file__}":{def_line}:1)'


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_lower_source_to_front_ir_preserves_control_flow_metadata() -> None:
    _ensure_hc_front_bindings_available()

    from hc.mlir import ir
    from hc.mlir.dialects import hc_front

    with ir.Context() as context:
        module = lower_source_to_front_ir(
            _CONTROL_FLOW_SOURCE,
            filename="control.py",
            context=context,
        )

        func_op = module.body.operations[0]
        body_ops = list(func_op.body.blocks[0].operations)

        assert isinstance(func_op, hc_front.FuncOp)
        assert _string_array_values(func_op.attributes["decorators"]) == ["kernel.func"]
        assert _parameter_records(func_op) == [("group", None), ("acc", None)]
        assert _parameter_passing_records(func_op) == ["positional", "positional"]
        assert any(isinstance(op, hc_front.IfOp) for op in body_ops)
        assert any(isinstance(op, hc_front.ForOp) for op in body_ops)


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_lower_module_to_front_ir_round_trips_if_without_else() -> None:
    _ensure_hc_front_bindings_available()

    from hc.mlir import ir
    from hc.mlir.dialects import hc_front

    parsed = ast.parse(_IF_WITHOUT_ELSE_SOURCE)

    with ir.Context() as context:
        module = lower_module_to_front_ir(
            parsed,
            filename="preparsed.py",
            context=context,
        )
        kernel_op = module.body.operations[0]
        if_op = kernel_op.body.blocks[0].operations[0]

        assert isinstance(kernel_op, hc_front.KernelOp)
        assert isinstance(if_op, hc_front.IfOp)
        assert str(module.body.operations[0].location) == 'loc("preparsed.py":2:1)'
        assert if_op.attributes["has_orelse"].value is False

        round_tripped = ir.Module.parse(str(module), context=context)
        round_tripped_if = round_tripped.body.operations[0].body.blocks[0].operations[0]

        assert isinstance(round_tripped_if, hc_front.IfOp)
        assert round_tripped_if.attributes["has_orelse"].value is False


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_lower_source_to_front_ir_emits_bare_return_for_return_none() -> None:
    _ensure_hc_front_bindings_available()

    from hc.mlir import ir
    from hc.mlir.dialects import hc_front

    with ir.Context() as context:
        module = lower_source_to_front_ir(
            KERNEL_RETURN_NONE_SOURCE,
            filename="return_none.py",
            context=context,
        )

        kernel_op = module.body.operations[0]
        body_ops = list(kernel_op.body.blocks[0].operations)
        assert len(body_ops) == 1
        assert isinstance(body_ops[0], hc_front.ReturnOp)
        assert len(body_ops[0].operation.operands) == 0


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_lower_function_to_front_ir_emits_decorator_metadata() -> None:
    _ensure_hc_front_bindings_available()

    from hc.mlir import ir
    from hc.mlir.dialects import hc_front

    with ir.Context() as context:
        module = lower_function_to_front_ir(_metadata_kernel, context=context)

        kernel_op = module.body.operations[0]

        assert isinstance(kernel_op, hc_front.KernelOp)
        # Symbolic expressions serialize via ixsimpl's canonical form, so the
        # pretty text we pin here is whatever `str(expr)` emits — not the
        # Python source `ceil_div(M, 16) * 32`.
        assert _string_array_values(kernel_op.attributes["work_shape"]) == [
            "32*ceiling(1/16*M)",
            "ceiling(1/16*K)",
        ]
        assert _string_array_values(kernel_op.attributes["group_shape"]) == ["32", "1"]
        assert kernel_op.attributes["subgroup_size"].value == 32
        assert str(kernel_op.attributes["subgroup_size"].type) == "i32"
        # `literals` is a `frozenset` on the Python side; the emitter sorts
        # it for stable round-trip.
        assert _string_array_values(kernel_op.attributes["literals"]) == [
            "WMMA_K",
            "WMMA_M",
        ]
        assert _parameter_structural_records(kernel_op) == [
            {"kind": "launch_context", "launch_context": "group"},
            {"kind": "buffer", "shape": ["M", "K"]},
            {"kind": "scalar", "dtype": "int"},
            {"kind": "symbol"},
        ]

    with ir.Context() as context:
        module = lower_function_to_front_ir(_metadata_helper, context=context)
        func_op = module.body.operations[0]

        assert isinstance(func_op, hc_front.FuncOp)
        assert func_op.attributes["scope"].value == "WorkGroup"
        assert "effects" not in func_op.attributes
        assert "const_kwargs" not in func_op.attributes

    with ir.Context() as context:
        module = lower_function_to_front_ir(_metadata_intrinsic, context=context)
        intrinsic_op = module.body.operations[0]

        assert isinstance(intrinsic_op, hc_front.IntrinsicOp)
        assert intrinsic_op.attributes["scope"].value == "WorkItem"
        assert intrinsic_op.attributes["effects"].value == "pure"
        assert _string_array_values(intrinsic_op.attributes["const_kwargs"]) == [
            "arch",
            "wave_size",
        ]
        assert _parameter_passing_records(intrinsic_op) == [
            "positional",
            "keyword_only",
            "keyword_only",
        ]


@_SKIP_HC_FRONT_DIALECT_TESTS
def test_lower_source_to_front_ir_reports_oversized_integer_literals() -> None:
    _ensure_hc_front_bindings_available()

    huge = 2**80
    source = textwrap.dedent(f"""\
        @kernel(work_shape=(4,), group_shape=(4,))
        def demo(group):
            return {huge}
        """)

    with pytest.raises(
        FrontendError,
        match="outside the supported signed 64-bit range",
    ) as exc_info:
        lower_source_to_front_ir(source, filename="huge.py")

    exc = exc_info.value

    assert exc.filename == "huge.py"
    assert exc.lineno == 3
    assert exc.offset == 12
    assert exc.text is not None
    assert exc.text.strip() == f"return {huge}"
