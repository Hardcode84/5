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

from build_tools.hc_native_tools import (
    ensure_hc_native_tools_built,
    export_hc_native_environment,
)
from build_tools.llvm_toolchain import ensure_llvm_toolchain
from hc import CurrentGroup, WorkItem, kernel
from hc._frontend import (
    FrontendError,
    lower_function_to_front_ir,
    lower_module_to_front_ir,
    lower_source_to_front_ir,
)

_SKIP_HC_FRONT_DIALECT_TESTS = pytest.mark.skipif(
    os.environ.get("HC_SKIP_HC_FRONT_DIALECT_TESTS") == "1",
    reason="native hc.front dialect smoke tests disabled by env",
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
        assert kernel_op.attributes["returns"].value == "int"
        assert _string_array_values(workitem_region.attributes["captures"]) == ["tmp"]
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
