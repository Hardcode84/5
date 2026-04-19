# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import inspect
import re

import pytest

from hc import CurrentGroup, WorkItem, kernel
from hc._frontend import (
    FrontendError,
    RecordingEmitter,
    lower_function,
    lower_module,
    lower_source,
)

_CONTROL_FLOW_SOURCE = """
@kernel.func(scope=WorkGroup)
def helper(group, acc):
    if acc[0] < 1:
        acc = acc[:, 0]
    else:
        acc = -acc
    for i in range(2):
        acc = acc + [i]
    return acc
"""

_CAPTURE_SOURCE = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group, acc):
    @group.workitems
    def lane(wi):
        tmp = wi.local_id()
        size = len(acc)
        for i in range(1):
            tmp = i
        return acc + tmp + size
    return lane()
"""

_NESTED_CAPTURE_SOURCE = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group, acc):
    @group.subgroups
    def wave(sg):
        @group.workitems
        def lane(wi):
            return acc
        return lane()
    return wave()
"""

_LATE_LOCAL_CAPTURE_SOURCE = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group):
    @group.workitems
    def lane(wi):
        return late
    late = 1
    return lane()
"""

_COMPREHENSION_POLLUTION_SOURCE = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group):
    @group.workitems
    def lane(wi):
        return mystery
    bogus = [0 for mystery in range(1)]
    return lane()
"""

_PARSE_ERROR_SOURCE = "def broken(:\n    pass\n"
_BAD_MODULE_SOURCE = """
@kernel(work_shape=(4,), group_shape=(4,))
def bad(group, x):
    while x:
        return x
    return x
"""
_SAMPLE_KERNEL_SPINE = [
    "module_begin",
    "kernel_begin",
    "assign_begin",
    "assign_end",
    "workitem_region_begin",
    "workitem_region_end",
    "return_begin",
    "call_begin",
    "call_end",
    "return_end",
    "kernel_end",
    "module_end",
]

_REJECTION_CASES = [
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        async def bad(group, x):
            return x
        """,
        "async functions are not supported",
        id="async-function",
    ),
    pytest.param(
        """
        def bad(group, x):
            return x
        """,
        "top-level functions must use @kernel, @kernel.func, or @kernel.intrinsic",
        id="missing-toplevel-decorator",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, x):
            def lane(wi):
                return x
            return lane(x)
        """,
        "nested functions must use @group.subgroups or @group.workitems",
        id="missing-region-decorator",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, items):
            for item in items:
                return item
            return items
        """,
        "only for-loops over bare range(...) are supported",
        id="non-range-loop",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, x):
            for i in range(1):
                return x
            else:
                return x
        """,
        "for-else is not supported",
        id="for-else",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, x):
            a = b = x
            return a
        """,
        "multiple assignment targets are not supported",
        id="multi-target-assign",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, *xs):
            return 0
        """,
        "varargs are not supported",
        id="varargs",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, /, x):
            return x
        """,
        "positional-only parameters are not supported",
        id="posonlyargs",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, x=0):
            return x
        """,
        "parameter defaults are not supported",
        id="parameter-default",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, kw):
            return use(**kw)
        """,
        "**kwargs are not supported",
        id="double-star-call",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, x):
            @group.workitems
            def lane(wi, extra):
                return x
            return lane(x)
        """,
        "collective regions must take exactly one parameter",
        id="region-arity",
    ),
    pytest.param(
        """
        @kernel(work_shape=(4,), group_shape=(4,))
        def bad(group, x):
            for i, j in range(2):
                return x
            return x
        """,
        "for-loop targets must be simple names",
        id="for-target-shape",
    ),
]


@kernel(work_shape=(4,), group_shape=(4,))
def _sample_kernel(group: CurrentGroup, x: int) -> int:
    tmp = x

    @group.workitems
    def lane(wi: WorkItem) -> int:
        return tmp

    return lane()


@kernel(work_shape=(4,), group_shape=(4,))
def _bad_file_kernel(group: CurrentGroup, x: int) -> int:
    while x:
        return x
    return x


def _event_kinds(emitter: RecordingEmitter) -> list[str]:
    return [event.kind for event in emitter.events]


def _payloads(
    emitter: RecordingEmitter,
    kind: str,
) -> list[dict[str, object]]:
    payloads = [event.payload for event in emitter.events if event.kind == kind]
    if payloads:
        return payloads
    raise AssertionError(f"missing event {kind!r}: {_event_kinds(emitter)}")


def _assert_contains_subsequence(
    values: list[str],
    expected: list[str],
) -> None:
    index = 0
    for value in values:
        if value == expected[index]:
            index += 1
            if index == len(expected):
                return
    raise AssertionError(f"missing subsequence {expected!r} in {values!r}")


def test_lower_function_records_kernel_and_collective_region() -> None:
    emitter = RecordingEmitter()

    lower_function(_sample_kernel, emitter)

    kinds = _event_kinds(emitter)
    kernel_payload = _payloads(emitter, "kernel_begin")[0]
    region_payload = _payloads(emitter, "workitem_region_begin")[0]

    _assert_contains_subsequence(kinds, _SAMPLE_KERNEL_SPINE)
    assert kernel_payload["name"] == "_sample_kernel"
    assert tuple(name for name, _ in kernel_payload["parameters"]) == ("group", "x")
    assert region_payload["captures"] == ("tmp",)
    assert tuple(name for name, _ in region_payload["parameters"]) == ("wi",)


def test_lower_source_records_structured_control_flow_and_slices() -> None:
    emitter = RecordingEmitter()

    lower_source(_CONTROL_FLOW_SOURCE, emitter, filename="testcase.py")

    kinds = _event_kinds(emitter)

    assert kinds[0] == "module_begin"
    assert kinds[-1] == "module_end"
    assert "func_begin" in kinds
    assert "if_begin" in kinds
    assert "condition_begin" in kinds
    assert "then_begin" in kinds
    assert "else_begin" in kinds
    assert "compare_begin" in kinds
    assert "subscript_begin" in kinds
    assert "slice_begin" in kinds
    assert "unaryop_begin" in kinds
    assert "for_range_begin" in kinds
    assert "iter_begin" in kinds
    assert "call_begin" in kinds
    assert "list_begin" in kinds


def test_nested_region_captures_see_all_enclosing_bindings() -> None:
    emitter = RecordingEmitter()

    lower_source(_NESTED_CAPTURE_SOURCE, emitter)

    assert _payloads(emitter, "subgroup_region_begin")[0]["captures"] == ()
    assert _payloads(emitter, "workitem_region_begin")[0]["captures"] == ("acc",)


def test_region_captures_include_later_supported_locals() -> None:
    emitter = RecordingEmitter()

    lower_source(_LATE_LOCAL_CAPTURE_SOURCE, emitter)

    assert _payloads(emitter, "workitem_region_begin")[0]["captures"] == ("late",)


def test_unsupported_comprehensions_do_not_pollute_capture_lists() -> None:
    emitter = RecordingEmitter()

    with pytest.raises(FrontendError, match="unsupported expression"):
        lower_source(_COMPREHENSION_POLLUTION_SOURCE, emitter)

    assert _payloads(emitter, "workitem_region_begin")[0]["captures"] == ()


def test_region_capture_list_ignores_region_locals_and_builtins() -> None:
    emitter = RecordingEmitter()

    lower_source(_CAPTURE_SOURCE, emitter)

    assert _payloads(emitter, "workitem_region_begin")[0]["captures"] == ("acc",)


def test_lower_function_reports_file_relative_diagnostics() -> None:
    emitter = RecordingEmitter()

    with pytest.raises(FrontendError, match="unsupported statement") as exc_info:
        lower_function(_bad_file_kernel, emitter)

    source_lines, start_line = inspect.getsourcelines(_bad_file_kernel)
    while_line = next(
        start_line + index
        for index, line in enumerate(source_lines)
        if line.lstrip().startswith("while ")
    )
    exc = exc_info.value

    assert exc.filename == __file__
    assert exc.lineno == while_line
    assert exc.offset == 5
    assert exc.text is not None
    assert exc.text.strip() == "while x:"


def test_lower_function_requires_importable_python_source() -> None:
    with pytest.raises(
        FrontendError,
        match="frontend source recovery requires an importable Python function",
    ):
        lower_function(len, RecordingEmitter())


def test_lower_module_accepts_preparsed_ast() -> None:
    emitter = RecordingEmitter()
    module = ast.parse(_CONTROL_FLOW_SOURCE)

    lower_module(module, emitter, filename="preparsed.py")

    assert _payloads(emitter, "func_begin")[0]["name"] == "helper"


def test_lower_module_uses_ast_coordinates_without_source_text() -> None:
    emitter = RecordingEmitter()
    module = ast.parse(_BAD_MODULE_SOURCE)
    while_stmt = module.body[0].body[0]
    before = (while_stmt.lineno, while_stmt.col_offset)

    with pytest.raises(FrontendError, match="unsupported statement") as exc_info:
        lower_module(module, emitter, filename="preparsed.py")

    exc = exc_info.value

    assert exc.filename == "preparsed.py"
    assert exc.lineno == before[0]
    assert exc.offset == before[1] + 1
    assert exc.text is None
    assert (while_stmt.lineno, while_stmt.col_offset) == before


def test_lower_source_wraps_parse_failures_with_coordinates() -> None:
    with pytest.raises(FrontendError, match="failed to parse source") as exc_info:
        lower_source(_PARSE_ERROR_SOURCE, RecordingEmitter(), filename="broken.py")

    exc = exc_info.value

    assert exc.filename == "broken.py"
    assert exc.lineno == 1
    assert exc.offset is not None


@pytest.mark.parametrize(("source", "match"), _REJECTION_CASES)
def test_lower_source_rejects_unsupported_constructs(
    source: str,
    match: str,
) -> None:
    with pytest.raises(FrontendError, match=re.escape(match)):
        lower_source(source, RecordingEmitter())
