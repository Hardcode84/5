# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import inspect
import re

import numpy as np
import pytest
from frontend_return_fixtures import (
    FUNC_RETURN_NONE_SOURCE,
    KERNEL_RETURN_NONE_SOURCE,
)

from examples.amdgpu_gfx11_wmma_matmul import wmma_gfx11
from hc import Buffer, CurrentGroup, WorkItem, kernel
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
    pair = (acc, acc)
    if acc[0] < 1:
        acc = acc[:, 0]
    else:
        acc = -acc
    left, right = pair
    for i, item in enumerate([left, right]):
        acc += [i]
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
_SUBSCRIPT_TARGET_SOURCE = """
@kernel.func(scope=WorkGroup)
def helper(group, values, row, col):
    values[row] = values[col]
    values[:, col] += [row]
    return values
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
    "target_name",
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
_CONTROL_FLOW_EXPECTED_KINDS = (
    "func_begin",
    "if_begin",
    "condition_begin",
    "then_begin",
    "else_begin",
    "compare_begin",
    "subscript_begin",
    "slice_begin",
    "unaryop_begin",
    "for_begin",
    "target_begin",
    "iter_begin",
    "call_begin",
    "list_begin",
    "aug_assign_begin",
    "target_tuple_begin",
)

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
        def bad(group, obj, x):
            obj.attr = x
            return x
        """,
        "unsupported assignment target",
        id="unsupported-assignment-target",
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


@kernel(work_shape=(4,), group_shape=(4,))
def _typed_buffer_kernel(group: CurrentGroup, data: Buffer[4, np.float32]) -> None:
    return None


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


def _assert_control_flow_trace(emitter: RecordingEmitter) -> None:
    kinds = _event_kinds(emitter)
    target_name_ids = {payload["id"] for payload in _payloads(emitter, "target_name")}

    assert kinds[0] == "module_begin"
    assert kinds[-1] == "module_end"
    for kind in _CONTROL_FLOW_EXPECTED_KINDS:
        assert kind in kinds
    assert {"pair", "left", "right", "i", "item"} <= target_name_ids
    assert _payloads(emitter, "aug_assign_begin")[0]["op"] == "Add"
    assert any(
        payload["length"] == 2 for payload in _payloads(emitter, "target_tuple_begin")
    )


def _assert_subscript_target_trace(emitter: RecordingEmitter) -> None:
    kinds = _event_kinds(emitter)

    assert kinds[0] == "module_begin"
    assert kinds[-1] == "module_end"
    assert kinds.count("target_subscript_begin") == 2
    assert "aug_assign_begin" in kinds
    assert "slice_begin" in kinds


def test_lower_function_records_kernel_and_collective_region() -> None:
    emitter = RecordingEmitter()

    lower_function(_sample_kernel, emitter)

    kinds = _event_kinds(emitter)
    kernel_payload = _payloads(emitter, "kernel_begin")[0]
    region_payload = _payloads(emitter, "workitem_region_begin")[0]

    _assert_contains_subsequence(kinds, _SAMPLE_KERNEL_SPINE)
    assert kernel_payload["name"] == "_sample_kernel"
    assert tuple(name for name, *_ in kernel_payload["parameters"]) == ("group", "x")
    assert tuple(passing for *_, passing in kernel_payload["parameters"]) == (
        "positional",
        "positional",
    )
    assert kernel_payload["metadata"] == {
        "work_shape": ("4",),
        "group_shape": ("4",),
    }
    assert kernel_payload["parameter_annotations"] == {
        "group": {"kind": "launch_context", "launch_context": "group"},
        "x": {"kind": "scalar", "dtype": "int"},
    }
    assert region_payload["captures"] == ("tmp",)
    assert tuple(name for name, *_ in region_payload["parameters"]) == ("wi",)
    assert _payloads(emitter, "target_name")[0]["id"] == "tmp"


def test_lower_function_records_buffer_dtype_annotations() -> None:
    emitter = RecordingEmitter()

    lower_function(_typed_buffer_kernel, emitter)

    kernel_payload = _payloads(emitter, "kernel_begin")[0]
    assert kernel_payload["parameter_annotations"] == {
        "group": {"kind": "launch_context", "launch_context": "group"},
        "data": {"kind": "buffer", "shape": ("4",), "dtype": "float32"},
    }


def test_lower_function_records_intrinsic_metadata_and_buffer_annotations() -> None:
    emitter = RecordingEmitter()

    lower_function(wmma_gfx11, emitter)

    intrinsic_payload = _payloads(emitter, "intrinsic_begin")[0]
    metadata = intrinsic_payload["metadata"]

    assert metadata["scope"] == "WorkItem"
    assert metadata["effects"] == "pure"
    assert metadata["const_kwargs"] == ("arch", "wave_size")
    assert tuple(name for name, *_ in intrinsic_payload["parameters"]) == (
        "group",
        "a_tile",
        "b_tile",
        "a_frag",
        "b_frag",
        "acc_frag",
        "lane",
        "wave_size",
        "arch",
    )
    assert tuple(passing for *_, passing in intrinsic_payload["parameters"]) == (
        "positional",
        "positional",
        "positional",
        "positional",
        "positional",
        "positional",
        "keyword_only",
        "keyword_only",
        "keyword_only",
    )
    # The intrinsic has no typed parameter annotations worth surfacing
    # structurally; the emitter should omit the key rather than emit an
    # empty mapping.
    assert "parameter_annotations" not in intrinsic_payload


def test_lower_source_does_not_fabricate_toplevel_metadata() -> None:
    emitter = RecordingEmitter()

    lower_source(_CONTROL_FLOW_SOURCE, emitter, filename="control.py")

    func_payload = _payloads(emitter, "func_begin")[0]
    # Source-only lowering has no Python function object to query for
    # decorator kwargs or resolved annotations, so these keys must stay
    # absent rather than leak stale values from a previous run.
    assert "metadata" not in func_payload
    assert "parameter_annotations" not in func_payload


def test_lower_source_records_structured_control_flow_and_slices() -> None:
    emitter = RecordingEmitter()

    lower_source(_CONTROL_FLOW_SOURCE, emitter, filename="testcase.py")

    _assert_control_flow_trace(emitter)


def test_lower_source_treats_return_none_as_operandless() -> None:
    emitter = RecordingEmitter()

    lower_source(KERNEL_RETURN_NONE_SOURCE, emitter, filename="return_none.py")

    assert _payloads(emitter, "return_begin")[0]["has_value"] is False
    assert "constant" not in _event_kinds(emitter)


def test_lower_source_keeps_func_return_none_value_for_current_signature() -> None:
    emitter = RecordingEmitter()

    lower_source(FUNC_RETURN_NONE_SOURCE, emitter, filename="func_return_none.py")

    assert _payloads(emitter, "return_begin")[0]["has_value"] is True
    assert "constant" in _event_kinds(emitter)


def test_lower_source_records_subscript_assignment_targets() -> None:
    emitter = RecordingEmitter()

    lower_source(_SUBSCRIPT_TARGET_SOURCE, emitter, filename="subscript.py")

    _assert_subscript_target_trace(emitter)


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


def test_name_load_payloads_classify_param_iv_local() -> None:
    # Covers the three frontend-owned ref kinds at once: ``x`` is a param,
    # ``k`` is an iv, ``tmp`` is a local. Captures + builtins are left
    # unclassified for the driver to fill in.
    emitter = RecordingEmitter()
    source = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group, x):
    tmp = x
    for k in range(1):
        tmp = tmp + k + CONST
    return tmp
"""

    lower_source(source, emitter, filename="demo.py")

    kinds_by_name: dict[str, set[str | None]] = {}
    for event in emitter.events:
        if event.kind != "name":
            continue
        if event.payload.get("ctx") != "load":
            continue
        ident = event.payload["id"]
        ref = event.payload.get("ref")
        kind = ref["kind"] if isinstance(ref, dict) else None
        kinds_by_name.setdefault(ident, set()).add(kind)

    assert kinds_by_name["x"] == {"param"}
    assert kinds_by_name["k"] == {"iv"}
    assert kinds_by_name["tmp"] == {"local"}
    # Unresolved captures — the driver fills in ``builtin`` / ``constant``.
    assert kinds_by_name["range"] == {None}
    assert kinds_by_name["CONST"] == {None}


def test_name_store_payloads_skip_ref() -> None:
    # Store-context names become ``target_name`` ops; the plain ``name`` op
    # only covers loads, so the ref classification shouldn't leak onto
    # stores even when the identifier happens to shadow a param.
    emitter = RecordingEmitter()
    source = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group, x):
    x = x + 1
    return x
"""

    lower_source(source, emitter, filename="demo.py")

    name_events = [event for event in emitter.events if event.kind == "name"]
    assert all("ref" in event.payload for event in name_events), name_events
    # No ``ref`` leaks onto ``target_name`` events, those have their own
    # payload shape because storing into a name doesn't resolve against the
    # function's scope the same way a load does.
    target_events = [event for event in emitter.events if event.kind == "target_name"]
    assert target_events
    assert all("ref" not in event.payload for event in target_events)


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


def test_lower_function_accepts_wmma_intrinsic_fallback() -> None:
    emitter = RecordingEmitter()

    lower_function(wmma_gfx11, emitter)

    assert _payloads(emitter, "intrinsic_begin")[0]["name"] == "wmma_gfx11"
    assert len(_payloads(emitter, "target_subscript_begin")) == 1
    assert "aug_assign_begin" in _event_kinds(emitter)


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
