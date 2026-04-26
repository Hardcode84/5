# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np
import pytest

from hc import idx_type, tensor_type, undef_type, vector_type
from hc._intrinsic_contracts import (
    serialize_intrinsic_type_spec,
    validate_intrinsic_type_contract_record,
)


def test_intrinsic_type_contract_schema_serializes_specs() -> None:
    assert serialize_intrinsic_type_spec(tensor_type((4, "N"), np.float32)) == {
        "kind": "tensor",
        "shape": ("4", "N"),
        "dtype": "float32",
    }
    assert serialize_intrinsic_type_spec(vector_type((8,), np.float16)) == {
        "kind": "vector",
        "shape": ("8",),
        "dtype": "float16",
    }
    assert serialize_intrinsic_type_spec(idx_type("lane")) == {
        "kind": "idx",
        "expr": "lane",
    }
    assert serialize_intrinsic_type_spec(undef_type()) == {"kind": "undef"}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({}, "missing string kind"),
        ({"kind": "buffer"}, "unsupported kind"),
        ({"kind": "tensor", "shape": ("4",)}, "missing string dtype"),
        ({"kind": "vector", "dtype": "float32"}, "missing string sequence shape"),
        (
            {"kind": "tensor", "shape": ("4",), "dtype": "float32", "expr": "M"},
            "unsupported key",
        ),
        ({"kind": "idx", "expr": 7}, "non-string expr"),
        ({"kind": "undef", "shape": ()}, "unsupported key"),
    ],
)
def test_intrinsic_type_contract_schema_rejects_invalid_records(
    record: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_intrinsic_type_contract_record(record)
