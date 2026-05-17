# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR-dump knobs for the compile pipeline.

`HC_DUMP_PASSES=1` enables per-pass IR printing on two surfaces:

* Device-side `PassManager` (`_GPU_LOWERING_PIPELINE`): standard
  `enable_ir_printing(...)`.
* Transform schedule: the interpreter spawns throwaway PMs per
  `transform.apply_registered_pass`, no instrumentation inherited.
  We splice `transform.print` after every payload-mutating op so the
  interpreter prints between passes.

Stderr only. `hc-lower-gpu-to-binary --dump-intermediates` lives
elsewhere.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DUMP_PASSES_ENV",
    "dump_passes_enabled",
    "splice_dump_passes",
]

DUMP_PASSES_ENV = "HC_DUMP_PASSES"

# Transform ops that mutate in place (0 results, 1 operand); probe
# fires on the operand. Explicit list avoids accidentally probing
# future structural ops.
_OPERAND_HANDLE_OPS: frozenset[str] = frozenset(
    {
        "transform.apply_patterns",
        "transform.apply_cse",
        "transform.apply_dce",
    }
)


def dump_passes_enabled() -> bool:
    """True when `HC_DUMP_PASSES=1`."""

    return os.environ.get(DUMP_PASSES_ENV) == "1"


def splice_dump_passes(module: Any) -> int:
    """Insert `transform.print` after each payload-mutating transform op.

    Probe hangs off the result handle for `apply_registered_pass`,
    off the operand handle for `apply_patterns`/`apply_cse`/`apply_dce`.
    Other ops (yield, nested pattern descriptors) skipped -- not
    inspect-worthy mid-schedule states.

    Returns probe count. Mutates `module` in place.
    """

    from .mlir import ir

    inserted = 0

    def visit(op: Any) -> None:
        nonlocal inserted
        for region in op.regions:
            for block in region.blocks:
                # Snapshot; inserting mid-loop would visit our probes.
                children = list(block.operations)
                for child in children:
                    visit(child)
                    name = child.operation.name
                    if name == "transform.apply_registered_pass":
                        pass_name = ir.StringAttr(child.attributes["pass_name"]).value
                        handle = child.results[0]
                        label = f"after-{pass_name}"
                    elif name in _OPERAND_HANDLE_OPS:
                        handle = child.operands[0]
                        label = f"after-{name.removeprefix('transform.')}"
                    else:
                        continue
                    with ir.InsertionPoint.after(child):
                        ir.Operation.create(
                            "transform.print",
                            results=[],
                            operands=[handle],
                            attributes={
                                "name": ir.StringAttr.get(label),
                            },
                        )
                    inserted += 1

    visit(module.operation)
    return inserted
