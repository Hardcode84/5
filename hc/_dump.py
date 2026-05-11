# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR-dumping knobs threaded through the compile pipeline.

`HC_DUMP_PASSES=1` turns on per-pass IR printing across the whole
compile, in two surfaces:

* The device-side `PassManager` (the appended `_GPU_LOWERING_PIPELINE`
  in `_pipeline.py`) gets `enable_ir_printing(...)` on it, which is
  the upstream `--mlir-print-ir-after-all` machinery applied to every
  pass in that chain. Catches `fold-memref-alias-ops`, the rocdl
  conversions, `gpu-to-llvm`, `hc-lower-gpu-to-binary`, and
  `hc-lower-launch-func-to-runtime`.

* The transform schedule (`hc/schedules/front_to_hc.mlir`) does NOT
  see the device PM's instrumentation: the transform interpreter
  spawns a fresh, throwaway `PassManager` per
  `transform.apply_registered_pass`, and it doesn't inherit
  instrumentation from the parent PM. We work around it by walking
  the parsed schedule and inserting a `transform.print` after every
  payload-mutating transform op. The interpreter then prints the
  payload between passes for free.

Output goes to stderr (matching `--mlir-print-ir-after-all` and
`transform.print` upstream conventions). Disk-backed dumping (the
`hc-lower-gpu-to-binary --dump-intermediates` half) is a separate
knob and not handled here.
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

# Transform ops that take a payload handle and mutate it in place
# (`results == 0`, `operands == 1`). The probe fires on the *operand*
# handle since there's no SSA result to chain off. We keep the list
# explicit instead of "anything with one operand and no result" so we
# don't accidentally start probing future structural ops that don't
# correspond to a pass step the user wants to inspect.
_OPERAND_HANDLE_OPS: frozenset[str] = frozenset(
    {
        "transform.apply_patterns",
        "transform.apply_cse",
        "transform.apply_dce",
    }
)


def dump_passes_enabled() -> bool:
    """True when `HC_DUMP_PASSES` is set to `1` in the process env."""

    return os.environ.get(DUMP_PASSES_ENV) == "1"


def splice_dump_passes(module: Any) -> int:
    """Insert `transform.print` after every payload-mutating transform op.

    Walks the module body, recursing into every region. For each
    `transform.apply_registered_pass` we hang the print off the
    *result* handle (`%mN+1 = apply_registered_pass ... to %mN`) and
    label it with the pass name. For `transform.apply_patterns` /
    `apply_cse` / `apply_dce` — which mutate the payload referenced
    by their operand and don't produce a result handle — we hang it
    off the operand. Other transform ops (`transform.yield`, nested
    apply-pattern descriptors like `transform.apply_patterns.canonicalization`,
    ...) are skipped because they don't represent an inspect-worthy
    mid-schedule state — `apply_patterns.canonicalization` only
    describes a pattern set, the parent `apply_patterns` is the
    actual mutator and the one we probe.

    Returns the number of probes inserted, mostly for tests/asserts.
    Mutates `module` in place.
    """

    from .mlir import ir

    inserted = 0

    def visit(op: Any) -> None:
        nonlocal inserted
        for region in op.regions:
            for block in region.blocks:
                # Snapshot before iterating: we'll be inserting new ops
                # immediately after each child, and rebinding the block's
                # operations list mid-loop would visit our own probes.
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
