# Compilation schedules

`hc.compile` drives MLIR pipelines through the
[transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/). A
*schedule* is a small MLIR module that exposes a named sequence
`@__transform_main` applying `transform.apply_registered_pass` ops in
order. The Python driver builds a two-pass pipeline —
`transform-preload-library` followed by `transform-interpreter` — and
hands it the schedule file, so any pass that is registered in the
process's MLIR pass registry (upstream canonicalize / cse / ...,
plus the three `hc` pass families) is fair game inside a schedule.

This is a pipeline manifest, not a full schedule language yet — no
knobs, no payload filtering, no scoping. It will grow as the compiler
earns knobs; today it only picks which passes run and in what order.

## Default schedule

`hc/schedules/front_to_hc.mlir` ships with the package and is applied
when `hc.compile(..., schedule=None)` (the default). It runs the
canonical `hc_front -> hc` pipeline:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%m: !transform.any_op) {
    %m1 = transform.apply_registered_pass "hc-front-fold-region-defs" to %m
        : (!transform.any_op) -> !transform.any_op
    %m2 = transform.apply_registered_pass "hc-front-inline" to %m1
        : (!transform.any_op) -> !transform.any_op
    %m3 = transform.apply_registered_pass "convert-hc-front-to-hc" to %m2
        : (!transform.any_op) -> !transform.any_op
    %m4 = transform.apply_registered_pass "hc-promote-names" to %m3
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

Each `apply_registered_pass` consumes its input handle and produces a
fresh one, which is why the entry-block argument is not marked
`{transform.readonly}` — the verifier refuses that combination.

## Overriding the schedule

`hc.compile(kernel_fn, symbols, schedule=...)` accepts either:

* `pathlib.Path` — read the schedule from that file. The path is
  resolved to absolute and checked for existence up front; a missing
  file raises `FileNotFoundError` immediately rather than surfacing as
  a far-away MLIR diagnostic. Paths with characters the MLIR option
  parser treats as delimiters (`,`, `}`, `=`) will still break the
  pipeline — avoid them.
* `str` — always treated as inline transform-module text. A path that
  happens to be stored as a string will be fed to the parser verbatim,
  not opened; wrap it with `Path(...)` first.
* `None` (default) — use the bundled schedule.

Anything else is a `TypeError`.

### Example: skip `hc-promote-names`

`hc-promote-names` folds `hc.name_load`/`hc.assign` ops into SSA
values. A caller interested in inspecting the name-based IR before that
rewrite (debugging, introspection tooling) can drop it from the
schedule:

```python
schedule = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%m: !transform.any_op) {
    %m1 = transform.apply_registered_pass "hc-front-fold-region-defs" to %m
        : (!transform.any_op) -> !transform.any_op
    %m2 = transform.apply_registered_pass "hc-front-inline" to %m1
        : (!transform.any_op) -> !transform.any_op
    %m3 = transform.apply_registered_pass "convert-hc-front-to-hc" to %m2
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""

handle = hc.compile(kernel_fn, {sym.W: 128}, schedule=schedule)
assert "hc.name_load" in handle.hc_ir_text  # survives without promote
```

A minimal variant — only `convert-hc-front-to-hc`, also a valid
override — is what `tests/test_hc_compile.py` exercises; both shapes
are legal so long as every pass name resolves.

The handle still carries `front_ir` / `front_ir_text` unchanged; only
`hc_ir` / `hc_ir_text` reflect the shorter pipeline.

## Failure handling

Schedule failures — parse errors, unknown pass names, pass-internal
verifier errors, exceptions from the interpreter — are non-fatal.
`hc.compile` returns a handle with `hc_ir = hc_ir_text = None`; the
`pipeline_diagnostics` tuple on the handle records what the MLIR
diagnostic handler captured during the run. Callers that need a hard
failure should inspect `pipeline_diagnostics` and raise themselves.

## Registered passes

The process-wide pass registry is populated by the MLIR python bindings
at import time (upstream passes) and by `hc.register_passes()` the
first time `hc.compile` is invoked (`hc`'s own three families —
`hc-front` transforms, `hc-front -> hc` conversion, `hc` transforms).
Any pass name a schedule references has to resolve against that
registry; `hc-opt --help` is the authoritative list.
