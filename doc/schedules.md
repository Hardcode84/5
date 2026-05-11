# Compilation schedules

`hc.compile` drives MLIR pipelines through the
[transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/). A
*schedule* is a small MLIR module that exposes a named sequence
`@__transform_main` applying `transform.apply_registered_pass` ops in
order. The Python driver builds a two-pass pipeline —
`transform-preload-library` followed by `transform-interpreter` — and
hands it the schedule file, so any pass that is registered in the
process's MLIR pass registry (upstream canonicalize / cse / ...,
plus the `hc` pass families) is fair game inside a schedule.

This is a pipeline manifest, not a full schedule language yet — pass
options are supported, but there is no payload filtering or scoping. It will
grow as the compiler earns knobs; today it mostly picks which passes run and
in what order.

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
    %m5 = transform.apply_registered_pass "hc-infer-types" to %m4
        : (!transform.any_op) -> !transform.any_op
    %m6 = transform.apply_registered_pass "hc-materialize-bound-exprs" to %m5
        : (!transform.any_op) -> !transform.any_op
    %m7 = transform.apply_registered_pass "hc-verify-static-shapes" to %m6
        : (!transform.any_op) -> !transform.any_op
    %m8 = transform.apply_registered_pass "hc-decompose-shaped-values"
        with options = { "strict" = false } to %m7
        : (!transform.any_op) -> !transform.any_op
    %m9 = transform.apply_registered_pass "hc-inline-helpers" to %m8
        : (!transform.any_op) -> !transform.any_op
    %m10 = transform.apply_registered_pass "hc-materialize-bound-exprs" to %m9
        : (!transform.any_op) -> !transform.any_op
    transform.apply_dce to %m10 : !transform.any_op
    %m11 = transform.apply_registered_pass "hc-normalize-scope-regions" to %m10
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m11 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m11 : !transform.any_op
    %m12 = transform.apply_registered_pass "hc-lower-kernels-to-gpu-launch" to %m11
        : (!transform.any_op) -> !transform.any_op
    %m13 = transform.apply_registered_pass "hc-lower-launch-body" to %m12
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m13 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m13 : !transform.any_op
    %m14 = transform.apply_registered_pass "hc-interpret-intrinsic-recipes"
        with options = { "target" = "__HC_TARGET__" } to %m13
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m14 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m14 : !transform.any_op
    %m15 = transform.apply_registered_pass "gpu-kernel-outlining" to %m14
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m15 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m15 : !transform.any_op
    %m16 = transform.apply_registered_pass "rocdl-attach-target"
        with options = { "chip" = "__HC_CHIP__" } to %m15
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %m16 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %m16 : !transform.any_op
    transform.yield
  }
}
```

The `__HC_TARGET__` and `__HC_CHIP__` tokens are substitution sentinels:
the Python driver replaces them before parsing the schedule.
`__HC_TARGET__` takes the value of `hc.compile(target=...)` (empty string
for the `None` default). `__HC_CHIP__` takes the AMDGPU chip name resolved
from the same `target=` (e.g. `amdgpu-gfx11` -> `gfx1100`); a bare
`gfx<chip>` works verbatim; anything else falls back to the default chip.
Custom schedules that keep the placeholders pick up the `target=` plumbing
for free; ones that drop them own their own pass invocations.

The closing trio splits each `gpu.launch` into a `gpu.module @<kernel>_kernel`
+ `gpu.func` + `gpu.launch_func` (`gpu-kernel-outlining`) and stamps the
module with `#rocdl.target<chip = "...">` (`rocdl-attach-target`) so the
binary-emission pass downstream has the chip info it needs. The actual
`gpu.module` body lowering chain (amdgpu/scf/memref/vector → llvm-dialect
inside the module) and `hc-lower-gpu-to-binary` are not yet in the default
schedule — they need a ROCDL-equivalent of upstream's
`gpu-lower-to-nvvm-pipeline` (workgroup memref address-space mapping,
kernel ABI, scf/vector lowering inside `gpu.module`) which is a
substantively separate piece of work tracked as a follow-up.

Bound symbolic expression materialization runs after type inference so it can
see pinned `!hc.idx<...>` / `!hc.pred<...>` facts and before later scope
normalization needs launch-context-independent SSA values. The static shape
verifier then validates SSA shape operands through their inferred tuple element
types before canonicalization and CSE can obscure the producer that carried a
bad shape. Shaped-value decomposition runs next in non-strict mode: supported
producers and users, including helper-call signatures, `hc.call` sites, stores,
and structured/collective region boundaries, are split into bare data/masks,
while remaining intrinsic boundaries are preserved with
`builtin.unrealized_conversion_cast` until those consumers grow decomposition
rules. Helper inlining can clone fresh launch-geometry producer chains from
callee bodies, so bound-expression materialization runs again before
scope-region normalization. DCE then removes now-dead scope-token geometry
producers; live workitem/subgroup token uses still diagnose in the normalization
pass. The final normalization removes supported `hc.func` call boundaries and
result-producing workitem regions from the executable HC body, leaving subgroup
and other unsupported scope cases to diagnose. Cleanup runs over that normalized
HC body before `hc-lower-kernels-to-gpu-launch` wraps each `hc.kernel` in a host
`func.func` containing a `gpu.launch` and exposes buffer ABI as
`(!hc.ptr<global, T>, dim*, stride*)`. `hc-lower-launch-body` then lowers the
scalar/index subset inside that launch to upstream `arith`/`scf` operations and
lowers static bare tensors to workgroup-memory `!hc.ptr<workgroup, T>` (via
`hc.alloc` + `hc.ptr_offset` + `hc.ptr_*`). Bare vectors lower to upstream
`vector` values, and final masked stores become `hc.ptr_store_pred` (vector
form) or guarded scalar `hc.ptr_store` inside `scf.if`.
Intrinsic boundaries remain HC ops with explicit casts at that point; the final
`hc-interpret-intrinsic-recipes` step then walks the sibling
`module @__hc_intrinsic_lowerings__` the frontend emitter planted, applies
every `transform.named_sequence` whose `hc.target` matches the requested
target (the default empty target runs every recipe — fine while each
intrinsic registers at most one recipe per compile), erases the spent
lowerings module, and DCEs `hc.intrinsic` decls whose last call site was
rewritten. The trailing canonicalize/cse pair folds the recipe-inserted
bridging UCCs into identity (they pair with the existing UCCs the
launch-body pass plants on either side of every call boundary), leaving the
final IR free of HC ops and bridging machinery. See **Intrinsics** in
`doc/lowering.md` for the recipe authoring + transform-op surface.

Each `apply_registered_pass` consumes its input handle and produces a
fresh one, which is why the entry-block argument is not marked
`{transform.readonly}` — the verifier refuses that combination.

## Selecting a target

`hc.compile(kernel_fn, symbols, target="amdgpu-gfx11")` substitutes the
target string into the default schedule's `__HC_TARGET__` placeholder so
`hc-interpret-intrinsic-recipes` only fires named sequences whose
`hc.target` attribute matches. The `None` default (no target) substitutes
the empty string, which makes the recipe interpreter apply every named
sequence regardless of `hc.target` — the right behaviour while each
intrinsic registers at most one recipe per compile. When multi-target
lowerings co-exist in one module, pass an explicit `target=` so the
interpreter picks the right recipe instead of running them all.

The same `target=` value is mapped to an AMDGPU chip name and substituted
into `__HC_CHIP__` so `rocdl-attach-target` can stamp every `gpu.module`
with `#rocdl.target<chip = "...">`. The current map covers `amdgpu-gfx11`
-> `gfx1100`; a bare `gfx<chip>` string is accepted verbatim for
chips the project hasn't catalogued yet. With `target=None` the chip
falls back to `gfx1100`, which is harmless because `rocdl-attach-target`
only runs on payloads that produced a `gpu.module` (i.e. non-trivial
kernels — and those will need to opt into a real target soon enough).

Strings containing `"`, `\`, `\n`, or `\r` are rejected up front: those
characters would either close the substituted MLIR string literal early
or break the transform option parser. The handle echoes the value back
as `CompiledKernel.target` for downstream stages and debugging.

A target the schedule's recipes don't cover surfaces as a hard
diagnostic from `hc-interpret-intrinsic-recipes` (`no intrinsic
lowering recipe matched @<callee> for target '<t>'`) rather than a
silent no-op — that's deliberate, since silent passthrough would just
move the gap to the next pass.

## Overriding the schedule

`hc.compile(kernel_fn, symbols, schedule=...)` accepts either:

* `pathlib.Path` — read the schedule from that file. The path is
  resolved to absolute and checked for existence up front; a missing
  file raises `FileNotFoundError` immediately rather than surfacing as
  a far-away MLIR diagnostic. The driver reads the file content and
  writes it to a tempfile it controls before handing it to the
  pipeline, so weird characters in the user-supplied path no longer
  bleed into the MLIR option parser.
* `str` — always treated as inline transform-module text. A path that
  happens to be stored as a string will be fed to the parser verbatim,
  not opened; wrap it with `Path(...)` first.
* `None` (default) — use the bundled schedule.

Anything else is a `TypeError`. Both file and string schedules go
through the same `__HC_TARGET__` and `__HC_CHIP__` substitutions as the
default, so a custom schedule that wants the `target=` plumbing only
needs to keep the placeholders where it makes sense.

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
