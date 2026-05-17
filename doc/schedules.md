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
when `hc.compile(..., schedule=None)` (the default). It is the
authoritative source for which passes run and in what order — the file
itself, with its inline per-pass justifications, is the contract. This
doc summarises the stages rather than pinning the pass list (a copy
here would rot the moment a pass is added, removed, or reordered).

In phase order, the schedule does:

1. **Frontend cleanup.** Fold region scaffolding, inline undecorated
   helpers (`hc_front.ref kind="inline"` markers), and convert
   `hc_front` ops into `hc` via `convert-hc-front-to-hc`.
2. **Literal specialisation + early lowerings.** Substitute
   `hc.compile(symbols={K: 4, ...})` bindings throughout reachable
   `#hc.expr` / `#hc.pred` / shape / layout carriers, then lower the
   `hc.pow` integer-literal-exponent carrier into a mul chain so type
   inference sees the muls.
3. **Type inference + symbolic materialisation.** Promote
   `hc.assign` / `hc.name_load` into SSA, run `hc-infer-types`,
   sever bound symbolic expressions from their producer chains via
   `hc-materialize-bound-exprs`, and verify static shape carriers.
4. **Shape decomposition + helper inlining.** Split semantic
   `!hc.tensor` / `!hc.vector` into bare data/mask pairs
   (`hc-decompose-shaped-values`), inline `@kernel.func` helpers,
   then re-run `hc-materialize-bound-exprs` to sever the
   launch-geometry chains the inliner brought along.
5. **Generic funnel.** Fold identity layouts, lower the
   `hc.strip_layout` boundary, and funnel `hc.matmul` / `hc.reduce` /
   per-element arith / `hc.builtin_call` / `hc.load` / `hc.store` /
   `hc.load_mask` into `hc.generic` via the four `*-to-generic`
   rewriters + `hc-lower-math`. Wave-cooperative layouts are then
   distributed to per-lane peers (`hc-distribute-wave-layouts`).
6. **Scope normalisation + launch wrap.** Fold supported `hc.func`
   call boundaries and workitem regions, then wrap each `hc.kernel`
   in a host `func.func` containing a `gpu.launch` with kernel-arg
   ABI `(!hc.ptr<global, T>, dim*, stride*)`
   (`hc-lower-kernels-to-gpu-launch`).
7. **Flatten + body lowering.** Collapse every shaped carrier to its
   1D storage form and compose every offset to a single 1D `#hc.expr`
   (`hc-flatten-with-layouts`), then lower the launch body
   (scalar / control flow, ptr family, masked stores) twice — once
   over the flat IR, then again after `hc-lower-generic` collapses
   `hc.generic` ops over `!hc.ptr` outs into `scf.parallel` /
   `scf.for` nests planting fresh `hc.idx_apply` ops. The
   `hc-insert-workgroup-barriers` pass between the launch-body and
   `hc-lower-generic` slots plants `gpu.barrier` between
   `hc.generic` ops that share a workgroup-AS storage root.
8. **Recipes + outlining + target stamping.** Interpret target
   intrinsic recipes (`hc-interpret-intrinsic-recipes` walks the
   sibling `@__hc_intrinsic_lowerings__` module), then run upstream
   `gpu-launch-sink-index-computations` + `gpu-kernel-outlining` to
   produce `gpu.module @<kernel>_kernel` + `gpu.func` +
   `gpu.launch_func`, and finally stamp every fresh `gpu.module` with
   `#rocdl.target` via `rocdl-attach-target` (chip + features
   resolved Python-side from `target=`).

After the schedule, a fixed GPU-lowering chain runs as a raw pipeline
string (appended by the Python driver — see `_GPU_LOWERING_PIPELINE`
in `hc/_pipeline.py`). It owns `hc-lower-to-llvm`, `convert-scf-to-cf`,
`convert-amdgpu-to-rocdl`, the `gpu.module(...)` nested pass manager,
`gpu-to-llvm`, the outer reconcile/canon/cse trio,
`hc-lower-gpu-to-binary` (LLVM IR → AMDGPU ISA → ELF → HSACO),
`hc-lower-launch-func-to-runtime` (`gpu.launch_func` → `hc_rt_*` calls
+ embedded HSACO global), an optional `hc-emit-bench-wrapper`
(`hc.compile(..., bench=True)`), and `symbol-dce`. The raw-string form
exists because `transform.apply_registered_pass` cannot express the
`gpu.module(...)` nest; the rest of the chain is otherwise flat and
might migrate into the schedule in the future.

The schedule's `__HC_TARGET__` and `__HC_CHIP__` tokens are substitution
sentinels the Python driver replaces before parsing.
`__HC_TARGET__` takes the value of `hc.compile(target=...)` (empty
string for the `None` default). `__HC_CHIP__` takes the AMDGPU chip
name resolved from the same `target=` (e.g. `amdgpu-gfx11` ->
`gfx1100`); a bare `gfx<chip>` works verbatim; anything else falls
back to the default chip. Custom schedules that keep the placeholders
pick up the `target=` plumbing for free; ones that drop them own
their own pass invocations.

Each `apply_registered_pass` consumes its input handle and produces a
fresh one, which is why the entry-block argument is not marked
`{transform.readonly}` — the verifier refuses that combination.

A handful of contracts ride on the schedule order; the file's inline
comments name them at the call site. Briefly:

* `hc-specialize-literals` runs immediately after `convert-hc-front-to-hc`
  so every later pass sees concrete dims. A schedule that drops the
  pass keeps unspecialised IR; downstream passes fail loud where they
  need a concrete dim they don't have.
* `hc-decompose-shaped-values` is the contract boundary between the
  semantic shaped surface and every downstream lowering: any survivor
  (including intrinsic boundaries that have no decomposition rule yet)
  fails the pass with the offending op named.
* `hc-flatten-with-layouts` runs after `hc-lower-kernels-to-gpu-launch`
  so flatten sees the kernel-arg UCC chain the launch wrapper plants
  and rewrites it through to the post-flatten layout retyper.
* `hc-lower-launch-body` runs twice: once over the flat post-launch
  IR, once after `hc-lower-generic` plants per-lane `hc.idx_apply` ops
  that still need to lower to plain `arith.*` / `index_cast`.
* See `doc/lowering.md`'s **Intrinsics** section for the recipe
  authoring + `transform.hc.*` op surface that
  `hc-interpret-intrinsic-recipes` consumes.

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
