# hc

Early Python package for the experimental high-level GPU kernel DSL described
in `doc/langref.md`.

## What is here

The project currently provides:

* package metadata and local tooling configuration in `pyproject.toml`,
* a pinned LLVM/MLIR bootstrap path plus a basic `hc-opt` native tool,
* a first project-owned MLIR frontend IR family plus native smoke tests,
* lightweight kernel/core scaffolding in `hc/`,
* a symbolic expression API in `hc.symbols`,
* a NumPy-backed reference executor in `hc.simulator`,
* pytest coverage for the public API, symbols layer, simulator, and native
  frontend integration.

The lowering/compiler pipeline is still TBD, but the symbols subsystem is
already backed by a real third-party engine rather than a pure stub.

## Optional simulator dependency

The base package keeps the simulator dependency optional. Install the simulator
extra when you want to use `hc.simulator`. It includes NumPy plus the
`greenlet` dependency used for workitem scheduling and barriers:

```bash
pip install -e ".[simulator]"
```

The test extra includes the simulator dependencies as well:

```bash
pip install -e ".[test]"
```

## Pinned LLVM/MLIR bootstrap

The repository now carries a pinned LLVM/MLIR lock file in
`third_party/llvm.lock.json`. Editable and wheel builds bootstrap that exact
revision into the gitignored project-local cache under `.hc/toolchains/llvm/`
before handing control to the normal Python package build.

That lock now enables MLIR's Python bindings in addition to the native tools.
This setting is part of the managed `<toolchain-key>` (for example, existing
checkouts that previously cached a `...-nopy-...` toolchain should expect a new
`...-py-...` install directory), so the first build after this change may
trigger a one-time fresh LLVM/MLIR rebuild even if a previous pinned toolchain
cache already exists.

Bootstrap assumes:

* `git` is available on `PATH`,
* the host has a working native toolchain suitable for building LLVM/MLIR,
* the first build may take a while and consume substantial disk space.

On a fresh checkout, the first:

```bash
pip install -e ".[dev]"
```

will:

1. clone the pinned `llvm-project` revision into
   `.hc/toolchains/llvm/src/llvm-project-<sha-prefix>/`,
2. configure and build it in
   `.hc/toolchains/llvm/build/<toolchain-key>/`,
3. stage the install in `.hc/toolchains/llvm/staging/<toolchain-key>/`,
4. promote the staged install into
   `.hc/toolchains/llvm/install/<toolchain-key>/`.

Subsequent installs reuse the cached install as long as the lock file and
Python minor version still match the recorded stamp.

If a full rebuild is required, the keyed `build/` and `staging/` directories
for that toolchain are recreated from scratch so stale CMake state cannot leak
into the next attempt.

To bootstrap the toolchain explicitly, run:

```bash
python -m build_tools.llvm_toolchain
```

Useful escape hatches during bring-up:

* set `HC_LLVM_FORCE_REBUILD=1` to force a rebuild of the pinned toolchain,
* set `HC_LLVM_CACHE_DIR=/path/to/cache` to relocate the otherwise
  project-local cache; this is especially useful for isolated builds or CI
  jobs that should reuse one persistent toolchain cache,
* set `HC_SKIP_LLVM_BOOTSTRAP=1` to skip the LLVM/MLIR bootstrap hook and the
  dependent `hc-opt` native build during package builds; the `ixsimpl` vendor
  bootstrap still runs. This affects the package build backend only, not direct
  toolchain bootstrap commands or pytest smoke tests that call the native
  helpers themselves.

To fully reset the managed local toolchain state, remove the cache directory
under `.hc/toolchains/llvm/` (or the directory pointed to by
`HC_LLVM_CACHE_DIR`).

## Basic MLIR Integration

The repository now also carries a small CMake project that builds `hc-opt`
against the pinned MLIR install. This initial bring-up intentionally uses only
stock MLIR registrations plus the first out-of-tree frontend `hc_front.*` IR
family. No custom conversions or passes are added yet.

MLIR now registers that bootstrap frontend as its own standalone dialect, so
textual frontend operations and types are spelled `hc_front.*` and
`!hc_front.*`.

Older textual IR that used the bootstrap `hc.front.*` / `!hc.front.*` spelling
must be renamed to `hc_front.*` / `!hc_front.*`.

Wheel and editable package builds place the native install under the gitignored
project-local cache at `.hc/native/install/<toolchain-key>/`. The resulting
driver binary lives at:

```text
.hc/native/install/<toolchain-key>/bin/hc-opt
```

To bootstrap the native tool explicitly from a source checkout, run:

```bash
python -m build_tools.hc_native_tools
```

That managed native install also carries a relocatable MLIR Python package
under:

```text
.hc/native/install/<toolchain-key>/python_packages/hc_front/hc_mlir/
```

The generated package embeds the core MLIR Python API plus the out-of-tree
frontend dialect bindings. From a source checkout, `hc.mlir` loads that managed
package on demand, so imports such as `from hc.mlir import ir` and
`from hc.mlir.dialects import hc_front` use the same pinned native build as
`hc-opt`.

Advanced override: set `HC_MLIR_PYTHON_PACKAGE_DIR` to the package root that
contains `hc_mlir/ir.py` when you need `hc.mlir` to load bindings from a
non-default managed install. In other words, point it at
`.hc/native/install/<toolchain-key>/python_packages/hc_front/`, not at the
nested `hc_mlir/` directory itself. Keep that override aligned with the same
native install as `hc-opt`.

`hc.mlir` expects to manage the underlying `hc_mlir` import itself. Import
`hc.mlir` before importing `hc_mlir` directly, and do not pre-import a
different `hc_mlir` from elsewhere on `PYTHONPATH`; mixed installs fail fast on
purpose instead of silently crossing runtime and native-build roots.

That `hc-opt` build now registers the frontend `hc_front.*` operations and can
parse textual frontend IR using placeholder types such as `!hc_front.value`.
The Python bindings expose the same placeholder surface through
`hc.mlir.dialects.hc_front.ValueType` and
`hc.mlir.dialects.hc_front.TypeExprType`.

The smoke tests in `tests/test_hc_front_dialect.py` and
`tests/test_hc_front_python_bindings.py` exercise that real native tool path.
They will reuse or bootstrap the pinned LLVM/MLIR toolchain and the native
`hc-opt` build on demand. To skip those `hc_front` native smoke tests, set
`HC_SKIP_HC_FRONT_DIALECT_TESTS=1`. Unlike `HC_SKIP_LLVM_BOOTSTRAP`, this only
affects those pytest modules; it does not change package-build bootstrap
behavior.

## Third-party symbols backend

`third_party/ixsimpl` is a git submodule containing the source for the
`ixsimpl` symbolic expression library from
[`Hardcode84/4`](https://github.com/Hardcode84/4). The repository name is
opaque, so this mapping is spelled out here intentionally.

Initialize it after cloning:

```bash
git submodule update --init --recursive
```

The local built copy lives under `.hc/vendor/ixsimpl` (gitignored) by default.
Set `HC_IXSIMPL_VENDOR_DIR` to use another empty or already hc-managed
directory. Symbols such as `sym.W` and `Context.sym("W")` are backed by this
engine, so they are real symbolic nodes rather than plain name-only tags.

Wheel and editable package builds now bootstrap that vendored copy up front.
In the normal editable-install flow, `pip install -e ".[dev]"` or
`pip install -e ".[test]"` will build `ixsimpl` before runtime symbol use.
Source archives and metadata-only hooks stay side-effect free.

To bootstrap the backend explicitly from a source checkout, run:

```bash
python -m build_tools.ixsimpl_toolchain
```

If the vendored backend is missing or stale, runtime symbol use now fails fast
instead of building on demand. Re-run the package build or the explicit
bootstrap command above after updating the submodule or clearing `.hc/`.
