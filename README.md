# hc

Early Python package for the experimental high-level GPU kernel DSL described
in `doc/langref.md`.

## What is here

The project currently provides:

* package metadata and local tooling configuration in `pyproject.toml`,
* lightweight kernel/core scaffolding in `hc/`,
* a symbolic expression API in `hc.symbols`,
* pytest coverage for both the public stubs and the symbols layer.

The runtime/compiler implementation is still TBD, but the symbols subsystem is
already backed by a real third-party engine rather than a pure stub.

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

To bootstrap the backend explicitly, run:

```bash
python -m hc._ixsimpl_loader
```

If the backend has not been bootstrapped yet, the first real symbol operation
will build it on demand from the submodule.
