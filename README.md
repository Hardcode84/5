# hc

Minimal Python stub for the experimental high-level GPU kernel DSL described in
`doc/langref.md`.

## What is here

The project currently provides:

* package metadata in `pyproject.toml`,
* a small public API surface in `hc/`,
* smoke tests for the exported stubs in `tests/`.

The runtime/compiler implementation is still TBD. The current package only
defines enough structure to hang real code on.
