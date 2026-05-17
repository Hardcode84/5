

<!-- br-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`/`bd`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View ready issues (unblocked, not deferred)
br ready              # or: bd ready

# List and search
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br search "keyword"   # Full-text search

# Create and update
br create --title="..." --description="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once

# Sync with git
br sync --flush-only  # Export DB to JSONL
br sync --status      # Check sync status
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run `br sync --flush-only` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git commit -m "..."     # Commit everything
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always sync before ending session

<!-- end-br-agent-instructions -->

## Code Review Follow-up

After running reviewers (multi-perspective or external):

1. **Fix the confusion at the source.** Go through every finding where a reviewer asked a question, expressed confusion, or had to work to understand the code. Add a code comment at the relevant location explaining the non-obvious intent. If a reviewer misunderstood the code, the code needs a comment — the reviewer is the proxy for every future reader. Fix it in the code, not in the review reply.

2. **File beads for everything not in the next action.** Any finding you aren't addressing in the immediately-following commit gets a bead: must-fix items you're deferring (rare, and only with a stated rationale), should-fix hygiene, architectural notes, speculative refactors, new test cases worth having. Include title, description, priority, type, then `br sync --flush-only`. Name the filed bead ids in the session reply so the reviewer can see nothing fell through. "I'll get to it later" without a bead means you won't.

Cut-off for "immediate" is the next commit on this branch — not "this sprint", not "before merge". If a finding crosses a commit boundary, it's deferred; file the bead.

## Code ↔ tracker boundary

Tracker artifacts (beads, design-doc delivery plans, sprint / milestone labels) live outside the source tree. Code lives in `lib/`, `include/`, `test/`, `tests/`. Keep them apart:

- No bead ids (`5-1k7`, `4-96o`, etc.) in source files, comments, docstrings, tests, or commit messages. Issue ids belong in `.beads/`, PRs, and the tracker.
- No delivery-plan ordinals (`slice 7`, `phase 2`, `milestone B`, ...) in source either. Numbering inside a design doc is a roadmap, not a stable contract — items renumber as the plan evolves and the references rot in place.
- No "this bead does X" / "this is slice 4" phrasing in code — say what the *code* does, in its own terms.
- For deferred work, describe the gap conceptually (e.g. "needs a separate pass", "ODS result extension") instead of naming a bead or slice that may move, merge, or close.
- Volatile labels rot. Code comments don't get updated when a bead gets renumbered or retired or a delivery plan reorders. The next reader in six months won't have `br show 5-2lf` memorised, or remember which slice was which.

Cross-links belong in the PR description, the bead, or the design doc — not in source. Bare references to a stable design doc by path (`doc/layouts.md` for the contract, not its slice list) are fine when they actually clarify intent.

## Tone

Comments earn their line or get cut. The bar is: would the next reader, staring at the code, miss this? If no, the comment is fluff.

Caveman style. Telegraph English. Substance > prose.

- **One line is the target.** Two is a stretch. Five is a smell — rename the symbol, split the function, or add a structural assert so the comment isn't load-bearing.
- **Say what's non-obvious**, not what the code already says. The function name carries the "what"; the comment carries the "why this shape" or "what would break otherwise".
- **No function-header essays.** Block comments above functions are the worst offender. If you find yourself writing five lines of intent above a `static` helper, the name is wrong or the function does too much.
- **No backstory.** "Previously / used to / old shape was / the X path now does Y" rots the moment the prior shape is forgotten. Describe the current constraint, not the rescue narrative.
- **No restating the signature**, no narrating the body, no "this function takes X and returns Y", no "Step 1, Step 2, Step 3" play-by-play.
- **No cross-references that go stale.** "Same as the load side", "mirrors the foo path", "see also bar()" — these snap when one side moves and the other doesn't. If the symmetry matters, factor a shared helper.
- **No hedging.** "Essentially", "basically", "more or less", "note that", "it's worth noting", "we should probably", "kind of" — cut them.
- **Drop articles and filler** when the meaning survives. "We", "the X path", "this approach", "in order to", "due to the fact that" — gone. `// rank parity required` beats `// We require that the ranks are parity-equal in order to ...`.
- **Don't apologise to the future**, don't explain well-known idioms (`// clone the body`, `// build the result vector`), don't paraphrase identifiers (`// loc is the location`).

Concrete contrasts, paraphrased from real diffs:

Bad:
```cpp
// Per-slot reduction carry fold: collapses to raw `yielded[oi]` on
// plain yield (`masks[oi]` null) and to `arith.select(mask, yielded,
// carry)` on predicated yield, matching the carry semantics the
// previous lowering produced inside the body.
```

Good:
```cpp
// Plain yield: passthrough. Predicated: select(mask, raw, carry).
```

Bad:
```cpp
// Reduction-iter shape for the chunk body: load each outs init at
// the parallel-only offset (the slot every thread owns
// post-partition), build a nested `scf.for` over reduction iters
// with the inits as `iter_args`, clone the body once per innermost
// iteration, propagate yielded values as carries through the nest,
// and store the outer loop's results back at the same outs offset.
```

Good:
```cpp
// Reduction iter syms bind to loop IVs through the nest and restore
// on unwind — siblings see no temp binding.
```
(Everything else the bad version says is plain in the code.)

Bad:
```cpp
// Capture the destination shape for the OOB-store guard so the
// downstream body can compose the bound conjunction. Only populate
// when the per-axis offset shape matches the destination rank
// because the bound zips per-axis and rank parity has to hold.
```

Good:
```cpp
// Rank parity required: bound zips per-axis. Empty indices skip
// (bound vacuously true on a whole-tile write).
```

Same rule covers docstrings, commit bodies, and PR descriptions. Wit is welcome, fluff is not. Neither is acceptable.

## Language and MLIR Guidelines

### Python

- Prefer `math.prod` over `reduce`.
- Iteration over `set` is not stable; sort or otherwise stabilize order when output must be deterministic.
- Underscore-prefixed names are module-private. Do not import them across modules; either drop the underscore or move the helper somewhere public.
- Use `contextlib.suppress(ExcType)` instead of bare `try` / `except` / `pass`.
- Prefer `pathlib.Path` over `os.path`; use `/`, `.exists()`, `.read_text()`, and related `Path` APIs.
- Avoid local imports unless they are needed to keep expensive or optional dependencies lazy.

### LLVM/MLIR C++

- Do not use braces for single-line `if` bodies.
- Avoid `auto` when the type is not trivial to infer; lambdas and iterators are fine.
- Do not name variables `module`, to avoid collision with C++ modules.
- Prefer `std::array` over `std::vector` / `llvm::SmallVector` when the count is known at compile time.
- Use descriptive asserts with `&& "message"`.
- Use `Op::create(builder, ...)` syntax.
- Use `cast<Type>(arg)` syntax.
- Prefer `llvm::seq` to C-style counted loops.
- For MLIR/C++ debug logging, include `llvm/Support/DebugLog.h` and use `LDBG()` / `LDBG_OS()` instead of raw `LLVM_DEBUG(llvm::dbgs() << ...)`.
- Mark TU-local free functions `static` even inside an `namespace { ... }`. Anonymous namespaces house struct/class definitions; free functions get an explicit `static` so the storage class is visible at the signature and not implied from a brace fifty lines up. Templates keep `template <...>` first then `static`; `[[noreturn]]` stays leftmost.

### MLIR

- Use `getConstantIntValue(Value/Attribute)` to extract a constant integer from either a value or an attribute instead of manually matching `arith.constant`.
- Prefer `StringRef` and `Twine` over `std::string` for string handling.
- Use `DenseMap::lookup(key)` when a missing key should return a default-constructed value without inserting into the map.
- Do not root passes on concrete ops until necessary; prefer broader interfaces or dialects.
- `op.walk(...)` lambdas can return `WalkResult::interrupt()` to stop early and propagate failure; check the result with `.wasInterrupted()`.
- Use `return signalPassFailure();` to abort a failed pass.
- Prefer named accessors to `getResult(0)` when possible.
- In LIT tests, never use raw SSA names like `%0` or `%1` in `CHECK` lines. Capture them with placeholders such as `[[VAL:%.*]]` and reuse the placeholder.
- For type-rewriting passes, drive the dialect-conversion infrastructure (`TypeConverter` + `applyPartialConversion`) instead of poking `Value::setType` from a walk; the conversion driver tracks materializations across boundaries that bare in-place mutation does not. Reuse upstream populators (`populateAnyFunctionOpInterfaceTypeConversionPattern`, `populateReturnOpTypeConversionPattern`, `populateCallOpTypeConversionPattern`, `scf::populateSCFStructuralTypeConversionsAndLegality`) rather than hand-rolling per-op clone-and-replace.
- Don't walk `unrealized_conversion_cast`. Even one `getDefiningOp<UnrealizedConversionCastOp>()` peek leans on the conversion driver's temporary bridge — fix the producer (source/target materialization, missing pattern, legality target) so the cast collapses.

### Symbolic expressions (ixsimpl)

- Never build or compare `#hc.expr` / `#hc.pred` payloads via string formatting + `sym::parseExpr` / `sym::parsePred`. Use the structural compose API: `composeExprSym(store, name)`, `composeExprInt(store, value)`, `composeExprBinary(store, lhs, op, rhs)`, `composeExprNeg`, `composeExprCeil`, `composePredCmp`. Hash-consing in the dialect-owned store gives pointer equality on the canonical handle as the right comparison; building from text re-parses every leaf and is brittle to spelling drift.
- `sym::parseExpr` / `sym::parsePred` are for the textual surface only — ODS parser hooks, attribute round-trip, frontend / Python ingestion. Production C++ paths inside passes build structurally.
- Same rule applies to LIT helpers, generated code, and Python: don't `f"{a} + {b}"`-then-parse. Construct via the API and serialize at the boundary if needed.

### Python -> MLIR attribute construction

- Don't smuggle structured payloads across the Python / MLIR boundary as `StringAttr`s of MLIR text that C++ then re-parses. Build the typed attribute in Python and let C++ readers cast it.
- Single source of truth for the construction is the MLIR Python bindings under `hc.mlir.ir`: prefer `ir.Attribute.parse('#hc.expr<"…">', context=ctx)` for `ExprAttr` / `PredAttr`, `ir.ArrayAttr.get([...])` for ordered name lists, and `ir.DictAttr.get({...})` for name->attr tables. Compose them into the final `DictAttr` ref the way `_OpClassifier._to_attr` does in `hc/_resolve.py`.
- Register every dialect whose attributes you intend to construct on the active context before building. Frontend emitters that touch hc payloads must call both `hc_front.register_dialects(ctx)` and `hc.register_dialects(ctx)`; otherwise `ir.Attribute.parse` rejects `#hc.…` with "unregistered dialect" and the silent fix is "round-trip as a string", which is the exact failure mode this rule forbids.
- The frontend boundary is allowed exactly one `parseExpr` per expression (mirroring the C++ side rule above). Don't widen that — never reconstruct an attribute by stringifying an existing typed attr and re-parsing it; pass the typed `Attribute` through.

## Testing

- Use `pytest` for Python tests.
- Write tests in free-function style with plain `assert`, not `unittest.TestCase`.
- Keep test modules importable by `pytest` without special runners or `if __name__ == "__main__"` blocks.

## Commits

- Small, focused commits. One logical change per commit. If you're wondering whether to split — split.
- **Acceptance gate**: every commit lands with LIT (`ninja check-hc`) and Python tests (`pytest`) green. Pre-commit runs linters, not tests — run both suites yourself before the commit, every time. No "tests were red on master, skipping" unless you explicitly state why in the message and file a bead against the regression.
- Stage files first, then run `pre-commit` — it only checks staged files. Fix issues and re-stage before committing.
- Sign commits: `git commit -s`.
- Commit messages should be descriptive, or at least funny. Not both is acceptable. Neither is not.
