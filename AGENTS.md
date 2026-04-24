

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

## Code ↔ beads boundary

Beads live in `.beads/`; code lives in `lib/`, `include/`, `test/`, `tests/`. Keep them apart:

- No bead ids (`5-1k7`, `4-96o`, etc.) in source files, comments, docstrings, tests, or commit messages. Issue ids belong in `.beads/`, PRs, and the tracker.
- No "this bead does X" phrasing in code — say what the *code* does.
- For deferred work, describe the gap conceptually (e.g. "needs a separate pass", "ODS result extension") instead of naming a bead that may move, merge, or close.
- Bead ids rot. Code comments don't get updated when a bead gets renumbered or retired. The next reader in six months won't have `br show 5-2lf` memorised — and it may not even exist.

If you need to cross-link, do it in the PR description or the bead itself, not in the source.

## Tone

Code comments, docstrings, and commit messages share the same voice: terse, dry, informative. Wit is welcome, fluff is not. Say what the thing does, not what you wish it did. If a comment doesn't earn its line, delete it.

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
