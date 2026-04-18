

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

After running reviewers (multi-perspective or external), go through every finding where a reviewer asked a question, expressed confusion, or had to work to understand the code. For each such finding, add a code comment at the relevant location explaining the non-obvious intent. If a reviewer misunderstood the code, the code needs a comment — the reviewer is the proxy for every future reader. Fix the confusion at the source, not in the review reply.

## Tone

Code comments, docstrings, and commit messages share the same voice: terse, dry, informative. Wit is welcome, fluff is not. Say what the thing does, not what you wish it did. If a comment doesn't earn its line, delete it.

## Testing

- Use `pytest` for Python tests.
- Write tests in free-function style with plain `assert`, not `unittest.TestCase`.
- Keep test modules importable by `pytest` without special runners or `if __name__ == "__main__"` blocks.

## Commits

- Small, focused commits. One logical change per commit. If you're wondering whether to split — split.
- Stage files first, then run `pre-commit` — it only checks staged files. Fix issues and re-stage before committing.
- Sign commits: `git commit -s`.
- Commit messages should be descriptive, or at least funny. Not both is acceptable. Neither is not.
