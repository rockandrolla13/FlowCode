# ADR 001: Monorepo Structure

## Status

Accepted

## Date

2026-02-03

## Context

FlowCode is a credit analytics platform for generating predictive signals from corporate bond flow data. We needed to decide on the repository structure to support:

1. Multiple independent packages (data, signals, metrics, backtest)
2. Clear dependency boundaries
3. Spec-driven development with golden test fixtures
4. Claude Code agent comprehension and navigation

Options considered:

**Option A: Polyrepo**
- Separate repos for each package
- Independent versioning
- Complex cross-repo testing

**Option B: Monorepo with Package-Level CLAUDE.md**
- Single repo, hierarchical CLAUDE.md files
- Package-specific context for Claude
- Requires Claude Code to traverse sibling packages (not supported)

**Option C: Monorepo with Root CLAUDE.md + Skills (Hybrid)**
- Single repo, root-level CLAUDE.md only
- Task workflows via skills/
- Clear package boundaries via imports

## Decision

We chose **Option C: Monorepo with Root CLAUDE.md + Skills**.

### Rationale

1. **Claude Code Context Discovery**: Claude Code reads CLAUDE.md from the current directory and parent directories, but NOT from sibling packages. A hierarchical CLAUDE.md structure would lose cross-package context.

2. **Clear Dependencies**: The monorepo structure enforces:
   ```
   data (no deps)
     ↑
   core (imports data)
     ↑
   signals (imports core, data)
     ↑
   metrics (imports signals output)
     ↑
   backtest (imports signals, metrics, data, core)
   ```

3. **Spec-Driven Testing**: Centralized `spec/` directory with:
   - `SPEC.md`: Single source of truth for all formulas
   - `fixtures/`: Golden test cases
   - `decisions/`: ADRs like this one

4. **Skills for Workflows**: Instead of package-level CLAUDE.md, we use `skills/` for task-oriented workflows:
   - `add-signal/SKILL.md`
   - `add-data-source/SKILL.md`
   - `run-backtest/SKILL.md`
   - `verify-change/SKILL.md`

## Consequences

### Positive

- Single source of truth (CLAUDE.md) prevents contradictory rules
- Clear package boundaries prevent circular dependencies
- Fixture-driven testing catches formula drift
- Skills provide task-specific guidance without context loss

### Negative

- Root CLAUDE.md grows as packages are added
- Developers must understand the full dependency graph
- No package-level isolation for different teams

### Mitigations

- Keep CLAUDE.md sections focused and well-organized
- Use Formula Registry to link spec ↔ code ↔ fixtures
- skills/ provide task-specific guidance

## Related

- [CLAUDE.md](../../CLAUDE.md) - Root project rules
- [SPEC.md](../SPEC.md) - Formula specifications
- [Implementation Plan](../../docs/IMPLEMENTATION_PLAN.md) - PR strategy
