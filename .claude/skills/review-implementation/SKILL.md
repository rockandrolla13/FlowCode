---
name: review-implementation
description: >
  Systematically process and implement changes based on code review feedback.
  Reads review files from research_projects/reviews/ and creates an implementation
  plan. REQUIRES user confirmation before making any changes. Trigger when user
  provides reviewer comments, PR feedback, code review notes, or asks to implement
  suggestions from reviews. This skill is PLAN-FIRST: it creates a detailed
  implementation strategy and waits for approval before touching any code.
---

# Review Feedback Implementation Skill

Process code review feedback files and implement requested changes systematically while maintaining spec compliance.

## Prerequisites

- Review file in `data_pipelines/us_credit/research_projects/reviews/`
- Review follows standard format (severity levels, findings, locations)
- Access to affected source files and spec requirements (if applicable)

## Core Principles (Invariants)

These are non-negotiable. Violations compromise trust and correctness.

### 1. No Invention Rule

Never invent requirements or make assumptions. No inferred requirements beyond what is explicitly stated in review text. No speculative fixes or "while we're at it" improvements. No behavior changes unless directly traceable to a specific finding. All changes must map directly to a finding with clear rationale.

- WRONG: Review says "Fix typo in docstring" → You also refactor the function logic
- RIGHT: Review says "Fix typo in docstring" → You only fix the typo, nothing else

### 2. Spec Compliance

All formula implementations must match `spec/SPEC.md` definitions. Run `verify_spec_parity.py` after any formula-related changes. Flag spec conflicts immediately — never resolve them unilaterally. If review conflicts with spec, escalate to user (spec is source of truth).

### 3. Behavior Preservation

Preserve existing functionality unless review explicitly requests change. Default assumption: code behavior stays the same, tests pass before and after, no side effects beyond stated changes. When behavior change IS intended: review must explicitly state the new behavior, tests may need updating (with user approval), document the change in implementation report.

### 4. Explicit Approval Gate

Never proceed to implementation without user confirmation. Always present complete plan before first code modification. Wait for explicit "approve", "proceed", or similar confirmation. Stop immediately on ambiguity or conflict — ask user. Forbidden phrases without approval: "I'll implement this now", "Let me make these changes", "Starting implementation..."

### 5. Performance Preservation

No performance degradation in performance-sensitive code. Trading signal calculations are hot paths — treat with care. Benchmark before/after for critical code changes. >10% slowdown requires explicit user approval.

Performance-sensitive areas: vectorized operations (NumPy/Pandas loops), signal calculation functions (imbalance, triggers, etc.), data loading and preprocessing steps.

When performance degrades: report metrics immediately, explain trade-off, get explicit approval, document cost in report.

---

## Workflow

## Phase B: Review Analysis (Read-Only)

**INVARIANT: No code modifications permitted in this phase**

### 1. Locate Review File

```bash
# Review files are stored in:
# data_pipelines/us_credit/research_projects/reviews/YYYY_mm_dd_order_flow_review.md
ls -t data_pipelines/us_credit/research_projects/reviews/*.md | head -5
```

If user doesn't specify which review: ask which file to process, show list of recent reviews, confirm selection before proceeding.

### 2. Read and Parse Review File

Read the entire review file, then extract from each finding: severity level (🔴 Critical, 🟠 Major, 🟡 Minor, 🔵 Suggestion), pillar (SRP, Conciseness, Clean Code, etc.), location (file path and line numbers), BEFORE/AFTER snippets, WHY rationale, and specific action required.

**Parsing rules:** Parse location references (`filename.py:line-range`). Group related changes by file. Preserve BEFORE/AFTER/WHY structure. Extract code snippets for reference.

**Clarify ambiguous items before starting.** If feedback is unclear, ask user for clarification with context and options. Do NOT proceed with ambiguous changes without confirmation.

**Parsing guarantees — MUST extract from each finding:**

| Required Field | Action if Absent | Enforcement |
|----------------|-----------------|-------------|
| Severity level | Ask user to classify | BLOCK until clarified |
| File location | Flag as incomplete | BLOCK until provided |
| Specific change requested | Escalate for clarification | BLOCK until understood |
| BEFORE/AFTER snippets | Infer from location if possible, flag if unclear | WARN, request if critical |
| WHY rationale | Extract from review text | OPTIONAL but prefer explicit |

**Never proceed with partial or assumed extraction.**

### 2.5: Early Conflict Detection

During parsing, actively check for:
1. **Location Conflicts:** Same line numbers referenced in multiple findings with different changes
2. **Semantic Conflicts:** One finding says "extract function", another says "inline function"
3. **Priority Conflicts:** Performance vs readability trade-offs without explicit priority

For each finding: check if location overlaps with previous findings, check if change contradicts previous findings' rationales, check if change creates incompatible requirements. If conflict detected: mark both findings as conflicting in notes, add to "Conflicts to Resolve" section, do NOT create todos for conflicting findings, present conflict to user during planning phase, do not create todos until conflict resolved.

### 3. Create Todo List

Use TaskCreate tool to create actionable tasks. Each feedback item becomes one or more todos. Break down complex feedback into smaller, specific tasks. Include file and line number references in task description. Specify expected outcome for each task.

**Priority ordering:** 🔴 Critical → 🟠 Major → 🟡 Minor → 🔵 Suggestions (if user approves)

**Task metadata for each todo:** Finding number, severity level, file path and line numbers, specific action required, verification step.

After creating the complete todo list: show user the full list for review, wait for approval, then mark the first task as `in_progress` before starting implementation.

## Phase C: Planning (Read-Only)

**INVARIANT: Still no code modifications permitted**

### 4. Create Implementation Plan

For each finding, create a structured entry with: severity, file, change description, risk assessment, spec impact (YES/NO), test impact, and performance risk level.

**Implementation order:** Specify the sequence — critical issues first, then tests for those changes, then major issues, run spec parity check, then minor/style issues, then full test suite.

**Risk assessment — flag:** Spec drift risk (changes that might violate SPEC.md), breaking changes (public API modifications), test coverage gaps (areas needing new tests), dependency impacts (changes affecting multiple files).

**Performance impact analysis:** For each finding, classify risk:

| Risk Level | Criteria | Action |
|------------|----------|--------|
| 🔴 HIGH | Changes vectorized operations, signal core, or adds loops in hot paths | Flag prominently, require benchmark |
| 🟡 MEDIUM | Changes data loading, adds allocations outside hot paths | Note in plan, monitor |
| 🟢 LOW | Changes tests, docs, utilities, or non-performance code | No special action |

Present plan to user with approval options:
- "approve" / "proceed" → implement all changes
- "critical only" → implement only 🔴 items
- "skip [finding #]" → exclude specific items
- "modify [finding #]" → adjust specific changes

**DO NOT PROCEED WITHOUT EXPLICIT USER APPROVAL**

## Phase D: Implementation (State-Changing)

**INVARIANT: Only proceed after explicit approval confirmation**

**Pre-implementation checklist (all must pass):**
- [ ] User explicitly approved plan
- [ ] All conflicts resolved
- [ ] All ambiguities clarified
- [ ] First todo marked `in_progress`
- [ ] Git status confirmed clean

If ANY item unchecked: STOP, report which item is blocking, wait for resolution.

### 5. Implement Changes

**Rules:** One severity tier at a time (critical → major → minor → suggestions). One todo at a time — mark `in_progress`, complete, then next. Run tests after each logical change. Stop if tests fail and report to user. Do NOT commit — leave git operations to user.

**For each todo:**

1. **Locate** — grep/find relevant code
2. **Read** — understand current implementation
3. **Edit** — use Edit tool; follow CLAUDE.md conventions; preserve existing functionality; maintain type hints and docstring format; keep style consistent
4. **Verify** — compare to BEFORE/AFTER from review; verify WHY rationale is satisfied; check edge cases mentioned are handled
5. **Test** — run affected tests; for formula changes: `uv run python tools/verify_spec_parity.py`
6. **Update status** — mark todo completed, move to next

**For 🔴 HIGH performance risk findings:** Run performance-critical tests with `--durations=10`. Compare before/after execution time. If >10% slowdown: flag to user with metrics, explain trade-off, present options (accept / optimize / reject), require explicit decision. Document outcome in implementation report.

**Progress update format:**
```
📋 Todo 3/10: Fix duplicate tag detection in triggers.py:82-90
✅ Change applied
✅ Tests passed (test_triggers.py)
📋 Marked todo #3 as completed
▶️ Moving to todo #4...
```

## Phase E: Validation

### 6. Final Verification

```bash
# ALL must pass:
uv run pytest packages/signals/tests/ -v          # 1. Full test suite
uv run python tools/verify_spec_parity.py          # 2. Spec parity
mypy packages/signals/src/                         # 3. Type checking
ruff check packages/                               # 4. Linting
# 5. If HIGH risk changes: compare --durations=10 to baseline
```

**Pass criteria:** ✅ All tests pass (0 failures), ✅ Spec parity verified (exit 0), ✅ Type checking clean (exit 0), ✅ Linting clean (exit 0 or only expected warnings), ✅ No performance regressions >10% (or user-approved trade-offs documented).

If any check fails: STOP, report failure with details, do NOT mark implementation as complete, investigate and fix before proceeding.

### Implementation Report

```markdown
# Implementation Report: [Review Date] Feedback

## Summary
- Review file: reviews/YYYY_mm_dd_order_flow_review.md
- Findings addressed: X critical, Y major, Z minor
- Files affected: N files

## Changes by Severity
### Critical (🔴)
- [X] Finding 1: [description] — IMPLEMENTED
### Major (🟠)
- [X] Finding 2: [description] — IMPLEMENTED
### Minor (🟡) / Suggestions (🔵)
- [ ] Finding 5: [description] — SKIPPED (optional)

## Conflicts Resolved
[If any — document finding numbers, resolution, rationale]

## Performance Impact
[If any HIGH risk changes — document metrics and user decisions]

## Test Results
All tests passed: ✅ | Spec parity: ✅ | Type checking: ✅ | Linting: ✅

## Files Modified
[List with line ranges]

## Next Steps
[Any remaining suggestions or follow-up items]
```

---

## Edge Cases

### Conflicting Findings

When two or more findings contradict each other or suggest incompatible changes:

1. **Detect** during parsing (Phase B §2.5). Create structured conflict entry with both findings quoted verbatim, severity levels, locations, and analysis of why they conflict.
2. **Analyze** implications — identify potential resolutions (prioritize by severity, ask reviewer, skip both, implement one and supersede other).
3. **Escalate** to user with concrete options:
   ```
   Findings #2 and #5 conflict on imbalance.py:45-60. Which should take precedence?
   A) Follow Finding #2 (extract function) — skip #5
   B) Follow Finding #5 (inline function) — skip #2
   C) Skip both until reviewer clarifies
   D) Other approach (please specify)
   ```
4. **Resolve** — document user's decision, update todos (create for chosen, mark other as skipped with note), update implementation plan.
5. **Verify** — in final report, document conflict, resolution rationale, what was implemented vs skipped.

**Critical rules:** Never guess which finding wins. Never silently skip a finding. Never assume severity determines priority. Always escalate to user.

### Breaking Changes

If change affects public API: flag as "BREAKING CHANGE" in plan with high visibility, list all downstream impacts and call sites, discuss alternatives, get explicit approval before implementing, update all call sites in same change.

### Spec-Related Changes

If review feedback touches formula implementation: read current SPEC.md, verify change aligns with spec, run `verify_spec_parity.py` after implementation, update Formula Registry if function signatures change.

### Test-Related Changes

If review requests test improvements: check for fixtures in `spec/fixtures/`, use `conftest.py` helpers, ensure tests validate against spec-defined expected outputs, add edge cases if review identified gaps. Write tests as functions, not classes.
