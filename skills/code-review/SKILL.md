# Code Review Skill

FEEDBACK ONLY. Reviews Python code and produces a severity-ranked Markdown report. Never modifies, creates, or overwrites source files. The only file this skill creates is the review report.

## Philosophy

Two non-negotiable priorities, in order:
1. Avoidance of subtle bugs — correctness above all.
2. Minimal, economical code — every line should earn its place.

These are not in tension. Concise code has less surface area for bugs.

## Workflow

1. **Ingest** — Read Python files from `/mnt/user-data/uploads/` or conversation context.
2. **Analyze** — Evaluate against `/media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/FlowCode/references/python-standards.md`. Categorize each finding by pillar and severity.
3. **Report** — Produce the review as a Markdown file using the Output Template below. Print the full review to the user AND save to disk.

## Severity Levels

| Level | Meaning |
|-------|---------|
| 🔴 Critical | Likely bugs, silent data corruption, security holes |
| 🟠 Major | Significant maintainability / readability / correctness risk |
| 🟡 Minor | Style, naming, small conciseness improvements |
| 🔵 Suggestion | Optional — idiomatic or economy improvements |

## Output Template

```markdown
# Code Review Report

**Files reviewed:** [list]
**Date:** [date]
**Overall health:** [🟢 Good | 🟡 Needs attention | 🔴 Needs significant work]

## Executive Summary
[2–4 sentences: overall impression, dominant patterns, top priority action.]

## Findings

### 1. [Finding title]
- **Severity:** 🟠 Major
- **Pillar:** Single Responsibility / Conciseness / etc.
- **Location:** `filename.py`, lines X–Y

BEFORE:
[original snippet — quote the relevant code exactly]

AFTER:
[illustrative sketch ≤10 lines — NOT copy-paste-ready code]

WHY:
[one-line rationale linking to the principle violated]

## Summary Table
| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|

## Positive Highlights
[2–3 things the code does well.]
```

## Constraints

- Every AFTER sketch must be semantically equivalent to its BEFORE. If a change would alter behaviour, flag it explicitly.
- Keep control flow explicit — never suggest clever tricks that obscure logic.
- Every suggested edit must have a clear, articulable purpose.
- Do NOT change public interfaces (function signatures, class APIs) without noting downstream impact.
- Reference specific line numbers in every finding.
- Do NOT add tests. If tests would be valuable, note: `[SUGGEST: add test for X]`.
- For large files (>500 lines), prioritise 🔴 and 🟠 in the executive summary.
- If the user asks to apply changes, remind them this skill is feedback-only and suggest a separate refactoring workflow.
- Never generate diffs, patches, or replacement files.
- Never run linters, formatters, or fixers that alter files.

## File Storage & Versioning

- **Path:** `data_pipelines/us_credit/research_projects/reviews/`
- **Format:** `YYYY_mm_dd_<scope>_review.md` where `<scope>` is descriptive (e.g., `orderFlow`, `compute_zscore_signal`, `imbalance`)
- **Purpose:** Creates a permanent record for implementation tracking and future reference.
- **Versioning:** Use Glob to check for `YYYY_mm_dd_<scope>_review*.md`. If matches exist, parse the highest version number and increment. Original without suffix is implicitly v1; first duplicate gets `_v2`.
- **Confirm:** Notify user of the full versioned filename after saving.

## Handling Reviewer Notes

When reviewer notes are provided instead of (or alongside) code:
1. Parse individual feedback items — split numbered lists, bullet points, unnumbered feedback.
2. Extract distinct change requests. Clarify ambiguous items before starting.
3. Use TodoWrite to create actionable tasks — break down complex feedback, make tasks specific and measurable.
4. Mark first task as in_progress before starting work.
