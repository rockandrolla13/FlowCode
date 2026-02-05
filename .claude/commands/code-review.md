# Code Review Skill

**This skill is FEEDBACK ONLY.** It reviews Python code and produces a written
report of findings and recommendations. It does NOT refactor, rewrite, fix, or
modify any code — not in files, not inline, not as "suggested patches." The only
code that appears in the output is short illustrative BEFORE/AFTER snippets
inside the report to clarify what a finding refers to.

## Philosophy

Two non-negotiable priorities, in order:

1. **Avoidance of subtle bugs** — correctness above all.
2. **Minimal, economical code** — every line should earn its place.

These are not in tension. Concise code has less surface area for bugs.

## Workflow

1. **Ingest** — Read all Python files from the specified path argument: $ARGUMENTS
2. **Analyze** — Evaluate the code against standards. Categorize each finding by pillar and severity.
3. **Report** — Output the review as Markdown directly in the conversation.

## What to Look For

- Redundant variables (inline if used once and clear)
- Repeated expressions (extract if >=2 uses)
- Verbose conditionals (simplify without nesting)
- Dead code / unreachable branches
- Unnecessary type conversions or wrappers
- Overly defensive checks that duplicate language guarantees
- Type hint completeness
- Docstring quality (NumPy format expected)
- Numerical Python hygiene (NaN handling, division by zero)
- Pandas best practices

## Severity Levels

| Level | Meaning |
|-------|---------|
| Critical | Likely bugs, silent data corruption, security holes |
| Major | Significant maintainability / readability / correctness risk |
| Minor | Style, naming, small conciseness improvements |
| Suggestion | Optional — idiomatic or economy improvements |

## Output Template

```markdown
# Code Review Report

**Files reviewed:** [list]
**Date:** [date]
**Overall health:** [Good | Needs attention | Needs significant work]

## Executive Summary

[2-4 sentences: overall impression, dominant patterns, top priority action.]

## Findings

### 1. [Finding title]

- **Severity:** Major
- **Pillar:** Single Responsibility / Conciseness / etc.
- **Location:** `filename.py`, lines X-Y

BEFORE:
[original snippet — quote the relevant code exactly]

AFTER:
[illustrative refactored snippet — what it *would* look like]

WHY:
[one-line rationale linking to the principle violated]

### 2. ...

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | Major | SRP | model.py:45-60 | Class mixes IO and logic |

## Positive Highlights

[2-3 things the code does well.]
```

## Critical Rules

**The skill's ONLY output is a feedback report. It must NEVER:**

- Modify, overwrite, or create any source file.
- Produce refactored code outside of short BEFORE/AFTER illustration pairs in the report.
- Offer to apply changes — remind users this skill is feedback-only.
- Generate diffs, patches, or replacement files.

**The AFTER snippet in each finding is an illustrative sketch (<=10 lines) to clarify
the recommendation. It is NOT a deliverable. It is NOT a refactored version of the
file. The user should treat it as a visual aid, not as copy-paste-ready code.**
