# Alphaeval Iteration 6 — Code Review Report

**Files reviewed:** 13 source + 7 test files in `packages/alphaeval/`
**Date:** 2026-02-16
**Overall health:** :green_circle: Good — 137 tests passing, 1 bug found + fixed
**Tests:** 137 (was 136 pre-iteration)
**Diff:** +15/-2 lines (incremental from iteration 5)

## Executive Summary

Fresh review pass after iteration 5. Found one genuine bug: `validate_panel` MultiIndex passthrough mutated the caller's DataFrame when date coercion was needed (string dates → datetime). The no-copy optimization from S5 didn't account for the coercion path. Fixed with a conditional copy (`if df is data`). All other code is clean — no new issues found.

---

## Findings

### 1. `validate_panel` mutates caller's DataFrame on MultiIndex date coercion
- **Severity:** :red_circle: Critical
- **Pillar:** Correctness / side-effect safety
- **Location:** `src/panel.py`, lines 63-70

BEFORE:
```python
df = data  # MultiIndex path, no copy
# ...later...
df.index = df.index.set_levels(...)  # mutates data too!
```

AFTER:
```python
if df is data:  # MultiIndex passthrough — copy to avoid mutating caller
    df = data.copy()
df.index = df.index.set_levels(...)
```

WHY:
When `df = data` (same object), `df.index = ...` mutates the caller's DataFrame. The S5 optimization (skip copy for MultiIndex) didn't account for the date coercion branch. The fix defers the copy to only when coercion is actually needed.

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | :red_circle: Critical | Correctness | panel.py:63-70 | MultiIndex coercion mutates caller |

## Positive Highlights

1. **All iteration 5 fixes remain solid** — 136→137 tests, no regressions.
2. **Error handling is comprehensive** — every raise/warn path has test coverage.
3. **`__all__` exports consistent** across all 5 metrics submodules.
4. **Performance optimization preserved** — no-copy path still works for the common case (datetime MultiIndex passthrough).

## Iteration 5 Issues — Status Tracker (Final)

| ID | Status | Notes |
|---|---|---|
| S1-S8 | :white_check_mark: All fixed | See iteration 5 report |
| NEW-1 | :white_check_mark: Fixed | MultiIndex date coercion mutation bug |
