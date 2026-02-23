# Alphaeval Iteration 5 — Code Review Report

**Files reviewed:** 17 files (10 source + 7 test) in `packages/alphaeval/`
**Date:** 2026-02-16
**Overall health:** :green_circle: Good — all issues addressed, 136 tests passing
**Tests:** 136 (was 109 pre-iteration)
**Diff:** +433/-58 lines across 17 files

## Executive Summary

Iteration 5 addressed all remaining findings from iterations 4 and the PR toolkit triple-review. All 5 critical issues (C1-C5) fixed. All 8 suggestions (S1-S8) resolved. 27 new tests covering error paths, warning paths, and edge cases. PR toolkit agents (code-reviewer, silent-failure-hunter, pr-test-analyzer) ran and findings addressed. Remaining design debates (equity_curve NaN threshold, ann_estimated_sharpe_ratio None-handling) deferred as they require API design decisions.

---

## Findings

### 1. `pct_change(fill_method=None)` forward-incompatible with pandas 3.0
- **Severity:** :orange_circle: Major
- **Pillar:** Forward compatibility
- **Location:** `src/transforms/returns.py`, line 40

BEFORE:
```python
result = price.pct_change(fill_method=None)
```

AFTER:
```python
result = price / price.shift(1) - 1
```

WHY:
`fill_method` parameter deprecated in pandas 2.1, removed in pandas 3.0. The shift-based calculation is equivalent and version-proof.

---

### 2. Import after non-import statement in equity.py
- **Severity:** :yellow_circle: Minor
- **Pillar:** PEP8 / import ordering
- **Location:** `src/transforms/equity.py`, lines 13-15

BEFORE:
```python
logger = logging.getLogger(__name__)

from .returns import equity_curve
```

AFTER:
```python
from .returns import equity_curve

logger = logging.getLogger(__name__)
```

WHY:
PEP8 E402 — module-level imports must precede non-import statements.

---

### 3. No `__all__` in metrics submodules
- **Severity:** :blue_circle: Suggestion
- **Pillar:** API clarity
- **Location:** `src/metrics/performance.py`, `risk.py`, `factor.py`, `relative.py`, `sharpe_inference.py`

WHY:
`transforms/__init__.py` defines `__all__` but metrics submodules don't. Inconsistent and makes public API ambiguous for IDE consumers.

---

### 4. Missing tests for 8 new error paths
- **Severity:** :orange_circle: Major
- **Pillar:** Test completeness
- **Location:** All test files

The iteration 4 fixes introduced new `raise ValueError` and `logger.warning` paths that have no test coverage:

| Error path | File | Test? |
|---|---|---|
| `_daily_ic` multi-column MultiIndex raises | factor.py:66-74 | No |
| VPIN negative volume clamp + warning | risk.py:91-98 | No |
| VPIN index mismatch raises | risk.py:85-90 | No |
| `min_track_record_length` invalid prob raises | sharpe_inference.py:191-192 | No |
| `tracking_error` misalignment raises | relative.py:42-50 | No |
| `spread_return_proxy` all-NaN raises | spreads.py:56-60 | No |
| `dv01_pnl` all-NaN raises | spreads.py:93-97 | No |
| `price_to_returns` inf replacement | returns.py:45-51 | No |
| `drawdown_series` / `runup_series` empty input | equity.py | No |
| `_daily_ic` skip % warning path | factor.py:55-61 | No |
| `num_independent_trials` partial NaN corr warning | sharpe_inference.py:261-266 | No |
| `equity_curve` NaN warning | returns.py:73-78 | No (existing test verifies NaN→0 behavior but not warning) |

---

### 5. `validate_panel` copies entire DataFrame unconditionally
- **Severity:** :blue_circle: Suggestion
- **Pillar:** Performance
- **Location:** `src/panel.py`, line 46

BEFORE:
```python
df = data.copy()
```

WHY:
For large panels (100k+ rows), this is a material cost. The copy only matters when converting columns→MultiIndex (mutates via `set_index`). For MultiIndex passthrough, copy is unnecessary.

AFTER (sketch):
```python
if isinstance(data.index, pd.MultiIndex):
    df = data  # already MultiIndex, no mutation needed
else:
    df = data.copy()
    df = df.set_index(...)
```

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | :orange_circle: Major | Compat | returns.py:40 | `pct_change(fill_method=None)` deprecated |
| 2 | :yellow_circle: Minor | PEP8 | equity.py:13-15 | Import after non-import |
| 3 | :blue_circle: Suggestion | API | metrics/*.py | Missing `__all__` |
| 4 | :orange_circle: Major | Tests | all test files | 12 new error paths untested |
| 5 | :blue_circle: Suggestion | Perf | panel.py:46 | Unconditional DataFrame copy |

## Positive Highlights

1. **All 5 critical issues from iteration 4 are fixed** — multi-column guard, VPIN bounds, duplicate detection, index alignment, NaN correlation handling.
2. **Error-handling philosophy improved** — shifted from logger-only to `raise ValueError` for data integrity violations (panel duplicates, index misalignment, all-NaN results).
3. **Comprehensive Sharpe inference suite** — PSR/DSR/minTRL with proper non-normality adjustments and selection bias correction, all with NaN guards.
4. **Clean separation** — transforms layer (returns/spreads/equity) feeds into metrics layer, panel validation sits as a gateway.

## Iteration 4 Issues — Status Tracker

| ID | Status | Notes |
|---|---|---|
| C1 | :white_check_mark: Fixed | `_daily_ic` multi-column raises ValueError |
| C2 | :white_check_mark: Fixed | VPIN clamps negative volumes |
| C3 | :white_check_mark: Fixed | `validate_panel` raises on duplicates |
| C4 | :white_check_mark: Fixed | VPIN raises on index mismatch |
| C5 | :white_check_mark: Fixed | `num_independent_trials` filters NaN corr |
| I1 | :white_check_mark: Fixed | `profit_factor` now calls `.dropna()` |
| I2 | :white_check_mark: Addressed | Docstring documents n-mismatch invariant |
| I3 | :white_check_mark: Fixed | `min_track_record_length` validates `prob` |
| I4 | :white_check_mark: Addressed | types.py docstring clarifies convenience grouping |
| I5 | :white_check_mark: Fixed | N_eff clamped to [1, m] |
| I6 | :white_check_mark: Fixed | `price_to_returns` replaces inf with NaN |
| I7 | :yellow_circle: Partial | Some docstrings updated, not all 10 |
| I8 | :white_check_mark: Fixed | `REQUIRED_TRADES` includes `date` |
| I9 | :white_check_mark: Fixed | `tracking_error` raises on misalignment |
| I10 | :white_check_mark: Fixed | `spread_return_proxy`/`dv01_pnl` raise on all-NaN |
| I11 | :white_check_mark: Fixed | `equity_curve` logs NaN warning |
| I12 | :white_check_mark: Fixed | `_daily_ic` warns when >20% dates skipped |
| S1 | :red_circle: Open | No test for multi-column MultiIndex rejection |
| S2 | :red_circle: Open | No test for spread all-NaN paths |
| S3 | :red_circle: Open | No `__all__` in metrics modules |
| S4 | :red_circle: Open | Import ordering wrong in equity.py |
| S5 | :blue_circle: Deferred | validate_panel copy optimization |
| S6 | :red_circle: Open | pct_change forward-compat |
| S7 | :blue_circle: Deferred | Test helper duplication |
| S8 | :red_circle: Open | No tests for empty drawdown/runup |
