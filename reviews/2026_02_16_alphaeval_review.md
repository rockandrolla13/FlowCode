# Code Review Report

**Files reviewed:**
- `packages/alphaeval/src/types.py`
- `packages/alphaeval/src/panel.py`
- `packages/alphaeval/src/metrics/performance.py`
- `packages/alphaeval/src/metrics/risk.py`
- `packages/alphaeval/src/metrics/factor.py`
- `packages/alphaeval/src/metrics/relative.py`
- `packages/alphaeval/src/metrics/sharpe_inference.py`
- `packages/alphaeval/src/transforms/returns.py`
- `packages/alphaeval/src/transforms/equity.py`
- `packages/alphaeval/src/transforms/spreads.py`
- Associated test files

**Date:** 2026-02-16
**Overall health:** Good

## Executive Summary

The alphaeval package implements corporate credit alpha evaluation metrics with solid correctness. Code is well-structured with consistent NaN/edge-case handling and good test coverage. The main concerns are: (1) a potential subtle bug in `_daily_ic` when handling edge cases with <3 instruments, (2) minor conciseness improvements, and (3) one missing negative test for empty panel validation. Overall implementation quality is high.

## Findings

### 1. _daily_ic: Loop-based IC computation should use vectorized groupby
- **Severity:** Minor
- **Pillar:** Performance / Conciseness
- **Location:** `factor.py`, lines 37-53

BEFORE:
```python
for dt in dates:
    s = signal.loc[dt].dropna()
    t = target.loc[dt].dropna()
    common = s.index.intersection(t.index)
    if len(common) < 3:
        logger.debug("IC: date %s has %d instruments (<3), skipping", dt, len(common))
        ics.append(np.nan)
        continue
    if method == "spearman":
        ics.append(float(s[common].rank().corr(t[common].rank())))
    else:
        ics.append(float(s[common].corr(t[common])))
return pd.Series(ics, index=dates, name="ic")
```

AFTER:
```python
def _ic_for_date(sig_row, tgt_row, method):
    common = sig_row.dropna().index.intersection(tgt_row.dropna().index)
    if len(common) < 3:
        return np.nan
    s, t = sig_row[common], tgt_row[common]
    return s.rank().corr(t.rank()) if method == "spearman" else s.corr(t)

ics = pd.Series({dt: _ic_for_date(signal.loc[dt], target.loc[dt], method) for dt in dates}, name="ic")
```

WHY:
Dict comprehension + helper is more Pythonic than manual list-append loop.

---

### 2. profit_factor: Near-zero comparison uses literal instead of named constant
- **Severity:** Suggestion
- **Pillar:** Consistency
- **Location:** `performance.py`, line 35

BEFORE:
```python
if abs(losses) < 1e-14:
```

AFTER:
```python
EPSILON = 1e-14  # module-level or from types.py
if abs(losses) < EPSILON:
```

WHY:
Other files use `1e-14` inline; centralizing as `EPSILON` improves maintainability.

---

### 3. validate_panel: Missing test for empty DataFrame edge case
- **Severity:** Minor
- **Pillar:** Testing / Edge Cases
- **Location:** `test_panel.py` (missing test)

BEFORE:
No test for empty input DataFrame.

AFTER:
```python
def test_empty_dataframe(self) -> None:
    df = pd.DataFrame({"date": [], "instrument": [], "returns": []})
    result = validate_panel(df, required_cols={"date", "instrument", "returns"})
    assert len(result) == 0
    assert isinstance(result.index, pd.MultiIndex)
```

WHY:
Empty input is a valid edge case that should be tested to ensure no exceptions.

---

### 4. vpin: Index alignment warning could be more precise
- **Severity:** Suggestion
- **Pillar:** Diagnostics
- **Location:** `risk.py`, lines 79-83

BEFORE:
```python
if not volume_buy.index.equals(volume_sell.index):
    logger.warning(
        "vpin: volume_buy and volume_sell have different indices; "
        "misalignment will introduce NaN"
    )
```

AFTER:
```python
if not volume_buy.index.equals(volume_sell.index):
    n_mismatch = len(volume_buy.index.symmetric_difference(volume_sell.index))
    logger.warning(
        "vpin: indices differ by %d entries; misalignment may produce NaN",
        n_mismatch,
    )
```

WHY:
Quantifying the mismatch helps debugging.

---

### 5. num_independent_trials: NaN avg_corr fallback to 1.0 may surprise users
- **Severity:** Minor
- **Pillar:** Semantics / Documentation
- **Location:** `sharpe_inference.py`, lines 247-252

BEFORE:
```python
if np.isnan(avg_corr):
    logger.warning(
        "num_independent_trials: avg_corr is NaN (constant returns?); "
        "defaulting to 1.0 (fully correlated, N_eff=1) as conservative estimate"
    )
    avg_corr = 1.0
```

AFTER:
Consider adding this to docstring:

```python
Notes
-----
If avg_corr is NaN (e.g., constant returns), defaults to 1.0 (fully correlated),
yielding N_eff=1. This is conservative for DSR but may over-penalize.
```

WHY:
The fallback behavior should be discoverable in the function signature, not just logs.

---

### 6. sortino_ratio: Convention mismatch between alphaeval and packages/metrics
- **Severity:** Minor
- **Pillar:** Documentation
- **Location:** `performance.py`, lines 163-166

BEFORE:
```python
"""Sortino ratio with population downside deviation (1/T).

Uses spec convention: downside_std = sqrt(1/T * sum(min(0, r-tau)^2)).
This differs from packages/metrics which uses ddof=1.
```

WHY:
This is correctly documented but worth highlighting: alphaeval uses 1/T (population) while packages/metrics uses ddof=1 (sample). Users mixing both packages may get inconsistent results. Consider adding a cross-reference or unifying conventions.

---

### 7. expected_maximum_sr: EMC constant could use math.e for Euler's number
- **Severity:** Suggestion
- **Pillar:** Readability
- **Location:** `sharpe_inference.py`, line 313

BEFORE:
```python
+ EMC * scipy_stats.norm.ppf(1 - 1.0 / (independent_trials * np.e))
```

AFTER:
Already uses `np.e`; EMC is correctly 0.5772156649 (Euler-Mascheroni). No change needed.

---

### 8. r_squared: Function arguments are (predicted, actual) — verify consistency
- **Severity:** Suggestion
- **Pillar:** API Consistency
- **Location:** `factor.py`, lines 142-145

BEFORE:
```python
def r_squared(
    predicted: pd.Series,
    actual: pd.Series,
) -> float:
```

WHY:
Standard convention is often (actual, predicted) or (y_true, y_pred). sklearn uses (y_true, y_pred). Current order is (predicted, actual). While internally consistent, document explicitly that order differs from sklearn.

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | Minor | Performance | `factor.py:37-53` | Loop-based IC → dict comprehension |
| 2 | Suggestion | Consistency | `performance.py:35` | Inline 1e-14 → named EPSILON |
| 3 | Minor | Testing | `test_panel.py` | Missing empty DataFrame test |
| 4 | Suggestion | Diagnostics | `risk.py:79-83` | Quantify index mismatch |
| 5 | Minor | Documentation | `sharpe_inference.py:247` | Document NaN fallback in docstring |
| 6 | Minor | Documentation | `performance.py:163` | Cross-package convention mismatch |
| 7 | N/A | Readability | `sharpe_inference.py:313` | EMC/np.e usage is correct |
| 8 | Suggestion | API | `factor.py:142` | Document (predicted, actual) order |

## Positive Highlights

1. **Robust NaN/edge-case handling** — Every metric guards against insufficient data, zero variance, and division by zero with explicit `if abs(x) < 1e-14` checks and returns `np.nan` appropriately.

2. **Comprehensive test coverage** — Tests cover happy path, edge cases (empty, single obs, zero std), and known-value verification. Population vs sample convention is explicitly tested (`test_population_convention`).

3. **Clear spec alignment** — The `sortino_ratio` docstring explicitly notes the 1/T vs ddof=1 convention difference, preventing silent mismatches. Formula references are traceable to `alpha_analytics.md`.

4. **Good separation of concerns** — `transforms/` (equity curves, spread changes) cleanly separates data transformation from metrics computation. Panel validation is isolated in `panel.py`.

5. **Logging discipline** — All warning-level logs include quantitative context (e.g., "dropped %d of %d observations"), aiding debugging without excessive verbosity.
