# Code Review Report (v4)

**Files reviewed:** 18 source files + 8 test files across `packages/{data,signals,metrics,backtest}`
**Date:** 2026-02-07
**Previous review:** `2026_02_07_FlowCode_packages_review_v3.md` (v3)
**Overall health:** 🟢 Good

## Executive Summary

All v3 findings were implemented successfully. This v4 review shifts focus to test quality gaps and remaining conciseness issues per `references/python-standards.md`. The primary concerns are: (1) test non-determinism from unseeded `np.random.randn()`, (2) a remaining dead branch in `compute_returns` that was only half-fixed in v3, (3) unused `Literal` import in `universe.py`, and (4) two `cusip` column checks that can be merged in `validate_reference`. No correctness bugs found.

## Findings

### 1. Non-deterministic tests using unseeded `np.random.randn()`

- **Severity:** 🟡 Minor
- **Pillar:** Testing / Determinism
- **Location:** `packages/backtest/tests/test_engine.py`, lines 109–116

BEFORE:
```python
def test_basic_backtest(self) -> None:
    dates = pd.date_range("2023-01-01", periods=10)
    signal = pd.DataFrame(
        {"A": np.random.randn(10), "B": np.random.randn(10)},
        index=dates,
    )
    prices = pd.DataFrame(
        {"A": 100 * (1 + np.random.randn(10).cumsum() * 0.01),
         "B": 100 * (1 + np.random.randn(10).cumsum() * 0.01)},
        index=dates,
    )
```

AFTER:
```python
def test_basic_backtest(self) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=10)
    signal = pd.DataFrame(
        {"A": rng.standard_normal(10), "B": rng.standard_normal(10)},
        index=dates,
    )
    prices = pd.DataFrame(
        {"A": 100 * (1 + rng.standard_normal(10).cumsum() * 0.01),
         "B": 100 * (1 + rng.standard_normal(10).cumsum() * 0.01)},
        index=dates,
    )
```

WHY: `python-standards.md` requires "Fixed RNG seeds, fixed timestamps, or mocked time sources." Unseeded random data makes tests non-reproducible — failures become unrepeatable.

---

### 2. Remaining dead branch in `compute_returns`

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness
- **Location:** `packages/backtest/src/engine.py`, lines 59–62

BEFORE:
```python
    # Strategy return = sum of (position * return) across assets
    if isinstance(asset_returns, pd.Series):
        strategy_returns = lagged_positions * asset_returns
    else:
        strategy_returns = (lagged_positions * asset_returns).sum(axis=1)
```

AFTER:
```python
    # Strategy return = sum of (position * return) across assets
    strategy_returns = (lagged_positions * asset_returns).sum(axis=1)
```

WHY: `compute_returns` takes `prices: pd.DataFrame`, so `prices.pct_change()` always returns a DataFrame. The `isinstance(..., pd.Series)` branch is unreachable. Additionally, `.sum(axis=1)` works correctly on single-column DataFrames, so the branch adds no value even as a guard.

---

### 3. Unused `Literal` import in `universe.py`

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness
- **Location:** `packages/data/src/universe.py`, line 10

BEFORE:
```python
from typing import Literal
```

AFTER:
```python
# Remove — Literal is not used anywhere in universe.py
```

WHY: `Literal` is imported but never used. Ruff/flake8 would flag this as `F401`.

---

### 4. Duplicate `if "cusip" in df.columns` guard in `validate_reference`

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness
- **Location:** `packages/data/src/validation.py`, lines 215–226

BEFORE:
```python
    # Check for duplicate CUSIPs
    if "cusip" in df.columns:
        duplicates = df["cusip"].duplicated().sum()
        stats["duplicate_cusips"] = duplicates
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate CUSIPs")

    # Validate CUSIP format (vectorized)
    if "cusip" in df.columns:
        n_invalid, cusip_warning = _validate_cusip_column(df)
        ...
```

AFTER:
```python
    if "cusip" in df.columns:
        # Check for duplicates
        duplicates = df["cusip"].duplicated().sum()
        stats["duplicate_cusips"] = duplicates
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate CUSIPs")

        # Validate format (vectorized)
        n_invalid, cusip_warning = _validate_cusip_column(df)
        stats["invalid_cusips"] = n_invalid
        if cusip_warning:
            warnings.append(cusip_warning)
```

WHY: Two consecutive `if "cusip" in df.columns` guards check the same condition. Merge into one block.

---

### 5. `results.py` missing `from __future__ import annotations`

- **Severity:** 🟡 Minor
- **Pillar:** Compatibility
- **Location:** `packages/backtest/src/results.py`, line 1

BEFORE:
```python
"""Backtest result container.
```

AFTER:
```python
from __future__ import annotations

"""Backtest result container.
```

WHY: All other source files in the codebase use `from __future__ import annotations` for Python 3.9 compatibility. `results.py` is the only source file missing it, and it uses `dict[str, float]` and `dict[str, Any]` type annotations that require it under Python 3.9.

---

### 6. `generate_trades` docstring lists `pnl` column that is never produced

- **Severity:** 🟡 Minor
- **Pillar:** Semantics & Correctness (documentation)
- **Location:** `packages/backtest/src/engine.py`, line 125

BEFORE:
```python
    Returns
    -------
    pd.DataFrame
        Trade log with columns: date, asset, side, size, price, pnl.
```

AFTER:
```python
    Returns
    -------
    pd.DataFrame
        Trade log with columns: date, asset, side, size, price.
```

WHY: The function returns `trades_df[["date", "asset", "side", "size", "price"]]` — there is no `pnl` column. The docstring is misleading and creates confusion about the API contract (note: `BacktestResult.win_rate` checks for `"pnl" in self.trades.columns` and finds it missing).

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | 🟡 Minor | Testing | `test_engine.py:109–116` | Unseeded `np.random.randn()` |
| 2 | 🟡 Minor | Conciseness | `engine.py:59–62` | Remaining dead branch |
| 3 | 🟡 Minor | Conciseness | `universe.py:10` | Unused `Literal` import |
| 4 | 🟡 Minor | Conciseness | `validation.py:215–226` | Duplicate `cusip` column guard |
| 5 | 🟡 Minor | Compatibility | `results.py:1` | Missing `from __future__ import annotations` |
| 6 | 🟡 Minor | Semantics | `engine.py:125` | Docstring mentions `pnl` column not produced |

## Positive Highlights

1. **All v3 findings cleanly resolved.** The extracted `_validate_cusip_column` helper, vectorized `top_n_positions`, and complete `__init__.py` exports are well-implemented.
2. **`compute_metrics` now uses explicit `ddof=1`** — consistency with the metrics package is maintained.
3. **118 tests pass across all 4 packages** — zero failures, solid foundation.
4. **Package boundaries remain clean** — no circular dependencies introduced.
