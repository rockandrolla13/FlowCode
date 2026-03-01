# Code Review Report

**Files reviewed:** 62 Python files across packages/, intraday modules, credit-analytics/, tools/
**Date:** 2026-03-01
**Overall health:** 🟡 Needs attention

## Executive Summary

The 5 packages (data, signals, metrics, backtest, alphaeval) are well-written: defensive NaN handling, consistent NumPy-style docstrings, type hints on all public functions, and 363 passing tests. The main concerns are: (1) a deprecated pandas API call in backtest that will break on pandas 3.0, (2) inconsistent side validation that silently drops trades, (3) the credit-analytics subtree containing unsafe code that could be imported accidentally, and (4) `verify_spec_parity.py` doesn't verify actual formula correctness. The intraday modules are stylistically older (using `typing.Optional` instead of `X | None`) but functionally sound.

## Findings

### CR-BUG-001: `prices.pct_change()` missing `fill_method=None`
- **Severity:** 🟠 Major
- **Pillar:** Correctness
- **Location:** `packages/backtest/src/engine.py:L52`

BEFORE:
```python
asset_returns = prices.pct_change()
```

AFTER:
```python
asset_returns = prices / prices.shift(1) - 1
```

WHY:
pandas 2.1+ deprecates the `fill_method` parameter defaulting to `'pad'`; pandas 3.0 will remove it. Already fixed in alphaeval (`transforms/returns.py:40`) but not in backtest. Will emit `FutureWarning` on every run and break on pandas upgrade.

---

### CR-BUG-002: `validate_trace` accepts sides that downstream functions ignore
- **Severity:** 🟠 Major
- **Pillar:** Correctness
- **Location:** `packages/data/src/validation.py:L153`

BEFORE:
```python
valid_sides = {"B", "S", "buy", "sell", "BUY", "SELL"}
```

AFTER:
```python
valid_sides = {"B", "S"}
# If other side conventions exist, normalize them first
```

WHY:
`compute_retail_imbalance` (retail.py:454-455) and `aggregate_daily_volume` (trace.py:138-139) filter on `side == "B"` and `side == "S"` only. Trades with `side == "buy"` pass validation but are silently dropped from all downstream aggregations. This is a data integrity gap — validation claims the data is valid, but computations silently lose rows.

---

### CR-BUG-003: `average_holding_time` mutates caller's DataFrame
- **Severity:** 🟠 Major
- **Pillar:** Correctness
- **Location:** `credit-analytics/packages/metrics/src/performance.py:L206`

BEFORE:
```python
trade_log['holding_time'] = trade_log['exit_time'] - trade_log['entry_time']
```

AFTER:
```python
holding_time = trade_log['exit_time'] - trade_log['entry_time']
return trade_log.groupby('isin')[holding_time].mean()  # or use .assign()
```

WHY:
Modifies the caller's DataFrame in-place via `trade_log['holding_time'] = ...`. Violates the "no side effects" convention established in CLAUDE.md. This is in the legacy credit-analytics code, reinforcing the case for deletion.

---

### CR-BUG-004: `verify_spec_parity.py` only checks file existence, not formula correctness
- **Severity:** 🟡 Minor
- **Pillar:** Correctness
- **Location:** `tools/verify_spec_parity.py:L59-73`

BEFORE:
```python
def check_code_location(code_location: str) -> bool:
    # checks if file exists at expected path
    expected_path = ROOT / "packages" / package / "src" / f"{module}.py"
    return expected_path.exists()
```

AFTER:
```python
# Should also verify: function exists in module, tests use fixture files,
# fixture values match code output
```

WHY:
The tool's name promises "spec parity" but only checks file existence. It doesn't verify that the named function exists in the file, that tests load from the fixture files, or that running the function with fixture inputs produces fixture outputs. `verify_spec_parity.py` could pass while formulas are completely wrong.

---

### CR-STYLE-001: Inconsistent `Optional` vs `X | None` syntax
- **Severity:** 🟡 Minor
- **Pillar:** Style
- **Location:** `intraday_microstructure_analytics.py:L39`, `intraday_quote_filters.py`

BEFORE:
```python
from typing import Optional
source: Optional[CompositeSource] = None
```

AFTER:
```python
source: CompositeSource | None = None
```

WHY:
All 5 packages use the modern `X | None` syntax (via `from __future__ import annotations`). The intraday modules use `typing.Optional`. Should be consistent per CLAUDE.md conventions.

---

### CR-STYLE-002: `from __future__ import annotations` before module docstring
- **Severity:** 🔵 Suggestion
- **Pillar:** Style
- **Location:** `packages/data/src/trace.py:L1-2`, `packages/data/src/reference.py:L1-2`

BEFORE:
```python
from __future__ import annotations

"""TRACE data loading utilities. ..."""
```

AFTER:
```python
"""TRACE data loading utilities. ..."""
from __future__ import annotations
```

WHY:
Per PEP 257, the module docstring should be the first statement. While Python accepts the future import before it (it's valid syntax), some documentation tools and IDEs may not pick up the docstring. All other modules in the repo place the docstring first.

---

### CR-DRY-001: `_safe_diff` logic duplicated
- **Severity:** 🟡 Minor
- **Pillar:** DRY
- **Location:** `intraday_quote_filters.py` (L77-104), `intraday_microstructure_analytics.py` (no explicit `_safe_diff` but same pattern inline)

BEFORE:
```python
# In intraday_quote_filters.py
def _safe_diff(data, col, isin_col="isin"):
    # prevents cross-ISIN contamination in diff()
```

AFTER:
```python
# If promoted to a package, extract to shared utility
```

WHY:
Cross-ISIN diff contamination is documented as a top failure mode in CLAUDE.md. The guard logic should live in exactly one place, not be copy-pasted between related modules.

---

### CR-PERF-001: `filter_by_rating` reconstructs valid_ratings sets on every call
- **Severity:** 🔵 Suggestion
- **Pillar:** Performance
- **Location:** `packages/data/src/universe.py:L163-170`

BEFORE:
```python
if min_rating is not None:
    min_idx = RATING_ORDER.index(min_rating)
    valid_ratings = set(RATING_ORDER[: min_idx + 1])
    mask &= df[rating_col].isin(valid_ratings)
```

AFTER:
```python
# Pre-compute rating sets or use pd.Categorical with ordered=True
```

WHY:
Minor performance note. For large DataFrames called repeatedly, converting `RATING_ORDER` to a `pd.CategoricalDtype(ordered=True)` would enable simple `<=` comparisons instead of set membership tests. Not urgent for current scale.

## Summary Table
| Finding ID | Severity | Pillar | Location | Finding |
|------------|----------|--------|----------|---------|
| CR-BUG-001 | 🟠 Major | Correctness | `backtest/src/engine.py:L52` | `pct_change()` deprecated, will break on pandas 3.0 |
| CR-BUG-002 | 🟠 Major | Correctness | `data/src/validation.py:L153` | Validation accepts side values that downstream functions silently ignore |
| CR-BUG-003 | 🟠 Major | Correctness | `credit-analytics/.../performance.py:L206` | `average_holding_time` mutates caller's DataFrame |
| CR-BUG-004 | 🟡 Minor | Correctness | `tools/verify_spec_parity.py:L59-73` | Only checks file existence, not formula correctness |
| CR-STYLE-001 | 🟡 Minor | Style | `intraday_*.py` | `Optional` vs `X | None` inconsistency with packages |
| CR-STYLE-002 | 🔵 Suggestion | Style | `data/src/trace.py:L1-2` | `__future__` import before module docstring |
| CR-DRY-001 | 🟡 Minor | DRY | `intraday_quote_filters.py:L77-104` | `_safe_diff` logic should be single-sourced |
| CR-PERF-001 | 🔵 Suggestion | Performance | `data/src/universe.py:L163` | Rating set reconstruction on each call |

## Positive Highlights

1. **Thorough NaN/inf guards** — Every numeric function in the 5 packages handles edge cases (zero std, empty input, insufficient history) with explicit `logger.warning()` and `np.nan` returns. No silent failures.

2. **NumPy-style docstrings throughout** — Consistent, complete docstrings with Parameters, Returns, Notes, and Examples sections. Meets CLAUDE.md conventions.

3. **Fixture-driven testing** — Tests load expected values from `spec/fixtures/*.json` rather than hardcoding, making it easy to verify spec parity and update expected values when formulas change.

---

## Handoff

| Severity | Pillar | Location | Finding | Finding ID |
|----------|--------|----------|---------|------------|
| 🟠 Major | Correctness | `backtest/src/engine.py:L52` | `pct_change()` deprecated; will break on pandas 3.0 | CR-BUG-001 |
| 🟠 Major | Correctness | `data/src/validation.py:L153` | Side validation accepts values that downstream drops silently | CR-BUG-002 |
| 🟠 Major | Correctness | `credit-analytics/.../performance.py:L206` | Input DataFrame mutation via column assignment | CR-BUG-003 |
| 🟡 Minor | Correctness | `tools/verify_spec_parity.py:L59-73` | Parity checker only verifies file existence, not formula output | CR-BUG-004 |
| 🟡 Minor | Style | `intraday_*.py` | `Optional` vs `X | None` inconsistency | CR-STYLE-001 |
| 🔵 Suggestion | Style | `data/src/trace.py:L1-2` | `__future__` import before docstring | CR-STYLE-002 |
| 🟡 Minor | DRY | `intraday_quote_filters.py:L77-104` | `_safe_diff` duplicated between intraday modules | CR-DRY-001 |
| 🔵 Suggestion | Performance | `data/src/universe.py:L163` | Rating set reconstructed per call | CR-PERF-001 |
