# Code Review Report — v5

**Files reviewed:** `packages/backtest/src/engine.py`, `packages/backtest/src/portfolio.py`, `packages/backtest/src/results.py`, `packages/backtest/src/__init__.py`, `packages/backtest/tests/test_engine.py`, `packages/data/src/trace.py`, `packages/data/src/universe.py`, `packages/data/src/validation.py`, `packages/data/src/reference.py`, `packages/data/src/config.py`, `packages/data/tests/test_validation.py`, `packages/data/tests/test_config.py`, `packages/signals/src/credit.py`, `packages/signals/src/retail.py`, `packages/signals/src/triggers.py`, `packages/signals/tests/test_credit.py`, `packages/signals/tests/test_retail.py`, `packages/signals/tests/test_triggers.py`, `packages/metrics/src/performance.py`, `packages/metrics/src/risk.py`, `packages/metrics/src/diagnostics.py`, `packages/metrics/src/__init__.py`, `packages/metrics/tests/test_performance.py`, `packages/metrics/tests/test_risk.py`
**Date:** 2026-02-07
**Overall health:** :green_circle: Good

## Executive Summary

After two rounds of review and implementation, the codebase is in good shape. The remaining findings are all minor: dead code (unused function, unused import, dead guard), non-deterministic RNG in test files (legacy `np.random` API without seed in several locations), and a minor consistency gap in `from __future__ import annotations`. No critical or major issues remain.

## Findings

### 1. Unused function `_rating_to_numeric` in universe.py

- **Severity:** :yellow_circle: Minor
- **Pillar:** Conciseness / Dead Code
- **Location:** `packages/data/src/universe.py`, lines 45–50

BEFORE:
```python
def _rating_to_numeric(rating: str) -> int:
    """Convert rating to numeric value (higher = better)."""
    try:
        return len(RATING_ORDER) - RATING_ORDER.index(rating)
    except ValueError:
        return -1  # Unknown rating
```

AFTER:
*(remove entirely)*

WHY:
Function is never called anywhere in the codebase. Dead code increases maintenance surface.

---

### 2. Unused `tempfile` import in test_config.py

- **Severity:** :yellow_circle: Minor
- **Pillar:** Conciseness / Dead Import
- **Location:** `packages/data/tests/test_config.py`, line 3

BEFORE:
```python
import tempfile
from pathlib import Path
```

AFTER:
```python
from pathlib import Path
```

WHY:
`tempfile` is imported but never used. Tests use `tmp_path` (pytest fixture) instead.

---

### 3. Dead `position_sizer` guard in config dict

- **Severity:** :yellow_circle: Minor
- **Pillar:** Conciseness / Dead Branch
- **Location:** `packages/backtest/src/engine.py`, line 238

BEFORE:
```python
"position_sizer": position_sizer.__name__ if position_sizer else "none",
```

AFTER:
```python
"position_sizer": position_sizer.__name__,
```

WHY:
By line 238, `position_sizer` is guaranteed non-None (defaulted to `equal_weight` on line 202). The `else "none"` branch is unreachable.

---

### 4. Non-deterministic RNG in test files (legacy `np.random` API)

- **Severity:** :yellow_circle: Minor
- **Pillar:** Testing / Determinism
- **Location:** Multiple test files

**4a.** `packages/metrics/tests/test_risk.py`, lines 104, 114–115, 121–122: Uses `np.random.seed(42)` + `np.random.randn()` (legacy global RNG).

**4b.** `packages/metrics/tests/test_risk.py`, line 168: Uses `np.random.randn(100)` with **no seed** — fully non-deterministic.

**4c.** `packages/metrics/tests/test_performance.py`, line 68: Uses `np.random.shuffle(returns.values)` with **no seed** — fully non-deterministic.

**4d.** `packages/signals/tests/test_triggers.py`, line 72: Uses `np.random.randn(100)` with **no seed** — fully non-deterministic.

BEFORE (example from test_risk.py:104):
```python
np.random.seed(42)
returns = pd.Series(np.random.randn(1000) * 0.01)
```

AFTER:
```python
rng = np.random.default_rng(42)
returns = pd.Series(rng.standard_normal(1000) * 0.01)
```

WHY:
Per python-standards.md: "Fixed RNG seeds, fixed timestamps, or mocked time sources." Legacy `np.random.seed()` uses mutable global state susceptible to cross-test contamination. Unseeded calls (4b, 4c, 4d) produce different results on each run, making test failures non-reproducible. `test_engine.py` was already fixed in v4 — these are the remaining occurrences.

---

### 5. Missing `from __future__ import annotations` in metrics source files

- **Severity:** :blue_circle: Suggestion
- **Pillar:** Consistency
- **Location:** `packages/metrics/src/performance.py`, line 1; `packages/metrics/src/risk.py`, line 1

BEFORE (performance.py):
```python
"""Performance metrics.
```

AFTER:
```python
from __future__ import annotations

"""Performance metrics.
```

WHY:
Every other source file across all 4 packages includes `from __future__ import annotations` (engine.py, portfolio.py, results.py, trace.py, universe.py, validation.py, reference.py, config.py, credit.py, retail.py, triggers.py, diagnostics.py). These two are the only source files missing it. While they don't currently use PEP 604 syntax, adding the import maintains consistency and prevents issues if union types are added later.

---

### 6. Redundant length check before regex in `validate_cusip`

- **Severity:** :blue_circle: Suggestion
- **Pillar:** Conciseness
- **Location:** `packages/data/src/validation.py`, lines 48–53

BEFORE:
```python
if len(cusip) != 9:
    return False

pattern = r"^[A-Z0-9]{9}$"
return bool(re.match(pattern, cusip.upper()))
```

AFTER:
```python
return bool(re.match(r"^[A-Z0-9]{9}$", cusip.upper()))
```

WHY:
The regex `^[A-Z0-9]{9}$` already enforces exactly 9 characters via `{9}` with anchors. The preceding length check is semantically redundant. However, it does serve as a micro-optimization (avoiding regex engine invocation for wrong-length strings), so this is purely a style preference.

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | :yellow_circle: Minor | Dead Code | universe.py:45–50 | Unused `_rating_to_numeric` function |
| 2 | :yellow_circle: Minor | Dead Import | test_config.py:3 | Unused `tempfile` import |
| 3 | :yellow_circle: Minor | Dead Branch | engine.py:238 | Unreachable `else "none"` guard |
| 4 | :yellow_circle: Minor | Testing/Determinism | test_risk.py, test_performance.py, test_triggers.py | Legacy/unseeded RNG in tests |
| 5 | :blue_circle: Suggestion | Consistency | performance.py:1, risk.py:1 | Missing `from __future__ import annotations` |
| 6 | :blue_circle: Suggestion | Conciseness | validation.py:48–53 | Redundant length check before regex |

## Positive Highlights

- Codebase is clean after two review iterations. No critical or major issues remain.
- Vectorized operations are well-implemented across all packages (portfolio ranking, CUSIP validation, QMP classification).
- Lookahead prevention rules in the backtest engine are properly documented and implemented.
- Zero-variance handling in z-score computation uses correct pre-masking pattern.
- Test coverage is solid: all major functions have tests, edge cases (zero spread, empty data, NaN) are covered.
