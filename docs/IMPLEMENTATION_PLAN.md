# FlowCode Credit Analytics: Implementation Plan

**Author**: Claude Code
**Date**: 2026-02-03
**Status**: Approved for Implementation

---

## Executive Summary

This document outlines the implementation plan for transforming FlowCode into a production-grade credit analytics monorepo. The work is divided into **6 Pull Requests**, each with clear scope, dependencies, and acceptance criteria.

**Total estimated files**: 45+
**Package count**: 5 (data, core, signals, metrics, backtest)

---

## Architecture Overview

```
FlowCode/
├── CLAUDE.md                      # Project rules, comprehension gates
├── conf/                          # YAML configuration
│   ├── base.yaml
│   └── signals/
├── packages/
│   ├── data/                      # Data loading (sole file reader)
│   ├── core/                      # Existing library (DO NOT MODIFY)
│   ├── signals/                   # Signal wrappers over core
│   ├── metrics/                   # Performance & risk metrics
│   └── backtest/                  # Strategy harness
├── spec/
│   ├── SPEC.md                    # Formula definitions
│   ├── fixtures/                  # Golden test cases
│   └── decisions/                 # ADRs
├── tools/
│   ├── verify_spec_parity.py
│   └── notebooks/
├── skills/                        # Claude Code skills
└── .github/workflows/
```

### Package Dependency Graph

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

---

## PR Strategy

### Why 6 PRs?

1. **Atomic changes**: Each PR is reviewable in isolation
2. **Rollback safety**: Issues in one package don't block others
3. **Parallel development**: After PR1, PR2-PR5 could theoretically parallelize
4. **CI validation**: Each PR must pass tests before merge

### PR Dependency Chain

```
PR1: Scaffold
  ↓
PR2: packages/data/
  ↓
PR3: packages/signals/  (depends on data)
  ↓
PR4: packages/metrics/  (depends on signals output format)
  ↓
PR5: packages/backtest/ (depends on all)
  ↓
PR6: spec/, tools/, skills/, CI
```

---

## PR1: Scaffold & Project Foundation

**Branch**: `feat/scaffold`
**Scope**: Project structure, CLAUDE.md, configuration

### Files Created

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project rules, comprehension gates, code standards |
| `conf/base.yaml` | Data paths, shared defaults |
| `conf/signals/zscore.yaml` | Z-score parameters |
| `conf/signals/streak.yaml` | Streak parameters |
| `packages/__init__.py` | Package namespace |
| `packages/data/` | Directory only (empty `__init__.py`) |
| `packages/signals/` | Directory only |
| `packages/metrics/` | Directory only |
| `packages/backtest/` | Directory only |
| `spec/` | Directory only |
| `tools/` | Directory only |
| `skills/` | Directory only |

### CLAUDE.md Sections

1. **Ownership Model** — Human vs Agent responsibilities
2. **Comprehension Gates** — 60-second explain, change simulation, 2am debug
3. **Code Standards** — Type hints, NumPy docstrings, DataFrame conventions
4. **Configuration** — conf/ structure, load_config() usage
5. **Package Boundaries** — Import rules, dependency direction
6. **Data Flow** — Who loads what, DataFrame contracts
7. **Spec as Single Source of Truth** — Formula registry (empty initially)
8. **Package sections** — data/, signals/, metrics/, backtest/
9. **Adding New Signals** — Checklist
10. **Adding New Data Sources** — Checklist
11. **Failure Mode Catalog** — Known issues
12. **Commands** — pytest, mypy, ruff, verify_spec_parity

### Acceptance Criteria

- [ ] `CLAUDE.md` exists and contains all sections
- [ ] `conf/base.yaml` loads without error
- [ ] Directory structure matches architecture diagram
- [ ] No Python syntax errors

---

## PR2: packages/data/

**Branch**: `feat/packages-data`
**Depends on**: PR1
**Scope**: Data loading layer — the only package that reads files

### Files Created

| File | Purpose |
|------|---------|
| `packages/data/pyproject.toml` | Package metadata, dependencies |
| `packages/data/src/__init__.py` | Public API exports |
| `packages/data/src/trace.py` | TRACE parquet loader |
| `packages/data/src/reference.py` | Bond reference data (CUSIP → issuer, rating) |
| `packages/data/src/universe.py` | IG/HY filters, liquidity screens |
| `packages/data/src/validation.py` | Data quality checks |
| `packages/data/src/config.py` | YAML config loader utility |
| `packages/data/tests/__init__.py` | Tests package |
| `packages/data/tests/test_config.py` | Config loader tests |
| `packages/data/tests/test_validation.py` | Validation tests |

### Key Interfaces

```python
# trace.py
def load_trace(path: str | Path) -> pd.DataFrame:
    """
    Load TRACE data from parquet.

    Returns
    -------
    pd.DataFrame
        Columns: date, cusip, price, volume, side
        Index: None (reset)
    """

# reference.py
def load_reference(path: str | Path) -> pd.DataFrame:
    """
    Load bond reference data.

    Returns
    -------
    pd.DataFrame
        Columns: cusip, issuer, rating, maturity, coupon
    """

# universe.py
def filter_ig(df: pd.DataFrame, rating_col: str = 'rating') -> pd.DataFrame:
    """Filter to investment-grade bonds (BBB- and above)."""

def filter_hy(df: pd.DataFrame, rating_col: str = 'rating') -> pd.DataFrame:
    """Filter to high-yield bonds (BB+ and below)."""

# config.py
def load_config(path: str | Path) -> dict:
    """Load YAML config file."""
```

### pyproject.toml

```toml
[project]
name = "flowcode-data"
version = "0.1.0"
description = "Data loading for FlowCode credit analytics"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "polars>=0.20",
    "pyarrow>=14.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

### Acceptance Criteria

- [ ] `pip install -e packages/data` succeeds
- [ ] `from data.config import load_config` works
- [ ] `pytest packages/data/tests/ -v` passes
- [ ] Type hints on all public functions
- [ ] Docstrings on all public functions

---

## PR3: packages/signals/

**Branch**: `feat/packages-signals`
**Depends on**: PR2
**Scope**: Signal computations — thin wrappers over core

### Files Created

| File | Purpose |
|------|---------|
| `packages/signals/pyproject.toml` | Package metadata |
| `packages/signals/src/__init__.py` | Public API exports |
| `packages/signals/src/retail.py` | Retail imbalance + QMP classification |
| `packages/signals/src/triggers.py` | Z-score and streak triggers |
| `packages/signals/tests/__init__.py` | Tests package |
| `packages/signals/tests/test_retail.py` | Retail signal tests |
| `packages/signals/tests/test_triggers.py` | Trigger tests |

### Key Interfaces

```python
# retail.py
def compute_retail_imbalance(
    trades: pd.DataFrame,
    price_col: str = 'price',
    volume_col: str = 'volume',
    side_col: str = 'side'
) -> pd.Series:
    """
    Compute retail order imbalance.

    I_t = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    Returns
    -------
    pd.Series
        Index: (date, cusip), Values: imbalance [-1, 1]
    """

def qmp_classify(
    price: float,
    mid: float,
    spread: float,
    threshold: float = 0.1
) -> str:
    """
    Quote Midpoint (QMP) classification.

    Returns 'buy' if price > mid + threshold * spread, else 'sell'.
    """

# triggers.py
def zscore_trigger(
    series: pd.Series,
    window: int = 252,
    threshold: float = 7.0
) -> pd.Series:
    """
    Z-score mean reversion trigger.

    Returns True when |z| > threshold.
    """

def streak_trigger(
    series: pd.Series,
    min_streak: int = 3
) -> pd.Series:
    """
    Momentum streak trigger.

    Returns True when same-sign streak >= min_streak.
    """
```

### Acceptance Criteria

- [ ] `pip install -e packages/signals` succeeds
- [ ] `from signals.retail import compute_retail_imbalance` works
- [ ] `pytest packages/signals/tests/ -v` passes
- [ ] Tests use fixtures (can be inline initially, moved to spec/fixtures/ in PR6)

---

## PR4: packages/metrics/

**Branch**: `feat/packages-metrics`
**Depends on**: PR3 (for output format alignment)
**Scope**: Performance and risk metrics — pure functions

### Files Created

| File | Purpose |
|------|---------|
| `packages/metrics/pyproject.toml` | Package metadata |
| `packages/metrics/src/__init__.py` | Public API exports |
| `packages/metrics/src/performance.py` | Sharpe, Sortino, Calmar |
| `packages/metrics/src/risk.py` | Drawdown, VaR |
| `packages/metrics/src/diagnostics.py` | Autocorrelation, decay, hit rate |
| `packages/metrics/tests/__init__.py` | Tests package |
| `packages/metrics/tests/test_performance.py` | Performance tests |
| `packages/metrics/tests/test_risk.py` | Risk tests |

### Key Interfaces

```python
# performance.py
def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio."""

def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Annualized Sortino ratio (downside deviation)."""

def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Annualized return / max drawdown."""

# risk.py
def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown series."""

def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """Historical VaR at given confidence level."""

# diagnostics.py
def autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Autocorrelation at given lag."""

def hit_rate(signals: pd.Series, returns: pd.Series) -> float:
    """Fraction of correct signal directions."""

def signal_decay(
    signal: pd.Series,
    returns: pd.Series,
    horizons: list[int] = [1, 5, 10, 21]
) -> pd.DataFrame:
    """Signal predictive power decay over horizons."""
```

### Design Principles

1. **Pure functions**: No side effects, no file I/O
2. **Pandas in, scalar/pandas out**: Consistent interface
3. **NaN handling**: Explicit behavior documented
4. **No magic numbers**: All parameters explicit

### Acceptance Criteria

- [ ] `pip install -e packages/metrics` succeeds
- [ ] `from metrics.performance import sharpe_ratio` works
- [ ] `pytest packages/metrics/tests/ -v` passes
- [ ] All functions are pure (no side effects)
- [ ] Edge cases tested (empty series, all NaN, single value)

---

## PR5: packages/backtest/

**Branch**: `feat/packages-backtest`
**Depends on**: PR3, PR4
**Scope**: Strategy harness — orchestrates signals and metrics

### Files Created

| File | Purpose |
|------|---------|
| `packages/backtest/pyproject.toml` | Package metadata |
| `packages/backtest/src/__init__.py` | Public API exports |
| `packages/backtest/src/engine.py` | Backtest engine |
| `packages/backtest/src/portfolio.py` | Position sizing |
| `packages/backtest/src/results.py` | BacktestResult dataclass |
| `packages/backtest/tests/__init__.py` | Tests package |
| `packages/backtest/tests/test_engine.py` | Engine tests |
| `packages/backtest/tests/test_portfolio.py` | Portfolio tests |

### Key Interfaces

```python
# results.py
@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]

# engine.py
def run_backtest(
    signal: pd.Series,
    prices: pd.DataFrame,
    position_sizer: Callable | None = None,
    transaction_cost: float = 0.0
) -> BacktestResult:
    """
    Run signal-based backtest.

    Parameters
    ----------
    signal : pd.Series
        Index: (date, cusip), Values: signal strength
    prices : pd.DataFrame
        Columns include 'mid_price', 'spread'
    position_sizer : Callable, optional
        Function(signal, prices) -> positions
    transaction_cost : float
        Cost per unit traded (pre-cost=0.0)

    Returns
    -------
    BacktestResult
    """

# portfolio.py
def equal_weight(
    signal: pd.Series,
    prices: pd.DataFrame,
    max_positions: int = 50
) -> pd.DataFrame:
    """Equal-weight position sizing."""

def risk_parity(
    signal: pd.Series,
    prices: pd.DataFrame,
    vol_window: int = 21
) -> pd.DataFrame:
    """Inverse-volatility position sizing."""
```

### Lookahead Prevention Rules

These rules are critical and will be documented in CLAUDE.md:

1. **Signal at t can only use data up to t-1**
2. **Position at t is based on signal at t-1**
3. **PnL at t is computed using price change from t-1 to t**
4. **Never use `.shift(-n)` except in PnL calculation**

### Acceptance Criteria

- [ ] `pip install -e packages/backtest` succeeds
- [ ] `from backtest.engine import run_backtest` works
- [ ] `pytest packages/backtest/tests/ -v` passes
- [ ] Lookahead bias tests included (intentional future leak should fail)
- [ ] Integration test with mock signal → metrics output

---

## PR6: spec/, tools/, skills/, CI

**Branch**: `feat/spec-tools-ci`
**Depends on**: PR1-PR5
**Scope**: Specification, tooling, skills, CI pipeline

### Files Created

| File | Purpose |
|------|---------|
| `spec/SPEC.md` | Formula definitions |
| `spec/fixtures/imbalance_cases.json` | Imbalance test cases |
| `spec/fixtures/zscore_cases.json` | Z-score test cases |
| `spec/fixtures/streak_cases.json` | Streak test cases |
| `spec/fixtures/sharpe_cases.json` | Sharpe test cases |
| `spec/fixtures/pnl_cases.json` | PnL test cases |
| `spec/decisions/001-monorepo-structure.md` | ADR: Why this structure |
| `tools/verify_spec_parity.py` | Spec ↔ code ↔ test checker |
| `tools/notebooks/.gitkeep` | Notebooks directory |
| `skills/add-signal/SKILL.md` | Add signal workflow |
| `skills/add-data-source/SKILL.md` | Add data source workflow |
| `skills/run-backtest/SKILL.md` | Run backtest workflow |
| `skills/verify-change/SKILL.md` | Verify change workflow |
| `.github/workflows/ci.yml` | CI pipeline |

### spec/SPEC.md Structure

```markdown
# Credit Analytics Specification

## §1 Data Definitions

### §1.1 Credit PnL
credit_pnl = -spread_change * (pvbp / mid_price)

### §1.2 Retail Order Imbalance
I_t = (buy_volume - sell_volume) / (buy_volume + sell_volume)

## §2 Signal Definitions

### §2.1 Z-Score Trigger
z_t = (I_t - mean(I, window)) / std(I, window)
trigger = |z_t| > threshold

### §2.2 Streak Trigger
streak_t = consecutive same-sign count
trigger = streak_t >= min_streak

## §3 Metric Definitions

### §3.1 Sharpe Ratio
sharpe = mean(r) / std(r) * sqrt(252)

### §3.2 Maximum Drawdown
dd_t = (peak_t - value_t) / peak_t
max_dd = max(dd_t)
```

### Fixture Format

```json
{
  "name": "zscore_cases",
  "cases": [
    {
      "id": "basic_trigger",
      "input": {
        "series": [0.1, 0.2, 0.15, ...],
        "window": 10,
        "threshold": 2.0
      },
      "expected": {
        "zscore": 2.5,
        "trigger": true
      }
    }
  ]
}
```

### verify_spec_parity.py

```python
"""
Verify that:
1. Every formula in SPEC.md has a code implementation
2. Every implementation has a test
3. Every test uses fixtures from spec/fixtures/
"""

def main():
    spec = parse_spec("spec/SPEC.md")
    registry = load_formula_registry("CLAUDE.md")

    for formula in spec.formulas:
        assert formula.id in registry, f"Missing from registry: {formula.id}"
        assert code_exists(registry[formula.id].code_location)
        assert fixture_exists(registry[formula.id].fixture)

    print("✓ Spec parity verified")
```

### CI Pipeline (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e packages/data[dev]
          pip install -e packages/signals[dev]
          pip install -e packages/metrics[dev]
          pip install -e packages/backtest[dev]

      - name: Run tests
        run: pytest packages/*/tests/ -v --cov

      - name: Type check
        run: |
          pip install mypy
          mypy packages/*/src/

      - name: Lint
        run: |
          pip install ruff
          ruff check packages/

      - name: Verify spec parity
        run: python tools/verify_spec_parity.py
```

### Acceptance Criteria

- [ ] `spec/SPEC.md` contains all formulas
- [ ] All fixtures are valid JSON
- [ ] `python tools/verify_spec_parity.py` passes
- [ ] All skills have complete workflow documentation
- [ ] CI passes on push

---

## Migration Notes

### packages/core/

The `core/` package is marked as **DO NOT MODIFY**. It is assumed to already exist with:
- `imbalance()` — Base imbalance calculation
- `pvbp()` — Price value of basis point
- `spread_pnl()` — Spread-based PnL

If `core/` does not exist, create a stub in PR2 with placeholder implementations.

### Existing FlowCode Content

Current FlowCode appears to be a meta-repo with other projects (DSPy, HayashiYoshida, etc.). The new structure will coexist:

```
FlowCode/
├── CLAUDE.md              # NEW
├── conf/                  # NEW
├── packages/              # NEW
├── spec/                  # NEW
├── tools/                 # NEW
├── skills/                # NEW
├── .github/               # NEW
├── DSPy/                  # EXISTING (unchanged)
├── HayashiYoshida/        # EXISTING (unchanged)
└── ...
```

---

## Testing Strategy

### Unit Tests
- Each package has its own `tests/` directory
- Tests are isolated, use mocks/fixtures
- Fast execution (<1s per test file)

### Integration Tests
- Located in `packages/backtest/tests/test_integration.py`
- End-to-end: data → signal → metrics → backtest
- Uses small synthetic datasets

### Fixture-Based Tests
- Golden test cases in `spec/fixtures/`
- Parametrized tests load fixtures
- Ensures code matches spec

### Property-Based Tests (Future)
- Hypothesis for edge cases
- Fuzz testing for data loaders

---

## Rollback Plan

If issues arise:

1. **PR-level rollback**: Revert specific PR
2. **Package isolation**: Broken package can be disabled without affecting others
3. **Feature flags**: Not needed initially (all pre-cost analysis)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Test coverage | >80% |
| Type coverage | 100% public API |
| CI pass rate | 100% before merge |
| Spec parity | 100% (all formulas have code + tests) |

---

## Timeline

| PR | Status |
|----|--------|
| PR1: Scaffold | Ready to implement |
| PR2: data/ | Blocked on PR1 |
| PR3: signals/ | Blocked on PR2 |
| PR4: metrics/ | Blocked on PR2 (output format) |
| PR5: backtest/ | Blocked on PR3, PR4 |
| PR6: spec/tools/CI | Blocked on PR1-PR5 |

---

## Appendix: Full File List

<details>
<summary>Click to expand (45 files)</summary>

```
CLAUDE.md
conf/base.yaml
conf/signals/zscore.yaml
conf/signals/streak.yaml
packages/__init__.py
packages/data/pyproject.toml
packages/data/src/__init__.py
packages/data/src/trace.py
packages/data/src/reference.py
packages/data/src/universe.py
packages/data/src/validation.py
packages/data/src/config.py
packages/data/tests/__init__.py
packages/data/tests/test_config.py
packages/data/tests/test_validation.py
packages/signals/pyproject.toml
packages/signals/src/__init__.py
packages/signals/src/retail.py
packages/signals/src/triggers.py
packages/signals/tests/__init__.py
packages/signals/tests/test_retail.py
packages/signals/tests/test_triggers.py
packages/metrics/pyproject.toml
packages/metrics/src/__init__.py
packages/metrics/src/performance.py
packages/metrics/src/risk.py
packages/metrics/src/diagnostics.py
packages/metrics/tests/__init__.py
packages/metrics/tests/test_performance.py
packages/metrics/tests/test_risk.py
packages/backtest/pyproject.toml
packages/backtest/src/__init__.py
packages/backtest/src/engine.py
packages/backtest/src/portfolio.py
packages/backtest/src/results.py
packages/backtest/tests/__init__.py
packages/backtest/tests/test_engine.py
packages/backtest/tests/test_portfolio.py
spec/SPEC.md
spec/fixtures/imbalance_cases.json
spec/fixtures/zscore_cases.json
spec/fixtures/streak_cases.json
spec/fixtures/sharpe_cases.json
spec/fixtures/pnl_cases.json
spec/decisions/001-monorepo-structure.md
tools/verify_spec_parity.py
tools/notebooks/.gitkeep
skills/add-signal/SKILL.md
skills/add-data-source/SKILL.md
skills/run-backtest/SKILL.md
skills/verify-change/SKILL.md
.github/workflows/ci.yml
```

</details>
