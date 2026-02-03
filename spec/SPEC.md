# FlowCode Credit Analytics Specification

## Overview

This document is the **single source of truth** for all formulas, algorithms, and business rules in FlowCode. Code implementations MUST match these specifications exactly.

---

## 1. Core Computations

### 1.1 Credit PnL (Spread-Based)

**Formula:**
```
PnL_t = -PVBP * ΔSpread_t
```

Where:
- `PVBP` = Price Value of a Basis Point (dollar duration)
- `ΔSpread_t` = Spread change from t-1 to t (in basis points)

**Code Location:** `packages/core/src/spread_pnl.py`
**Fixture:** `spec/fixtures/pnl_cases.json`

### 1.2 Retail Order Imbalance

**Formula:**
```
I_t = (B_t - S_t) / (B_t + S_t)
```

Where:
- `B_t` = Buy volume at time t (retail)
- `S_t` = Sell volume at time t (retail)
- Result: [-1, +1], where +1 = all buys, -1 = all sells

**Edge Cases:**
- If `B_t + S_t = 0`, return `NaN`
- If only buys: return `+1`
- If only sells: return `-1`

**Code Location:** `packages/signals/src/retail.py::compute_retail_imbalance()`
**Fixture:** `spec/fixtures/imbalance_cases.json`

### 1.3 QMP Classification

**Definition:**
QMP (Qualifying Market Participant) classification distinguishes retail vs institutional trades.

**Rules:**
- Trade size ≤ $100,000 → Retail
- Trade size > $100,000 → Institutional

**Code Location:** `packages/signals/src/retail.py::qmp_classify()`
**Fixture:** `spec/fixtures/qmp_cases.json`

---

## 2. Signal Triggers

### 2.1 Z-Score Trigger

**Formula:**
```
Z_t = (X_t - μ) / σ

where:
  μ = rolling mean over window
  σ = rolling std over window
```

**Trigger:**
```
signal = 1  if Z_t > threshold
       = -1 if Z_t < -threshold
       = 0  otherwise
```

**Parameters:**
- `window`: 252 (default, ~1 year trading days)
- `threshold`: 7.0 (default)
- `min_periods`: 126 (minimum observations before computing)

**Code Location:** `packages/signals/src/triggers.py::zscore_trigger()`
**Fixture:** `spec/fixtures/zscore_cases.json`

### 2.2 Streak Trigger

**Formula:**
```
streak_t = consecutive days with same sign(X)
```

**Trigger:**
```
signal = sign(X_t) if |streak_t| >= min_streak
       = 0         otherwise
```

**Parameters:**
- `min_streak`: 3 (default)

**Code Location:** `packages/signals/src/triggers.py::streak_trigger()`
**Fixture:** `spec/fixtures/streak_cases.json`

---

## 3. Performance Metrics

### 3.1 Sharpe Ratio

**Formula:**
```
Sharpe = (μ_r - r_f) / σ_r * √252

where:
  μ_r = mean daily return
  r_f = risk-free rate (default 0)
  σ_r = std of daily returns
  252 = trading days per year
```

**Code Location:** `packages/metrics/src/performance.py::sharpe_ratio()`
**Fixture:** `spec/fixtures/sharpe_cases.json`

### 3.2 Sortino Ratio

**Formula:**
```
Sortino = (μ_r - r_f) / σ_d * √252

where:
  σ_d = downside deviation (std of negative returns only)
```

**Code Location:** `packages/metrics/src/performance.py::sortino_ratio()`
**Fixture:** `spec/fixtures/sortino_cases.json`

### 3.3 Calmar Ratio

**Formula:**
```
Calmar = Annualized_Return / |Max_Drawdown|
```

**Code Location:** `packages/metrics/src/performance.py::calmar_ratio()`
**Fixture:** `spec/fixtures/calmar_cases.json`

---

## 4. Risk Metrics

### 4.1 Maximum Drawdown

**Formula:**
```
Drawdown_t = (Peak_t - Value_t) / Peak_t

where:
  Peak_t = max(Value_0, Value_1, ..., Value_t)

Max_Drawdown = min(Drawdown_t) for all t
```

**Note:** Returns negative value (e.g., -0.15 = 15% drawdown)

**Code Location:** `packages/metrics/src/risk.py::max_drawdown()`
**Fixture:** `spec/fixtures/drawdown_cases.json`

### 4.2 Value at Risk (VaR)

**Formula (Historical):**
```
VaR_α = -quantile(returns, α)

where:
  α = confidence level (default 0.05 for 95% VaR)
```

**Code Location:** `packages/metrics/src/risk.py::value_at_risk()`
**Fixture:** `spec/fixtures/var_cases.json`

### 4.3 Expected Shortfall (CVaR)

**Formula:**
```
ES_α = -mean(returns where returns ≤ -VaR_α)
```

**Code Location:** `packages/metrics/src/risk.py::expected_shortfall()`
**Fixture:** `spec/fixtures/es_cases.json`

---

## 5. Diagnostics

### 5.1 Hit Rate

**Formula:**
```
Hit_Rate = count(returns > 0) / count(returns)
```

**Code Location:** `packages/metrics/src/diagnostics.py::hit_rate()`
**Fixture:** `spec/fixtures/hit_rate_cases.json`

### 5.2 Autocorrelation

**Formula:**
```
AC_k = corr(X_t, X_{t-k})
```

**Code Location:** `packages/metrics/src/diagnostics.py::autocorrelation()`
**Fixture:** `spec/fixtures/autocorr_cases.json`

### 5.3 Information Coefficient

**Formula:**
```
IC = corr(signal_t, return_{t+1})
```

**Note:** Forward return, not concurrent.

**Code Location:** `packages/metrics/src/diagnostics.py::information_coefficient()`
**Fixture:** `spec/fixtures/ic_cases.json`

---

## 6. Backtest Engine

### 6.1 Return Calculation

**Formula:**
```
Portfolio_Return_t = Σ(Position_{t-1,i} * Asset_Return_{t,i})
```

**Lookahead Prevention:**
- Position at t-1 earns return from t-1 to t
- Signal at t determines position at t+1

**Code Location:** `packages/backtest/src/engine.py::compute_returns()`

### 6.2 Transaction Costs

**Formula:**
```
Cost_t = Σ|ΔPosition_{t,i}| * cost_rate
Net_Return_t = Gross_Return_t - Cost_t
```

**Code Location:** `packages/backtest/src/engine.py::run_backtest()`

---

## 7. Data Validation

### 7.1 TRACE Data Requirements

| Field | Type | Constraints |
|-------|------|-------------|
| date | datetime | Not null, valid date |
| cusip | string | 9 characters |
| price | float | > 0 |
| volume | float | ≥ 0 |
| side | string | 'B' or 'S' |

### 7.2 Reference Data Requirements

| Field | Type | Constraints |
|-------|------|-------------|
| cusip | string | 9 characters, unique |
| issuer | string | Not null |
| rating | string | Valid rating (AAA to D) |
| maturity | datetime | > issue_date |

---

## Formula Registry

| Formula | Spec § | Code Location | Fixture |
|---------|--------|---------------|---------|
| Credit PnL | §1.1 | core.spread_pnl() | pnl_cases.json |
| Imbalance | §1.2 | signals.retail.compute_retail_imbalance() | imbalance_cases.json |
| QMP | §1.3 | signals.retail.qmp_classify() | qmp_cases.json |
| Z-score | §2.1 | signals.triggers.zscore_trigger() | zscore_cases.json |
| Streak | §2.2 | signals.triggers.streak_trigger() | streak_cases.json |
| Sharpe | §3.1 | metrics.performance.sharpe_ratio() | sharpe_cases.json |
| Sortino | §3.2 | metrics.performance.sortino_ratio() | sortino_cases.json |
| Calmar | §3.3 | metrics.performance.calmar_ratio() | calmar_cases.json |
| Max DD | §4.1 | metrics.risk.max_drawdown() | drawdown_cases.json |
| VaR | §4.2 | metrics.risk.value_at_risk() | var_cases.json |
| ES | §4.3 | metrics.risk.expected_shortfall() | es_cases.json |
| Hit Rate | §5.1 | metrics.diagnostics.hit_rate() | hit_rate_cases.json |
| Autocorr | §5.2 | metrics.diagnostics.autocorrelation() | autocorr_cases.json |
| IC | §5.3 | metrics.diagnostics.information_coefficient() | ic_cases.json |
