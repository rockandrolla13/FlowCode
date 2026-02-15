# alphaeval — Corporate Credit Alpha Evaluation Library (Price + Spread)

A unified evaluation toolkit for **corporate bonds**, **corporate bond ETFs**, and **CDX indices**, supporting evaluation in:

- **Price space**: returns / currency PnL
- **Spread space**: spread changes (bp), spread-return proxies (duration-based), and DV01-based spread PnL

It also includes **factor-mining / cross-sectional** evaluation (IC/RankIC/IR*, R²) and **Sharpe inference** tools (PSR/DSR) to handle **non-normal returns** and **selection bias** after trying many alphas.

---

## 1) Goals & non-goals

### Goals
- Standardized evaluation across instruments and horizons:
  - Performance: Sharpe, Sortino, PF, WinRate, Expectancy, CAGR, MDD, Run-up, CAGR/MDD
  - Risk: VaR
  - Execution: latency, fill probability, message traffic, quote-to-trade
  - Factor quality: R², IC*, RankIC*, IR*
  - Benchmark-relative: Information Ratio
  - Sharpe inference: SR stdev (non-normal), PSR, minTRL, independent trials, expected max SR, DSR

### Non-goals (v1)
- Full backtesting engine
- Full microstructure simulation
- Full instrument pricing models (beyond spread-to-return proxies)

---

## 2) Core concepts

### 2.1 Price space vs Spread space

**Price space** typically uses:
- `price` (clean price or total return index) → `returns`
- strategy `position` / `pnl` / `trades`

**Spread space** typically uses:
- `spread` (e.g., OAS/Z/ASW — you label convention)
- `duration` and/or `dv01`

Spread primitives (library supplies helpers):

1) **Spread change (bp)**
\[
\Delta s^{bp}_t = (s_t - s_{t-1})\cdot 10^4
\]

2) **Duration spread-return proxy** (approx price return from spread move)
\[
r^{spread}_t \approx -D_t \cdot \Delta s_t
\]
where \(\Delta s_t\) is in **decimal** (e.g., 1bp = 0.0001)

3) **DV01 spread PnL proxy**
\[
\text{PnL}^{spread}_t \approx -\text{DV01}_t \cdot \Delta s^{bp}_t
\]

---

## 3) Data model (inputs)

### 3.1 Panel schema (recommended)
A panel is keyed by `(date, instrument)` (MultiIndex or columns).

Required minimum (time-series performance):
- `date`
- `instrument`
- either `returns` OR `price`

Optional (spread evaluation):
- `spread`, `duration`, `dv01`

Optional (benchmark / IR):
- `rf` (risk-free per period)
- `benchmark_return`

Optional (factor evaluation):
- `signal` (factor value)
- `target` (next-period outcome: next-day RV, next-day spread change, next-day excess return, etc.)

Optional (strategy reconstruction):
- `position` (EOP holdings), `pnl`

### 3.2 Execution logs (optional)
Needed for execution KPIs.

Orders/Quotes:
- `order_id`, `instrument`, `submit_ts`, `cancel_ts` (optional)
- `side`, `limit_price`, `quoted_size`

Fills:
- `order_id`, `fill_ts`, `filled_size`, `fill_price`

Message log:
- `ts`, `msg_type` (NEW/CANCEL/REPLACE/UPDATE), `instrument`, `order_id`

---

## 4) Library layout (suggested)

alphaeval/
init.py
core/
types.py # dataclasses + protocols
panel.py # Panel wrapper + validation
registry.py # MetricRegistry
runner.py # evaluate() engine
config.py # defaults (annualization, ddof, policies)
transforms/
returns.py # price_return, log_return
spreads.py # delta_spread_bp, spread_return_proxy, dv01_pnl
equity.py # equity curve, drawdown, runup
metrics/
performance.py # sharpe, sortino, pf, win_rate, expectancy, cagr, mdd, runup, calmar
risk.py # var_historical, var_parametric
execution.py # latency, p_fill, message_traffic, quote_to_trade
factor.py # r2, ic_star, rank_ic_star, ir_star
relative.py # information_ratio, tracking_error
sharpe_inference.py # SR stdev, PSR, minTRL, N_eff, expected max SR, DSR
reports/
report.py # MetricReport
render_md.py # markdown renderer
tests/
test_*.py


---

## 5) Metric engine (how evaluation works)

### 5.1 Metric contract (concept)
Each metric declares:
- `name`
- `requires` columns
- `scope`: `timeseries` | `cross_section` | `execution`
- `compute(panel, trades, orders, config) -> result`

### 5.2 Runner contract (concept)
`evaluate(panel, metrics=[...], trades=None, orders=None, messages=None, config=...) -> MetricReport`

Report contains:
- `summary`: scalar metrics
- `series`: time series outputs (equity, drawdown, rolling stats, daily IC)
- `tables`: cross-sectional breakdowns

---

## 6) Formulas (complete)

### 6.0 Notation
- Period returns: \(r_t\), \(t=1,\dots,T\)
- Risk-free per period: \(r_{f,t}\) (or constant \(r_f\))
- Excess return: \(x_t = r_t - r_{f,t}\)
- Mean: \(\mu_r = \frac{1}{T}\sum_{t=1}^{T} r_t\)
- Sample std (ddof=1): \(\sigma_r = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_t-\mu_r)^2}\)
- Equity curve: \(E_0=1,\; E_t = E_{t-1}(1+r_t)\)

---

### A) Performance metrics

#### Sharpe Ratio
\[
\text{Sharpe}=\frac{\mathbb{E}[R_p-R_f]}{\sigma(R_p-R_f)}=\frac{\mu_x}{\sigma_x}
\]
Annualized with \(A\) periods/year:
\[
\text{Sharpe}_{ann}=\sqrt{A}\cdot \frac{\mu_x}{\sigma_x}
\]

#### Sortino Ratio ✅
Downside deviation vs threshold \(\tau\) (often \(\tau=r_f\) or 0):
\[
\sigma_{down}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(0,r_t-\tau)^2}
\]
\[
\text{Sortino}=\frac{\mathbb{E}[r_t-\tau]}{\sigma_{down}}
\]
Annualized:
\[
\text{Sortino}_{ann}=\sqrt{A}\cdot \frac{\mathbb{E}[r_t-\tau]}{\sigma_{down}}
\]

#### Profit Factor (PF)
Trade PnLs \(p_i,\, i=1,\dots,N\):
\[
\text{GrossProfit}=\sum_{i:p_i>0}p_i,\quad
\text{GrossLoss}=\sum_{i:p_i<0}|p_i|
\]
\[
\text{PF}=\frac{\text{GrossProfit}}{\text{GrossLoss}}
\]

#### Win Rate (%)
\[
\text{WinRate}=\frac{\#\{i:p_i>0\}}{N}\times 100
\]

#### Expectancy / Expected Value per trade ✅
Let \(N_w\) wins, \(N_l\) losses, \(N=N_w+N_l\):
\[
\text{AvgWin}=\frac{1}{N_w}\sum_{i:p_i>0}p_i,\quad
\text{AvgLoss}=\frac{1}{N_l}\sum_{i:p_i<0}|p_i|
\]
\[
\text{WinRate}=\frac{N_w}{N},\quad \text{LossRate}=1-\text{WinRate}
\]
\[
\text{Expectancy}=\text{WinRate}\cdot \text{AvgWin}-\text{LossRate}\cdot \text{AvgLoss}
\]
Equivalent:
\[
\text{Expectancy}=\frac{1}{N}\sum_{i=1}^{N}p_i
\]

#### Maximum Drawdown (MDD) ✅
Running peak:
\[
P_t=\max_{0\le u\le t}E_u
\]
Drawdown series:
\[
DD_t=\frac{P_t-E_t}{P_t}
\]
Maximum drawdown:
\[
\text{MDD}=\max_{0\le t\le T}DD_t
\]

#### Maximum Run-up (MaxRU) ✅
Running trough:
\[
Q_t=\min_{0\le u\le t}E_u
\]
Run-up series:
\[
RU_t=\frac{E_t-Q_t}{Q_t}
\]
Maximum run-up:
\[
\text{MaxRU}=\max_{0\le t\le T}RU_t
\]

#### CAGR
If horizon is \(Y\) years:
\[
\text{CAGR}=\left(\frac{E_T}{E_0}\right)^{1/Y}-1
\]

#### CAGR / MaxDD ratio (aka Calmar-style)
\[
\frac{\text{CAGR}}{\text{MDD}}
\]

---

### B) Execution / operational KPIs

#### Latency
\[
\text{Latency}_i=t^{(i)}_{fill}-t^{(i)}_{submit}
\]
Report mean, p50/p90/p99.

#### Fill Probability \(P(\text{Fill})_T\)
Volume-based:
\[
P(\text{Fill})_T=\frac{\text{FilledVolume}_T}{\text{SubmittedVolume}_T}
\]
Order-based (optional):
\[
P(\text{Fill})_T=\frac{\#\{\text{orders filled within }T\}}{\#\{\text{orders submitted}\}}
\]

#### Message traffic
\[
\text{Mess}(\Delta)=\#\{\text{messages in bucket }\Delta\}
\]

#### Quote-to-trade ratio
\[
\text{QTR}=\frac{\#\{\text{quotes}\}}{\#\{\text{trades}\}}
\]

---

### C) Risk metrics

#### Value at Risk (VaR) — historical
Define losses \(L_t\) (e.g., \(L_t=-r_t\) or \(-\text{PnL}_t\)):
\[
\text{VaR}_\alpha=\text{Quantile}_\alpha(L)
\]

#### Value at Risk (VaR) — Gaussian parametric
If \(L\sim\mathcal{N}(\mu_L,\sigma_L^2)\):
\[
\text{VaR}_\alpha=\mu_L+z_\alpha\sigma_L
\]

#### Position duration
If position \(j\) has open/close timestamps:
\[
\text{Dur}_j=t^{(j)}_{close}-t^{(j)}_{open},\quad
\overline{\text{Dur}}=\frac{1}{J}\sum_{j=1}^{J}\text{Dur}_j
\]

#### VPIN (spec-level)
Volume buckets \(b\) with fixed bucket volume \(V\), buy \(V_b^+\), sell \(V_b^-\):
\[
\text{Imb}_b=|V_b^+-V_b^-|
\]
Rolling VPIN over last \(n\) buckets:
\[
\text{VPIN}_t=\frac{\sum_{b=t-n+1}^{t}\text{Imb}_b}{nV}
\]

---

### D) Key drivers / signal quality

#### Signal quality \(\pi\) (directional accuracy interpretation)
\[
\pi=\mathbb{P}(\mathrm{sign}(\hat r_{t+1})=\mathrm{sign}(r_{t+1}))
\]
Uninformative \(\pi=0.5\), perfect \(\pi=1\).

---

### E) Factor-mining / cross-sectional metrics

These are computed **daily across instruments**, then averaged over days.

#### \(R^2\)
\[
R^2=
1-\frac{\sum_{i=1}^{M_{test}}(y_i-\hat y_i)^2}{\sum_{i=1}^{M_{test}}(y_i-\bar y)^2}
\]

#### \(IC^*\)
Composite factor:
\[
g_{t,i}=\sum_{j=1}^{k}w_j f_j(X_{t,i})
\]
Daily IC:
\[
IC_t=\rho(g_{t,\cdot},y_{t+1,\cdot})
\]
Averaged:
\[
IC^*=\mathbb{E}_t[IC_t],\quad IC_t\in[-1,1]
\]

#### Rank \(IC^*\)
\[
RankIC_t=\rho(r(g_{t,\cdot}),r(y_{t+1,\cdot}))
\]
\[
RankIC^*=\mathbb{E}_t[RankIC_t]
\]

#### \(IR^*\) (IC information ratio)
\[
IR^*=\frac{\mathbb{E}_t[IC_t]}{\mathrm{Std}_t(IC_t)}
\]

---

### F) Benchmark-relative metric

#### Information Ratio (IR)
Active return \(a_t=r_{p,t}-r_{b,t}\):
\[
IR=\frac{\mathbb{E}[a_t]}{\sigma(a_t)}
\]
Tracking error:
\[
\text{TE}=\sigma(a_t)
\]
So:
\[
IR=\frac{R_p-R_b}{\text{TE}}
\]

---

## 7) Advanced Sharpe inference (PSR / DSR) — formulas + code

These methods address:
- non-normality (skew/kurtosis)
- selection bias after running many trials and picking the best

### 7.1 Notation
- Returns \(r_t\), \(t=1,\dots,n\)
- \(\widehat{SR}\): estimated Sharpe (rf=0)
- \(\gamma_3\): skew (normal ⇒ 0)
- \(\gamma_4\): kurtosis (non-excess; normal ⇒ 3)
- Normal CDF \(\Phi\), inverse \(\Phi^{-1}\)

### 7.2 Formulas

#### Estimated Sharpe Ratio (rf=0)
\[
\widehat{SR}=\frac{\bar r}{s_r}
\]

#### Annualized estimated Sharpe
\[
\widehat{SR}_{ann}=\sqrt{A}\,\widehat{SR}
\]

#### Standard deviation of \(\widehat{SR}\) (non-normal)
\[
\sigma_{\widehat{SR}}=
\sqrt{\frac{
1+\frac{1}{2}\widehat{SR}^2-\gamma_3\widehat{SR}+\frac{\gamma_4-3}{4}\widehat{SR}^2
}{n-1}}
\]

#### Probabilistic Sharpe Ratio (PSR)
\[
PSR(SR^\*)=\Phi\left(\frac{\widehat{SR}-SR^\*}{\sigma_{\widehat{SR}}}\right)
\]

#### Minimum Track Record Length (minTRL)
\[
\text{minTRL}=1+\sigma_{\widehat{SR}}^2(n-1)\left(\frac{\Phi^{-1}(p)}{\widehat{SR}-SR^\*}\right)^2
\]

#### Effective number of independent trials \(N_{\text{eff}}\)
Given \(m\) trials and average correlation \(\bar\rho\):
\[
N_{\text{eff}}\approx 1+\bar\rho+(1-\bar\rho)m
\]

#### Expected Maximum Sharpe Ratio after search (EMS)
Euler–Mascheroni constant \(\gamma\approx0.5772156649\):
\[
\mathbb{E}[\max SR]\approx SR_{mean}+\sigma_{trial}\cdot Z_{\max}
\]
\[
Z_{\max}=(1-\gamma)\Phi^{-1}\left(1-\frac{1}{N_{\text{eff}}}\right)+
\gamma\,\Phi^{-1}\left(1-\frac{1}{N_{\text{eff}}e}\right)
\]

#### Deflated Sharpe Ratio (DSR)
\[
DSR = \Phi\left(\frac{\widehat{SR}-\mathbb{E}[\max SR]}{\sigma_{\widehat{SR}}}\right)
\]

### 7.3 Reference implementation (paste into `alphaeval/metrics/sharpe_inference.py`)

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def estimated_sharpe_ratio(returns):
    """Calculate the estimated sharpe ratio (risk_free=0)."""
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(returns=None, periods=261, *, sr=None):
    """Calculate the annualized estimated sharpe ratio (risk_free=0)."""
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sharpe_ratio_stdev(returns=None, *, n=None, skew=None, kurtosis=None, sr=None):
    """
    Calculate the standard deviation of the sharpe ratio estimation.
    Generalizes for both normal and non-normal returns.
    """
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if n is None:
        n = len(_returns)
    if skew is None:
        skew = pd.Series(scipy_stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy_stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = estimated_sharpe_ratio(_returns)

    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) == pd.DataFrame:
        sr_std = pd.Series(sr_std, index=returns.columns)
    elif type(sr_std) not in (float, np.float64, pd.DataFrame):
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
    """Calculate the Probabilistic Sharpe Ratio (PSR)."""
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    psr = scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
    """Calculate the Min Track Record Length (minTRL)."""
    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy_stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def num_independent_trials(trials_returns=None, *, m=None, p=None):
    """Calculate the number of independent trials."""
    if m is None:
        m = trials_returns.shape[1]

    if p is None:
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

    n = p + (1 - p) * m
    n = int(n) + 1  # round up
    return n


def expected_maximum_sr(trials_returns=None, expected_mean_sr=0.0, *, independent_trials=None, trials_sr_std=None):
    """Compute the expected maximum Sharpe ratio (analytical)."""
    emc = 0.5772156649  # Euler-Mascheroni constant

    if independent_trials is None:
        independent_trials = num_independent_trials(trials_returns)

    if trials_sr_std is None:
        srs = estimated_sharpe_ratio(trials_returns)
        trials_sr_std = srs.std()

    maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1.0 / independent_trials) + emc * scipy_stats.norm.ppf(
        1 - 1.0 / (independent_trials * np.e)
    )
    expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
    return expected_max_sr


def deflated_sharpe_ratio(trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
    """Calculate the Deflated Sharpe Ratio (DSR)."""
    if expected_max_sr is None:
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)

    dsr = probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)
    return dsr
8) API sketch (what you implement)
8.1 Transforms
price_to_returns(price, method="simple"|"log")

spread_to_delta_bp(spread)

spread_to_return_proxy(spread, duration) (uses 
−
𝐷
Δ
𝑠
−DΔs)

spread_to_dv01_pnl(delta_bp, dv01) (uses 
−
𝐷
𝑉
01
⋅
Δ
𝑠
𝑏
𝑝
−DV01⋅Δs 
bp
 )

8.2 Evaluate
evaluate_timeseries(panel, metrics=[...], config=...)

evaluate_cross_section(panel, factor_metrics=[...], config=...)

evaluate_execution(orders, fills, messages, metrics=[...], config=...)

8.3 Reports
report.to_markdown()

report.to_json()

9) Implementation plan (roadmap)
Phase 0 — Conventions
Freeze:

annualization factor 
𝐴
A

ddof convention (use ddof=1 consistently)

NaN policy (drop per metric)

spread units (bp vs decimal; enforce via transform naming)

Phase 1 — Core data wrappers + validation
Panel wrapper + schema checks

Simple adapters: pandas / polars / parquet

Phase 2 — Transforms
price returns, equity, drawdown, run-up

spread transforms (bp change, duration proxy returns, DV01 PnL)

Phase 3 — Metrics
performance, risk, execution, factor, relative

sharpe inference module (PSR/DSR suite above)

Phase 4 — Runner + registry
MetricRegistry + evaluate() engine

consistent outputs + markdown rendering

Phase 5 — Tests
toy series tests + edge cases (std=0, no trades, NaNs)

monotonicity tests (PSR increases with SR, etc.)

selection-bias sanity (DSR ≤ PSR typically when many trials)

Phase 6 — Docs + examples
cookbook: “price vs spread”, “IC pipeline”, “PSR/DSR workflow”

10) Testing checklist (must-have)
Sharpe / Sortino:

constant returns ⇒ std=0 policy (NaN/inf) documented

MDD:

monotonic equity ⇒ MDD=0

MaxRU:

monotonic up equity ⇒ MaxRU = (E_T - min(E))/min(E)

PF:

no losing trades ⇒ PF policy (inf or NaN) documented

IC / RankIC:

perfect monotonic ranks ⇒ ±1

IR:

zero tracking error ⇒ policy (NaN/inf) documented

PSR/minTRL/DSR:

PSR monotonic in SR

minTRL increases as prob increases

DSR decreases as #trials increases (all else equal)

11) Practical workflows
11.1 Time-series alpha eval (price)
Build strategy returns 
𝑟
𝑡
r 
t
​
  (or PnL then returns)

Compute equity 
𝐸
𝑡
E 
t
​
 

Run metrics: Sharpe, Sortino, MDD, MaxRU, CAGR, PF, WinRate, Expectancy

11.2 Time-series alpha eval (spread)
Convert spread → 
Δ
𝑠
𝑏
𝑝
Δs 
bp
 

Choose proxy:

duration proxy returns, or

DV01 PnL

Run same metrics as price space (now on proxy returns / PnL)

11.3 Factor eval (IC / RankIC / IR*)
For each date, compute cross-section factor signal

Define target y_{t+1} (next-day RV, spread change, excess return)

Compute daily IC / RankIC, then average + IR*

11.4 “Is this SR real?” (PSR / DSR)
Compute 
𝑆
𝑅
^
SR
  of selected strategy

Compute 
𝜎
𝑆
𝑅
^
σ 
SR
 
​
  (skew/kurtosis)

Compute PSR vs benchmark SR*

If you tried many alphas:

build trials_returns of all candidate strategies

compute expected max SR

compute DSR (deflated)


If you want the combined `.md` to be **exactly a README** (with a tighter “Quickstart” and a stricter input schema table with dtypes), tell me your preferred stack (pandas-only vs pandas+polars+arrow) and I’ll compress the sections into a “buildable in one morning” README format.

You said:
include a t-stat

ChatGPT said:

Thought for 12s
# alphaeval — Corporate Credit Alpha Evaluation Library (Price + Spread)

A unified evaluation toolkit for **corporate bonds**, **corporate bond ETFs**, and **CDX indices**, supporting evaluation in:

- **Price space**: returns / currency PnL
- **Spread space**: spread changes (bp), spread-return proxies (duration-based), and DV01-based spread PnL

It also includes **factor-mining / cross-sectional** evaluation (IC/RankIC/IR*, R²) and **Sharpe inference** tools (PSR/DSR) to handle **non-normal returns** and **selection bias** after trying many alphas.

---

## 1) Goals & non-goals

### Goals
- Standardized evaluation across instruments and horizons:
  - Performance: Sharpe, Sortino, PF, WinRate, Expectancy, CAGR, MDD, Run-up, CAGR/MDD
  - Significance: t-stat of mean (returns / IC)
  - Risk: VaR
  - Execution: latency, fill probability, message traffic, quote-to-trade
  - Factor quality: R², IC*, RankIC*, IR*
  - Benchmark-relative: Information Ratio
  - Sharpe inference: SR stdev (non-normal), PSR, minTRL, independent trials, expected max SR, DSR

### Non-goals (v1)
- Full backtesting engine
- Full microstructure simulation
- Full instrument pricing models (beyond spread-to-return proxies)

---

## 2) Core concepts

### 2.1 Price space vs Spread space

**Price space** typically uses:
- `price` (clean price or total return index) → `returns`
- strategy `position` / `pnl` / `trades`

**Spread space** typically uses:
- `spread` (e.g., OAS/Z/ASW — you label convention)
- `duration` and/or `dv01`

Spread primitives (library supplies helpers):

1) **Spread change (bp)**
\[
\Delta s^{bp}_t = (s_t - s_{t-1})\cdot 10^4
\]

2) **Duration spread-return proxy** (approx price return from spread move)
\[
r^{spread}_t \approx -D_t \cdot \Delta s_t
\]
where \(\Delta s_t\) is in **decimal** (e.g., 1bp = 0.0001)

3) **DV01 spread PnL proxy**
\[
\text{PnL}^{spread}_t \approx -\text{DV01}_t \cdot \Delta s^{bp}_t
\]

---

## 3) Data model (inputs)

### 3.1 Panel schema (recommended)
A panel is keyed by `(date, instrument)` (MultiIndex or columns).

Required minimum (time-series performance):
- `date`
- `instrument`
- either `returns` OR `price`

Optional (spread evaluation):
- `spread`, `duration`, `dv01`

Optional (benchmark / IR):
- `rf` (risk-free per period)
- `benchmark_return`

Optional (factor evaluation):
- `signal` (factor value)
- `target` (next-period outcome: next-day RV, next-day spread change, next-day excess return, etc.)

Optional (strategy reconstruction):
- `position` (EOP holdings), `pnl`

### 3.2 Execution logs (optional)
Needed for execution KPIs.

Orders/Quotes:
- `order_id`, `instrument`, `submit_ts`, `cancel_ts` (optional)
- `side`, `limit_price`, `quoted_size`

Fills:
- `order_id`, `fill_ts`, `filled_size`, `fill_price`

Message log:
- `ts`, `msg_type` (NEW/CANCEL/REPLACE/UPDATE), `instrument`, `order_id`

---

## 4) Library layout (suggested)

alphaeval/
init.py
core/
types.py # dataclasses + protocols
panel.py # Panel wrapper + validation
registry.py # MetricRegistry
runner.py # evaluate() engine
config.py # defaults (annualization, ddof, policies)
transforms/
returns.py # price_return, log_return
spreads.py # delta_spread_bp, spread_return_proxy, dv01_pnl
equity.py # equity curve, drawdown, runup
metrics/
performance.py # sharpe, sortino, pf, win_rate, expectancy, cagr, mdd, runup, calmar, tstat
risk.py # var_historical, var_parametric
execution.py # latency, p_fill, message_traffic, quote_to_trade
factor.py # r2, ic_star, rank_ic_star, ir_star, ic_tstat
relative.py # information_ratio, tracking_error
sharpe_inference.py # SR stdev, PSR, minTRL, N_eff, expected max SR, DSR
reports/
report.py # MetricReport
render_md.py # markdown renderer
tests/
test_*.py


---

## 5) Metric engine (how evaluation works)

### 5.1 Metric contract (concept)
Each metric declares:
- `name`
- `requires` columns
- `scope`: `timeseries` | `cross_section` | `execution`
- `compute(panel, trades, orders, config) -> result`

### 5.2 Runner contract (concept)
`evaluate(panel, metrics=[...], trades=None, orders=None, messages=None, config=...) -> MetricReport`

Report contains:
- `summary`: scalar metrics
- `series`: time series outputs (equity, drawdown, rolling stats, daily IC)
- `tables`: cross-sectional breakdowns

---

## 6) Formulas (complete)

### 6.0 Notation
- Period returns: \(r_t\), \(t=1,\dots,T\)
- Risk-free per period: \(r_{f,t}\) (or constant \(r_f\))
- Excess return: \(x_t = r_t - r_{f,t}\)
- Mean: \(\mu_r = \frac{1}{T}\sum_{t=1}^{T} r_t\)
- Sample std (ddof=1): \(\sigma_r = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_t-\mu_r)^2}\)
- Equity curve: \(E_0=1,\; E_t = E_{t-1}(1+r_t)\)

---

### A) Performance metrics

#### Sharpe Ratio
\[
\text{Sharpe}=\frac{\mathbb{E}[R_p-R_f]}{\sigma(R_p-R_f)}=\frac{\mu_x}{\sigma_x}
\]
Annualized with \(A\) periods/year:
\[
\text{Sharpe}_{ann}=\sqrt{A}\cdot \frac{\mu_x}{\sigma_x}
\]

#### Sortino Ratio
Downside deviation vs threshold \(\tau\) (often \(\tau=r_f\) or 0):
\[
\sigma_{down}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(0,r_t-\tau)^2}
\]
\[
\text{Sortino}=\frac{\mathbb{E}[r_t-\tau]}{\sigma_{down}}
\]
Annualized:
\[
\text{Sortino}_{ann}=\sqrt{A}\cdot \frac{\mathbb{E}[r_t-\tau]}{\sigma_{down}}
\]

#### Profit Factor (PF)
Trade PnLs \(p_i,\, i=1,\dots,N\):
\[
\text{GrossProfit}=\sum_{i:p_i>0}p_i,\quad
\text{GrossLoss}=\sum_{i:p_i<0}|p_i|
\]
\[
\text{PF}=\frac{\text{GrossProfit}}{\text{GrossLoss}}
\]

#### Win Rate (%)
\[
\text{WinRate}=\frac{\#\{i:p_i>0\}}{N}\times 100
\]

#### Expectancy / Expected Value per trade
Let \(N_w\) wins, \(N_l\) losses, \(N=N_w+N_l\):
\[
\text{AvgWin}=\frac{1}{N_w}\sum_{i:p_i>0}p_i,\quad
\text{AvgLoss}=\frac{1}{N_l}\sum_{i:p_i<0}|p_i|
\]
\[
\text{WinRate}=\frac{N_w}{N},\quad \text{LossRate}=1-\text{WinRate}
\]
\[
\text{Expectancy}=\text{WinRate}\cdot \text{AvgWin}-\text{LossRate}\cdot \text{AvgLoss}
\]
Equivalent:
\[
\text{Expectancy}=\frac{1}{N}\sum_{i=1}^{N}p_i
\]

#### Maximum Drawdown (MDD)
Running peak:
\[
P_t=\max_{0\le u\le t}E_u
\]
Drawdown series:
\[
DD_t=\frac{P_t-E_t}{P_t}
\]
Maximum drawdown:
\[
\text{MDD}=\max_{0\le t\le T}DD_t
\]

#### Maximum Run-up (MaxRU)
Running trough:
\[
Q_t=\min_{0\le u\le t}E_u
\]
Run-up series:
\[
RU_t=\frac{E_t-Q_t}{Q_t}
\]
Maximum run-up:
\[
\text{MaxRU}=\max_{0\le t\le T}RU_t
\]

#### CAGR
If horizon is \(Y\) years:
\[
\text{CAGR}=\left(\frac{E_T}{E_0}\right)^{1/Y}-1
\]

#### CAGR / MaxDD ratio (aka Calmar-style)
\[
\frac{\text{CAGR}}{\text{MDD}}
\]

#### t-stat of mean return (or mean excess return) ✅
Test \(H_0: \mathbb{E}[r_t]=0\) (or \(H_0:\mathbb{E}[x_t]=0\)):
\[
t = \frac{\bar r}{s_r/\sqrt{T}}
\quad\text{or}\quad
t = \frac{\bar x}{s_x/\sqrt{T}}
\]
- Use **\(\bar x\)** if you want “skill vs risk-free”.
- In practice, you may optionally add HAC / Newey–West later if returns are autocorrelated.

---

### B) Execution / operational KPIs

#### Latency
\[
\text{Latency}_i=t^{(i)}_{fill}-t^{(i)}_{submit}
\]

#### Fill Probability \(P(\text{Fill})_T\)
Volume-based:
\[
P(\text{Fill})_T=\frac{\text{FilledVolume}_T}{\text{SubmittedVolume}_T}
\]

#### Message traffic
\[
\text{Mess}(\Delta)=\#\{\text{messages in bucket }\Delta\}
\]

#### Quote-to-trade ratio
\[
\text{QTR}=\frac{\#\{\text{quotes}\}}{\#\{\text{trades}\}}
\]

---

### C) Risk metrics

#### Value at Risk (VaR) — historical
Define losses \(L_t\) (e.g., \(L_t=-r_t\) or \(-\text{PnL}_t\)):
\[
\text{VaR}_\alpha=\text{Quantile}_\alpha(L)
\]

#### Value at Risk (VaR) — Gaussian parametric
If \(L\sim\mathcal{N}(\mu_L,\sigma_L^2)\):
\[
\text{VaR}_\alpha=\mu_L+z_\alpha\sigma_L
\]

#### Position duration
\[
\overline{\text{Dur}}=\frac{1}{J}\sum_{j=1}^{J}(t^{(j)}_{close}-t^{(j)}_{open})
\]

#### VPIN (spec-level)
\[
\text{VPIN}_t=\frac{\sum_{b=t-n+1}^{t}|V_b^+-V_b^-|}{nV}
\]

---

### D) Key drivers / signal quality

#### Signal quality \(\pi\) (directional accuracy interpretation)
\[
\pi=\mathbb{P}(\mathrm{sign}(\hat r_{t+1})=\mathrm{sign}(r_{t+1}))
\]

---

### E) Factor-mining / cross-sectional metrics

These are computed **daily across instruments**, then averaged over days.

#### \(R^2\)
\[
R^2=
1-\frac{\sum_{i=1}^{M_{test}}(y_i-\hat y_i)^2}{\sum_{i=1}^{M_{test}}(y_i-\bar y)^2}
\]

#### \(IC^*\)
\[
g_{t,i}=\sum_{j=1}^{k}w_j f_j(X_{t,i}),\qquad
IC_t=\rho(g_{t,\cdot},y_{t+1,\cdot}),\qquad
IC^*=\mathbb{E}_t[IC_t]
\]

#### Rank \(IC^*\)
\[
RankIC_t=\rho(r(g_{t,\cdot}),r(y_{t+1,\cdot})),\qquad
RankIC^*=\mathbb{E}_t[RankIC_t]
\]

#### \(IR^*\) (IC information ratio)
\[
IR^*=\frac{\mathbb{E}_t[IC_t]}{\mathrm{Std}_t(IC_t)}
\]

#### t-stat of daily IC ✅
Let the daily IC series be \(\{IC_t\}_{t=1}^{T_d}\):
\[
t_{IC}=\frac{\overline{IC}}{s_{IC}/\sqrt{T_d}}
\]
This is the common “is IC reliably non-zero?” diagnostic.

---

### F) Benchmark-relative metric

#### Information Ratio (IR)
Active return \(a_t=r_{p,t}-r_{b,t}\):
\[
IR=\frac{\mathbb{E}[a_t]}{\sigma(a_t)}
\]

---

## 7) Advanced Sharpe inference (PSR / DSR) — formulas + code

These methods address:
- non-normality (skew/kurtosis)
- selection bias after running many trials and picking the best

### 7.1 Notation
- Returns \(r_t\), \(t=1,\dots,n\)
- \(\widehat{SR}\): estimated Sharpe (rf=0)
- \(\gamma_3\): skew (normal ⇒ 0)
- \(\gamma_4\): kurtosis (non-excess; normal ⇒ 3)
- Normal CDF \(\Phi\), inverse \(\Phi^{-1}\)

### 7.2 Formulas

#### Estimated Sharpe Ratio (rf=0)
\[
\widehat{SR}=\frac{\bar r}{s_r}
\]

#### Annualized estimated Sharpe
\[
\widehat{SR}_{ann}=\sqrt{A}\,\widehat{SR}
\]

#### Standard deviation of \(\widehat{SR}\) (non-normal)
\[
\sigma_{\widehat{SR}}=
\sqrt{\frac{
1+\frac{1}{2}\widehat{SR}^2-\gamma_3\widehat{SR}+\frac{\gamma_4-3}{4}\widehat{SR}^2
}{n-1}}
\]

#### Probabilistic Sharpe Ratio (PSR)
\[
PSR(SR^\*)=\Phi\left(\frac{\widehat{SR}-SR^\*}{\sigma_{\widehat{SR}}}\right)
\]

#### Minimum Track Record Length (minTRL)
\[
\text{minTRL}=1+\sigma_{\widehat{SR}}^2(n-1)\left(\frac{\Phi^{-1}(p)}{\widehat{SR}-SR^\*}\right)^2
\]

#### Effective number of independent trials \(N_{\text{eff}}\)
Given \(m\) trials and average correlation \(\bar\rho\):
\[
N_{\text{eff}}\approx 1+\bar\rho+(1-\bar\rho)m
\]

#### Expected Maximum Sharpe Ratio after search (EMS)
Euler–Mascheroni constant \(\gamma\approx0.5772156649\):
\[
\mathbb{E}[\max SR]\approx SR_{mean}+\sigma_{trial}\cdot Z_{\max}
\]
\[
Z_{\max}=(1-\gamma)\Phi^{-1}\left(1-\frac{1}{N_{\text{eff}}}\right)+
\gamma\,\Phi^{-1}\left(1-\frac{1}{N_{\text{eff}}e}\right)
\]

#### Deflated Sharpe Ratio (DSR)
\[
DSR = \Phi\left(\frac{\widehat{SR}-\mathbb{E}[\max SR]}{\sigma_{\widehat{SR}}}\right)
\]

### 7.3 Reference implementation (paste into `alphaeval/metrics/sharpe_inference.py`)
```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def estimated_sharpe_ratio(returns):
    """Calculate the estimated sharpe ratio (risk_free=0)."""
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(returns=None, periods=261, *, sr=None):
    """Calculate the annualized estimated sharpe ratio (risk_free=0)."""
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sharpe_ratio_stdev(returns=None, *, n=None, skew=None, kurtosis=None, sr=None):
    """
    Calculate the standard deviation of the sharpe ratio estimation.
    Generalizes for both normal and non-normal returns.
    """
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if n is None:
        n = len(_returns)
    if skew is None:
        skew = pd.Series(scipy_stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy_stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = estimated_sharpe_ratio(_returns)

    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) == pd.DataFrame:
        sr_std = pd.Series(sr_std, index=returns.columns)
    elif type(sr_std) not in (float, np.float64, pd.DataFrame):
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
    """Calculate the Probabilistic Sharpe Ratio (PSR)."""
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    psr = scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
    """Calculate the Min Track Record Length (minTRL)."""
    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy_stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def num_independent_trials(trials_returns=None, *, m=None, p=None):
    """Calculate the number of independent trials."""
    if m is None:
        m = trials_returns.shape[1]

    if p is None:
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

    n = p + (1 - p) * m
    n = int(n) + 1  # round up
    return n


def expected_maximum_sr(trials_returns=None, expected_mean_sr=0.0, *, independent_trials=None, trials_sr_std=None):
    """Compute the expected maximum Sharpe ratio (analytical)."""
    emc = 0.5772156649  # Euler-Mascheroni constant

    if independent_trials is None:
        independent_trials = num_independent_trials(trials_returns)

    if trials_sr_std is None:
        srs = estimated_sharpe_ratio(trials_returns)
        trials_sr_std = srs.std()

    maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1.0 / independent_trials) + emc * scipy_stats.norm.ppf(
        1 - 1.0 / (independent_trials * np.e)
    )
    expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
    return expected_max_sr


def deflated_sharpe_ratio(trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
    """Calculate the Deflated Sharpe Ratio (DSR)."""
    if expected_max_sr is None:
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)

    dsr = probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)
    return dsr
8) API sketch (what you implement)
8.1 Transforms
price_to_returns(price, method="simple"|"log")

spread_to_delta_bp(spread)

spread_to_return_proxy(spread, duration) (uses 
−
𝐷
Δ
𝑠
−DΔs)

spread_to_dv01_pnl(delta_bp, dv01) (uses 
−
𝐷
𝑉
01
⋅
Δ
𝑠
𝑏
𝑝
−DV01⋅Δs 
bp
 )

8.2 Evaluate
evaluate_timeseries(panel, metrics=[...], config=...)

evaluate_cross_section(panel, factor_metrics=[...], config=...)

evaluate_execution(orders, fills, messages, metrics=[...], config=...)

8.3 Reports
report.to_markdown()

report.to_json()

9) Implementation plan (roadmap)
Phase 0 — Conventions
Freeze:

annualization factor 
𝐴
A

ddof convention (use ddof=1 consistently)

NaN policy (drop per metric)

spread units (bp vs decimal; enforce via transform naming)

Phase 1 — Core data wrappers + validation
Panel wrapper + schema checks

Simple adapters: pandas / polars / parquet

Phase 2 — Transforms
price returns, equity, drawdown, run-up

spread transforms (bp change, duration proxy returns, DV01 PnL)

Phase 3 — Metrics
performance, risk, execution, factor, relative

add tstat_mean_return, tstat_ic

sharpe inference module (PSR/DSR suite above)

Phase 4 — Runner + registry
MetricRegistry + evaluate() engine

consistent outputs + markdown rendering

Phase 5 — Tests
toy series tests + edge cases (std=0, no trades, NaNs)

monotonicity tests (PSR increases with SR, etc.)

selection-bias sanity (DSR ≤ PSR typically when many trials)

Phase 6 — Docs + examples
cookbook: “price vs spread”, “IC pipeline”, “PSR/DSR workflow”

10) Testing checklist (must-have)
Sharpe / Sortino:

constant returns ⇒ std=0 policy (NaN/inf) documented

t-stat:

mean zero synthetic ⇒ t around 0 (distribution sanity)

MDD:

monotonic equity ⇒ MDD=0

MaxRU:

monotonic up equity ⇒ MaxRU = (E_T - min(E))/min(E)

PF:

no losing trades ⇒ PF policy (inf or NaN) documented

IC / RankIC:

perfect monotonic ranks ⇒ ±1

IC t-stat:

iid IC with mean 0 ⇒ t around 0

IR:

zero tracking error ⇒ policy (NaN/inf) documented

PSR/minTRL/DSR:

PSR monotonic in SR

minTRL increases as prob increases

DSR decreases as #trials increases (all else equal)

11) Practical workflows
11.1 Time-series alpha eval (price)
Build strategy returns 
𝑟
𝑡
r 
t
​
  (or PnL then returns)

Compute equity 
𝐸
𝑡
E 
t
​
 

Run metrics: Sharpe, Sortino, t-stat, MDD, MaxRU, CAGR, PF, WinRate, Expectancy

11.2 Time-series alpha eval (spread)
Convert spread → 
Δ
𝑠
𝑏
𝑝
Δs 
bp
 

Choose proxy:

duration proxy returns, or

DV01 PnL

Run same metrics as price space (now on proxy returns / PnL)

11.3 Factor eval (IC / RankIC / IR*)
For each date, compute cross-section factor signal

Define target y_{t+1} (next-day RV, spread change, excess return)

Compute daily IC / RankIC and IC t-stat; then average + IR*

11.4 “Is this SR real?” (PSR / DSR)
Compute 
𝑆
𝑅
^
SR
  of selected strategy

Compute 
𝜎
𝑆
𝑅
^
σ 
SR
 
​
  (skew/kurtosis)

Compute PSR vs benchmark SR*

If you tried many alphas:

build trials_returns of all candidate strategies

compute expected max SR

compute DSR (deflated)


You said:
i need a buildable  very detailed .md to develop a plan

ChatGPT said:

Thought for 13s
# alphaeval — Build Plan (Corporate Bonds / Bond ETFs / CDX)  
**Goal:** build a library that evaluates alpha strategies in **price** and **spread** space, with **time-series**, **cross-sectional (factor)**, and optional **execution** evaluation. This document is written to be **buildable**: you can follow it top-to-bottom and ship a v1.

---

## 0) What you are building (scope statement)

### Instruments
- **Corporate bonds**
- **Corporate bond ETFs**
- **CDX indices**

### Evaluation spaces (must support both)
1) **Price space**
- returns and/or PnL based on price moves (or total-return index)

2) **Spread space**
- spread changes (bp)
- spread-return proxy using duration
- DV01-based PnL from spread moves

### Evaluation modes
A) **Time-series** (one strategy over time)  
B) **Cross-sectional / factor** (signal across instruments per day; IC/RankIC/IR*, R²)  
C) **Execution KPIs** (optional if you have orders/fills/messages)

### Deliverable
- A Python package `alphaeval` with:
  - stable input schema
  - transforms (price→returns, spread→bp / proxy returns / DV01 PnL, equity→DD/RU)
  - metric suite with formulas + tests
  - a runner that produces a structured report
  - markdown renderer for reports

---

## 1) Repo bootstrapping (buildable checklist)

### 1.1 Create repo + env
**Commands (example):**
```bash
mkdir alphaeval && cd alphaeval
git init
python -m venv .venv
source .venv/bin/activate
pip install -U pip
1.2 Dependencies (v1)
Required:

numpy, pandas, scipy

Dev:

pytest, ruff, mypy (optional), pre-commit (optional)

1.3 pyproject.toml (minimal)
[project]
name = "alphaeval"
version = "0.1.0"
description = "Alpha evaluation library for corporate bonds, bond ETFs, and CDX indices"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "pandas>=2.0", "scipy>=1.10"]

[project.optional-dependencies]
dev = ["pytest>=7.4", "ruff>=0.4", "mypy>=1.7"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 110
1.4 File tree (create now)
alphaeval/
  alphaeval/
    __init__.py
    core/
      __init__.py
      config.py
      panel.py
      runner.py
      registry.py
      types.py
      warnings.py
    transforms/
      __init__.py
      returns.py
      spreads.py
      equity.py
    metrics/
      __init__.py
      performance.py
      risk.py
      execution.py
      factor.py
      relative.py
      sharpe_inference.py
      tstats.py
    reports/
      __init__.py
      report.py
      render_md.py
  tests/
    test_transforms.py
    test_performance.py
    test_factor.py
    test_sharpe_inference.py
    test_tstats.py
  README.md
  BUILD_PLAN.md  (this file)
2) Conventions (decide once, encode in config)
2.1 Periodicity / annualization
periods_per_year default:

daily = 252 (or 261; pick one and standardize)

all annualized ratios should use sqrt(periods_per_year).

2.2 Standard deviation convention
Use ddof=1 for sample std by default (matches your Sharpe inference code).

2.3 Units and naming (avoid footguns)
spread stored in decimal (e.g., 100bp = 0.0100) OR in bp (choose one).

Recommended: store spread in decimal, produce delta_spread_bp explicitly.

Provide transform names that encode units:

delta_spread_bp(...) returns bp

spread_return_proxy_duration(...) expects decimal Δs

2.4 Missing data policy
Metric-level dropping:

each metric only drops rows where its required columns are missing

do not globally drop everything

2.5 “No trades / zero variance” policy
Document and enforce:

if denom is 0 → return NaN (recommended) + warning

PF with no losses: return inf (or NaN), but be explicit and consistent

3) Input schema (buildable spec)
3.1 Panel (primary input)
A Pandas DataFrame with either:

MultiIndex (date, instrument); or

columns date, instrument plus values

Minimum required columns (time-series eval):

returns OR price

Optional columns:

Spread evaluation: spread, duration, dv01

Benchmark: rf, benchmark_return

Strategy: position, pnl

Factor: signal, target

3.2 Execution inputs (optional)
orders_df

order_id, instrument, submit_ts, cancel_ts (optional), side, limit_price, quoted_size

fills_df

order_id, fill_ts, filled_size, fill_price

messages_df

ts, msg_type, instrument, order_id

4) Core architecture (what to implement)
4.1 Config (alphaeval/core/config.py)
Create a dataclass:

periods_per_year: int = 252

ddof: int = 1

risk_free_col: str = "rf"

benchmark_col: str = "benchmark_return"

nan_policy: str = "drop_required"

pf_no_loss_policy: str = "inf"

zero_div_policy: str = "nan"

sortino_threshold: float | None = None (None means use rf if present else 0)

var_alpha: float = 0.05

psr_benchmark_sr: float = 0.0

mintrl_prob: float = 0.95

4.2 Panel wrapper (alphaeval/core/panel.py)
Responsibilities:

validate schema

ensure sorted index

helpers:

get_series(col, instrument=None) for time-series

groupby_date() for cross-section

4.3 Metric interface (alphaeval/core/types.py)
Define:

MetricResult (name, value, optional series/table, warnings)

Metric protocol:

name: str

requires: set[str]

scope: Literal["timeseries","cross_section","execution"]

compute(panel, orders=None, fills=None, messages=None, config=None) -> MetricResult

4.4 Registry (alphaeval/core/registry.py)
register(name, metric_callable_or_class)

get(name)

list()

4.5 Runner (alphaeval/core/runner.py)
evaluate(panel, metrics=[...], orders=None, fills=None, messages=None, config=Config()) -> Report

5) Transforms (build these first)
5.1 Price returns (alphaeval/transforms/returns.py)
Simple returns

𝑟
𝑡
=
𝑃
𝑡
𝑃
𝑡
−
1
−
1
r 
t
​
 = 
P 
t−1
​
 
P 
t
​
 
​
 −1
Log returns

𝑟
𝑡
=
log
⁡
(
𝑃
𝑡
𝑃
𝑡
−
1
)
r 
t
​
 =log( 
P 
t−1
​
 
P 
t
​
 
​
 )
5.2 Spread transforms (alphaeval/transforms/spreads.py)
Spread change in bp

Δ
𝑠
𝑡
𝑏
𝑝
=
(
𝑠
𝑡
−
𝑠
𝑡
−
1
)
⋅
10
4
Δs 
t
bp
​
 =(s 
t
​
 −s 
t−1
​
 )⋅10 
4
 
Duration spread-return proxy

𝑟
𝑡
𝑠
𝑝
𝑟
𝑒
𝑎
𝑑
≈
−
𝐷
𝑡
⋅
Δ
𝑠
𝑡
r 
t
spread
​
 ≈−D 
t
​
 ⋅Δs 
t
​
 
(Δs in decimal)

DV01 PnL proxy

PnL
𝑡
𝑠
𝑝
𝑟
𝑒
𝑎
𝑑
≈
−
DV01
𝑡
⋅
Δ
𝑠
𝑡
𝑏
𝑝
PnL 
t
spread
​
 ≈−DV01 
t
​
 ⋅Δs 
t
bp
​
 
5.3 Equity + drawdown/run-up (alphaeval/transforms/equity.py)
Equity curve

𝐸
0
=
1
,
𝐸
𝑡
=
𝐸
𝑡
−
1
(
1
+
𝑟
𝑡
)
E 
0
​
 =1,E 
t
​
 =E 
t−1
​
 (1+r 
t
​
 )
Max Drawdown

𝑃
𝑡
=
max
⁡
0
≤
𝑢
≤
𝑡
𝐸
𝑢
,
𝐷
𝐷
𝑡
=
𝑃
𝑡
−
𝐸
𝑡
𝑃
𝑡
,
MDD
=
max
⁡
𝑡
𝐷
𝐷
𝑡
P 
t
​
 = 
0≤u≤t
max
​
 E 
u
​
 ,DD 
t
​
 = 
P 
t
​
 
P 
t
​
 −E 
t
​
 
​
 ,MDD= 
t
max
​
 DD 
t
​
 
Max Run-up

𝑄
𝑡
=
min
⁡
0
≤
𝑢
≤
𝑡
𝐸
𝑢
,
𝑅
𝑈
𝑡
=
𝐸
𝑡
−
𝑄
𝑡
𝑄
𝑡
,
MaxRU
=
max
⁡
𝑡
𝑅
𝑈
𝑡
Q 
t
​
 = 
0≤u≤t
min
​
 E 
u
​
 ,RU 
t
​
 = 
Q 
t
​
 
E 
t
​
 −Q 
t
​
 
​
 ,MaxRU= 
t
max
​
 RU 
t
​
 
6) Metrics (implement by module)
Rule: each metric has (1) formula, (2) required columns, (3) tests, (4) edge policy.

6.1 Performance metrics (alphaeval/metrics/performance.py)
Sharpe (estimated)
Sharpe
=
𝜇
𝑥
𝜎
𝑥
,
𝑥
𝑡
=
𝑟
𝑡
−
𝑟
𝑓
,
𝑡
Sharpe= 
σ 
x
​
 
μ 
x
​
 
​
 ,x 
t
​
 =r 
t
​
 −r 
f,t
​
 
Annualized:

Sharpe
𝑎
𝑛
𝑛
=
𝐴
𝜇
𝑥
𝜎
𝑥
Sharpe 
ann
​
 = 
A
​
  
σ 
x
​
 
μ 
x
​
 
​
 
Requires: returns (+ optional rf)

Sortino
Downside deviation vs threshold 
𝜏
τ:

𝜎
𝑑
𝑜
𝑤
𝑛
=
1
𝑇
∑
𝑡
=
1
𝑇
min
⁡
(
0
,
𝑟
𝑡
−
𝜏
)
2
σ 
down
​
 = 
T
1
​
  
t=1
∑
T
​
 min(0,r 
t
​
 −τ) 
2
 
​
 
Sortino
=
𝐸
[
𝑟
𝑡
−
𝜏
]
𝜎
𝑑
𝑜
𝑤
𝑛
,
Sortino
𝑎
𝑛
𝑛
=
𝐴
⋅
Sortino
Sortino= 
σ 
down
​
 
E[r 
t
​
 −τ]
​
 ,Sortino 
ann
​
 = 
A
​
 ⋅Sortino
Requires: returns (+ optional rf)

Profit Factor (PF)
With trade PnLs 
𝑝
𝑖
p 
i
​
 :

PF
=
∑
𝑖
:
𝑝
𝑖
>
0
𝑝
𝑖
∑
𝑖
:
𝑝
𝑖
<
0
∣
𝑝
𝑖
∣
PF= 
∑ 
i:p 
i
​
 <0
​
 ∣p 
i
​
 ∣
∑ 
i:p 
i
​
 >0
​
 p 
i
​
 
​
 
Requires: trade_pnl OR derived trade list

Win Rate
WinRate
=
#
{
𝑖
:
𝑝
𝑖
>
0
}
𝑁
×
100
WinRate= 
N
#{i:p 
i
​
 >0}
​
 ×100
Requires: trade PnLs

Expectancy (EV per trade)
Expectancy
=
WinRate
⋅
AvgWin
−
LossRate
⋅
AvgLoss
Expectancy=WinRate⋅AvgWin−LossRate⋅AvgLoss
Equivalent:

Expectancy
=
1
𝑁
∑
𝑖
𝑝
𝑖
Expectancy= 
N
1
​
  
i
∑
​
 p 
i
​
 
CAGR
CAGR
=
(
𝐸
𝑇
𝐸
0
)
1
/
𝑌
−
1
CAGR=( 
E 
0
​
 
E 
T
​
 
​
 ) 
1/Y
 −1
Max Drawdown (MDD), Max Run-up (MaxRU)
(see transforms section)

CAGR / MDD
CAGR
MDD
MDD
CAGR
​
 
6.2 t-stats (alphaeval/metrics/tstats.py)
t-stat of mean return
Test 
𝐻
0
:
𝐸
[
𝑟
]
=
0
H 
0
​
 :E[r]=0 (or excess return 
𝑥
x):

𝑡
=
𝑟
ˉ
𝑠
𝑟
/
𝑇
or
𝑡
=
𝑥
ˉ
𝑠
𝑥
/
𝑇
t= 
s 
r
​
 / 
T
​
 
r
ˉ
 
​
 ort= 
s 
x
​
 / 
T
​
 
x
ˉ
 
​
 
Requires: returns (+ optional rf)

Implementation note (v1): iid t-stat.
Future: HAC/Newey–West if autocorrelation matters.

t-stat of daily IC
Given daily IC series 
𝐼
𝐶
𝑡
IC 
t
​
 , 
𝑡
=
1
,
…
,
𝑇
𝑑
t=1,…,T 
d
​
 :

𝑡
𝐼
𝐶
=
𝐼
𝐶
‾
𝑠
𝐼
𝐶
/
𝑇
𝑑
t 
IC
​
 = 
s 
IC
​
 / 
T 
d
​
 
​
 
IC
 
​
 
6.3 Risk metrics (alphaeval/metrics/risk.py)
VaR (historical)
Loss series 
𝐿
𝑡
L 
t
​
  (e.g., 
−
𝑟
𝑡
−r 
t
​
  or 
−
PnL
𝑡
−PnL 
t
​
 ):

VaR
𝛼
=
Quantile
𝛼
(
𝐿
)
VaR 
α
​
 =Quantile 
α
​
 (L)
VaR (Gaussian parametric)
If 
𝐿
∼
𝑁
(
𝜇
𝐿
,
𝜎
𝐿
2
)
L∼N(μ 
L
​
 ,σ 
L
2
​
 ):

VaR
𝛼
=
𝜇
𝐿
+
𝑧
𝛼
𝜎
𝐿
VaR 
α
​
 =μ 
L
​
 +z 
α
​
 σ 
L
​
 
6.4 Execution metrics (alphaeval/metrics/execution.py) (optional)
Latency
Latency
𝑖
=
𝑡
𝑓
𝑖
𝑙
𝑙
(
𝑖
)
−
𝑡
𝑠
𝑢
𝑏
𝑚
𝑖
𝑡
(
𝑖
)
Latency 
i
​
 =t 
fill
(i)
​
 −t 
submit
(i)
​
 
Fill probability
Volume-based:

𝑃
(
Fill
)
𝑇
=
FilledVolume
𝑇
SubmittedVolume
𝑇
P(Fill) 
T
​
 = 
SubmittedVolume 
T
​
 
FilledVolume 
T
​
 
​
 
Message traffic
Mess
(
Δ
)
=
#
{
messages in bucket 
Δ
}
Mess(Δ)=#{messages in bucket Δ}
Quote-to-trade
QTR
=
#
{
quotes
}
#
{
trades
}
QTR= 
#{trades}
#{quotes}
​
 
VPIN (spec stub)
Volume buckets 
𝑏
b with fixed bucket volume 
𝑉
V:

VPIN
𝑡
=
∑
𝑏
=
𝑡
−
𝑛
+
1
𝑡
∣
𝑉
𝑏
+
−
𝑉
𝑏
−
∣
𝑛
𝑉
VPIN 
t
​
 = 
nV
∑ 
b=t−n+1
t
​
 ∣V 
b
+
​
 −V 
b
−
​
 ∣
​
 
6.5 Factor metrics (alphaeval/metrics/factor.py)
𝑅
2
R 
2
 
𝑅
2
=
1
−
∑
𝑖
=
1
𝑀
𝑡
𝑒
𝑠
𝑡
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
∑
𝑖
=
1
𝑀
𝑡
𝑒
𝑠
𝑡
(
𝑦
𝑖
−
𝑦
ˉ
)
2
R 
2
 =1− 
∑ 
i=1
M 
test
​
 
​
 (y 
i
​
 − 
y
ˉ
​
 ) 
2
 
∑ 
i=1
M 
test
​
 
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
​
 
Requires: y_true, y_pred (or build a helper)

𝐼
𝐶
∗
IC 
∗
  and RankIC*
Composite factor:

𝑔
𝑡
,
𝑖
=
∑
𝑗
=
1
𝑘
𝑤
𝑗
𝑓
𝑗
(
𝑋
𝑡
,
𝑖
)
g 
t,i
​
 = 
j=1
∑
k
​
 w 
j
​
 f 
j
​
 (X 
t,i
​
 )
Daily IC:

𝐼
𝐶
𝑡
=
𝜌
(
𝑔
𝑡
,
⋅
,
𝑦
𝑡
+
1
,
⋅
)
IC 
t
​
 =ρ(g 
t,⋅
​
 ,y 
t+1,⋅
​
 )
Averaged:

𝐼
𝐶
∗
=
𝐸
𝑡
[
𝐼
𝐶
𝑡
]
IC 
∗
 =E 
t
​
 [IC 
t
​
 ]
RankIC:

𝑅
𝑎
𝑛
𝑘
𝐼
𝐶
𝑡
=
𝜌
(
𝑟
(
𝑔
𝑡
,
⋅
)
,
𝑟
(
𝑦
𝑡
+
1
,
⋅
)
)
RankIC 
t
​
 =ρ(r(g 
t,⋅
​
 ),r(y 
t+1,⋅
​
 ))
𝐼
𝑅
∗
IR 
∗
 
𝐼
𝑅
∗
=
𝐸
𝑡
[
𝐼
𝐶
𝑡
]
S
t
d
𝑡
(
𝐼
𝐶
𝑡
)
IR 
∗
 = 
Std 
t
​
 (IC 
t
​
 )
E 
t
​
 [IC 
t
​
 ]
​
 
6.6 Benchmark-relative (alphaeval/metrics/relative.py)
Information Ratio (IR)
Active return 
𝑎
𝑡
=
𝑟
𝑝
,
𝑡
−
𝑟
𝑏
,
𝑡
a 
t
​
 =r 
p,t
​
 −r 
b,t
​
 :

𝐼
𝑅
=
𝐸
[
𝑎
𝑡
]
𝜎
(
𝑎
𝑡
)
IR= 
σ(a 
t
​
 )
E[a 
t
​
 ]
​
 
Requires: returns, benchmark_return

7) Advanced Sharpe inference (PSR / DSR) — ship as a module
7.1 What it solves
non-normal returns (skew/kurtosis)

selection bias from picking best alpha among many

7.2 Formulas
Estimated Sharpe (rf=0):

𝑆
𝑅
^
=
𝑟
ˉ
𝑠
𝑟
SR
 = 
s 
r
​
 
r
ˉ
 
​
 
Annualized:

𝑆
𝑅
^
𝑎
𝑛
𝑛
=
𝐴
𝑆
𝑅
^
SR
  
ann
​
 = 
A
​
  
SR
 
Std dev of estimated SR (non-normal):

𝜎
𝑆
𝑅
^
=
1
+
1
2
𝑆
𝑅
^
2
−
𝛾
3
𝑆
𝑅
^
+
𝛾
4
−
3
4
𝑆
𝑅
^
2
𝑛
−
1
σ 
SR
 
​
 = 
n−1
1+ 
2
1
​
  
SR
  
2
 −γ 
3
​
  
SR
 + 
4
γ 
4
​
 −3
​
  
SR
  
2
 
​
 
​
 
PSR vs benchmark 
𝑆
𝑅
\*
SR 
\*
 :

𝑃
𝑆
𝑅
(
𝑆
𝑅
\*
)
=
Φ
(
𝑆
𝑅
^
−
𝑆
𝑅
\*
𝜎
𝑆
𝑅
^
)
PSR(SR 
\*
 )=Φ( 
σ 
SR
 
​
 
SR
 −SR 
\*
 
​
 )
minTRL for target probability 
𝑝
p:

minTRL
=
1
+
𝜎
𝑆
𝑅
^
2
(
𝑛
−
1
)
(
Φ
−
1
(
𝑝
)
𝑆
𝑅
^
−
𝑆
𝑅
\*
)
2
minTRL=1+σ 
SR
 
2
​
 (n−1)( 
SR
 −SR 
\*
 
Φ 
−1
 (p)
​
 ) 
2
 
Effective independent trials:

𝑁
eff
≈
1
+
𝜌
ˉ
+
(
1
−
𝜌
ˉ
)
𝑚
N 
eff
​
 ≈1+ 
ρ
ˉ
​
 +(1− 
ρ
ˉ
​
 )m
Expected max SR after search:

𝐸
[
max
⁡
𝑆
𝑅
]
≈
𝑆
𝑅
𝑚
𝑒
𝑎
𝑛
+
𝜎
𝑡
𝑟
𝑖
𝑎
𝑙
⋅
𝑍
max
⁡
E[maxSR]≈SR 
mean
​
 +σ 
trial
​
 ⋅Z 
max
​
 
Deflated SR:

𝐷
𝑆
𝑅
=
Φ
(
𝑆
𝑅
^
−
𝐸
[
max
⁡
𝑆
𝑅
]
𝜎
𝑆
𝑅
^
)
DSR=Φ( 
σ 
SR
 
​
 
SR
 −E[maxSR]
​
 )
7.3 Implementation (paste as-is)
Create: alphaeval/metrics/sharpe_inference.py

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def estimated_sharpe_ratio(returns):
    """Calculate the estimated sharpe ratio (risk_free=0)."""
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(returns=None, periods=261, *, sr=None):
    """Calculate the annualized estimated sharpe ratio (risk_free=0)."""
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sharpe_ratio_stdev(returns=None, *, n=None, skew=None, kurtosis=None, sr=None):
    """
    Calculate the standard deviation of the sharpe ratio estimation.
    Generalizes for both normal and non-normal returns.
    """
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if n is None:
        n = len(_returns)
    if skew is None:
        skew = pd.Series(scipy_stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy_stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = estimated_sharpe_ratio(_returns)

    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) == pd.DataFrame:
        sr_std = pd.Series(sr_std, index=returns.columns)
    elif type(sr_std) not in (float, np.float64, pd.DataFrame):
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
    """Calculate the Probabilistic Sharpe Ratio (PSR)."""
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    psr = scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
    """Calculate the Min Track Record Length (minTRL)."""
    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy_stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def num_independent_trials(trials_returns=None, *, m=None, p=None):
    """Calculate the number of independent trials."""
    if m is None:
        m = trials_returns.shape[1]

    if p is None:
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

    n = p + (1 - p) * m
    n = int(n) + 1  # round up
    return n


def expected_maximum_sr(trials_returns=None, expected_mean_sr=0.0, *, independent_trials=None, trials_sr_std=None):
    """Compute the expected maximum Sharpe ratio (analytical)."""
    emc = 0.5772156649  # Euler-Mascheroni constant

    if independent_trials is None:
        independent_trials = num_independent_trials(trials_returns)

    if trials_sr_std is None:
        srs = estimated_sharpe_ratio(trials_returns)
        trials_sr_std = srs.std()

    maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1.0 / independent_trials) + emc * scipy_stats.norm.ppf(
        1 - 1.0 / (independent_trials * np.e)
    )
    expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
    return expected_max_sr


def deflated_sharpe_ratio(trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
    """Calculate the Deflated Sharpe Ratio (DSR)."""
    if expected_max_sr is None:
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)

    dsr = probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)
    return dsr
8) Reports (buildable output contract)
8.1 Report object (alphaeval/reports/report.py)
Must contain:

summary: dict[str, float]

series: dict[str, pd.Series]

tables: dict[str, pd.DataFrame]

warnings: list[str]

8.2 Markdown renderer (alphaeval/reports/render_md.py)
A to_markdown() method that prints:

headline metrics (Sharpe/Sortino/t-stat/MDD/CAGR/IR)

factor summary (IC*/RankIC*/IR*/tIC)

Sharpe inference (PSR/minTRL/DSR)

warnings block

9) Tests (make it correct)
9.1 Transform tests
price→returns correctness on a tiny sequence

spread decimal → delta_bp correctness

duration proxy: sign and scaling

dv01 pnl: units and sign

9.2 Metric tests (toy data you can hand-calc)
Sharpe on simple known returns

Sortino with threshold 0 vs rf

MDD on monotonic equity → 0

MaxRU on monotonic up equity → expected

t-stat on mean-zero synthetic → around 0

IR with constant active returns → denom=0 policy

IC perfect monotonic ranks → ±1

9.3 Sharpe inference tests
PSR monotonic with SR

minTRL increases as prob increases

DSR ≤ PSR when expected_max_sr > sr_benchmark (typical)

10) Implementation order (the actual build plan)
Milestone M0 — Skeleton (30–60 min)
 Create tree, pyproject, basic imports, empty modules

 Add Config dataclass

 Add Report object + to_markdown() stub

Milestone M1 — Transforms (1–2 hours)
 returns.py (simple/log)

 spreads.py (delta_bp, duration proxy, dv01 pnl)

 equity.py (equity, dd, runup)

Acceptance: unit tests for transforms pass.

Milestone M2 — Core performance + t-stats (2–4 hours)
 Sharpe, Sortino

 MDD, MaxRU

 CAGR, CAGR/MDD

 t-stat mean return/excess return

Acceptance: evaluate_timeseries() produces summary + equity/dd series.

Milestone M3 — Factor metrics (2–4 hours)
 IC daily, mean IC*, RankIC*

 IR* of IC series

 t-stat of IC series

 R² helper (optional for v1 unless you already have preds)

Acceptance: evaluate_cross_section() outputs IC table + summary.

Milestone M4 — Sharpe inference (1 hour)
 Add sharpe_inference.py (paste code)

 Wrap as metrics (PSR/minTRL/DSR) in registry

Acceptance: PSR/minTRL/DSR show up in report.

Milestone M5 — Execution metrics (optional; 2–6 hours)
 latency stats

 fill probability

 message counts, QTR

 VPIN (stub or implement after bucket spec)

11) Acceptance criteria (definition of done)
v1 must:
 Evaluate a strategy in price space from returns or price

 Evaluate in spread space using:

 Δspread(bp), and at least one of:

 duration proxy returns

 DV01 pnl proxy

 Produce:

 Sharpe, Sortino, t-stat (mean return or excess)

 MDD, MaxRU, CAGR, CAGR/MDD

 IR (if benchmark provided)

 Cross-sectional factor evaluation:

 IC*, RankIC*, IR*, IC t-stat

 Sharpe inference:

 PSR, minTRL, DSR

 Provide report.to_markdown() output

 Have tests covering transforms + core metrics + sharpe inference

12) Usage examples (buildable API examples)
12.1 Time-series (price)
from alphaeval.core.config import Config
from alphaeval.core.runner import evaluate

cfg = Config(periods_per_year=252)

report = evaluate(
    panel=df,  # includes returns or price
    metrics=[
        "perf.sharpe",
        "perf.sortino",
        "tstat.mean_return",
        "perf.mdd",
        "perf.max_runup",
        "perf.cagr",
        "perf.cagr_mdd",
    ],
    config=cfg,
)

print(report.to_markdown())
12.2 Time-series (spread)
from alphaeval.transforms.spreads import delta_spread_bp, spread_return_proxy_duration

df["delta_spread_bp"] = delta_spread_bp(df["spread"])
df["spread_proxy_return"] = spread_return_proxy_duration(df["spread"], df["duration"])

report = evaluate(
    panel=df.rename(columns={"spread_proxy_return": "returns"}),
    metrics=["perf.sharpe", "perf.mdd", "tstat.mean_return"],
)
12.3 Cross-sectional factor (IC)
# df has columns: date, instrument, signal, target
report = evaluate(
    panel=df,
    metrics=["factor.ic_star", "factor.rank_ic_star", "factor.ir_star", "tstat.ic"],
)
12.4 DSR workflow
from alphaeval.metrics.sharpe_inference import deflated_sharpe_ratio

dsr = deflated_sharpe_ratio(
    trials_returns=trials_df,         # many strategies tried
    returns_selected=selected_series, # the winner
)
13) Notes for spread vs price correctness (practical)
Spread proxies are approximations. Always label which proxy you used:

duration proxy returns: 
−
𝐷
Δ
𝑠
−DΔs

DV01 PnL: # alphaeval — Per-metric spec tables (requires / outputs / edge-cases)

This document defines **explicit, implementable contracts** for every v1 metric.
Each metric includes:
- **Requires**: exact input columns or objects
- **Outputs**: scalar(s) and optional series/tables
- **Edge-cases**: exact behavior and required warnings
- **Formula**: canonical definition (to match tests)

> Conventions assumed:
> - `ddof = 1` for sample std unless stated
> - `periods_per_year = A` (default 252 unless you pick 261)
> - `zero_div_policy = "nan"` → return NaN + warning
> - NaNs: drop only rows needed for the metric (“drop_required”)

---

## 0) Shared definitions

### 0.1 Standard inputs
- `returns`: per-period strategy return series \(r_t\)
- `rf`: per-period risk-free rate \(r_{f,t}\) (optional)
- `benchmark_return`: per-period benchmark return \(r_{b,t}\) (optional)

### 0.2 Standard derived series (helpers)
- Excess return: \(x_t = r_t - r_{f,t}\)
- Equity curve: \(E_0=1,\;E_t=E_{t-1}(1+r_t)\)
- Running peak: \(P_t = \max_{u\le t} E_u\)
- Running trough: \(Q_t = \min_{u\le t} E_u\)

### 0.3 Standard output structure
Each metric returns a `MetricResult`:
- `name: str`
- `value: float | pd.Series` (scalar for single series; series for column-wise)
- `series: dict[str, pd.Series]` (optional)
- `table: pd.DataFrame | None` (optional)
- `warnings: list[str]` (optional; must include edge-case messages)

---

## 1) Transforms (required for v1 spread/price symmetry)

### T1 — `price_to_returns`
| Field | Spec |
|---|---|
| **Requires** | `price: pd.Series` (sorted by time) |
| **Outputs** | `returns: pd.Series` aligned to price index (first value NaN) |
| **Formula** | simple: \(r_t = P_t/P_{t-1}-1\); log: \(r_t=\log(P_t/P_{t-1})\) |
| **Edge-cases** | non-positive prices for log returns → NaN + warning; missing price → drop_required |

### T2 — `delta_spread_bp`
| Field | Spec |
|---|---|
| **Requires** | `spread: pd.Series` (spread in **decimal**) |
| **Outputs** | `delta_spread_bp: pd.Series` (first value NaN) |
| **Formula** | \(\Delta s_t^{bp}=(s_t-s_{t-1})\cdot10^4\) |
| **Edge-cases** | if spread appears in bp (user error) → cannot detect reliably; provide optional `assume_units` config |

### T3 — `spread_return_proxy_duration`
| Field | Spec |
|---|---|
| **Requires** | `spread: pd.Series` in decimal, `duration: pd.Series` |
| **Outputs** | `proxy_return: pd.Series` |
| **Formula** | \(r^{spread}_t \approx -D_t\cdot\Delta s_t\) (Δs in decimal) |
| **Edge-cases** | missing duration → NaN + warning; duration ≤ 0 → NaN + warning |

### T4 — `dv01_pnl_from_spread_bp`
| Field | Spec |
|---|---|
| **Requires** | `delta_spread_bp: pd.Series`, `dv01: pd.Series` |
| **Outputs** | `pnl: pd.Series` |
| **Formula** | \(\text{PnL}^{spread}_t\approx -\text{DV01}_t\cdot\Delta s_t^{bp}\) |
| **Edge-cases** | dv01 missing → NaN + warning |

### T5 — `equity_curve`
| Field | Spec |
|---|---|
| **Requires** | `returns: pd.Series` (NaNs allowed) |
| **Outputs** | `equity: pd.Series` (start at 1) |
| **Formula** | \(E_0=1;\;E_t=E_{t-1}(1+r_t)\) |
| **Edge-cases** | returns ≤ -1 breaks multiplicative equity → set equity to NaN forward + warning |

### T6 — `drawdown_series`
| Field | Spec |
|---|---|
| **Requires** | `equity: pd.Series` |
| **Outputs** | `dd: pd.Series` |
| **Formula** | \(P_t=\max_{u\le t}E_u;\;DD_t=(P_t-E_t)/P_t\) |
| **Edge-cases** | equity ≤ 0 → NaN + warning |

### T7 — `runup_series`
| Field | Spec |
|---|---|
| **Requires** | `equity: pd.Series` |
| **Outputs** | `ru: pd.Series` |
| **Formula** | \(Q_t=\min_{u\le t}E_u;\;RU_t=(E_t-Q_t)/Q_t\) |
| **Edge-cases** | trough \(Q_t\le 0\) → NaN + warning |

---

## 2) Time-series performance metrics

### P1 — `perf.sharpe` (estimated, rf optional)
| Field | Spec |
|---|---|
| **Requires** | `returns` (Series). Optional `rf` |
| **Outputs** | scalar Sharpe; optionally `series["excess_returns"]` |
| **Formula** | \(x_t=r_t-r_{f,t}\);\; \(\text{Sharpe}=\bar x/s_x\). Annualized: \(\sqrt{A}\bar x/s_x\) if metric is `perf.sharpe_ann` |
| **Edge-cases** | std=0 → NaN + warning `"sharpe: zero variance"`; insufficient obs (<2) → NaN + warning |
| **Notes** | use `ddof=1` for std |

### P2 — `perf.sortino` (rf or threshold)
| Field | Spec |
|---|---|
| **Requires** | `returns`. Optional `rf` or `config.sortino_threshold` |
| **Outputs** | scalar Sortino |
| **Formula** | threshold \(\tau = config.sortino_threshold\) else \(\tau=\bar r_f\) if `rf` exists else 0. Downside: \(\sigma_{down}=\sqrt{\frac{1}{T}\sum \min(0,r_t-\tau)^2}\). Sortino \(=\mathbb{E}[r_t-\tau]/\sigma_{down}\). Annualized multiply by \(\sqrt{A}\) |
| **Edge-cases** | no downside (σ_down=0) → NaN + warning `"sortino: zero downside deviation"`; insufficient obs → NaN |

### P3 — `perf.mdd` (max drawdown)
| Field | Spec |
|---|---|
| **Requires** | `returns` OR `equity` (if provided). Prefer equity if present. |
| **Outputs** | scalar MDD; optionally `series["drawdown"]` |
| **Formula** | \(E_t\) from returns; \(DD_t=(P_t-E_t)/P_t\); MDD \(=\max_t DD_t\) |
| **Edge-cases** | equity invalid (≤0) or returns ≤ -1 → NaN + warning `"mdd: invalid equity"` |

### P4 — `perf.max_runup`
| Field | Spec |
|---|---|
| **Requires** | `returns` OR `equity` |
| **Outputs** | scalar MaxRU; optionally `series["runup"]` |
| **Formula** | \(RU_t=(E_t-Q_t)/Q_t\); MaxRU \(=\max_t RU_t\) |
| **Edge-cases** | trough \(Q_t\le0\) → NaN + warning `"runup: invalid trough"` |

### P5 — `perf.cagr`
| Field | Spec |
|---|---|
| **Requires** | `returns` OR `equity`, plus `config.periods_per_year` |
| **Outputs** | scalar CAGR |
| **Formula** | let \(T\) periods; years \(Y=T/A\). \( \text{CAGR}=(E_T/E_0)^{1/Y}-1 \) |
| **Edge-cases** | \(E_T\le0\) or \(Y\le0\) → NaN + warning `"cagr: invalid horizon"` |

### P6 — `perf.cagr_mdd` (Calmar-style)
| Field | Spec |
|---|---|
| **Requires** | `perf.cagr` and `perf.mdd` (computed or recompute internally) |
| **Outputs** | scalar CAGR/MDD |
| **Formula** | \(\text{CAGR}/\text{MDD}\) |
| **Edge-cases** | MDD=0 → NaN (or inf if you prefer) + warning `"cagr_mdd: zero drawdown"` |

---

## 3) Trade-based performance metrics (optional v1 if you have trade PnL)

> If you don’t have trade-level PnL, keep these as v1.1 or accept an input column `trade_pnl` list/series.

### T1 — `trade.profit_factor`
| Field | Spec |
|---|---|
| **Requires** | `trade_pnl: pd.Series` (one value per trade) |
| **Outputs** | scalar PF |
| **Formula** | \(GP=\sum_{p_i>0}p_i\); \(GL=\sum_{p_i<0}|p_i|\); \(PF=GP/GL\) |
| **Edge-cases** | no losing trades (GL=0): return `inf` if `pf_no_loss_policy="inf"` else NaN; warning `"pf: no losses"`; no trades → NaN + warning |

### T2 — `trade.win_rate`
| Field | Spec |
|---|---|
| **Requires** | `trade_pnl` |
| **Outputs** | scalar WinRate (%) |
| **Formula** | \(100\cdot \#\{p_i>0\}/N\) |
| **Edge-cases** | N=0 → NaN + warning `"win_rate: no trades"` |

### T3 — `trade.expectancy`
| Field | Spec |
|---|---|
| **Requires** | `trade_pnl` |
| **Outputs** | scalar Expectancy |
| **Formula** | \(WR=N_w/N\), \(LR=1-WR\), \(AvgWin=\frac{1}{N_w}\sum_{p_i>0}p_i\), \(AvgLoss=\frac{1}{_

−
𝐷
𝑉
01
⋅
Δ
𝑠
𝑏
𝑝
−DV01⋅Δs 
bp
 

For ETFs and CDX indices, spread inputs may be:

implied spreads or index OAS proxies

in v1 treat them uniformly as spread + duration when available

14) Next steps (v1.1+)
HAC / Newey–West t-stats

bootstrap confidence intervals for Sharpe / IC

turnover, costs, slippage models

richer execution analytics (fill curves, adverse selection)

standardized CDX roll handling and ETF NAV vs price decomposition

# alphaeval — Per-metric spec tables (requires / outputs / edge-cases)

This document defines **explicit, implementable contracts** for every v1 metric.
Each metric includes:
- **Requires**: exact input columns or objects
- **Outputs**: scalar(s) and optional series/tables
- **Edge-cases**: exact behavior and required warnings
- **Formula**: canonical definition (to match tests)

> Conventions assumed:
> - `ddof = 1` for sample std unless stated
> - `periods_per_year = A` (default 252 unless you pick 261)
> - `zero_div_policy = "nan"` → return NaN + warning
> - NaNs: drop only rows needed for the metric (“drop_required”)

---

## 0) Shared definitions

### 0.1 Standard inputs
- `returns`: per-period strategy return series \(r_t\)
- `rf`: per-period risk-free rate \(r_{f,t}\) (optional)
- `benchmark_return`: per-period benchmark return \(r_{b,t}\) (optional)

### 0.2 Standard derived series (helpers)
- Excess return: \(x_t = r_t - r_{f,t}\)
- Equity curve: \(E_0=1,\;E_t=E_{t-1}(1+r_t)\)
- Running peak: \(P_t = \max_{u\le t} E_u\)
- Running trough: \(Q_t = \min_{u\le t} E_u\)

### 0.3 Standard output structure
Each metric returns a `MetricResult`:
- `name: str`
- `value: float | pd.Series` (scalar for single series; series for column-wise)
- `series: dict[str, pd.Series]` (optional)
- `table: pd.DataFrame | None` (optional)
- `warnings: list[str]` (optional; must include edge-case messages)

---

## 1) Transforms (required for v1 spread/price symmetry)

### T1 — `price_to_returns`
| Field | Spec |
|---|---|
| **Requires** | `price: pd.Series` (sorted by time) |
| **Outputs** | `returns: pd.Series` aligned to price index (first value NaN) |
| **Formula** | simple: \(r_t = P_t/P_{t-1}-1\); log: \(r_t=\log(P_t/P_{t-1})\) |
| **Edge-cases** | non-positive prices for log returns → NaN + warning; missing price → drop_required |

### T2 — `delta_spread_bp`
| Field | Spec |
|---|---|
| **Requires** | `spread: pd.Series` (spread in **decimal**) |
| **Outputs** | `delta_spread_bp: pd.Series` (first value NaN) |
| **Formula** | \(\Delta s_t^{bp}=(s_t-s_{t-1})\cdot10^4\) |
| **Edge-cases** | if spread appears in bp (user error) → cannot detect reliably; provide optional `assume_units` config |

### T3 — `spread_return_proxy_duration`
| Field | Spec |
|---|---|
| **Requires** | `spread: pd.Series` in decimal, `duration: pd.Series` |
| **Outputs** | `proxy_return: pd.Series` |
| **Formula** | \(r^{spread}_t \approx -D_t\cdot\Delta s_t\) (Δs in decimal) |
| **Edge-cases** | missing duration → NaN + warning; duration ≤ 0 → NaN + warning |

### T4 — `dv01_pnl_from_spread_bp`
| Field | Spec |
|---|---|
| **Requires** | `delta_spread_bp: pd.Series`, `dv01: pd.Series` |
| **Outputs** | `pnl: pd.Series` |
| **Formula** | \(\text{PnL}^{spread}_t\approx -\text{DV01}_t\cdot\Delta s_t^{bp}\) |
| **Edge-cases** | dv01 missing → NaN + warning |

### T5 — `equity_curve`
| Field | Spec |
|---|---|
| **Requires** | `returns: pd.Series` (NaNs allowed) |
| **Outputs** | `equity: pd.Series` (start at 1) |
| **Formula** | \(E_0=1;\;E_t=E_{t-1}(1+r_t)\) |
| **Edge-cases** | returns ≤ -1 breaks multiplicative equity → set equity to NaN forward + warning |

### T6 — `drawdown_series`
| Field | Spec |
|---|---|
| **Requires** | `equity: pd.Series` |
| **Outputs** | `dd: pd.Series` |
| **Formula** | \(P_t=\max_{u\le t}E_u;\;DD_t=(P_t-E_t)/P_t\) |
| **Edge-cases** | equity ≤ 0 → NaN + warning |

### T7 — `runup_series`
| Field | Spec |
|---|---|
| **Requires** | `equity: pd.Series` |
| **Outputs** | `ru: pd.Series` |
| **Formula** | \(Q_t=\min_{u\le t}E_u;\;RU_t=(E_t-Q_t)/Q_t\) |
| **Edge-cases** | trough \(Q_t\le 0\) → NaN + warning |

---

## 2) Time-series performance metrics

### P1 — `perf.sharpe` (estimated, rf optional)
| Field | Spec |
|---|---|
| **Requires** | `returns` (Series). Optional `rf` |
| **Outputs** | scalar Sharpe; optionally `series["excess_returns"]` |
| **Formula** | \(x_t=r_t-r_{f,t}\);\; \(\text{Sharpe}=\bar x/s_x\). Annualized: \(\sqrt{A}\bar x/s_x\) if metric is `perf.sharpe_ann` |
| **Edge-cases** | std=0 → NaN + warning `"sharpe: zero variance"`; insufficient obs (<2) → NaN + warning |
| **Notes** | use `ddof=1` for std |

### P2 — `perf.sortino` (rf or threshold)
| Field | Spec |
|---|---|
| **Requires** | `returns`. Optional `rf` or `config.sortino_threshold` |
| **Outputs** | scalar Sortino |
| **Formula** | threshold \(\tau = config.sortino_threshold\) else \(\tau=\bar r_f\) if `rf` exists else 0. Downside: \(\sigma_{down}=\sqrt{\frac{1}{T}\sum \min(0,r_t-\tau)^2}\). Sortino \(=\mathbb{E}[r_t-\tau]/\sigma_{down}\). Annualized multiply by \(\sqrt{A}\) |
| **Edge-cases** | no downside (σ_down=0) → NaN + warning `"sortino: zero downside deviation"`; insufficient obs → NaN |

### P3 — `perf.mdd` (max drawdown)
| Field | Spec |
|---|---|
| **Requires** | `returns` OR `equity` (if provided). Prefer equity if present. |
| **Outputs** | scalar MDD; optionally `series["drawdown"]` |
| **Formula** | \(E_t\) from returns; \(DD_t=(P_t-E_t)/P_t\); MDD \(=\max_t DD_t\) |
| **Edge-cases** | equity invalid (≤0) or returns ≤ -1 → NaN + warning `"mdd: invalid equity"` |

### P4 — `perf.max_runup`
| Field | Spec |
|---|---|
| **Requires** | `returns` OR `equity` |
| **Outputs** | scalar MaxRU; optionally `series["runup"]` |
| **Formula** | \(RU_t=(E_t-Q_t)/Q_t\); MaxRU \(=\max_t RU_t\) |
| **Edge-cases** | trough \(Q_t\le0\) → NaN + warning `"runup: invalid trough"` |

### P5 — `perf.cagr`
| Field | Spec |
|---|---|
| **Requires** | `returns` OR `equity`, plus `config.periods_per_year` |
| **Outputs** | scalar CAGR |
| **Formula** | let \(T\) periods; years \(Y=T/A\). \( \text{CAGR}=(E_T/E_0)^{1/Y}-1 \) |
| **Edge-cases** | \(E_T\le0\) or \(Y\le0\) → NaN + warning `"cagr: invalid horizon"` |

### P6 — `perf.cagr_mdd` (Calmar-style)
| Field | Spec |
|---|---|
| **Requires** | `perf.cagr` and `perf.mdd` (computed or recompute internally) |
| **Outputs** | scalar CAGR/MDD |
| **Formula** | \(\text{CAGR}/\text{MDD}\) |
| **Edge-cases** | MDD=0 → NaN (or inf if you prefer) + warning `"cagr_mdd: zero drawdown"` |

---

## 3) Trade-based performance metrics (optional v1 if you have trade PnL)

> If you don’t have trade-level PnL, keep these as v1.1 or accept an input column `trade_pnl` list/series.

### T1 — `trade.profit_factor`
| Field | Spec |
|---|---|
| **Requires** | `trade_pnl: pd.Series` (one value per trade) |
| **Outputs** | scalar PF |
| **Formula** | \(GP=\sum_{p_i>0}p_i\); \(GL=\sum_{p_i<0}|p_i|\); \(PF=GP/GL\) |
| **Edge-cases** | no losing trades (GL=0): return `inf` if `pf_no_loss_policy="inf"` else NaN; warning `"pf: no losses"`; no trades → NaN + warning |

### T2 — `trade.win_rate`
| Field | Spec |
|---|---|
| **Requires** | `trade_pnl` |
| **Outputs** | scalar WinRate (%) |
| **Formula** | \(100\cdot \#\{p_i>0\}/N\) |
| **Edge-cases** | N=0 → NaN + warning `"win_rate: no trades"` |

### T3 — `trade.expectancy`
| Field | Spec |
|---|---|
| **Requires** | `trade_pnl` |
| **Outputs** | scalar Expectancy |
| **Formula** | \(WR=N_w/N\), \(LR=1-WR\), \(AvgWin=\frac{1}{N_w}\sum_{p_i>0}p_i\), \(AvgLoss=\frac{1}{_



