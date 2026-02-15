---
name: extract-algorithm
description: "Extract the core algorithm from a research paper and produce an implementation-ready technical specification. Triggers: user shares a paper (PDF, URL, arXiv link, or pasted text) and says 'extract the algorithm', 'how do I implement this', 'what's the method', 'give me the algorithm', 'implementation spec', 'how does this work technically', 'break down the method', 'pseudocode for this', or 'I want to code this up'. Also triggers when user says 'what exactly are they doing' about a paper, or asks for a 'technical breakdown'. Do NOT trigger for general summaries, literature reviews, or positioning analysis — those belong to other skills."
---

# Algorithm Extractor

## Purpose

Read a research paper and produce a complete, implementation-ready technical specification of the algorithm. The output should be precise enough that a senior quant or ML engineer could implement it without reading the original paper. This means: exact mathematical definitions, unambiguous pseudocode, explicit data structures, computational complexity, numerical pitfalls, and the specific choices the paper makes (and the ones it leaves to the implementer).

## Perspective

Write as an expert who lives at the intersection of statistical learning theory, financial econometrics, and production trading systems. Assume the reader:
- Is fluent in measure theory, stochastic processes, optimisation, and asymptotic statistics
- Knows standard ML architectures and training procedures
- Understands market microstructure (LOB dynamics, tick data, latency, transaction costs)
- Wants to implement this in Python/NumPy/PyTorch (or R where relevant)
- Cares about numerical stability, computational cost, and edge cases in real financial data

Do NOT simplify the mathematics. Do NOT hand-wave over implementation details. The value of this skill is precision.
Your output should be a clear .md, and a .tex file that has clear color coded explanation of the algorithm

## Workflow

### Step 1: Obtain and read the paper

If user provides:
- **arXiv link** → fetch the paper
- **PDF upload** → read from `/mnt/user-data/uploads/`
- **SSRN / journal URL** → fetch and extract
- **Pasted text / abstract** → work with what's provided, flag if critical sections are missing

Read the full paper. Identify:
- The **core algorithm** (there may be several; extract the main contribution first)
- The **mathematical framework** (probability space, data-generating process, loss function, estimator)
- The **computational procedure** (what you actually compute, in what order)
- The **theoretical guarantees** (what the paper proves about the algorithm)
- The **experimental setup** (how they tested it — this reveals implementation choices the theory section omits)

### Step 2: Produce the Algorithm Specification

Generate a structured document with the following sections. Every section is mandatory — if the paper doesn't specify something, flag it as an **implementation decision** the user must make.

---

#### 2.1 — Problem Setup

State the mathematical problem the algorithm solves. Be precise about:

**Input space.** What is X? (e.g., "X ∈ ℝ^{d×T}, a matrix of d features observed at T timestamps, where features include...")

**Output space.** What is the algorithm producing? (e.g., "A point estimate ĥ(q) ∈ ℝ for each moment order q ∈ Q", or "A policy π: S → A mapping states to actions", or "A test statistic T_n and critical value c_α")

**Objective.** What is being optimised or estimated? Write the explicit objective function, loss, or estimand:

$$\hat{\theta} = \arg\min_{\theta \in \Theta} \frac{1}{n} \sum_{i=1}^{n} \ell(y_i, f_\theta(x_i)) + \lambda \cdot R(\theta)$$

If the paper uses multiple objectives (e.g., a two-stage procedure), state each one.

**Data assumptions.** What does the algorithm require of the data?
- Stationarity? Mixing conditions? (specify: α-mixing, β-mixing, with rate?)
- Moment conditions? (E[|X|^p] < ∞ for which p?)
- Sampling frequency? (tick-by-tick, fixed intervals, irregular?)
- Missing data handling? (the paper may assume none; real LOB data always has gaps)

---

#### 2.2 — Algorithm: Step-by-Step

This is the core section. Write the algorithm as numbered steps with full mathematical precision. Each step must specify:

1. **What** is computed (mathematical expression)
2. **How** it is computed (computational method — closed form, iterative, approximate?)
3. **Data structures** involved (matrix, tensor, rolling window, tree, hash map?)
4. **Complexity** of this step (time and space)

Format:

```
ALGORITHM: [Name]

Input:  [precise specification of all inputs with types and dimensions]
Output: [precise specification of all outputs]
Hyperparameters: [list every tunable parameter with the paper's default/recommended value]

1. [STEP NAME]
   Compute: [mathematical expression]
   Method:  [how — e.g., "solve via Newton-Raphson with Hessian H_n(θ)", 
             or "compute via FFT of length N padded to next power of 2"]
   Cost:    O(...)
   Notes:   [numerical considerations, e.g., "log-sum-exp trick for stability"]

2. [STEP NAME]
   ...

Return: [what is returned and in what form]
```

**Critical rules:**
- If the paper says "solve the optimisation problem" without specifying the solver, flag this: "IMPLEMENTATION DECISION: Paper does not specify solver. Options: [L-BFGS for smooth problems / Adam for neural net parameters / CVXPY for convex programs]. Recommended: [your recommendation based on problem structure]."
- If the paper uses a subroutine from another paper (e.g., "we use the DML procedure of Chernozhukov et al. 2018"), expand it. Don't just cite — write out the cross-fitting steps, the Neyman orthogonal score, the variance estimator. The reader should not need to read the other paper.
- If the paper has a training loop (neural network, RL agent), specify: initialisation, batch construction, gradient computation, update rule, convergence criterion, and early stopping rule.

---

#### 2.3 — Hyperparameters and Tuning

For EVERY hyperparameter, state:

| Parameter | Symbol | Paper's Value | Sensitivity | Tuning Guidance |
|-----------|--------|--------------|-------------|-----------------|
| [name] | [symbol] | [what they used] | [how sensitive is performance?] | [how to tune — CV, theory-driven, grid search?] |

**Critical:** Many papers bury hyperparameter choices in the appendix or experimental section. Extract ALL of them. Common hidden hyperparameters in financial ML:
- Lookback window length
- Number of cross-fitting folds (for DML-type procedures)
- Kernel bandwidth (for nonparametric methods)
- Regularisation strength
- Number of bootstrap replicates
- Threshold for truncation/winsorisation
- Learning rate schedule
- Batch size and number of epochs
- Random seed (if results are seed-sensitive, flag this)

---

#### 2.4 — Data Pipeline Specification

Describe the exact data transformations from raw input to algorithm input:

```
RAW DATA
  │
  ├─ [Step 1: Cleaning]
  │   What: [e.g., remove cancelled orders, handle auction periods]
  │   Filter: [specific conditions]
  │
  ├─ [Step 2: Feature Construction]
  │   Features: [list every feature with exact formula]
  │   f_1 = log(ask_1) - log(bid_1)           # log spread
  │   f_2 = (V_ask - V_bid) / (V_ask + V_bid)  # order imbalance
  │   ...
  │
  ├─ [Step 3: Normalisation / Transformation]
  │   Method: [e.g., rolling z-score with 500-tick window, rank transform]
  │   Pitfall: [e.g., lookahead bias if using full-sample statistics]
  │
  ├─ [Step 4: Train/Val/Test Split]
  │   Method: [temporal split — never shuffle financial time series]
  │   Sizes: [what the paper uses]
  │   Gap: [embargo period between train and test to prevent leakage]
  │
  └─ ALGORITHM INPUT: X ∈ ℝ^{n×d}, y ∈ ℝ^n (or whatever the format is)
```

**Financial data pitfalls to flag:**
- Lookahead bias in feature construction (using future information)
- Survivorship bias in asset selection
- Time-of-day effects and market microstructure noise at open/close
- Corporate actions (splits, dividends) affecting price series
- Asynchronous observation across assets
- Bid-ask bounce contaminating returns at high frequency

---

#### 2.5 — Theoretical Guarantees

State every formal result (proposition, theorem, corollary) that the paper proves about the algorithm. For each:

**Result:** [formal statement, using the paper's notation]

**Assumptions required:** [numbered list — be explicit about which assumptions are standard and which are restrictive]

**Rate / bound:** [the quantitative result — convergence rate, regret bound, coverage level]

**Practical implication:** [what this means for implementation — e.g., "this tells you the minimum sample size needed for the estimator to be reliable is roughly n > d²/ε² where ε is your tolerance"]

**When it breaks:** [conditions under which the guarantee fails — e.g., "the bound degrades to O(1) if the mixing coefficient decays slower than polynomial"]

---

#### 2.6 — Numerical and Implementation Considerations

This section covers everything the paper doesn't tell you but that you'll discover when implementing:

**Numerical stability:**
- Where can overflow/underflow occur? (e.g., computing likelihoods in log-space)
- Where is cancellation error a risk? (e.g., subtracting nearly equal quantities)
- What precision is needed? (float32 sufficient or need float64?)

**Computational bottlenecks:**
- What's the dominant cost? (e.g., "the kernel matrix is O(n²) — for n > 50k you need Nyström or random features")
- Can anything be parallelised? (e.g., cross-fitting folds are embarrassingly parallel)
- What can be precomputed and cached?

**Edge cases in financial data:**
- Zero-volume periods (no trades for extended time)
- Extreme returns (flash crashes — does the algorithm handle fat tails?)
- Regime changes (does the algorithm need to be re-estimated periodically?)
- Market closures (weekends, holidays — how to handle the gap?)

**Memory requirements:**
- For a typical dataset size (state what "typical" means — e.g., "1 year of TAQ data at tick level ≈ 500M rows"), estimate peak memory usage.

---

#### 2.7 — Implementation Skeleton

Produce a Python implementation skeleton. NOT full production code — a structural skeleton with:
- Class/function signatures with type hints
- Docstrings stating what each function does mathematically
- The computational flow
- Placeholder comments for the core computations
- References to which algorithm step (§2.2) each function implements

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class AlgorithmConfig:
    """Hyperparameters (see §2.3)."""
    window_length: int = 500       # Paper's default; sensitive (see §2.3)
    n_folds: int = 5               # Cross-fitting folds
    bandwidth: float = 1.0         # Kernel bandwidth; tune via median heuristic
    alpha: float = 0.05            # Significance level

class AlgorithmName:
    """
    Implements [Paper Author (Year)] [Algorithm Name].
    
    Solves: [one-line problem statement from §2.1]
    Guarantees: [key result from §2.5, e.g., "consistent at rate O(n^{-1/3}) 
                 under β-mixing"]
    
    References ARTIFACTS.md: [which existing modules this uses, e.g., C-01, D-01]
    """
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AlgorithmName':
        """
        Step 1–3 of Algorithm (§2.2).
        
        Args:
            X: Feature matrix, shape (n_samples, n_features). 
               Assumes features are pre-processed per §2.4.
            y: Target vector, shape (n_samples,).
        
        Returns:
            self (fitted estimator)
        """
        # Step 1: [description from §2.2]
        # TODO: implement [mathematical expression]
        
        # Step 2: [description]  
        # TODO: implement — NOTE: [numerical pitfall from §2.6]
        
        # Step 3: [description]
        # TODO: implement
        
        return self
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Apply fitted model to new data.
        
        Args:
            X_new: shape (n_new, n_features)
        
        Returns:
            Predictions, shape (n_new,)
        """
        # TODO: implement
        pass
    
    def test(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        If the algorithm includes a hypothesis test (§2.5),
        return test statistic, p-value, and critical value.
        """
        # TODO: implement
        pass
```

**Cross-reference with ARTIFACTS.md:** Note which existing code modules (C-01 through C-09) can be used as dependencies, and which new module this implementation would create.

---

#### 2.8 — Reproduction Checklist

A checklist for verifying your implementation matches the paper:

```
□ Synthetic data test: Generate data from the paper's DGP (§2.4) 
  and verify the algorithm recovers known parameters
□ Table reproduction: Replicate the paper's Table [X] 
  (the key results table) within [tolerance]
□ Figure reproduction: Replicate Figure [X] 
  (usually the convergence plot or performance comparison)
□ Edge case test: Run on [specific edge case from §2.6] 
  and verify behaviour matches paper's claims
□ Complexity verification: Time the algorithm on increasing n 
  and verify empirical complexity matches §2.2 claims
```

---

#### 2.9 — Adaptation Notes for Your Programme

Map the extracted algorithm to the user's research tracks:

- **Which track(s) could use this?** (A through E)
- **How does it relate to existing artifacts?** (Does it replace C-xx? Extend T-xx? Require D-xx?)
- **What modifications are needed for your specific setting?** (e.g., "the paper assumes i.i.d. data; for your LOB application under Track A, you'd need the dependent-data extension, which requires modifying Step 3 to use a block bootstrap instead of standard bootstrap")
- **Does this affect any formal claim in your quasi-theory templates?** (e.g., "if you adopt their kernel, your power result in T-02 may need to account for their bandwidth selection procedure")

---

### Step 3: Produce output files

Generate two files:

1. **`algo_[short_name]_spec.md`** — The full specification document (all sections above)
2. **`algo_[short_name]_skeleton.py`** — The implementation skeleton from §2.7

Place in the relevant track directory or in a shared `algorithms/` directory if it spans tracks.

### Step 4: Flag implementation decisions

At the end, produce a numbered list of every **IMPLEMENTATION DECISION** the user must make — choices the paper left unspecified or where multiple valid approaches exist. For each, state the options and a recommendation. Example:

```
IMPLEMENTATION DECISIONS (requires your input):

1. SOLVER for Step 3 optimisation
   Options: L-BFGS (fast, needs gradient), Adam (robust, slower), CVXPY (exact, only if convex)
   Recommendation: L-BFGS — the objective is smooth and d < 100 in your setting
   
2. BANDWIDTH selection for kernel in Step 5
   Options: Median heuristic (fast, parameter-free), cross-validation (data-adaptive, expensive), 
            paper's fixed value of h=0.5 (reproducible but may not transfer to your data)
   Recommendation: Median heuristic for initial implementation; CV for final results

3. BOOTSTRAP variant for variance estimation in Step 7
   Options: Standard bootstrap (assumes i.i.d.), block bootstrap (handles dependence), 
            subsampling (handles long memory)
   Recommendation: Block bootstrap — your LOB data has short-range dependence. 
                   Block length via Politis-White (2004) automatic selection.
```

## Quality Standards

- **Every mathematical symbol must be defined.** If the paper uses θ without defining it, define it yourself from context.
- **Every matrix dimension must be stated.** "Compute X^T X" is not sufficient — state "X ∈ ℝ^{n×d}, so X^T X ∈ ℝ^{d×d}."
- **Every "standard" procedure must be specified.** If the paper says "we standardise features," state the exact transformation (z-score? rank? min-max?) and whether it's computed on the training set only.
- **Computational complexity for every step.** Not just big-O — also the constants that matter in practice (e.g., "O(n²d) but the constant is small because the inner loop is vectorisable").
- **No hand-waving.** If you can't determine how a step is implemented from the paper, say so explicitly and list the plausible options.

## Anti-patterns

1. **Summarising instead of specifying.** "They use a neural network" is a summary. "They use a 3-layer MLP with hidden dimensions [128, 64, 32], ReLU activations, BatchNorm after each layer, dropout p=0.1, trained with Adam (lr=1e-3, β₁=0.9, β₂=0.999) for 200 epochs with early stopping on validation loss (patience=20)" is a specification.

2. **Citing instead of expanding.** "They use DML (Chernozhukov et al. 2018)" is a citation. Write out the K-fold cross-fitting, the Neyman orthogonal score, the debiased estimator, and the variance formula.

3. **Ignoring the experiments section.** The experiments section of a paper reveals implementation choices that the method section omits — batch sizes, learning rate schedules, data splits, preprocessing, and which hyperparameters were actually tuned.

4. **Assuming standard means obvious.** In financial ML, "standard" preprocessing can mean five different things. Specify exactly which one.

5. **Skipping edge cases.** The algorithm may work beautifully on the paper's data but fail on yours because of market microstructure noise, zero-volume periods, or overnight gaps. Flag every place where financial data is non-standard.
