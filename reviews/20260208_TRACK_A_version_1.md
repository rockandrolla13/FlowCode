# Conversation Summary: Track A

**Date**: 2026-02-08
**Version**: 1
**Previous Version**: N/A

## Key Topics Discussed

- Full codebase audit: 4-package monorepo (data, signals, metrics, backtest) reviewed for spec parity, test coverage, and code quality.
- Systematic code review: 4 parallel review agents each focused on one package, findings consolidated into a unified priority list.

## Technical Contributions

### Concepts Developed

- **Repo State Assessment**: FlowCode is in post-build/pre-polish state. All 4 packages are implemented with 19 formulas from SPEC.md, but spec-code-test parity chain is broken at the test layer.
- **Cross-Package Issue Taxonomy**: Issues categorized as Critical (spec formula mismatches, missing tests), Major (sign conventions, boundary violations), Minor (naming, magic numbers).

### Methods / Approaches

- **Parallel Agent Review**: 4 `superpowers:code-reviewer` agents ran simultaneously, one per package, each checking against CLAUDE.md standards, SPEC.md formulas, and fixture files.
- **Spec Parity Analysis**: Compared every formula in SPEC.md against its implementation and test coverage. Found systematic gaps.

## Code / Implementation Notes

### Critical Formula Mismatches Found

```
signals/retail.py:  qmp_classify() → 2-state (buy/sell), spec says 3-state (buy/sell/neutral)
signals/triggers.py: zscore_trigger() → bool, spec says ternary (1/-1/0)
signals/triggers.py: streak_trigger() → bool, spec says ternary (1/-1/0)
metrics/risk.py:    value_at_risk() → negative value, spec says positive loss magnitude
metrics/risk.py:    expected_shortfall() → compounds VaR sign error
backtest/engine.py: compute_metrics() → reimplements Sharpe/drawdown inline instead of importing metrics pkg
```

### Missing Fixture Files (Referenced in CLAUDE.md but don't exist)
```
pnl_cases.json, range_position_cases.json, retail_id_cases.json,
drawdown_cases.json, sortino_cases.json, calmar_cases.json,
var_cases.json, es_cases.json
```

### Test Coverage Gaps
```
data/: trace.py, reference.py, universe.py → 0 tests
metrics/: diagnostics.py → 7 functions, 0 tests
backtest/: portfolio.py → risk_parity, top_n_positions untested
All packages: Tests don't load spec/fixtures/*.json at all
```

## Open Questions

- Should data loaders return empty DataFrame on missing file (per CLAUDE.md) or raise FileNotFoundError (current behavior)?
- Should DataFrame index convention (DatetimeIndex named 'date') be enforced in loaders or deferred to callers?
- Should the `src/` package directory naming be changed to avoid namespace collisions when multiple packages are installed together?
- Sortino downside deviation: should denominator use count of negative returns only, or total return count?

## Next Steps

- [ ] P0: Fix qmp_classify() to 3-state classifier matching spec §1.5
- [ ] P0: Fix VaR/ES sign convention in metrics/risk.py to match spec §4.2/§4.3
- [ ] P0: Wire all test files to load golden cases from spec/fixtures/*.json
- [ ] P1: Create 8 missing fixture files referenced in CLAUDE.md Formula Registry
- [ ] P1: Add tests for untested modules (trace, reference, universe, diagnostics, portfolio)
- [ ] P1: Replace inline metrics in backtest/engine.py with imports from metrics package
- [ ] P1: Make trigger functions return ternary signals per spec §2.1/§2.2
- [ ] P2: Resolve DataFrame index convention decision
- [ ] P2: Resolve missing-file behavior decision
- [ ] P2: Remove unused polars dependency from data/pyproject.toml
- [ ] P3: Rename src/ directories to proper package names

## References / Resources Mentioned

- CLAUDE.md: Project standards, ownership model, comprehension gates, package boundaries
- spec/SPEC.md: 19 formulas across 7 sections — single source of truth
- docs/IMPLEMENTATION_PLAN.md: Original 6-PR implementation strategy

## Cross-Track Connections

- **Track C**: Backtest engine's lookahead prevention is critical infrastructure for any offline RL / decision-focused learning built on top.
- **Track D**: Signal triggers (zscore, streak) feed into multiscale memory representations — ternary vs boolean return type matters for downstream consumers.

## Agent Memory Tags

#track-a #code-review #spec-parity #test-coverage #audit #microstructure
