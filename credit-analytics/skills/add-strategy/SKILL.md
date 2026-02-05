# Skill: Add Strategy

This skill guides you through the process of creating a new backtesting strategy.

## Steps
1.  **Define Strategy Logic**: Create a new file in `packages/backtest/src/strategies/` (you may need to create the `strategies` subdirectory).
2.  **Assemble Components**: Your strategy should import and use components from the other packages:
    - `packages/data`: To get instrument data.
    - `packages/signals`: To generate entry/exit signals.
    - `packages/metrics`: To calculate performance during or after the backtest.
3.  **Configure the Strategy**: Define the strategy's parameters using the TOML/Pydantic configuration pattern.
4.  **Create a Backtest Runner**: Add a script in `tools/` or a notebook in `tools/notebooks/` to execute your strategy using the backtesting engine.
5.  **Add Tests**: Add unit tests for your strategy's specific logic in `packages/backtest/tests/`.
