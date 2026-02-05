# Skill: Extend Signal

This skill helps you add a new signal or modify an existing one within the `packages/signals` directory.

## Steps
1.  **Identify the Core Logic**: Determine if the new signal is a novel calculation or an extension of a function from `packages/core`.
2.  **Create/Modify Signal File**: Add or edit a file in `packages/signals/src/`. The new file should wrap the `core` function and add the desired business logic (e.g., QMP rules, parameterization).
3.  **Add Tests**: Create a corresponding test file in `packages/signals/tests/` to verify the new signal's behavior. Use fixtures from `spec/fixtures/` where applicable.
4.  **Update Specification**: If this change modifies the formal specification, update `spec/SPEC.md`.
5.  **Run Verification**: Run `tools/verify_spec_parity.py` to ensure your changes are aligned with the specification.
