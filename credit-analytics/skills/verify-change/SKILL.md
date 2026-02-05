# Skill: Verify Change

This skill outlines the process for verifying any change to ensure it is consistent with the repository's standards and specifications. This is the final step before merging a pull request.

## Steps
1.  **Run All Tests**: Execute `pytest` from the root directory to run all tests across all packages.
2.  **Check Static Analysis**: Run any linting or static analysis tools configured for the project.
3.  **Verify Specification Parity**: Run the verification script to ensure your code has not diverged from the formal specification and golden fixtures.
    ```bash
    python tools/verify_spec_parity.py
    ```
4.  **Review CI Results**: Check that the full CI pipeline passes on your branch or pull request. The CI pipeline automates the steps above.
