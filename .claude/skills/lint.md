---
description: Run linting and type checking on the codebase
user_invocable: true
---

# Lint and Type Check

1. Check if linting tools are available, install if needed:
   ```bash
   pip install ruff mypy
   ```

2. Run ruff for linting:
   ```bash
   ruff check .
   ```

3. Run mypy for type checking:
   ```bash
   mypy --ignore-missing-imports *.py
   ```

4. Report all findings to the user, grouped by severity.

5. Offer to auto-fix safe issues:
   ```bash
   ruff check --fix .
   ```
