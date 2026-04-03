---
description: Run linting and type checking on the codebase
---

# Lint and Type Check

1. Run ESLint:
   ```bash
   cd frontend && npx eslint .
   ```

2. Run TypeScript type checking:
   ```bash
   cd frontend && npx tsc --noEmit
   ```

3. Report all findings to the user, grouped by severity.

4. Offer to auto-fix safe ESLint issues:
   ```bash
   cd frontend && npx eslint . --fix
   ```
