---
description: Git workflow conventions for this project
disable-model-invocation: true
---

# Git Conventions

## Branch Naming
- `feature/<short-description>` for new features
- `fix/<short-description>` for bug fixes
- `refactor/<short-description>` for refactoring

## Commit Messages
- Use imperative mood: "Add upload validation" not "Added upload validation"
- Keep first line under 72 characters
- Reference the module affected: "filterStore: fix reset behavior"

## Safety Rules
- Never force push to `main`
- Never use `git reset --hard` without explicit user confirmation
- Never skip pre-commit hooks (`--no-verify`)
- Always review `git diff` before committing
- Never add `Co-Authored-By` lines for Claude or any AI assistant in commit messages

## Files to Never Commit
- `.env` or credentials files
- `settings.local.json`
- Large data files (>10 MB) — the CSV is already tracked but avoid adding more
- `node_modules/` directory
- `frontend/dist/` build output
