---
description: Git workflow conventions for this project
user_invocable: false
---

# Git Conventions

## Branch Naming
- `feature/<short-description>` for new features
- `fix/<short-description>` for bug fixes
- `refactor/<short-description>` for refactoring

## Commit Messages
- Use imperative mood: "Add upload validation" not "Added upload validation"
- Keep first line under 72 characters
- Reference the module affected: "callbacks: fix presentation mode toggle"

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
- `__pycache__/` directories
- `exports/` output files (generated, not source)
