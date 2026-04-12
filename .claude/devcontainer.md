# Dev Container Setup

## Architecture

Two lifecycle scripts in `.devcontainer/`:

| Script | Lifecycle | Runs when |
|--------|-----------|-----------|
| `post-create.sh` | `postCreateCommand` | Once, when the container is first built |
| `start-dev.sh` | `postStartCommand` | Every time the container starts |

**`post-create.sh`** handles one-time setup: system packages (tmux), Python deps + pre-commit, and Node deps.

**`start-dev.sh`** starts the Vite dev server with safety nets: verifies `node_modules` exists (retries npm install 3x if missing), then runs Vite with auto-restart on crash (up to 5 times). Logs to `/tmp/dev-server.log`.

## Design Principles

- **Single script per lifecycle** — no `&&` chains or object splits in `devcontainer.json`. Each script is self-contained and readable.
- **Idempotent guards** — `command -v` before installing packages, `[[ -d node_modules ]]` before npm install, `[[ -f parquet ]]` before regenerating data. Safe to re-run.
- **`log()` helper** — all output prefixed with `[post-create]` or `[start-dev]` for easy debugging.
- **`start-dev.sh` is a safety net** — it re-checks deps in case `post-create.sh` failed. The dev server must always come up.
- **`nohup` + background** — `postStartCommand` runs the start script via `nohup ... &` so it doesn't block the terminal and survives shell exits. Output goes to `/tmp/dev-server.log`.

## Troubleshooting

### Dev server not running
```bash
cat /tmp/dev-server.log          # check start-dev.sh output
pgrep -af vite                   # is Vite alive?
ls frontend/node_modules/.vite   # does the Vite cache exist?
```

If `node_modules` is missing, `start-dev.sh` will auto-install. If it keeps failing, check network connectivity and run `cd frontend && npm install` manually.

### Parquet data file missing
Parquet files are committed to the repo via the `add-dataset/` workflow. To add a new version, drop a CSV into `add-dataset/` and commit — the pre-commit hook converts it to Parquet automatically.

## History

### "Can't connect to app after container rebuild" (2026-03-22)

**Symptom:** Dev server starts but app is unreachable in browser.

**Root cause:** `postCreateCommand` used `&&` chaining, so if `pip install` failed, `npm install` never ran. Vite then failed to start because node_modules was missing.

**Fix applied:** Replaced fragile `&&` chains and JSON object splits with dedicated shell scripts (`post-create.sh`, `start-dev.sh`) that have idempotent guards, retries, and auto-restart.

**Not the issue:** Vite binding to localhost is fine — VS Code port forwarding handles localhost-bound ports. No need to add `server: { host: '0.0.0.0' }` to vite.config.ts.
