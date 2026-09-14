#!/bin/bash
set -euo pipefail

# =============================================================================
# postCreateCommand — runs once when the container is first built
# 1. Install system packages (tmux)
# 2. Install Python dependencies + pre-commit hooks
# 3. Install Node dependencies
# 4. Install Playwright's Chromium + its system libraries
# =============================================================================

log() { echo "[post-create] $*"; }

cd /workspaces/zScore-App

# ---------------------------------------------------------------------------
# 1. Install system packages
# ---------------------------------------------------------------------------
if ! command -v tmux &>/dev/null; then
  log "Installing tmux..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq tmux > /dev/null
  log "tmux installed."
else
  log "tmux already installed — skipping."
fi

# ---------------------------------------------------------------------------
# 2. Install Python dependencies + pre-commit hooks
#
# Python is only needed for scripts/version_dataset.py and the pre-commit
# hooks — the frontend itself does not use it. Some base images ship without
# pip, so warn and carry on rather than aborting the whole setup under `set -e`.
# ---------------------------------------------------------------------------
if command -v pip &>/dev/null; then
  PIP="pip"
elif python3 -m pip --version &>/dev/null; then
  PIP="python3 -m pip"
else
  PIP=""
fi

if [ -n "$PIP" ]; then
  log "Installing Python dependencies..."
  $PIP install -q -r requirements.txt

  if [ -f .pre-commit-config.yaml ]; then
    log "Installing pre-commit hooks..."
    pre-commit install
  fi
else
  log "WARNING: pip not found — skipping Python dependencies and pre-commit hooks."
  log "WARNING: scripts/version_dataset.py and 'pre-commit run' will not work."
fi

# ---------------------------------------------------------------------------
# 3. Install Node dependencies
# ---------------------------------------------------------------------------
log "Installing Node dependencies..."
cd frontend
npm install
cd ..

# ---------------------------------------------------------------------------
# 4. Install Playwright's Chromium + its system libraries
#
# The npm package is a devDependency, but the browser binary and the ~20 system
# libraries it links against (libnss3, libglib-2.0, libX11, ...) are not covered
# by npm install. Without them `chromium.launch()` fails at startup.
# ---------------------------------------------------------------------------
if ! dpkg -s libnss3 &>/dev/null; then
  log "Installing Playwright system libraries..."
  sudo env "PATH=$PATH" npx --prefix frontend playwright install-deps chromium
else
  log "Playwright system libraries already installed — skipping."
fi

log "Installing Playwright Chromium (no-op if already cached)..."
npx --prefix frontend playwright install chromium

log "Post-create setup complete."
