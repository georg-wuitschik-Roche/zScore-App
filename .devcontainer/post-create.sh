#!/bin/bash
set -euo pipefail

# =============================================================================
# postCreateCommand — runs once when the container is first built
# 1. Install system packages (tmux)
# 2. Install Python dependencies + pre-commit hooks
# 3. Install Node dependencies
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
# ---------------------------------------------------------------------------
log "Installing Python dependencies..."
pip install -q -r requirements.txt

if [ -f .pre-commit-config.yaml ]; then
  log "Installing pre-commit hooks..."
  pre-commit install
fi

# ---------------------------------------------------------------------------
# 3. Install Node dependencies
# ---------------------------------------------------------------------------
log "Installing Node dependencies..."
cd frontend
npm install
cd ..

log "Post-create setup complete."
