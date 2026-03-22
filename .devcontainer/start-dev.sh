#!/bin/bash
set -euo pipefail

# =============================================================================
# postStartCommand — runs every time the container starts
# 1. Ensure Node dependencies exist (safety net for failed postCreate)
# 2. Start Vite dev server with auto-restart on crash
# =============================================================================

log() { echo "[start-dev] $*"; }

cd /workspaces/zScore-App/frontend

# ---------------------------------------------------------------------------
# 1. Ensure Node dependencies exist
# ---------------------------------------------------------------------------
if [[ ! -d node_modules ]]; then
  log "node_modules missing — running npm install..."
  for attempt in 1 2 3; do
    npm install && break
    log "npm install failed (attempt $attempt/3), retrying in 5s..."
    sleep 5
  done
  if [[ ! -d node_modules ]]; then
    log "ERROR: npm install failed after 3 attempts"
    exit 1
  fi
else
  log "node_modules present."
fi

# ---------------------------------------------------------------------------
# 2. Start Vite dev server (auto-restart on crash, max 5 times)
# ---------------------------------------------------------------------------
MAX_RESTARTS=5
restarts=0

while (( restarts < MAX_RESTARTS )); do
  log "Starting Vite dev server..."
  npx vite && break  # clean exit (Ctrl-C) → stop loop
  restarts=$((restarts + 1))
  log "Vite exited unexpectedly (restart $restarts/$MAX_RESTARTS), restarting in 2s..."
  sleep 2
done

if (( restarts >= MAX_RESTARTS )); then
  log "ERROR: Vite crashed $MAX_RESTARTS times, giving up."
  exit 1
fi
