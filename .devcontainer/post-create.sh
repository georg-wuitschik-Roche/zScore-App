#!/bin/bash
set -euo pipefail

# =============================================================================
# postCreateCommand — runs once when the container is first built
# 1. Install system packages (tmux)
# 2. Install Python dependencies + pre-commit hooks
# 3. Install Node dependencies
# 4. Generate Parquet data file from CSV
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
pip install -q -r requirements.txt -r paper/requirements.txt

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

# ---------------------------------------------------------------------------
# 4. Generate Parquet data file from CSV (if missing)
# ---------------------------------------------------------------------------
PARQUET="frontend/public/data/z-score-peaks.parquet"
CSV="z-Score Peaks with FG.csv"

if [[ ! -f "$PARQUET" ]] && [[ -f "$CSV" ]]; then
  log "Generating parquet from CSV..."
  python3 -c "
import pandas as pd
df = pd.read_csv('$CSV', encoding='utf-8')
USED = ['ELN_ID','PLATENUMBER','Coordinate','AREA_TOTAL_REDUCED',
        'Base','Catalyst','Solvent','Ligand','Additive',
        'Coupling Reagent','Secondary Solvent','Tertiary Solvent',
        'Reaction Type','FG A','FG B','FG_sorted','z-Score','output_column']
df[[c for c in USED if c in df.columns]].to_parquet(
    '$PARQUET', compression='zstd', index=False)
"
  log "Parquet generated."
elif [[ -f "$PARQUET" ]]; then
  log "Parquet already exists — skipping."
else
  log "WARNING: CSV not found, cannot generate parquet."
fi

log "Post-create setup complete."
