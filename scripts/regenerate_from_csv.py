#!/usr/bin/env python3
"""Regenerate all derived artifacts from the source CSV.

Triggered automatically by the pre-commit hook when
``z-Score Peaks with FG.csv`` is staged, or run manually:

    python3 scripts/regenerate_from_csv.py

Artifacts produced:
  1. frontend/public/data/z-score-peaks.parquet   (Parquet dataset)
  2. frontend/public/data/dropdown-index.json      (runtime dropdown data)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "z-Score Peaks with FG.csv"
PARQUET_PATH = ROOT / "frontend" / "public" / "data" / "z-score-peaks.parquet"
DROPDOWN_INDEX_PATH = ROOT / "frontend" / "public" / "data" / "dropdown-index.json"
FRONTEND_GOLDEN_DIR = ROOT / "frontend" / "golden"

USED_COLUMNS = [
    "ELN_ID",
    "PLATENUMBER",
    "Coordinate",
    "AREA_TOTAL_REDUCED",
    "Base",
    "Catalyst",
    "Solvent",
    "Ligand",
    "Additive",
    "Coupling Reagent",
    "Secondary Solvent",
    "Tertiary Solvent",
    "Reaction Type",
    "FG A",
    "FG B",
    "FG_sorted",
    "z-Score",
    "output_column",
]


def generate_parquet() -> int:
    """Read the CSV and write a slim, ZSTD-compressed Parquet file."""
    import pandas as pd

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    cols = [c for c in USED_COLUMNS if c in df.columns]
    df_slim = df[cols].copy()

    # European decimal separators → proper floats
    for col in ("z-Score", "AREA_TOTAL_REDUCED"):
        if col in df_slim.columns:
            df_slim[col] = pd.to_numeric(
                df_slim[col].astype(str).str.replace(",", ".").str.strip(),
                errors="coerce",
            )

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_slim.to_parquet(PARQUET_PATH, compression="zstd", index=False)
    print(f"  Parquet: {len(df_slim)} rows, {df_slim.shape[1]} cols → {PARQUET_PATH.relative_to(ROOT)}")
    return 0


def generate_dropdown_index() -> None:
    """Derive dropdown-index.json from the golden dropdown_conditioning.json.

    Strips ``row_count`` (not needed at runtime) and writes minified JSON.
    """
    src = FRONTEND_GOLDEN_DIR / "dropdown_conditioning.json"
    if not src.exists():
        print("  WARNING: dropdown_conditioning.json not found, skipping index", file=sys.stderr)
        return

    with open(src) as f:
        data = json.load(f)

    # Strip row_count from each reaction type entry
    for rt_data in data.values():
        rt_data.pop("row_count", None)

    DROPDOWN_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DROPDOWN_INDEX_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"  Dropdown index: {len(data)} reaction types → {DROPDOWN_INDEX_PATH.relative_to(ROOT)}")


def _is_git_repo() -> bool:
    """Check if we're inside a git working tree (not bare CI builds)."""
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def stage_generated_files() -> None:
    """Stage all regenerated files for the commit (skipped in CI)."""
    if not _is_git_repo():
        print("  Skipping git stage (not in a git working tree)")
        return

    files_to_stage = [
        str(PARQUET_PATH.relative_to(ROOT)),
        str(DROPDOWN_INDEX_PATH.relative_to(ROOT)),
        str(FRONTEND_GOLDEN_DIR.relative_to(ROOT)),
    ]
    subprocess.run(
        ["git", "add"] + files_to_stage,
        cwd=str(ROOT),
        check=True,
    )
    print("  Staged all generated files")


def main() -> int:
    print("Regenerating artifacts from CSV...")

    rc = generate_parquet()
    if rc != 0:
        return rc

    generate_dropdown_index()
    stage_generated_files()

    print("\nDone — all artifacts regenerated and staged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
