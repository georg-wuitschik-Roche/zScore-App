#!/usr/bin/env python3
"""Auto-version new dataset CSVs dropped into datasets/.

Triggered by a pre-commit hook when a CSV is staged in datasets/:

    git add datasets/my-new-data.csv && git commit

For each CSV found:
  1. Determines the next version number from versions.json
  2. Converts CSV → frontend/public/data/v{N}.parquet
  3. Computes dropdown index → frontend/public/data/v{N}-dropdown-index.json
  4. Updates frontend/public/data/versions.json
  5. Removes the source CSV from datasets/
  6. Stages all generated files
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / 'datasets'
DATA_DIR = ROOT / 'frontend' / 'public' / 'data'
VERSIONS_PATH = DATA_DIR / 'versions.json'

USED_COLUMNS = [
    'ELN_ID',
    'PLATENUMBER',
    'Coordinate',
    'AREA_TOTAL_REDUCED',
    'Base',
    'Catalyst',
    'Solvent',
    'Ligand',
    'Additive',
    'Coupling Reagent',
    'Secondary Solvent',
    'Tertiary Solvent',
    'Reaction Type',
    'FG A',
    'FG B',
    'FG_sorted',
    'z-Score',
    'output_column',
]

CATEGORY_OPTIONS = [
    'Additive',
    'Base',
    'Catalyst',
    'Coupling Reagent',
    'Solvent',
    'Ligand',
    'Secondary Solvent',
]


def _load_versions() -> dict:
    """Load existing versions.json or return empty structure."""
    if VERSIONS_PATH.exists():
        with open(VERSIONS_PATH) as f:
            return json.load(f)
    return {'versions': [], 'latest': 'default'}


def _next_version_number(manifest: dict) -> int:
    """Determine the next version number from existing manifest."""
    max_num = 0
    for v in manifest.get('versions', []):
        vid = v.get('id', '')
        if vid.startswith('v') and vid[1:].isdigit():
            max_num = max(max_num, int(vid[1:]))
    return max_num + 1


def _prepare_dataframe(csv_path: Path) -> 'pd.DataFrame':
    """Read CSV and apply same cleaning as regenerate_from_csv.py."""
    import pandas as pd

    df = pd.read_csv(csv_path, encoding='utf-8')
    cols = [c for c in USED_COLUMNS if c in df.columns]
    df_slim = df[cols].copy()

    # European decimal separators → proper floats
    for col in ('z-Score', 'AREA_TOTAL_REDUCED'):
        if col in df_slim.columns:
            df_slim[col] = pd.to_numeric(
                df_slim[col].astype(str).str.replace(',', '.').str.strip(),
                errors='coerce',
            )

    return df_slim


def _compute_dropdown_index(df: 'pd.DataFrame') -> dict[str, dict]:
    """Compute dropdown index from DataFrame.

    Mirrors the logic in frontend/src/data/dropdownOptions.ts:
    - reactant_availability: which CATEGORY_OPTIONS columns have non-null data
    - fg_all_options: sorted unique FG A + FG B values
    - fg_b_conditioned: for each FG A value, co-occurring FG B values (and vice versa)
    """
    index: dict[str, dict] = {}

    if 'Reaction Type' not in df.columns:
        return index

    for rt, group in df.groupby('Reaction Type'):
        # reactant_availability
        availability: list[str] = []
        for cat in CATEGORY_OPTIONS:
            if cat in group.columns and group[cat].notna().any() and (group[cat] != '').any():
                availability.append(cat)

        # fg_all_options
        fgs: set[str] = set()
        if 'FG A' in group.columns:
            fgs.update(group['FG A'].dropna().astype(str).loc[lambda s: s != ''].unique())
        if 'FG B' in group.columns:
            fgs.update(group['FG B'].dropna().astype(str).loc[lambda s: s != ''].unique())
        fg_all = sorted(fgs)

        # fg_b_conditioned
        fg_b_cond: dict[str, list[str]] = defaultdict(set)  # type: ignore[assignment]
        if 'FG A' in group.columns and 'FG B' in group.columns:
            for _, row in group[['FG A', 'FG B']].dropna().iterrows():
                fa, fb = str(row['FG A']), str(row['FG B'])
                if fa and fb and fa != 'nan' and fb != 'nan':
                    fg_b_cond[fa].add(fb)  # type: ignore[union-attr]
                    fg_b_cond[fb].add(fa)  # type: ignore[union-attr]

        # Convert sets to sorted lists
        fg_b_cond_sorted: dict[str, list[str]] = {
            k: sorted(v) for k, v in sorted(fg_b_cond.items())  # type: ignore[arg-count]
        }

        index[str(rt)] = {
            'reactant_availability': availability,
            'fg_all_options': fg_all,
            'fg_b_conditioned': fg_b_cond_sorted,
        }

    return index


def _is_git_repo() -> bool:
    """Check if we're inside a git working tree."""
    result = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == 'true'


def process_csv(csv_path: Path, manifest: dict) -> int:
    """Process a single CSV file into a versioned dataset.

    Returns the version number assigned.
    """
    import pandas as pd

    version_num = _next_version_number(manifest)
    version_id = f'v{version_num}'

    print(f'  Processing {csv_path.name} → {version_id}')

    # 1. Read and clean
    df = _prepare_dataframe(csv_path)
    print(f'    {len(df)} rows, {df.shape[1]} columns')

    # 2. Write parquet
    parquet_path = DATA_DIR / f'{version_id}.parquet'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, compression='zstd', index=False)
    print(f'    Parquet → {parquet_path.relative_to(ROOT)}')

    # 3. Compute and write dropdown index
    index = _compute_dropdown_index(df)
    index_path = DATA_DIR / f'{version_id}-dropdown-index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, separators=(',', ':'))
    print(f'    Dropdown index: {len(index)} reaction types → {index_path.relative_to(ROOT)}')

    # 4. Update manifest
    # Use CSV filename (without extension) as label
    label = csv_path.stem.replace('_', ' ').replace('-', ' ')
    manifest['versions'].append({
        'id': version_id,
        'parquet': f'/data/{version_id}.parquet',
        'index': f'/data/{version_id}-dropdown-index.json',
        'label': f'v{version_num}',
        'date': date.today().isoformat(),
    })
    manifest['latest'] = version_id

    return version_num


def main() -> int:
    """Scan datasets/ for CSVs and version them."""
    if not DATASETS_DIR.exists():
        print('No datasets/ directory found, nothing to do.')
        return 0

    csv_files = sorted(DATASETS_DIR.glob('*.csv'))
    if not csv_files:
        print('No CSV files in datasets/, nothing to do.')
        return 0

    print(f'Found {len(csv_files)} CSV file(s) to version...')

    manifest = _load_versions()
    files_to_stage: list[str] = []
    files_to_remove: list[str] = []

    for csv_path in csv_files:
        try:
            process_csv(csv_path, manifest)
            files_to_remove.append(str(csv_path.relative_to(ROOT)))
        except Exception as e:
            print(f'  ERROR processing {csv_path.name}: {e}', file=sys.stderr)
            return 1

    # Write updated versions.json
    with open(VERSIONS_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'  Updated {VERSIONS_PATH.relative_to(ROOT)}')

    # Stage generated files and remove source CSVs
    if _is_git_repo():
        # Stage all generated data files
        files_to_stage = [str(DATA_DIR.relative_to(ROOT))]
        subprocess.run(['git', 'add'] + files_to_stage, cwd=str(ROOT), check=True)

        # Remove source CSVs
        for csv_rel in files_to_remove:
            csv_abs = ROOT / csv_rel
            csv_abs.unlink(missing_ok=True)
            subprocess.run(['git', 'rm', '--cached', '-f', csv_rel], cwd=str(ROOT), capture_output=True)
        print('  Staged generated files and removed source CSVs')

    print('\nDone — dataset(s) versioned successfully.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
