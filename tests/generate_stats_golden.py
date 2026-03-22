#!/usr/bin/env python3
"""Generate golden fixtures for stats table (describe()) output.

Captures count, mean, std, min, quartiles, max for filtered data —
the exact values shown in the Statistics tab.

Run manually:  python3 tests/generate_stats_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import data_utils as du

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'golden'

# Test scenarios: (label, filter_data kwargs)
# The stats tab uses min_eln=None, max_components=None to get full data
SCENARIOS = [
    (
        'buchwald_catalyst',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Catalyst'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'buchwald_ligand',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Ligand'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'buchwald_fg_pair',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Catalyst'],
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'suzuki_catalyst',
        {
            'reaction_types': ['Suzuki-Miyaura'],
            'reactant_types': ['Catalyst'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'suzuki_solvent',
        {
            'reaction_types': ['Suzuki-Miyaura'],
            'reactant_types': ['Solvent'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'amide_coupling_reagent',
        {
            'reaction_types': ['Amide coupling'],
            'reactant_types': ['Coupling Reagent'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'ch_activation_catalyst',
        {
            'reaction_types': ['CH-Activation'],
            'reactant_types': ['Catalyst'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'all_reactions_base',
        {
            'reactant_types': ['Base'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 5,
        },
    ),
    (
        'buchwald_strict',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Catalyst'],
            'fg_a': ['RNH2'],
            'fg_b': ['ArBr'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': None,
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 3,
        },
    ),
    (
        'buchwald_relaxed',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Catalyst'],
            'exclude_cui': None,
            'exclude_scaleup': None,
            'include_null_categories': [True],
            'min_eln': None,
            'max_components': None,
            'topn_zscore': 10,
        },
    ),
]


def _compute_stats_snapshot(source_df, kwargs: dict) -> dict | None:
    """Compute describe() stats matching the Statistics tab logic."""
    try:
        result = du.filter_data(source_df=source_df, **kwargs)
    except Exception:
        return None

    dff: pd.DataFrame = result[0] if isinstance(result, tuple) else result  # type: ignore[assignment]

    if dff is None or dff.empty:
        return None

    numeric_cols: list[str] = []
    for col in ['z-Score', 'AREA_TOTAL_REDUCED']:
        if col in dff.columns:
            numeric_cols.append(col)

    if not numeric_cols:
        return None

    desc = dff[numeric_cols].describe()
    eln_count = int(dff['ELN_ID'].nunique()) if 'ELN_ID' in dff.columns else len(dff)

    # Serialize describe() output
    stats: dict = {
        'row_count': len(dff),
        'eln_count': eln_count,
        'columns': {},
    }

    for col in numeric_cols:
        col_stats: dict[str, int | float | None] = {}
        for stat_name in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            val = desc.loc[stat_name, col]
            if np.isnan(val):
                col_stats[stat_name] = None
            elif stat_name == 'count':
                col_stats[stat_name] = int(val)
            else:
                col_stats[stat_name] = round(float(val), 6)
        stats['columns'][col] = col_stats

    return stats


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    source_df = du.DF
    results = {}
    total = 0

    for label, kwargs in SCENARIOS:
        data = _compute_stats_snapshot(source_df, kwargs)
        if data is not None:
            results[label] = {
                'params': {k: v for k, v in kwargs.items() if v is not None},
                **data,
            }
            total += 1
            print(f'  {label}: {data["row_count"]} rows, {data["eln_count"]} ELNs')
        else:
            print(f'  {label}: SKIPPED (no data)')

    output_path = GOLDEN_DIR / 'stats_table.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nGenerated {total} stats snapshots')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
