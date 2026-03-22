#!/usr/bin/env python3
"""Generate golden snapshot files from the real dataset.

Run manually and commit the results to git:
    python3 tests/generate_golden.py

Golden files are JSON containing filter params, expected row counts,
z-score statistics, and category value counts. They serve as the
acceptance criteria for a future React rewrite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import data_utils as du

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'golden'


# Golden test scenarios: (name, filter_data kwargs)
SCENARIOS = [
    (
        'buchwald_hartwig_catalyst',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Catalyst'],
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
        },
    ),
    (
        'buchwald_hartwig_ligand',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Ligand'],
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
        },
    ),
    (
        'suzuki_miyaura_catalyst',
        {
            'reaction_types': ['Suzuki-Miyaura'],
            'reactant_types': ['Catalyst'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
        },
    ),
    (
        'all_reactions_base',
        {
            'reactant_types': ['Base'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
        },
    ),
    (
        'multi_category_catalyst_solvent',
        {
            'reaction_types': ['Buchwald-Hartwig'],
            'reactant_types': ['Catalyst', 'Solvent'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 3,
            'topn_zscore': 3,
            'max_components': 10,
        },
    ),
    (
        'no_filters',
        {},
    ),
]


def _generate_snapshot(name: str, kwargs: dict) -> dict:
    """Run filter_data with given kwargs and capture output statistics."""
    result = du.filter_data(**kwargs, return_stats=True, source_df=du.DF)

    if isinstance(result, tuple):
        dff, stats = result
    else:
        dff, stats = result, {}

    z_scores = dff['z-Score'].dropna()

    snapshot: dict = {
        'params': kwargs,
        'row_count': len(dff),
        'eln_count': int(dff['ELN_ID'].nunique()) if 'ELN_ID' in dff.columns else 0,
        'column_count': len(dff.columns),
        'z_score_stats': {
            'mean': float(z_scores.mean()) if len(z_scores) > 0 else None,
            'std': float(z_scores.std()) if len(z_scores) > 0 else None,
            'min': float(z_scores.min()) if len(z_scores) > 0 else None,
            'max': float(z_scores.max()) if len(z_scores) > 0 else None,
            'median': float(z_scores.median()) if len(z_scores) > 0 else None,
        },
        'stats_dict': _serialize_stats(stats),
    }

    # Add category value counts for reactant types
    reactant_types = kwargs.get('reactant_types', [])
    if reactant_types:
        snapshot['category_value_counts'] = {}
        for rt in reactant_types:
            if rt in dff.columns:
                counts = dff[rt].value_counts().to_dict()
                snapshot['category_value_counts'][rt] = {str(k): int(v) for k, v in counts.items()}

    return snapshot


def _serialize_stats(stats: dict) -> dict:
    """Convert stats dict values to JSON-serializable types."""
    result: dict = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            result[k] = {str(k2): int(v2) if isinstance(v2, int | np.integer) else v2 for k2, v2 in v.items()}
        elif isinstance(v, int | np.integer):
            result[k] = int(v)
        else:
            result[k] = v
    return result


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    for name, kwargs in SCENARIOS:
        print(f'Generating {name}...')
        try:
            snapshot = _generate_snapshot(name, kwargs)
            output_path = GOLDEN_DIR / f'{name}.json'
            with open(output_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            print(f'  -> {output_path} ({snapshot["row_count"]} rows, {snapshot["eln_count"]} ELNs)')
        except Exception as e:
            print(f'  ERROR: {e}')

    print(f'\nDone. Generated {len(SCENARIOS)} golden files in {GOLDEN_DIR}')


if __name__ == '__main__':
    main()
