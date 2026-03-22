#!/usr/bin/env python3
"""Generate golden median fixtures for all reaction type × reactant type × filter combos.

Run manually and commit the results to git:
    python3 tests/generate_median_golden.py

The output is a single JSON file containing median z-scores per category
for every combination. Tests compare current medians against these values
to catch any drift in filtering or computation logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import data_utils as du

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'golden'

# Reactant types to test (exclude FG categories — they're not reactant types)
REACTANT_TYPES = [
    'Catalyst',
    'Solvent',
    'Base',
    'Ligand',
    'Additive',
    'Coupling Reagent',
    'Secondary Solvent',
]

# Common FG pairs for testing (most frequent in dataset)
_COMMON_FG_A = ['ArBr', 'ArCl', 'ArNH2', 'RNH2 a-branch', 'RNH2', 'ArI']
_COMMON_FG_B = ['ArBr', 'ArCl', 'RNH2 a-branch', 'ArB(OR)2', 'ArNH2', 'RNH2']

# Filter parameter sets: (label, kwargs to override)
FILTER_COMBOS = [
    # --- Baseline: default dashboard settings ---
    (
        'defaults',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- CuI exclusion toggle ---
    (
        'no_cui_filter',
        {
            'exclude_cui': None,
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- Min ELN variations ---
    (
        'min_eln_1',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 1,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'min_eln_3',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 3,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'min_eln_10',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 10,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'min_eln_15',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 15,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- Top-N z-score variations ---
    (
        'topn_1',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 1,
            'max_components': None,
        },
    ),
    (
        'topn_3',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 3,
            'max_components': None,
        },
    ),
    (
        'topn_10',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 10,
            'max_components': None,
        },
    ),
    # --- Max components cap ---
    (
        'max_components_3',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 3,
        },
    ),
    (
        'max_components_5',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 5,
        },
    ),
    (
        'max_components_10',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
        },
    ),
    (
        'max_components_20',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 20,
        },
    ),
    # --- Scale-up plate toggle ---
    (
        'no_scaleup_filter',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': None,
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- Null categories toggle ---
    (
        'exclude_null_categories',
        {
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': None,
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- All checkboxes OFF ---
    (
        'all_checkboxes_off',
        {
            'exclude_cui': None,
            'exclude_scaleup': None,
            'include_null_categories': None,
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- Minimal filtering (just reaction type + reactant type) ---
    (
        'minimal_filters',
        {
            'exclude_cui': None,
            'exclude_scaleup': None,
            'include_null_categories': [True],
            'min_eln': 1,
            'topn_zscore': 10,
            'max_components': None,
        },
    ),
    # --- FG A single selections ---
    (
        'fg_a_ArBr',
        {
            'fg_a': ['ArBr'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_a_ArCl',
        {
            'fg_a': ['ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_a_RNH2',
        {
            'fg_a': ['RNH2'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_a_ArNH2',
        {
            'fg_a': ['ArNH2'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- FG A multi selections ---
    (
        'fg_a_ArBr_ArCl',
        {
            'fg_a': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_a_RNH2_RNH2_abranch',
        {
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- FG A + FG B pair selections ---
    (
        'fg_pair_RNH2_ArBr',
        {
            'fg_a': ['RNH2'],
            'fg_b': ['ArBr'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_pair_RNH2_ArCl',
        {
            'fg_a': ['RNH2'],
            'fg_b': ['ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_pair_ArNH2_ArBr',
        {
            'fg_a': ['ArNH2'],
            'fg_b': ['ArBr'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_pair_RNH2abranch_ArBr_ArCl',
        {
            'fg_a': ['RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_pair_multi_a_multi_b',
        {
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- FG B only (no FG A constraint) ---
    (
        'fg_b_only_ArBr',
        {
            'fg_b': ['ArBr'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_b_only_ArCl',
        {
            'fg_b': ['ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    # --- Combined edge cases ---
    (
        'fg_pair_strict_eln',
        {
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 10,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'fg_pair_topn1_max5',
        {
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 1,
            'max_components': 5,
        },
    ),
    (
        'fg_pair_no_cui_no_scaleup',
        {
            'fg_a': ['RNH2'],
            'fg_b': ['ArBr'],
            'exclude_cui': None,
            'exclude_scaleup': None,
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': None,
        },
    ),
    (
        'all_strict',
        {
            'fg_a': ['RNH2', 'RNH2 a-branch'],
            'fg_b': ['ArBr', 'ArCl'],
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': None,
            'min_eln': 10,
            'topn_zscore': 3,
            'max_components': 5,
        },
    ),
    (
        'all_relaxed',
        {
            'exclude_cui': None,
            'exclude_scaleup': None,
            'include_null_categories': [True],
            'min_eln': 1,
            'topn_zscore': 10,
            'max_components': None,
        },
    ),
]


def _compute_medians(
    source_df,
    reaction_type: str,
    reactant_type: str,
    filter_kwargs: dict,
) -> dict | None:
    """Compute median z-scores per category value for one combination.

    Returns None if the reactant column doesn't exist or has no data
    for this reaction type.
    """
    if reactant_type not in source_df.columns:
        return None

    # Check if this reaction type + reactant type has data
    rt_mask = source_df['Reaction Type'] == reaction_type
    if not rt_mask.any():
        return None
    if not source_df.loc[rt_mask, reactant_type].notna().any():
        return None

    # Call filter_data WITHOUT max_components first, then apply our own deterministic
    # max_components selection with alphabetical tie-breaking. This avoids pandas'
    # Categorical-order-dependent tiebreaking in _filter_max_components.
    fkw = dict(filter_kwargs)
    max_components = fkw.pop('max_components', None)

    try:
        result = du.filter_data(
            source_df=source_df,
            reaction_types=[reaction_type],
            reactant_types=[reactant_type],
            max_components=None,
            **fkw,
        )
    except Exception:
        return None

    dff: pd.DataFrame = result[0] if isinstance(result, tuple) else result  # type: ignore[assignment]

    if dff is None or dff.empty or reactant_type not in dff.columns:
        return None

    # Compute median z-score per category value (include null categories)
    cat_series = dff[reactant_type].astype('object').fillna('(no value)').astype(str)
    medians_raw = dff.groupby(cat_series)['z-Score'].median()
    if medians_raw.empty:
        return None

    # Sort by median descending, then alphabetically for deterministic tie-breaking.
    medians_df = medians_raw.to_frame('median').reset_index()
    medians_df.columns = ['category', 'median']
    medians_df['sort_key'] = medians_df['median'].round(9)
    medians_df = medians_df.sort_values(['sort_key', 'category'], ascending=[False, True])
    medians_df = medians_df.drop(columns=['sort_key'])

    # Apply max_components cap using the same deterministic ordering
    if max_components and max_components > 0 and len(medians_df) > max_components:
        top_categories = medians_df['category'].head(max_components).tolist()
        dff = dff[cat_series.isin(top_categories)]
        medians_df = medians_df.head(max_components)

    medians = medians_df.set_index('category')['median']

    return {
        'row_count': len(dff),
        'n_categories': len(medians),
        'medians': {str(k): round(float(v), 6) for k, v in medians.items()},
        'category_order': [str(k) for k in medians.index],
    }


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    source_df = du.DF
    reaction_types = sorted(source_df['Reaction Type'].dropna().unique().tolist())

    all_results = {}
    total = 0
    skipped = 0

    for filter_label, filter_kwargs in FILTER_COMBOS:
        all_results[filter_label] = {}

        for rt in reaction_types:
            all_results[filter_label][rt] = {}

            for reactant in REACTANT_TYPES:
                result = _compute_medians(source_df, rt, reactant, filter_kwargs)
                if result is not None:
                    all_results[filter_label][rt][reactant] = result
                    total += 1
                else:
                    skipped += 1

            # Remove empty reaction type entries
            if not all_results[filter_label][rt]:
                del all_results[filter_label][rt]

    output_path = GOLDEN_DIR / 'median_consistency.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'Generated {total} median snapshots ({skipped} skipped)')
    print(f'Filter combos: {len(FILTER_COMBOS)}')
    print(f'Reaction types: {len(reaction_types)}')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
