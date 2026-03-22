#!/usr/bin/env python3
"""Generate golden fixtures for heatmap pivot values.

Captures the 2D median z-score matrix, axis orderings, and ELN counts
for each reaction type × reactant type pair combination.

Run manually:  python3 tests/generate_heatmap_golden.py
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

# Reactant type pairs to test (first = y-axis, second = x-axis)
REACTANT_PAIRS = [
    ['Catalyst', 'Solvent'],
    ['Catalyst', 'Base'],
    ['Catalyst', 'Ligand'],
    ['Solvent', 'Base'],
    ['Solvent', 'Ligand'],
    ['Base', 'Ligand'],
    ['Catalyst', 'Solvent', 'Base'],  # 3-way
]

# Filter settings
FILTER_KWARGS = {
    'exclude_cui': ['exclude_cui'],
    'exclude_scaleup': [True],
    'include_null_categories': [True],
    'min_eln': 3,
    'topn_zscore': 3,
    'max_components': None,
}


def _compute_heatmap_data(
    source_df,
    reaction_type: str,
    reactant_pair: list[str],
) -> dict | None:
    """Compute heatmap pivot data for one combination."""
    # Check that all reactant columns exist and have data
    rt_df = source_df[source_df['Reaction Type'] == reaction_type]
    for col in reactant_pair:
        if col not in rt_df.columns or not rt_df[col].notna().any():
            return None

    try:
        result = du.filter_data(
            source_df=source_df,
            reaction_types=[reaction_type],
            reactant_types=reactant_pair,
            **FILTER_KWARGS,
        )
    except Exception:
        return None

    dff: pd.DataFrame = result[0] if isinstance(result, tuple) else result  # type: ignore[assignment]

    if dff is None or dff.empty:
        return None

    y_category = reactant_pair[0]
    x_category = reactant_pair[1]

    # Check both columns still have data after filtering
    if dff[y_category].notna().sum() == 0 or dff[x_category].notna().sum() == 0:
        return None

    # Compute pivot table (same logic as plot_utils.create_heatmap)
    heatmap_df = dff.pivot_table(
        index=y_category,
        columns=x_category,
        values='z-Score',
        aggfunc='median',
    )
    eln_df = dff.pivot_table(
        index=y_category,
        columns=x_category,
        values='ELN_ID',
        aggfunc='nunique',
    )

    # Order axes by median (y ascending, x descending) — matches plot_utils
    y_medians = dff.groupby(y_category)['z-Score'].median().sort_values(ascending=True)
    x_medians = dff.groupby(x_category)['z-Score'].median().sort_values(ascending=False)
    y_order = y_medians.index.tolist()
    x_order = x_medians.index.tolist()

    heatmap_df = heatmap_df.reindex(index=y_order, columns=x_order)
    eln_df = eln_df.reindex(index=y_order, columns=x_order).fillna(0).astype(int)

    # Serialize the matrix
    cell_values = {}
    eln_values = {}
    for y_val in y_order:
        for x_val in x_order:
            key = f'{y_val}|{x_val}'
            val = heatmap_df.loc[y_val, x_val]
            cell_values[key] = round(float(val), 6) if not np.isnan(val) else None
            eln_values[key] = int(eln_df.loc[y_val, x_val])

    return {
        'row_count': len(dff),
        'y_order': [str(v) for v in y_order],
        'x_order': [str(v) for v in x_order],
        'n_y': len(y_order),
        'n_x': len(x_order),
        'cell_medians': cell_values,
        'cell_eln_counts': eln_values,
    }


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    source_df = du.DF
    reaction_types = sorted(source_df['Reaction Type'].dropna().unique().tolist())

    results = {}
    total = 0

    for rt in reaction_types:
        for pair in REACTANT_PAIRS:
            key = f'{rt}|{"|".join(pair)}'
            data = _compute_heatmap_data(source_df, rt, pair)
            if data is not None:
                results[key] = data
                total += 1

    output_path = GOLDEN_DIR / 'heatmap_pivots.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Generated {total} heatmap pivot snapshots')
    print(f'Reactant pairs: {len(REACTANT_PAIRS)}')
    print(f'Reaction types: {len(reaction_types)}')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
