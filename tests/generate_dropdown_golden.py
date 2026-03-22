#!/usr/bin/env python3
"""Generate golden fixtures for dropdown conditioning logic.

Captures what FG B options appear for each FG A selection, per reaction type.
Also captures reactant type availability per reaction type.

Run manually:  python3 tests/generate_dropdown_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import data_utils as du

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'golden'

# FG A values to test conditioning on
FG_A_SELECTIONS = [
    ['RNH2'],
    ['ArNH2'],
    ['RNH2 a-branch'],
    ['ArBr'],
    ['ArCl'],
    ['ArI'],
    ['RNH2', 'RNH2 a-branch'],
    ['ArBr', 'ArCl'],
    ['ArNH2', 'ArBr'],
    ['RNH2', 'ArNH2', 'RNH2 a-branch'],
]


def _compute_fg_b_options(dff: pd.DataFrame, fg_a_list: list[str]) -> list[str]:
    """Replicate the FG B conditioning logic from callbacks.py."""
    if not {'FG A', 'FG B'}.issubset(dff.columns):
        return []

    all_other_fgs: list = []
    for fg_a_val in fg_a_list:
        mask = (dff['FG A'] == fg_a_val) | (dff['FG B'] == fg_a_val)
        dff_sub = dff[mask]
        other_fgs: list = []
        other_fgs.extend(dff_sub.loc[dff_sub['FG A'] == fg_a_val, 'FG B'])
        other_fgs.extend(dff_sub.loc[dff_sub['FG B'] == fg_a_val, 'FG A'])
        all_other_fgs.extend(other_fgs)

    return sorted(pd.Series(all_other_fgs).dropna().unique().tolist())


def _compute_fg_all_options(dff: pd.DataFrame) -> list[str]:
    """Get all FG options (when FG A = 'All')."""
    if not {'FG A', 'FG B'}.issubset(dff.columns):
        return []
    fg_values = pd.concat([dff['FG A'], dff['FG B']]).dropna().unique()
    return sorted(fg_values.tolist())


def _compute_reactant_availability(dff: pd.DataFrame) -> list[str]:
    """Which reactant type columns have data for this reaction type."""
    categories = [
        'Additive',
        'Base',
        'Catalyst',
        'Coupling Reagent',
        'Ligand',
        'Secondary Solvent',
        'Solvent',
    ]
    available = []
    for cat in categories:
        if cat in dff.columns and dff[cat].notna().any():
            available.append(cat)
    return available


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    source_df = du.DF
    reaction_types = sorted(source_df['Reaction Type'].dropna().unique().tolist())

    results = {}
    total = 0

    for rt in reaction_types:
        dff = source_df[source_df['Reaction Type'] == rt]
        if dff.empty:
            continue

        rt_data = {
            'row_count': len(dff),
            'reactant_availability': _compute_reactant_availability(dff),
            'fg_all_options': _compute_fg_all_options(dff),
            'fg_b_conditioned': {},
        }

        for fg_a_list in FG_A_SELECTIONS:
            key = '+'.join(fg_a_list)
            fg_b_options = _compute_fg_b_options(dff, fg_a_list)
            if fg_b_options:
                rt_data['fg_b_conditioned'][key] = fg_b_options
                total += 1

        results[rt] = rt_data

    output_path = GOLDEN_DIR / 'dropdown_conditioning.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Generated {total} FG B conditioning snapshots')
    print(f'Reaction types: {len(results)}')
    print(f'FG A selections tested: {len(FG_A_SELECTIONS)}')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
