"""Heatmap pivot value tests — verify cell medians, axis ordering, and ELN
counts for all reaction type × reactant pair combinations.

These tests catch bugs where the React rewrite's pivot table logic produces
different cell values or axis orderings than the current implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import data_utils as du

GOLDEN_PATH = Path(__file__).parent / 'fixtures' / 'golden' / 'heatmap_pivots.json'

with open(GOLDEN_PATH) as _f:
    _GOLDEN = json.load(_f)

FILTER_KWARGS = {
    'exclude_cui': ['exclude_cui'],
    'exclude_scaleup': [True],
    'include_null_categories': [True],
    'min_eln': 3,
    'topn_zscore': 3,
    'max_components': None,
}


def _build_params():
    params = []
    for key, expected in _GOLDEN.items():
        parts = key.split('|')
        rt = parts[0]
        pair = parts[1:]
        params.append((rt, pair, expected, key))
    return params


_PARAMS = _build_params()


@pytest.mark.slow
@pytest.mark.golden
class TestHeatmapPivots:
    """Verify heatmap pivot values match golden fixtures."""

    @pytest.mark.parametrize(
        'reaction_type,reactant_pair,expected',
        [(p[0], p[1], p[2]) for p in _PARAMS],
        ids=[p[3] for p in _PARAMS],
    )
    def test_pivot_values_match(self, full_dataset, reaction_type, reactant_pair, expected):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=[reaction_type],
            reactant_types=reactant_pair,
            **FILTER_KWARGS,
        )

        assert len(dff) == expected['row_count'], f'Row count: got {len(dff)}, expected {expected["row_count"]}'

        y_category = reactant_pair[0]
        x_category = reactant_pair[1]

        # Verify axis ordering
        y_medians = dff.groupby(y_category)['z-Score'].median().sort_values(ascending=True)
        x_medians = dff.groupby(x_category)['z-Score'].median().sort_values(ascending=False)

        actual_y_order = [str(v) for v in y_medians.index]
        actual_x_order = [str(v) for v in x_medians.index]

        assert (
            actual_y_order == expected['y_order']
        ), f'Y-axis order changed:\n  got:      {actual_y_order}\n  expected: {expected["y_order"]}'
        assert (
            actual_x_order == expected['x_order']
        ), f'X-axis order changed:\n  got:      {actual_x_order}\n  expected: {expected["x_order"]}'

        # Verify pivot cell values
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
        heatmap_df = heatmap_df.reindex(index=y_medians.index, columns=x_medians.index)
        eln_df = eln_df.reindex(index=y_medians.index, columns=x_medians.index).fillna(0).astype(int)

        for y_val in y_medians.index:
            for x_val in x_medians.index:
                key = f'{y_val}|{x_val}'
                actual_val = heatmap_df.loc[y_val, x_val]
                expected_val = expected['cell_medians'].get(key)

                if expected_val is None:
                    assert np.isnan(actual_val), f'Cell {key}: expected NaN, got {actual_val}'
                else:
                    assert not np.isnan(actual_val), f'Cell {key}: expected {expected_val}, got NaN'
                    assert (
                        abs(round(float(actual_val), 6) - expected_val) < 1e-4
                    ), f'Cell {key}: got {actual_val:.6f}, expected {expected_val}'

                # Verify ELN count
                actual_eln = int(eln_df.loc[y_val, x_val])
                expected_eln = expected['cell_eln_counts'].get(key, 0)
                assert actual_eln == expected_eln, f'ELN count for {key}: got {actual_eln}, expected {expected_eln}'
