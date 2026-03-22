"""Median consistency tests — verify median z-scores stay identical across all
reaction type × reactant type × filter combinations.

These tests catch any drift in the filter chain, deduplication, or groupby
logic that would change category rankings or median values.

Run with: python3 -m pytest tests/test_median_consistency.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import data_utils as du

GOLDEN_PATH = Path(__file__).parent / 'fixtures' / 'golden' / 'median_consistency.json'

# Load golden data once at module level
with open(GOLDEN_PATH) as _f:
    _GOLDEN = json.load(_f)


def _build_test_params():
    """Build parameterized test cases from the golden file."""
    params = []
    for filter_label, reactions in _GOLDEN.items():
        for reaction_type, reactants in reactions.items():
            for reactant_type, expected in reactants.items():
                test_id = f'{filter_label}/{reaction_type}/{reactant_type}'
                params.append((filter_label, reaction_type, reactant_type, expected, test_id))
    return params


# Filter kwargs matching generate_median_golden.py
def _build_filter_kwargs():
    """Import filter combos from the generator to stay in sync."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'generate_median_golden',
        Path(__file__).parent / 'generate_median_golden.py',
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {label: kwargs for label, kwargs in mod.FILTER_COMBOS}


_FILTER_KWARGS = _build_filter_kwargs()

_TEST_PARAMS = _build_test_params()


@pytest.mark.slow
@pytest.mark.golden
class TestMedianConsistency:
    """Verify median z-scores match golden values for every combination."""

    @pytest.mark.parametrize(
        'filter_label,reaction_type,reactant_type,expected',
        [(p[0], p[1], p[2], p[3]) for p in _TEST_PARAMS],
        ids=[p[4] for p in _TEST_PARAMS],
    )
    def test_medians_match(self, full_dataset, filter_label, reaction_type, reactant_type, expected):
        filter_kwargs = _FILTER_KWARGS[filter_label]

        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=[reaction_type],
            reactant_types=[reactant_type],
            **filter_kwargs,
        )

        assert dff is not None and not dff.empty, f'No data for {reaction_type}/{reactant_type} with {filter_label}'

        # Verify row count
        assert len(dff) == expected['row_count'], (
            f'Row count changed for {reaction_type}/{reactant_type}/{filter_label}: '
            f'got {len(dff)}, expected {expected["row_count"]}'
        )

        # Verify number of categories
        medians = dff.groupby(reactant_type)['z-Score'].median().sort_values(ascending=False)
        assert len(medians) == expected['n_categories'], (
            f'Category count changed for {reaction_type}/{reactant_type}/{filter_label}: '
            f'got {len(medians)}, expected {expected["n_categories"]}'
        )

        # Verify category ordering (most important — drives boxplot display)
        actual_order = [str(k) for k in medians.index]
        assert actual_order == expected['category_order'], (
            f'Category ordering changed for {reaction_type}/{reactant_type}/{filter_label}:\n'
            f'  got:      {actual_order}\n'
            f'  expected: {expected["category_order"]}'
        )

        # Verify each median value (tolerance: 1e-4 for float rounding)
        for cat_name, expected_median in expected['medians'].items():
            actual_median = round(float(medians[cat_name]), 6)
            assert abs(actual_median - expected_median) < 1e-4, (
                f'Median changed for {cat_name} in {reaction_type}/{reactant_type}/{filter_label}: '
                f'got {actual_median}, expected {expected_median}'
            )
