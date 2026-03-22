"""Stats table output tests — verify describe() values (count, mean, std,
min, quartiles, max) match golden fixtures for various filter combinations.

These tests catch bugs where the React rewrite's statistics computation
produces different summary values than the current Python implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import data_utils as du

GOLDEN_PATH = Path(__file__).parent / 'fixtures' / 'golden' / 'stats_table.json'

with open(GOLDEN_PATH) as _f:
    _GOLDEN = json.load(_f)


def _build_params():
    params = []
    for label, data in _GOLDEN.items():
        params.append((label, data['params'], data))
    return params


_PARAMS = _build_params()


@pytest.mark.slow
@pytest.mark.golden
class TestStatsTable:
    """Verify describe() output matches golden fixtures."""

    @pytest.mark.parametrize(
        'label,filter_params,expected',
        _PARAMS,
        ids=[p[0] for p in _PARAMS],
    )
    def test_stats_match(self, full_dataset, label, filter_params, expected):
        dff = du.filter_data(source_df=full_dataset, **filter_params)

        # Row count
        assert len(dff) == expected['row_count'], f'Row count: got {len(dff)}, expected {expected["row_count"]}'

        # ELN count
        eln_count = int(dff['ELN_ID'].nunique())
        assert eln_count == expected['eln_count'], f'ELN count: got {eln_count}, expected {expected["eln_count"]}'

        # Verify describe() for each numeric column
        for col_name, expected_stats in expected['columns'].items():
            assert col_name in dff.columns, f'Missing column: {col_name}'

            desc = dff[col_name].describe()

            for stat_name, expected_val in expected_stats.items():
                actual_val = desc[stat_name]

                if expected_val is None:
                    assert np.isnan(actual_val), f'{col_name}.{stat_name}: expected NaN, got {actual_val}'
                elif stat_name == 'count':
                    assert (
                        int(actual_val) == expected_val
                    ), f'{col_name}.{stat_name}: got {int(actual_val)}, expected {expected_val}'
                else:
                    assert (
                        abs(round(float(actual_val), 6) - expected_val) < 1e-4
                    ), f'{col_name}.{stat_name}: got {actual_val:.6f}, expected {expected_val}'
