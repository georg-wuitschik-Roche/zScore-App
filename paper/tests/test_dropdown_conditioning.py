"""Dropdown conditioning tests — verify FG B options given FG A selections
and reactant type availability per reaction type.

These tests catch bugs where the React rewrite shows wrong options in
dependent dropdowns, leading users to select invalid filter combinations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

GOLDEN_PATH = Path(__file__).parent / 'fixtures' / 'golden' / 'dropdown_conditioning.json'

with open(GOLDEN_PATH) as _f:
    _GOLDEN = json.load(_f)


def _compute_fg_b_options(dff: pd.DataFrame, fg_a_list: list[str]) -> list[str]:
    """Replicate FG B conditioning logic from callbacks.py."""
    all_other_fgs: list = []
    for fg_a_val in fg_a_list:
        mask = (dff['FG A'] == fg_a_val) | (dff['FG B'] == fg_a_val)
        dff_sub = dff[mask]
        all_other_fgs.extend(dff_sub.loc[dff_sub['FG A'] == fg_a_val, 'FG B'])
        all_other_fgs.extend(dff_sub.loc[dff_sub['FG B'] == fg_a_val, 'FG A'])
    return sorted(pd.Series(all_other_fgs).dropna().unique().tolist())


# ---------------------------------------------------------------------------
# FG B conditioned on FG A
# ---------------------------------------------------------------------------


def _build_fg_b_params():
    params = []
    for rt, rt_data in _GOLDEN.items():
        for fg_a_key, expected_fg_b in rt_data.get('fg_b_conditioned', {}).items():
            fg_a_list = fg_a_key.split('+')
            test_id = f'{rt}/FG_A={fg_a_key}'
            params.append((rt, fg_a_list, expected_fg_b, test_id))
    return params


_FG_B_PARAMS = _build_fg_b_params()


@pytest.mark.slow
@pytest.mark.golden
class TestFgBConditionedOnFgA:
    """Verify FG B options match golden values for each FG A selection."""

    @pytest.mark.parametrize(
        'reaction_type,fg_a_list,expected_fg_b',
        [(p[0], p[1], p[2]) for p in _FG_B_PARAMS],
        ids=[p[3] for p in _FG_B_PARAMS],
    )
    def test_fg_b_options_match(self, full_dataset, reaction_type, fg_a_list, expected_fg_b):
        dff = full_dataset[full_dataset['Reaction Type'] == reaction_type]
        actual_fg_b = _compute_fg_b_options(dff, fg_a_list)
        assert actual_fg_b == expected_fg_b, (
            f'FG B options changed for {reaction_type} with FG A={fg_a_list}:\n'
            f'  got:      {actual_fg_b}\n'
            f'  expected: {expected_fg_b}'
        )


# ---------------------------------------------------------------------------
# Reactant type availability
# ---------------------------------------------------------------------------


def _build_reactant_params():
    params = []
    for rt, rt_data in _GOLDEN.items():
        expected = rt_data.get('reactant_availability', [])
        params.append((rt, expected))
    return params


_REACTANT_PARAMS = _build_reactant_params()


@pytest.mark.slow
@pytest.mark.golden
class TestReactantAvailability:
    """Verify which reactant types are available per reaction type."""

    @pytest.mark.parametrize(
        'reaction_type,expected_reactants',
        _REACTANT_PARAMS,
        ids=[p[0] for p in _REACTANT_PARAMS],
    )
    def test_reactant_types_available(self, full_dataset, reaction_type, expected_reactants):
        dff = full_dataset[full_dataset['Reaction Type'] == reaction_type]
        categories = [
            'Additive',
            'Base',
            'Catalyst',
            'Coupling Reagent',
            'Ligand',
            'Secondary Solvent',
            'Solvent',
        ]
        actual = [c for c in categories if c in dff.columns and dff[c].notna().any()]
        assert actual == expected_reactants, (
            f'Reactant availability changed for {reaction_type}:\n'
            f'  got:      {actual}\n'
            f'  expected: {expected_reactants}'
        )


# ---------------------------------------------------------------------------
# FG A options (all FGs available per reaction type)
# ---------------------------------------------------------------------------


def _build_fg_all_params():
    params = []
    for rt, rt_data in _GOLDEN.items():
        expected = rt_data.get('fg_all_options', [])
        if expected:
            params.append((rt, expected))
    return params


_FG_ALL_PARAMS = _build_fg_all_params()


@pytest.mark.slow
@pytest.mark.golden
class TestFgAllOptions:
    """Verify the full set of FG options available per reaction type."""

    @pytest.mark.parametrize(
        'reaction_type,expected_fgs',
        _FG_ALL_PARAMS,
        ids=[p[0] for p in _FG_ALL_PARAMS],
    )
    def test_fg_options_match(self, full_dataset, reaction_type, expected_fgs):
        dff = full_dataset[full_dataset['Reaction Type'] == reaction_type]
        fg_values = pd.concat([dff['FG A'], dff['FG B']]).dropna().unique()
        actual = sorted(fg_values.tolist())
        assert (
            actual == expected_fgs
        ), f'FG options changed for {reaction_type}:\n  got:      {actual}\n  expected: {expected_fgs}'
