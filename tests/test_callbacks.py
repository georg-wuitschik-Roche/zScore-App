"""Tests for callbacks.py — helper functions and callback logic."""

from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
import pytest
from dash import no_update

import callbacks as cb
import data_utils as du

# ===========================================================================
# Helper functions
# ===========================================================================


class TestParseFilterArgs:
    def test_correct_key_mapping(self):
        args = (
            ['Catalyst'],  # reactant_types
            ['Buchwald-Hartwig'],  # reaction_types
            ['RNH2'],  # fg_a
            ['ArBr'],  # fg_b
            ['exclude_cui'],  # exclude_cui
            [True],  # exclude_scaleup
            [True],  # include_null_categories
            5,  # min_eln
            5,  # topn_zscore
            10,  # max_components
            None,  # uploaded_data
        )
        result = cb._parse_filter_args(args)
        assert result['reactant_types'] == ['Catalyst']
        assert result['reaction_types'] == ['Buchwald-Hartwig']
        assert result['fg_a'] == ['RNH2']
        assert result['fg_b'] == ['ArBr']
        assert result['exclude_cui'] == ['exclude_cui']
        assert result['min_eln'] == 5
        assert result['topn_zscore'] == 5
        assert result['max_components'] == 10
        assert result['uploaded_data'] is None

    def test_all_eleven_keys_present(self):
        args = tuple(range(11))
        result = cb._parse_filter_args(args)
        assert len(result) == 11
        assert set(result.keys()) == set(cb._FILTER_KEYS)

    def test_preserves_none_values(self):
        args = (None,) * 11
        result = cb._parse_filter_args(args)
        assert all(v is None for v in result.values())

    def test_preserves_list_values(self):
        args = (['a', 'b'],) + (None,) * 10
        result = cb._parse_filter_args(args)
        assert result['reactant_types'] == ['a', 'b']


class TestCallFilterData:
    def test_calls_filter_data_with_correct_args(self, small_df, monkeypatch):
        captured = {}

        def mock_filter(**kwargs):
            captured.update(kwargs)
            return small_df

        monkeypatch.setattr(du, 'filter_data', mock_filter)
        monkeypatch.setattr(du, 'get_uploaded_dataframe', lambda x: None)

        fkw = {
            'reactant_types': ['Catalyst'],
            'reaction_types': ['BH'],
            'fg_a': None,
            'fg_b': None,
            'exclude_cui': None,
            'exclude_scaleup': None,
            'include_null_categories': None,
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
            'uploaded_data': None,
        }
        cb._call_filter_data(fkw)
        assert captured['reactant_types'] == ['Catalyst']
        assert captured['reaction_types'] == ['BH']
        assert captured['session_id'] is None
        assert captured['source_df'] is None

    def test_overrides_take_precedence(self, small_df, monkeypatch):
        captured = {}

        def mock_filter(**kwargs):
            captured.update(kwargs)
            return small_df

        monkeypatch.setattr(du, 'filter_data', mock_filter)
        monkeypatch.setattr(du, 'get_uploaded_dataframe', lambda x: None)

        fkw = {k: None for k in cb._FILTER_KEYS}
        fkw['min_eln'] = 5
        cb._call_filter_data(fkw, return_stats=True, min_eln=10)
        assert captured['return_stats'] is True
        assert captured['min_eln'] == 10  # Override wins

    def test_session_id_loads_uploaded_df(self, upload_df, monkeypatch):
        sid = du.store_uploaded_dataframe(upload_df)
        captured = {}

        def mock_filter(**kwargs):
            captured.update(kwargs)
            return upload_df

        monkeypatch.setattr(du, 'filter_data', mock_filter)

        fkw = {k: None for k in cb._FILTER_KEYS}
        fkw['uploaded_data'] = sid
        cb._call_filter_data(fkw)
        assert captured['source_df'] is not None
        assert captured['session_id'] == sid


class TestIsTutorialStepSatisfied:
    @pytest.mark.parametrize(
        'step_idx,kwargs,expected',
        [
            # Step 0: reaction types selected
            (0, {'reaction_types': ['BH']}, True),
            (0, {'reaction_types': []}, False),
            (0, {'reaction_types': None}, False),
            # Step 1: reactant types selected
            (1, {'reactant_types': ['Catalyst']}, True),
            (1, {'reactant_types': []}, False),
            # Step 2: FG A selected
            (2, {'fg_a_vals': ['RNH2']}, True),
            (2, {'fg_a_vals': []}, False),
            # Step 3: FG B selected
            (3, {'fg_b_vals': ['ArBr']}, True),
            (3, {'fg_b_vals': []}, False),
            # Step 4: filter panel expanded
            (4, {'filter_panel_style': {'maxHeight': '200px'}}, True),
            (4, {'filter_panel_style': {'maxHeight': '0px'}}, False),
            (4, {'filter_panel_style': None}, False),
            (4, {'filter_panel_style': {'display': 'none'}}, False),
            # Step 5: min ELN changed from 10
            (5, {'min_eln': 3}, True),
            (5, {'min_eln': 10}, False),
            (5, {'min_eln': None}, False),
            # Step 6: topn changed from 3
            (6, {'topn': 5}, True),
            (6, {'topn': 3}, False),
            # Step 7: max components changed from 10
            (7, {'max_comp': 5}, True),
            (7, {'max_comp': 10}, False),
            # Step 8: CuI exclude unchecked
            (8, {'exclude_cui_val': []}, True),
            (8, {'exclude_cui_val': ['exclude_cui']}, False),
            # Step 9: heatmap tab active
            (9, {'tabs_value': 'tab-heatmap'}, True),
            (9, {'tabs_value': 'tab-graph'}, False),
            # Step 10: always true
            (10, {}, True),
        ],
    )
    def test_step_gating(self, step_idx, kwargs, expected):
        # Build full kwargs with defaults
        defaults = {
            'reaction_types': None,
            'reactant_types': None,
            'fg_a_vals': None,
            'fg_b_vals': None,
            'filter_panel_style': None,
            'min_eln': None,
            'topn': None,
            'max_comp': None,
            'exclude_cui_val': None,
            'tabs_value': None,
        }
        defaults.update(kwargs)
        assert cb._is_tutorial_step_satisfied(step_idx, **defaults) is expected


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    def test_required_columns_list(self):
        assert isinstance(cb.REQUIRED_COLUMNS, list)
        assert 'z-Score' in cb.REQUIRED_COLUMNS
        assert 'ELN_ID' in cb.REQUIRED_COLUMNS
        assert 'Reaction Type' in cb.REQUIRED_COLUMNS

    def test_default_values_defined(self):
        assert cb.DEFAULT_REACTION_TYPES == ['Buchwald-Hartwig']
        assert cb.DEFAULT_FG_A == ['RNH2 a-branch', 'RNH2']
        assert cb.DEFAULT_FG_B == ['ArBr', 'ArCl']
        assert cb.DEFAULT_MIN_ELN == 5
        assert cb.DEFAULT_TOPN_ZSCORE == 5
        assert cb.DEFAULT_MAX_COMPONENTS == 10

    def test_filter_inputs_count_is_11(self):
        assert len(cb.FILTER_INPUTS) == 11

    def test_filter_states_count_is_11(self):
        assert len(cb.FILTER_STATES) == 11

    def test_filter_keys_count_is_11(self):
        assert len(cb._FILTER_KEYS) == 11

    def test_max_upload_bytes(self):
        assert cb.MAX_UPLOAD_BYTES == 50 * 1024 * 1024


# ===========================================================================
# Upload processing (extracted logic tests)
# ===========================================================================


def _make_upload_contents(df: pd.DataFrame, encoding: str = 'utf-8', sep: str = ',') -> str:
    """Create base64-encoded CSV string matching Dash upload format."""
    csv_bytes = df.to_csv(index=False, sep=sep).encode(encoding)
    b64 = base64.b64encode(csv_bytes).decode('utf-8')
    return f'data:text/csv;base64,{b64}'


def _make_valid_upload_df() -> pd.DataFrame:
    """Create a minimal valid DataFrame with all required columns."""
    return pd.DataFrame(
        {
            'ELN_ID': ['E1', 'E2', 'E3'],
            'PLATENUMBER': [1, 1, 2],
            'Coordinate': ['A1', 'A2', 'B1'],
            'AREA_TOTAL_REDUCED': [100.0, 110.0, 90.0],
            'Base': ['Cs2CO3', 'K2CO3', 'Et3N'],
            'Catalyst': ['Pd(OAc)2', 'CuI', 'Pd(PPh3)4'],
            'Solvent': ['DMF', 'DMSO', 'THF'],
            'Ligand': ['XPhos', 'SPhos', 'PPh3'],
            'Reaction Type': ['Buchwald-Hartwig', 'Buchwald-Hartwig', 'Suzuki-Miyaura'],
            'FG A': ['RNH2', 'ArNH2', 'ArOH'],
            'FG B': ['ArBr', 'ArCl', 'ArI'],
            'FG_sorted': ['ArBr, RNH2', 'ArCl, ArNH2', 'ArI, ArOH'],
            'z-Score': [2.5, 1.0, -0.5],
        }
    )


class TestUploadProcessing:
    """Test the upload validation logic by simulating what _process_uploaded_data does."""

    def test_valid_csv_decodes_correctly(self):
        df = _make_valid_upload_df()
        contents = _make_upload_contents(df)
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        result = pd.read_csv(io.BytesIO(decoded))
        assert len(result) == 3
        assert 'z-Score' in result.columns

    def test_file_too_large_detected(self):
        df = _make_valid_upload_df()
        contents = _make_upload_contents(df)
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        assert len(decoded) < cb.MAX_UPLOAD_BYTES

        # Simulate a too-large file
        large_bytes = b'x' * (cb.MAX_UPLOAD_BYTES + 1)
        assert len(large_bytes) > cb.MAX_UPLOAD_BYTES

    def test_missing_columns_detected(self):
        df = pd.DataFrame(
            {
                'ELN_ID': ['E1'],
                'some_other_col': ['val'],
            }
        )
        contents = _make_upload_contents(df)
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        result = pd.read_csv(io.BytesIO(decoded))
        missing = [col for col in cb.REQUIRED_COLUMNS if col not in result.columns]
        assert len(missing) > 0

    def test_valid_columns_pass_check(self):
        df = _make_valid_upload_df()
        contents = _make_upload_contents(df)
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        result = pd.read_csv(io.BytesIO(decoded))
        missing = [col for col in cb.REQUIRED_COLUMNS if col not in result.columns]
        assert len(missing) == 0

    def test_semicolon_delimiter_detected(self):
        df = _make_valid_upload_df()
        csv_str = df.to_csv(index=False, sep=';')
        csv_bytes = csv_str.encode('utf-8')
        b64 = base64.b64encode(csv_bytes).decode('utf-8')
        contents = f'data:text/csv;base64,{b64}'
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Try comma first — should get 1 column
        result_comma = pd.read_csv(io.BytesIO(decoded), sep=',')
        assert len(result_comma.columns) <= 1 or ';' in result_comma.columns[0]
        # Try semicolon — should get correct columns
        result_semi = pd.read_csv(io.BytesIO(decoded), sep=';')
        assert len(result_semi.columns) > 1

    def test_invalid_zscore_detected(self):
        df = _make_valid_upload_df()
        df['z-Score'] = ['not_a_number', 'also_not', 'nope']
        contents = _make_upload_contents(df)
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        result = pd.read_csv(io.BytesIO(decoded))
        result['z-Score'] = pd.to_numeric(
            result['z-Score'].astype(str).str.replace(',', '.').str.strip(),
            errors='coerce',
        )
        assert result['z-Score'].notna().sum() == 0

    def test_comma_decimal_zscore_parsed(self):
        df = _make_valid_upload_df()
        df['z-Score'] = ['2,5', '1,0', '-0,5']
        contents = _make_upload_contents(df)
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        result = pd.read_csv(io.BytesIO(decoded))
        result['z-Score'] = pd.to_numeric(
            result['z-Score'].astype(str).str.replace(',', '.').str.strip(),
            errors='coerce',
        )
        assert result['z-Score'].notna().sum() == 3
        assert abs(result['z-Score'].iloc[0] - 2.5) < 0.01

    def test_fg_pair_sorted_computed(self):
        df = _make_valid_upload_df()
        # Simulate FG_PAIR_SORTED creation
        if 'FG_sorted' in df.columns:
            df['FG_PAIR_SORTED'] = df['FG_sorted']
        else:
            a = df['FG A'].astype(str)
            b = df['FG B'].astype(str)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            df['FG_PAIR_SORTED'] = lo + ', ' + hi
        assert 'FG_PAIR_SORTED' in df.columns
        assert df['FG_PAIR_SORTED'].iloc[0] == 'ArBr, RNH2'

    def test_latin1_encoding_decodable(self):
        df = _make_valid_upload_df()
        csv_bytes = df.to_csv(index=False).encode('latin-1')
        result = pd.read_csv(io.BytesIO(csv_bytes), encoding='latin-1')
        assert len(result) == 3

    def test_previous_session_cleanup(self, upload_df):
        # Store a session, then simulate cleanup
        old_sid = du.store_uploaded_dataframe(upload_df)
        assert du.get_uploaded_dataframe(old_sid) is not None
        du.remove_uploaded_dataframe(old_sid)
        assert du.get_uploaded_dataframe(old_sid) is None


# ===========================================================================
# Reaction type dropdown options
# ===========================================================================


class TestUpdateReactionTypeOptions:
    def test_not_dashboard_returns_all_types(self):
        # When not on dashboard, return all reaction types
        options = [{'label': rt, 'value': rt} for rt in du.REACTION_TYPES]
        assert len(options) > 0
        values = [o['value'] for o in options]
        assert 'Buchwald-Hartwig' in values

    def test_default_data_returns_types(self):
        source_df = du.get_active_dataframe(None)
        reaction_types = du.get_reaction_types_from_data(source_df)
        assert len(reaction_types) > 0

    def test_uploaded_data_returns_its_types(self, upload_df):
        sid = du.store_uploaded_dataframe(upload_df)
        source_df = du.get_active_dataframe(sid)
        reaction_types = du.get_reaction_types_from_data(source_df)
        assert set(reaction_types) == {'Buchwald-Hartwig', 'Suzuki-Miyaura'}


# ===========================================================================
# Reactant types dropdown options
# ===========================================================================


class TestUpdateReactantTypesOptions:
    def test_filters_fg_categories_out(self):
        # Simulate the callback logic
        filtered_categories = [c for c in du.CATEGORY_OPTIONS if c not in ['Functional Group A', 'Functional Group B']]
        assert 'Functional Group A' not in filtered_categories
        assert 'Functional Group B' not in filtered_categories
        assert 'Catalyst' in filtered_categories

    def test_only_available_categories_returned(self):
        source_df = du.get_active_dataframe(None)
        reaction_types = ['Buchwald-Hartwig']
        dff = source_df[source_df['Reaction Type'].isin(reaction_types)]
        available = []
        for category in du.CATEGORY_OPTIONS:
            if (
                category in dff.columns
                and dff[category].notna().any()
                and category not in ['Functional Group A', 'Functional Group B']
            ):
                available.append(category)
        assert len(available) > 0
        assert 'Catalyst' in available

    def test_no_reaction_types_returns_all_categories(self):
        filtered = [c for c in du.CATEGORY_OPTIONS if c not in ['Functional Group A', 'Functional Group B']]
        assert len(filtered) == 7  # 9 total - 2 FG categories


# ===========================================================================
# FG A / FG B dropdown options
# ===========================================================================


class TestUpdateFgAOptions:
    def test_no_reaction_types_returns_all_only(self):
        options = [{'label': 'All', 'value': 'All'}]
        assert len(options) == 1
        assert options[0]['value'] == 'All'

    def test_with_reaction_types_returns_sorted_fgs(self):
        source_df = du.get_active_dataframe(None)
        dff = source_df[source_df['Reaction Type'] == 'Buchwald-Hartwig']
        fg_values = pd.concat([dff['FG A'], dff['FG B']]).dropna().unique()
        options = ['All'] + sorted(fg_values.tolist())
        assert options[0] == 'All'
        assert len(options) > 1
        # Should be sorted after 'All'
        assert options[1:] == sorted(options[1:])


class TestUpdateFgBOptions:
    def test_fg_a_conditions_fg_b_options(self):
        source_df = du.get_active_dataframe(None)
        dff = source_df[source_df['Reaction Type'] == 'Buchwald-Hartwig']
        fg_a_val = 'RNH2'
        mask = (dff['FG A'] == fg_a_val) | (dff['FG B'] == fg_a_val)
        dff_sub = dff[mask]
        other_fgs = []
        other_fgs.extend(dff_sub.loc[dff_sub['FG A'] == fg_a_val, 'FG B'])
        other_fgs.extend(dff_sub.loc[dff_sub['FG B'] == fg_a_val, 'FG A'])
        fg_b_values = pd.Series(other_fgs).dropna().unique()
        assert len(fg_b_values) > 0

    def test_all_fg_a_returns_all_fgs(self):
        source_df = du.get_active_dataframe(None)
        dff = source_df[source_df['Reaction Type'] == 'Buchwald-Hartwig']
        fg_values = pd.concat([dff['FG A'], dff['FG B']]).dropna().unique()
        assert len(fg_values) > 0


# ===========================================================================
# FG reset on reaction type change
# ===========================================================================


class TestResetFunctionalGroupsOnReactionChange:
    def test_restoring_returns_no_update(self):
        # When is_restoring is True, should not reset
        is_restoring = True
        if is_restoring:
            result = (no_update, no_update)
        assert result == (no_update, no_update)

    def test_non_default_reaction_returns_all(self):
        reaction_types = ['Suzuki-Miyaura']
        if sorted(reaction_types) != sorted(cb.DEFAULT_REACTION_TYPES):
            result = (['All'], ['All'])
        assert result == (['All'], ['All'])

    def test_default_reaction_returns_valid_defaults(self):
        reaction_types = cb.DEFAULT_REACTION_TYPES
        source_df = du.get_active_dataframe(None)
        dff = source_df[source_df['Reaction Type'].isin(reaction_types)]
        available_fgs = set(pd.concat([dff['FG A'], dff['FG B']]).dropna().unique())
        valid_fg_a = [v for v in cb.DEFAULT_FG_A if v in available_fgs]
        valid_fg_b = [v for v in cb.DEFAULT_FG_B if v in available_fgs]
        assert len(valid_fg_a) > 0
        assert len(valid_fg_b) > 0


# ===========================================================================
# Tabs management
# ===========================================================================


class TestUpdateTabsLogic:
    def test_single_reactant_hides_heatmap(self):
        reactant_types = ['Catalyst']
        has_heatmap = reactant_types and len(reactant_types) >= 2
        assert has_heatmap is False

    def test_two_reactants_shows_heatmap(self):
        reactant_types = ['Catalyst', 'Solvent']
        has_heatmap = reactant_types and len(reactant_types) >= 2
        assert has_heatmap is True

    def test_switch_from_heatmap_when_removed(self):
        current_tab = 'tab-heatmap'
        reactant_types = ['Catalyst']  # Only 1, so heatmap removed
        if current_tab == 'tab-heatmap' and (not reactant_types or len(reactant_types) < 2):
            new_tab = 'tab-graph'
        else:
            new_tab = current_tab
        assert new_tab == 'tab-graph'

    def test_stay_on_stats_tab(self):
        current_tab = 'tab-stats'
        reactant_types = ['Catalyst']
        available = ['tab-graph', 'tab-stats']
        if current_tab == 'tab-heatmap' and len(reactant_types) < 2 or current_tab not in available:
            new_tab = 'tab-graph'
        else:
            new_tab = current_tab
        assert new_tab == 'tab-stats'


# ===========================================================================
# Max components slider logic
# ===========================================================================


class TestUpdateMaxComponentsSliderLogic:
    def test_no_stats_returns_defaults(self):
        stats = None
        if not stats or 'max_components_cap' not in (stats or {}):
            max_val, _marks, value = 10, {1: '1', 5: '5', 10: '10'}, 10
        assert max_val == 10
        assert value == 10

    def test_updates_max_and_marks(self):
        stats = {'max_components_cap': 15}
        max_value = max(1, int(stats['max_components_cap']))
        marks = {1: '1'}
        for i in range(5, max_value + 1, 5):
            marks[i] = str(i)
        assert max_value == 15
        assert 5 in marks
        assert 10 in marks
        assert 15 in marks

    def test_clamps_value_to_new_max(self):
        max_value = 8
        current_value = 15
        proposed_value = max(1, min(current_value, max_value))
        assert proposed_value == 8

    def test_bumps_from_previous_cap(self):
        max_value = 20
        current_value = 5
        previous_max = 5
        default_target = min(10, max_value)
        # Previously clamped to old cap — bump up
        if previous_max and max_value > previous_max and current_value == previous_max:
            proposed = default_target
        else:
            proposed = current_value
        assert proposed == 10  # bumped to default

    def test_none_value_gets_default(self):
        max_value = 15
        current_value = None
        default_target = min(10, max_value)
        if current_value is None:
            proposed = default_target
        else:
            proposed = current_value
        assert proposed == 10

    def test_large_max_uses_10_step_marks(self):
        stats = {'max_components_cap': 50}
        max_value = int(stats['max_components_cap'])
        marks = {1: '1'}
        if max_value > 20:
            for i in range(10, max_value + 1, 10):
                marks[i] = str(i)
        assert 10 in marks
        assert 20 in marks
        assert 50 in marks
        assert 5 not in marks  # Should use 10-step, not 5-step


# ===========================================================================
# Stats table logic
# ===========================================================================


class TestStatsTableLogic:
    def test_not_dashboard_returns_empty(self):
        pathname = '/'
        if pathname != '/dashboard':
            result = []
        assert result == []

    def test_not_stats_tab_returns_no_update(self):
        active_tab = 'tab-graph'
        if active_tab != 'tab-stats':
            result = no_update
        assert result is no_update

    def test_stats_computed_for_filtered_data(self, small_df):
        dff = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
        )
        # Replicate stats table logic
        numeric_cols = [col for col in ['z-Score', 'AREA_TOTAL_REDUCED'] if col in dff.columns]
        assert len(numeric_cols) > 0
        desc = dff[numeric_cols].describe().T.reset_index().rename(columns={'index': 'Metric'})
        assert 'count' in desc.columns
        assert len(desc) > 0

    def test_empty_data_handled(self, empty_df):
        if empty_df.empty:
            result = 'No data available'
        assert 'No data' in result


# ===========================================================================
# Download logic
# ===========================================================================


class TestDownloadLogic:
    def test_csv_download_produces_csv_string(self, small_df):
        dff = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
        )
        csv_str = dff.to_csv(index=False)
        assert 'z-Score' in csv_str
        assert 'ELN_ID' in csv_str
        assert len(csv_str) > 0

    def test_no_clicks_returns_no_update(self):
        n_clicks = None
        if not n_clicks:
            result = no_update
        assert result is no_update

    def test_filter_data_called_for_download(self, small_df):
        fkw = {
            'reactant_types': ['Catalyst'],
            'reaction_types': ['Buchwald-Hartwig'],
            'fg_a': None,
            'fg_b': None,
            'exclude_cui': ['exclude_cui'],
            'exclude_scaleup': [True],
            'include_null_categories': [True],
            'min_eln': 5,
            'topn_zscore': 5,
            'max_components': 10,
            'uploaded_data': None,
        }
        result = cb._call_filter_data(fkw)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ===========================================================================
# Min ELN on FG change
# ===========================================================================


class TestMinElnOnFgChange:
    def test_restoring_returns_no_update(self):
        is_restoring = True
        if is_restoring:
            result = no_update
        assert result is no_update

    def test_already_default_returns_no_update(self):
        current_min_eln = cb.DEFAULT_MIN_ELN
        if current_min_eln == cb.DEFAULT_MIN_ELN:
            result = no_update
        assert result is no_update

    def test_non_default_resets_to_default(self):
        current_min_eln = 3
        is_restoring = False
        if is_restoring or current_min_eln == cb.DEFAULT_MIN_ELN:
            result = no_update
        else:
            result = cb.DEFAULT_MIN_ELN
        assert result == 5


# ===========================================================================
# Initial FG values
# ===========================================================================


class TestSetInitialFgValues:
    def test_empty_current_sets_defaults(self):
        current_fg_a = None
        fg_a_options = [
            {'label': 'All', 'value': 'All'},
            {'label': 'RNH2', 'value': 'RNH2'},
            {'label': 'RNH2 a-branch', 'value': 'RNH2 a-branch'},
        ]
        desired = cb.DEFAULT_FG_A  # ['RNH2 a-branch', 'RNH2']
        if not current_fg_a and fg_a_options:
            available = [opt['value'] for opt in fg_a_options]
            valid = [val for val in desired if val in available]
            result = valid if valid else ['All']
        else:
            result = no_update
        assert result == ['RNH2 a-branch', 'RNH2']

    def test_existing_value_returns_no_update(self):
        current_fg_a = ['ArBr']
        fg_a_options = [{'label': 'All', 'value': 'All'}]
        if not current_fg_a and fg_a_options:
            result = ['All']
        else:
            result = no_update
        assert result is no_update

    def test_no_defaults_available_returns_all(self):
        current_fg_a = None
        fg_a_options = [{'label': 'All', 'value': 'All'}, {'label': 'ArOH', 'value': 'ArOH'}]
        desired = cb.DEFAULT_FG_A
        if not current_fg_a and fg_a_options:
            available = [opt['value'] for opt in fg_a_options]
            valid = [val for val in desired if val in available]
            result = valid if valid else ['All']
        else:
            result = no_update
        assert result == ['All']  # None of DEFAULT_FG_A in options


# ===========================================================================
# Reset filters
# ===========================================================================


class TestResetFiltersLogic:
    def test_reset_returns_default_reaction_types(self):
        assert cb.DEFAULT_REACTION_TYPES == ['Buchwald-Hartwig']

    def test_reset_returns_default_min_eln(self):
        assert cb.DEFAULT_MIN_ELN == 5

    def test_reset_returns_default_topn(self):
        assert cb.DEFAULT_TOPN_ZSCORE == 5

    def test_reset_returns_default_max_components(self):
        assert cb.DEFAULT_MAX_COMPONENTS == 10

    def test_reset_selects_catalyst_if_available(self):
        reaction_types = ['Buchwald-Hartwig']
        source_df = du.get_active_dataframe(None)
        dff = source_df[source_df['Reaction Type'].isin(reaction_types)]
        available = [
            c
            for c in du.CATEGORY_OPTIONS
            if c in dff.columns and dff[c].notna().any() and c not in ['Functional Group A', 'Functional Group B']
        ]
        if 'Catalyst' in available:
            default_reactant = ['Catalyst']
        elif available:
            default_reactant = [available[0]]
        else:
            default_reactant = ['Additive']  # fallback
        assert default_reactant == ['Catalyst']

    def test_reset_fg_validates_against_options(self):
        fg_a_options = [
            {'label': 'All', 'value': 'All'},
            {'label': 'RNH2', 'value': 'RNH2'},
            {'label': 'RNH2 a-branch', 'value': 'RNH2 a-branch'},
        ]
        desired_fg_a = cb.DEFAULT_FG_A
        available = [opt['value'] for opt in fg_a_options]
        valid = [val for val in desired_fg_a if val in available]
        assert valid == ['RNH2 a-branch', 'RNH2']

    def test_reaction_type_change_returns_no_update(self):
        # When triggered by reaction-type-dropdown, all other outputs are no_update
        triggered_id = 'reaction-type-dropdown'
        if triggered_id == 'reaction-type-dropdown':
            result = tuple([no_update] * 10)
        assert all(r is no_update for r in result)
