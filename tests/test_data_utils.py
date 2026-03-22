"""Tests for data_utils.py — filter chain, cache, upload store, statistics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import data_utils as du

# ===========================================================================
# Utility functions
# ===========================================================================


class TestConvertCheckboxToBool:
    def test_none_returns_false(self):
        assert du._convert_checkbox_to_bool(None) is False

    def test_empty_list_returns_false(self):
        assert du._convert_checkbox_to_bool([]) is False

    def test_nonempty_list_returns_true(self):
        assert du._convert_checkbox_to_bool(['exclude_cui']) is True

    def test_single_item_list_returns_true(self):
        assert du._convert_checkbox_to_bool([True]) is True

    def test_string_returns_true(self):
        assert du._convert_checkbox_to_bool('yes') is True


class TestCreateCacheKey:
    def test_same_args_same_key(self):
        k1 = du._create_cache_key('a', 'b', 1)
        k2 = du._create_cache_key('a', 'b', 1)
        assert k1 == k2

    def test_different_args_different_key(self):
        k1 = du._create_cache_key('a', 'b')
        k2 = du._create_cache_key('a', 'c')
        assert k1 != k2

    def test_order_matters(self):
        k1 = du._create_cache_key('a', 'b')
        k2 = du._create_cache_key('b', 'a')
        assert k1 != k2

    def test_none_args_produce_valid_key(self):
        k = du._create_cache_key(None, None)
        assert isinstance(k, str) and len(k) == 32

    def test_key_is_md5_hex_string(self):
        k = du._create_cache_key('test')
        assert len(k) == 32
        assert all(c in '0123456789abcdef' for c in k)


class TestNormalizeFgInput:
    def test_none_returns_empty_list(self):
        assert du._normalize_fg_input(None) == []

    def test_empty_string_returns_empty_list(self):
        assert du._normalize_fg_input('') == []

    def test_empty_list_returns_empty_list(self):
        assert du._normalize_fg_input([]) == []

    def test_single_string_returns_list(self):
        assert du._normalize_fg_input('RNH2') == ['RNH2']

    def test_all_string_returns_empty_list(self):
        assert du._normalize_fg_input('All') == []

    def test_list_with_all_filters_it_out(self):
        assert du._normalize_fg_input(['All', 'RNH2']) == ['RNH2']

    def test_list_without_all_passes_through(self):
        assert du._normalize_fg_input(['RNH2', 'ArBr']) == ['RNH2', 'ArBr']

    def test_mixed_list_filters_all_only(self):
        result = du._normalize_fg_input(['All', 'RNH2', 'All'])
        assert result == ['RNH2']

    def test_non_string_non_list_returns_empty(self):
        assert du._normalize_fg_input(123) == []


class TestMaskContainsFg:
    def test_match_in_fg_a_column(self, small_df):
        mask = du._mask_contains_fg(small_df, 'RNH2')
        assert mask.any()
        assert small_df.loc[mask, 'FG A'].eq('RNH2').any()

    def test_match_in_fg_b_column(self, small_df):
        mask = du._mask_contains_fg(small_df, 'ArBr')
        assert mask.any()
        assert small_df.loc[mask, 'FG B'].eq('ArBr').any()

    def test_no_match_returns_all_false(self, small_df):
        mask = du._mask_contains_fg(small_df, 'NONEXISTENT_FG')
        assert not mask.any()

    def test_handles_nan_values(self, small_df):
        # Should not raise even with NaN values
        df = small_df.copy()
        df.loc[0, 'FG A'] = np.nan
        mask = du._mask_contains_fg(df, 'RNH2')
        assert isinstance(mask, pd.Series)


class TestFillnaSafe:
    def test_regular_series(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = du._fillna_safe(s, 0)
        assert result.isna().sum() == 0

    def test_categorical_series(self):
        s = pd.Series(pd.Categorical(['a', None, 'b']))
        result = du._fillna_safe(s, 'FILL')
        assert result.isna().sum() == 0
        assert (result == 'FILL').sum() == 1

    def test_dataframe_with_categorical_columns(self):
        df = pd.DataFrame(
            {
                'cat_col': pd.Categorical(['a', None, 'b']),
                'str_col': ['x', None, 'z'],
            }
        )
        result = du._fillna_safe(df, 'FILL')
        assert result.isna().sum().sum() == 0

    def test_dataframe_without_categorical_columns(self):
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': ['x', None]})
        result = du._fillna_safe(df, 'FILL')
        assert result.isna().sum().sum() == 0


# ===========================================================================
# Individual filter steps
# ===========================================================================


class TestFilterByReactionTypes:
    def test_none_returns_all_rows(self, small_df):
        result = du._filter_by_reaction_types(small_df, None)
        assert len(result) == len(small_df)

    def test_empty_list_returns_all_rows(self, small_df):
        result = du._filter_by_reaction_types(small_df, [])
        assert len(result) == len(small_df)

    def test_single_type_filters_correctly(self, small_df):
        result = du._filter_by_reaction_types(small_df, ['Buchwald-Hartwig'])
        assert len(result) > 0
        assert set(result['Reaction Type'].dropna().unique()) == {'Buchwald-Hartwig'}

    def test_multiple_types_union(self, small_df):
        result = du._filter_by_reaction_types(small_df, ['Buchwald-Hartwig', 'Suzuki-Miyaura'])
        assert len(result) > 0
        types = set(result['Reaction Type'].dropna().unique())
        assert types <= {'Buchwald-Hartwig', 'Suzuki-Miyaura'}

    def test_nonexistent_type_returns_empty(self, small_df):
        result = du._filter_by_reaction_types(small_df, ['NONEXISTENT'])
        assert len(result) == 0


class TestFilterByReactantColumns:
    def test_none_reactant_types_returns_all(self, small_df):
        result = du._filter_by_reactant_columns(small_df, None, include_null=False)
        assert len(result) == len(small_df)

    def test_include_null_true_returns_all(self, small_df):
        result = du._filter_by_reactant_columns(small_df, ['Catalyst'], include_null=True)
        assert len(result) == len(small_df)

    def test_filters_out_null_values(self, small_df):
        result = du._filter_by_reactant_columns(small_df, ['Catalyst'], include_null=False)
        # Should have fewer rows since some have null Catalyst
        assert result['Catalyst'].isna().sum() == 0

    def test_multiple_columns_all_must_be_populated(self, small_df):
        result = du._filter_by_reactant_columns(small_df, ['Catalyst', 'Ligand'], include_null=False)
        assert result['Catalyst'].isna().sum() == 0
        assert result['Ligand'].isna().sum() == 0


class TestFilterExcludeCui:
    def test_none_returns_all(self, small_df):
        result = du._filter_exclude_cui(small_df, None)
        assert len(result) == len(small_df)

    def test_empty_list_returns_all(self, small_df):
        result = du._filter_exclude_cui(small_df, [])
        assert len(result) == len(small_df)

    def test_exclude_cui_removes_cui_rows(self, small_df):
        original_cui_count = (small_df['Catalyst'] == 'CuI').sum()
        assert original_cui_count > 0, 'Test data should contain CuI rows'
        result = du._filter_exclude_cui(small_df, ['exclude_cui'])
        assert (result['Catalyst'] == 'CuI').sum() == 0
        assert len(result) < len(small_df)

    def test_preserves_null_catalyst_rows(self, small_df):
        result = du._filter_exclude_cui(small_df, ['exclude_cui'])
        # Null catalyst rows should be preserved
        assert result['Catalyst'].isna().any()


class TestFilterFgA:
    def test_none_returns_all(self, small_df):
        result, fg_list = du._filter_fg_a(small_df, None)
        assert len(result) == len(small_df)
        assert fg_list == []

    def test_all_returns_all(self, small_df):
        result, fg_list = du._filter_fg_a(small_df, 'All')
        assert len(result) == len(small_df)
        assert fg_list == []

    def test_single_fg_filters(self, small_df):
        result, fg_list = du._filter_fg_a(small_df, ['RNH2'])
        assert len(result) > 0
        assert len(result) < len(small_df)
        assert fg_list == ['RNH2']
        # Every row should have RNH2 in FG A or FG B
        for _, row in result.iterrows():
            assert row['FG A'] == 'RNH2' or row['FG B'] == 'RNH2'

    def test_multiple_fg_union(self, small_df):
        result, fg_list = du._filter_fg_a(small_df, ['RNH2', 'ArNH2'])
        assert len(result) > 0
        assert fg_list == ['RNH2', 'ArNH2']

    def test_nonexistent_fg_returns_empty(self, small_df):
        result, fg_list = du._filter_fg_a(small_df, ['NONEXISTENT_FG'])
        assert len(result) == 0


class TestFilterFgB:
    def test_none_returns_all(self, small_df):
        result, fg_list = du._filter_fg_b(small_df, None, [])
        assert len(result) == len(small_df)
        assert fg_list == []

    def test_with_fg_a_list_matches_pairs(self, small_df):
        result, fg_list = du._filter_fg_b(small_df, ['ArBr'], ['RNH2'])
        assert len(result) > 0
        # Rows should have FG_PAIR_SORTED matching the pair
        expected_pair = ', '.join(sorted(['RNH2', 'ArBr']))
        assert (result['FG_PAIR_SORTED'] == expected_pair).all()

    def test_without_fg_a_list_filters_both_columns(self, small_df):
        result, fg_list = du._filter_fg_b(small_df, ['ArBr'], [])
        assert len(result) > 0
        # Every row should have ArBr in FG A or FG B
        for _, row in result.iterrows():
            assert row['FG A'] == 'ArBr' or row['FG B'] == 'ArBr'

    def test_returns_normalized_fg_list(self, small_df):
        _, fg_list = du._filter_fg_b(small_df, ['All', 'ArBr'], [])
        assert fg_list == ['ArBr']


class TestFilterScaleupPlates:
    def test_none_returns_all(self, small_df):
        result = du._filter_scaleup_plates(small_df, None)
        assert len(result) == len(small_df)

    def test_false_returns_all(self, small_df):
        result = du._filter_scaleup_plates(small_df, [])
        assert len(result) == len(small_df)

    def test_removes_plates_with_no_variability(self, small_df):
        result = du._filter_scaleup_plates(small_df, [True])
        # The SCALEUP plate should be removed (no reagent variability)
        assert 'ELN-SCALEUP' not in result['ELN_ID'].values

    def test_keeps_plates_with_variability(self, small_df):
        result = du._filter_scaleup_plates(small_df, [True])
        # Regular plates with variability should be kept
        assert len(result) > 0


class TestDeduplicateBestZscore:
    def test_keeps_highest_zscore_per_combo(self, minimal_df):
        result = du._deduplicate_best_zscore(minimal_df)
        # Should have fewer or equal rows
        assert len(result) <= len(minimal_df)

    def test_single_row_per_combo_unchanged(self):
        df = pd.DataFrame(
            {
                'ELN_ID': pd.Categorical(['E1', 'E2']),
                'Catalyst': pd.Categorical(['A', 'B']),
                'z-Score': [1.0, 2.0],
            }
        )
        result = du._deduplicate_best_zscore(df)
        assert len(result) == 2


class TestFilterTopnZscore:
    def test_none_topn_returns_all(self, small_df):
        result = du._filter_topn_zscore(small_df, None, ['Catalyst'], include_null=False)
        assert len(result) == len(small_df)

    def test_none_reactant_types_returns_all(self, small_df):
        result = du._filter_topn_zscore(small_df, 3, None, include_null=False)
        assert len(result) == len(small_df)

    def test_topn_1_keeps_best_per_group(self, small_df):
        result = du._filter_topn_zscore(small_df, 1, ['Catalyst'], include_null=False)
        assert len(result) < len(small_df)
        assert len(result) > 0

    def test_topn_limits_correctly(self, small_df):
        result_1 = du._filter_topn_zscore(small_df, 1, ['Catalyst'], include_null=False)
        result_3 = du._filter_topn_zscore(small_df, 3, ['Catalyst'], include_null=False)
        assert len(result_1) <= len(result_3)


class TestFilterMinEln:
    def test_none_min_eln_returns_all(self, small_df):
        result = du._filter_min_eln(small_df, None, ['Catalyst'], include_null=False)
        assert len(result) == len(small_df)

    def test_none_reactant_types_returns_all(self, small_df):
        result = du._filter_min_eln(small_df, 5, None, include_null=False)
        assert len(result) == len(small_df)

    def test_min_eln_1_keeps_most(self, small_df):
        result = du._filter_min_eln(small_df, 1, ['Catalyst'], include_null=False)
        # min_eln=1 keeps all groups with at least 1 ELN, but null catalyst rows
        # are dropped when include_null=False
        assert len(result) > 0
        assert len(result) >= len(small_df) * 0.9

    def test_min_eln_filters_sparse_groups(self, small_df):
        # High min_eln should filter out groups with few ELNs
        result = du._filter_min_eln(small_df, 10, ['Catalyst'], include_null=False)
        assert len(result) <= len(small_df)


class TestFilterMaxComponents:
    def test_none_returns_all(self, small_df):
        result = du._filter_max_components(small_df, None, ['Catalyst'], include_null=False)
        assert len(result) == len(small_df)

    def test_zero_returns_all(self, small_df):
        result = du._filter_max_components(small_df, 0, ['Catalyst'], include_null=False)
        assert len(result) == len(small_df)

    def test_negative_returns_all(self, small_df):
        result = du._filter_max_components(small_df, -1, ['Catalyst'], include_null=False)
        assert len(result) == len(small_df)

    def test_none_reactant_types_returns_all(self, small_df):
        result = du._filter_max_components(small_df, 3, None, include_null=False)
        assert len(result) == len(small_df)

    def test_max_exceeds_unique_returns_all(self, small_df):
        n_unique = small_df['Catalyst'].nunique()
        result = du._filter_max_components(small_df, n_unique + 10, ['Catalyst'], include_null=False)
        assert len(result) == len(small_df)

    def test_limits_by_median_zscore(self, small_df):
        result = du._filter_max_components(small_df, 2, ['Catalyst'], include_null=False)
        assert result['Catalyst'].nunique() <= 2
        assert len(result) < len(small_df)

    def test_multiple_key_cols(self, small_df):
        result = du._filter_max_components(small_df, 3, ['Catalyst', 'Solvent'], include_null=False)
        assert len(result) > 0


# ===========================================================================
# Full filter_data()
# ===========================================================================


class TestFilterData:
    def test_no_args_returns_dataframe(self, small_df):
        result = du.filter_data(source_df=small_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_return_stats_true_returns_tuple(self, small_df):
        result = du.filter_data(source_df=small_df, return_stats=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)

    def test_return_stats_false_returns_df(self, small_df):
        result = du.filter_data(source_df=small_df, return_stats=False)
        assert isinstance(result, pd.DataFrame)

    def test_stats_dict_has_expected_keys(self, small_df):
        _, stats = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            fg_a=['RNH2'],
            fg_b=['ArBr'],
            return_stats=True,
        )
        assert 'whole_dataset' in stats
        assert 'elns' in stats['whole_dataset']

    def test_reaction_type_filter(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
        )
        assert all(result['Reaction Type'] == 'Buchwald-Hartwig')

    def test_cache_hit_returns_copy(self, small_df):
        r1 = du.filter_data(source_df=small_df, reaction_types=['Buchwald-Hartwig'])
        r2 = du.filter_data(source_df=small_df, reaction_types=['Buchwald-Hartwig'])
        assert r1 is not r2
        pd.testing.assert_frame_equal(r1, r2)

    def test_cache_miss_after_different_params(self, small_df):
        r1 = du.filter_data(source_df=small_df, reaction_types=['Buchwald-Hartwig'])
        r2 = du.filter_data(source_df=small_df, reaction_types=['Suzuki-Miyaura'])
        assert len(r1) != len(r2)

    def test_source_df_overrides_default(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
        )
        # Result should come from small_df, not du.DF
        assert set(result['ELN_ID'].unique()) <= set(small_df['ELN_ID'].unique())

    def test_full_filter_chain(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reactant_types=['Catalyst'],
            reaction_types=['Buchwald-Hartwig'],
            exclude_cui=['exclude_cui'],
            include_null_categories=[True],
            min_eln=1,
            topn_zscore=10,
            max_components=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert (result['Catalyst'] == 'CuI').sum() == 0

    def test_thread_safety_concurrent_calls(self, small_df):
        def call_filter(rt):
            return len(du.filter_data(source_df=small_df, reaction_types=[rt]))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(call_filter, rt) for rt in ['Buchwald-Hartwig', 'Suzuki-Miyaura', 'C-H Activation'] * 3
            ]
            results = [f.result() for f in futures]
        assert all(r > 0 for r in results)


# ===========================================================================
# Cache
# ===========================================================================


class TestFilterCache:
    def test_clear_filter_cache_empties_cache(self, small_df):
        du.filter_data(source_df=small_df, reaction_types=['Buchwald-Hartwig'])
        assert du.get_cache_info()['cache_size'] > 0
        du.clear_filter_cache()
        assert du.get_cache_info()['cache_size'] == 0

    def test_get_cache_info_returns_dict(self):
        info = du.get_cache_info()
        assert isinstance(info, dict)
        assert 'cache_size' in info
        assert 'max_size' in info

    def test_get_cache_info_max_size_is_50(self):
        info = du.get_cache_info()
        assert info['max_size'] == 50

    def test_lru_eviction_order(self, small_df):
        # Fill cache beyond max size to test eviction
        original_max = du._CACHE_MAX_SIZE
        du._CACHE_MAX_SIZE = 3
        try:
            du.filter_data(source_df=small_df, reaction_types=['Buchwald-Hartwig'])
            du.filter_data(source_df=small_df, reaction_types=['Suzuki-Miyaura'])
            du.filter_data(source_df=small_df, reaction_types=['C-H Activation'])
            assert du.get_cache_info()['cache_size'] == 3

            # Adding a 4th should evict the oldest
            du.filter_data(source_df=small_df, reaction_types=['Buchwald-Hartwig', 'Suzuki-Miyaura'])
            assert du.get_cache_info()['cache_size'] == 3
        finally:
            du._CACHE_MAX_SIZE = original_max


# ===========================================================================
# Upload store
# ===========================================================================


class TestUploadStore:
    def test_store_returns_uuid(self, upload_df):
        sid = du.store_uploaded_dataframe(upload_df)
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID format

    def test_get_returns_stored_df(self, upload_df):
        sid = du.store_uploaded_dataframe(upload_df)
        result = du.get_uploaded_dataframe(sid)
        assert result is not None
        assert len(result) == len(upload_df)

    def test_get_nonexistent_returns_none(self):
        result = du.get_uploaded_dataframe('nonexistent-uuid')
        assert result is None

    def test_get_empty_session_id_returns_none(self):
        assert du.get_uploaded_dataframe('') is None
        assert du.get_uploaded_dataframe(None) is None

    def test_remove_clears_entry(self, upload_df):
        sid = du.store_uploaded_dataframe(upload_df)
        du.remove_uploaded_dataframe(sid)
        assert du.get_uploaded_dataframe(sid) is None

    def test_max_sessions_eviction(self, upload_df):
        sids = []
        for _ in range(du._UPLOAD_MAX_SESSIONS + 2):
            sids.append(du.store_uploaded_dataframe(upload_df.copy()))
        # Oldest sessions should have been evicted
        assert len(du._UPLOAD_STORE) <= du._UPLOAD_MAX_SESSIONS

    def test_thread_safety_concurrent_stores(self, upload_df):
        def store():
            return du.store_uploaded_dataframe(upload_df.copy())

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(store) for _ in range(8)]
            sids = [f.result() for f in futures]
        assert len(set(sids)) == 8  # All unique UUIDs


class TestGetActiveDataframe:
    def test_none_session_returns_default_df(self):
        result = du.get_active_dataframe(None)
        assert result is du.DF

    def test_valid_session_returns_uploaded_df(self, upload_df):
        sid = du.store_uploaded_dataframe(upload_df)
        result = du.get_active_dataframe(sid)
        assert len(result) == len(upload_df)

    def test_expired_session_returns_default_df(self):
        result = du.get_active_dataframe('nonexistent')
        assert result is du.DF


# ===========================================================================
# Dropdown helpers
# ===========================================================================


class TestGetReactionTypesFromData:
    def test_none_uses_default_df(self):
        result = du.get_reaction_types_from_data()
        assert len(result) > 0

    def test_custom_df_extracts_types(self, small_df):
        result = du.get_reaction_types_from_data(small_df)
        assert set(result) == {'Buchwald-Hartwig', 'Suzuki-Miyaura', 'C-H Activation'}

    def test_missing_column_returns_empty(self):
        df = pd.DataFrame({'other': [1, 2, 3]})
        assert du.get_reaction_types_from_data(df) == []


class TestGetCategoryOptionsFromData:
    def test_none_uses_default_df(self):
        result = du.get_category_options_from_data()
        assert isinstance(result, list)

    def test_returns_only_populated_columns(self, small_df):
        result = du.get_category_options_from_data(small_df)
        for cat in result:
            assert cat in small_df.columns
            assert small_df[cat].notna().any()


# ===========================================================================
# Statistical functions
# ===========================================================================


class TestComputeDistributionStats:
    def test_returns_dataframe_with_expected_columns(self, small_df):
        result = du.compute_distribution_stats(small_df, 'Catalyst', min_samples=5)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {'group', 'n', 'mean', 'std', 'skewness', 'kurtosis', 'shapiro_stat', 'shapiro_p', 'is_normal'}
        assert expected_cols <= set(result.columns)

    def test_min_samples_filter(self, small_df):
        result_low = du.compute_distribution_stats(small_df, 'Catalyst', min_samples=1)
        result_high = du.compute_distribution_stats(small_df, 'Catalyst', min_samples=100)
        assert len(result_low) >= len(result_high)

    def test_shapiro_wilk_runs(self, small_df):
        result = du.compute_distribution_stats(small_df, 'Catalyst', min_samples=5)
        if len(result) > 0:
            assert result['shapiro_p'].notna().any()


class TestComputeSignificanceTests:
    def test_returns_dict_with_expected_keys(self, small_df):
        result = du.compute_significance_tests(small_df, 'Catalyst')
        assert isinstance(result, dict)
        assert 'kruskal_wallis' in result
        assert 'n_groups' in result
        assert 'group_stats' in result

    def test_kruskal_wallis_runs(self, small_df):
        result = du.compute_significance_tests(small_df, 'Catalyst')
        kw = result['kruskal_wallis']
        assert 'statistic' in kw
        assert 'p_value' in kw
        assert 'significant' in kw

    def test_fewer_than_two_groups(self):
        df = pd.DataFrame(
            {
                'cat': ['A'] * 10,
                'z-Score': range(10),
            }
        )
        result = du.compute_significance_tests(df, 'cat')
        assert result['n_groups'] <= 1

    def test_pairwise_mann_whitney(self, small_df):
        result = du.compute_significance_tests(small_df, 'Catalyst')
        if result['n_groups'] >= 2:
            assert 'pairwise' in result
            assert isinstance(result['pairwise'], pd.DataFrame)

    def test_bonferroni_correction(self, small_df):
        result = du.compute_significance_tests(small_df, 'Catalyst')
        if result['n_groups'] >= 2:
            assert 'alpha_corrected' in result
            assert result['alpha_corrected'] <= 0.05


class TestComputePermutationTest:
    def test_returns_expected_keys(self, small_df):
        # Use small permutation count for speed
        bh = small_df[small_df['Reaction Type'] == 'Buchwald-Hartwig']
        result = du.compute_permutation_test(bh, 'Catalyst', n_permutations=100)
        expected_keys = {
            'observed_h',
            'standard_p',
            'empirical_p',
            'n_permutations',
            'permuted_h_mean',
            'permuted_h_std',
            'permuted_h_95th',
            'significant_permutation',
        }
        assert expected_keys <= set(result.keys())

    def test_reproducible_with_seed(self, small_df):
        bh = small_df[small_df['Reaction Type'] == 'Buchwald-Hartwig']
        r1 = du.compute_permutation_test(bh, 'Catalyst', n_permutations=50, random_state=42)
        r2 = du.compute_permutation_test(bh, 'Catalyst', n_permutations=50, random_state=42)
        assert r1['empirical_p'] == r2['empirical_p']

    def test_empirical_p_value_range(self, small_df):
        bh = small_df[small_df['Reaction Type'] == 'Buchwald-Hartwig']
        result = du.compute_permutation_test(bh, 'Catalyst', n_permutations=100)
        assert 0.0 <= result['empirical_p'] <= 1.0


class TestInterpretEffectSize:
    def test_negligible(self):
        assert du._interpret_effect_size(0.05) == 'negligible'

    def test_small(self):
        assert du._interpret_effect_size(0.2) == 'small'

    def test_medium(self):
        assert du._interpret_effect_size(0.4) == 'medium'

    def test_large(self):
        assert du._interpret_effect_size(0.6) == 'large'

    def test_boundary_values(self):
        assert du._interpret_effect_size(0.0) == 'negligible'
        assert du._interpret_effect_size(0.1) == 'small'
        assert du._interpret_effect_size(0.3) == 'medium'
        assert du._interpret_effect_size(0.5) == 'large'


class TestGetDistributionSummary:
    def test_returns_expected_keys(self, small_df):
        result = du.get_distribution_summary(small_df, 'Catalyst')
        expected_keys = {
            'n_groups',
            'n_normal',
            'pct_normal',
            'n_symmetric',
            'pct_symmetric',
            'n_moderate_skew',
            'n_high_skew',
            'median_skewness',
            'median_kurtosis',
        }
        assert expected_keys <= set(result.keys())

    def test_empty_data_returns_zeros(self, empty_df):
        result = du.get_distribution_summary(empty_df, 'Reaction Type')
        assert result['n_groups'] == 0
        assert result['n_normal'] == 0

    def test_skewness_categories_sum(self, small_df):
        result = du.get_distribution_summary(small_df, 'Catalyst')
        total = result['n_symmetric'] + result['n_moderate_skew'] + result['n_high_skew']
        assert total == result['n_groups']


# ===========================================================================
# Data loading
# ===========================================================================


class TestDataLoading:
    def test_load_and_prepare_creates_fg_pair_sorted(self):
        assert 'FG_PAIR_SORTED' in du.DF.columns

    def test_load_and_prepare_zscore_is_numeric(self):
        assert pd.api.types.is_numeric_dtype(du.DF['z-Score'])

    def test_load_and_prepare_area_is_numeric(self):
        assert pd.api.types.is_numeric_dtype(du.DF['AREA_TOTAL_REDUCED'])

    def test_load_and_prepare_has_categorical_columns(self):
        for col in ['Catalyst', 'Solvent', 'Base', 'Reaction Type']:
            if col in du.DF.columns:
                assert du.DF[col].dtype.name == 'category', f'{col} should be categorical'

    def test_df_not_empty(self):
        assert len(du.DF) > 0

    def test_df_has_required_columns(self):
        required = [
            'ELN_ID',
            'PLATENUMBER',
            'Coordinate',
            'AREA_TOTAL_REDUCED',
            'Reaction Type',
            'z-Score',
            'FG A',
            'FG B',
        ]
        for col in required:
            assert col in du.DF.columns, f'Missing required column: {col}'


# ===========================================================================
# Integration tests with full dataset
# ===========================================================================


@pytest.mark.slow
class TestFullDatasetFilterChain:
    @pytest.mark.parametrize(
        'reaction_type',
        [
            'Buchwald-Hartwig',
            'Suzuki-Miyaura',
        ],
    )
    def test_each_reaction_type_filters_without_error(self, full_dataset, reaction_type):
        if reaction_type not in full_dataset['Reaction Type'].cat.categories.tolist():
            pytest.skip(f'{reaction_type} not in dataset')
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=[reaction_type],
            reactant_types=['Catalyst'],
            min_eln=5,
            topn_zscore=5,
            max_components=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_empty_filters_return_deduplicated_dataset(self, full_dataset):
        result = du.filter_data(source_df=full_dataset)
        # Dedup step 7 always runs, so result <= full dataset
        assert len(result) > 0
        assert len(result) <= len(full_dataset)

    def test_cui_exclusion_removes_cui(self, full_dataset):
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            exclude_cui=['exclude_cui'],
        )
        if 'Catalyst' in result.columns:
            assert (result['Catalyst'] == 'CuI').sum() == 0

    def test_stats_dict_keys(self, full_dataset):
        _, stats = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            fg_a=['RNH2', 'RNH2 a-branch'],
            fg_b=['ArBr', 'ArCl'],
            return_stats=True,
        )
        assert 'whole_dataset' in stats
        assert 'after_fg_a' in stats
        assert 'after_fg_b' in stats
        assert 'max_components_cap' in stats
