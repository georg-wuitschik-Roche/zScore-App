"""Edge case tests — 0 rows, 1 category, all-NaN, degenerate inputs.

These tests verify graceful handling of boundary conditions that a React
rewrite must also handle without crashing or showing incorrect results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import data_utils as du
import plot_utils as pu

# ===========================================================================
# Filter chain edge cases
# ===========================================================================


class TestFilterChainEdgeCases:
    """Test filter_data with inputs that produce 0 or very few rows."""

    def test_nonexistent_reaction_type(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['THIS_DOES_NOT_EXIST'],
        )
        assert len(result) == 0

    def test_nonexistent_fg_a(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            fg_a=['NONEXISTENT_FG'],
        )
        assert len(result) == 0

    def test_nonexistent_fg_b(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            fg_a=['RNH2'],
            fg_b=['NONEXISTENT_FG'],
        )
        assert len(result) == 0

    def test_impossible_fg_pair(self, small_df):
        """FG A and FG B that never co-occur."""
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            fg_a=['RCOOH'],
            fg_b=['ArBpin'],
        )
        # May or may not be empty — but should not crash
        assert isinstance(result, pd.DataFrame)

    def test_very_high_min_eln(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=9999,
        )
        assert len(result) == 0

    def test_max_components_1(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            max_components=1,
        )
        if len(result) > 0:
            assert result['Catalyst'].nunique() <= 1

    def test_topn_0_treated_as_no_filter(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            topn_zscore=0,
        )
        # topn=0 is falsy, should skip the filter
        assert len(result) > 0

    def test_all_filters_contradictory(self, small_df):
        """CuI excluded + only CuI rows should produce empty."""
        cui_only = small_df[small_df['Catalyst'] == 'CuI'].copy()
        if len(cui_only) > 0:
            result = du.filter_data(
                source_df=cui_only,
                exclude_cui=['exclude_cui'],
            )
            assert (result['Catalyst'] == 'CuI').sum() == 0

    def test_return_stats_on_empty_result(self, small_df):
        result = du.filter_data(
            source_df=small_df,
            reaction_types=['NONEXISTENT'],
            return_stats=True,
        )
        assert isinstance(result, tuple)
        dff, stats = result
        assert len(dff) == 0
        assert isinstance(stats, dict)

    def test_empty_dataframe_input(self, empty_df):
        result = du.filter_data(source_df=empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        df = pd.DataFrame(
            {
                'ELN_ID': pd.Categorical(['E1']),
                'PLATENUMBER': [1],
                'Coordinate': ['A1'],
                'AREA_TOTAL_REDUCED': [100.0],
                'Reaction Type': pd.Categorical(['BH']),
                'Catalyst': pd.Categorical(['Pd']),
                'Solvent': pd.Categorical(['DMF']),
                'Base': pd.Categorical(['Cs2CO3']),
                'Ligand': pd.Categorical(['XPhos']),
                'FG A': pd.Categorical(['RNH2']),
                'FG B': pd.Categorical(['ArBr']),
                'FG_sorted': 'ArBr, RNH2',
                'FG_PAIR_SORTED': pd.Categorical(['ArBr, RNH2']),
                'z-Score': [1.5],
            }
        )
        result = du.filter_data(source_df=df, reaction_types=['BH'])
        assert len(result) == 1


# ===========================================================================
# NaN edge cases
# ===========================================================================


class TestNaNEdgeCases:
    """Test handling of NaN/null values throughout the pipeline."""

    def test_all_nan_zscores(self):
        df = pd.DataFrame(
            {
                'ELN_ID': pd.Categorical(['E1', 'E2', 'E3']),
                'PLATENUMBER': [1, 1, 1],
                'Coordinate': ['A1', 'A2', 'A3'],
                'AREA_TOTAL_REDUCED': [100.0, 110.0, 90.0],
                'Reaction Type': pd.Categorical(['BH', 'BH', 'BH']),
                'Catalyst': pd.Categorical(['Pd', 'Pd', 'Cu']),
                'Solvent': pd.Categorical(['DMF', 'DMF', 'DMF']),
                'Base': pd.Categorical(['Cs', 'Cs', 'Cs']),
                'Ligand': pd.Categorical([None, None, None]),
                'FG A': pd.Categorical(['RNH2', 'RNH2', 'RNH2']),
                'FG B': pd.Categorical(['ArBr', 'ArBr', 'ArBr']),
                'FG_sorted': 'ArBr, RNH2',
                'FG_PAIR_SORTED': pd.Categorical(['ArBr, RNH2'] * 3),
                'z-Score': [np.nan, np.nan, np.nan],
            }
        )
        result = du.filter_data(source_df=df, reaction_types=['BH'])
        assert isinstance(result, pd.DataFrame)

    def test_all_null_reactant_column(self):
        df = pd.DataFrame(
            {
                'ELN_ID': pd.Categorical(['E1', 'E2']),
                'PLATENUMBER': [1, 1],
                'Coordinate': ['A1', 'A2'],
                'AREA_TOTAL_REDUCED': [100.0, 110.0],
                'Reaction Type': pd.Categorical(['BH', 'BH']),
                'Catalyst': pd.Categorical([None, None]),
                'Solvent': pd.Categorical(['DMF', 'DMF']),
                'Base': pd.Categorical(['Cs', 'Cs']),
                'Ligand': pd.Categorical([None, None]),
                'FG A': pd.Categorical(['RNH2', 'RNH2']),
                'FG B': pd.Categorical(['ArBr', 'ArBr']),
                'FG_sorted': 'ArBr, RNH2',
                'FG_PAIR_SORTED': pd.Categorical(['ArBr, RNH2'] * 2),
                'z-Score': [1.0, 2.0],
            }
        )
        # With include_null=False, filtering on Catalyst should return 0 rows
        result = du.filter_data(
            source_df=df,
            reaction_types=['BH'],
            reactant_types=['Catalyst'],
            include_null_categories=None,
        )
        assert len(result) == 0

        # With include_null=True, should return all rows
        result = du.filter_data(
            source_df=df,
            reaction_types=['BH'],
            reactant_types=['Catalyst'],
            include_null_categories=[True],
        )
        assert len(result) == 2

    def test_mixed_nan_and_valid_zscores(self):
        df = pd.DataFrame(
            {
                'ELN_ID': pd.Categorical(['E1', 'E2', 'E3']),
                'PLATENUMBER': [1, 1, 1],
                'Coordinate': ['A1', 'A2', 'A3'],
                'AREA_TOTAL_REDUCED': [100.0, np.nan, 90.0],
                'Reaction Type': pd.Categorical(['BH', 'BH', 'BH']),
                'Catalyst': pd.Categorical(['Pd', 'Pd', 'Cu']),
                'Solvent': pd.Categorical(['DMF', 'DMF', 'DMF']),
                'Base': pd.Categorical(['Cs', 'Cs', 'Cs']),
                'Ligand': pd.Categorical([None, None, None]),
                'FG A': pd.Categorical(['RNH2', 'RNH2', 'RNH2']),
                'FG B': pd.Categorical(['ArBr', 'ArBr', 'ArBr']),
                'FG_sorted': 'ArBr, RNH2',
                'FG_PAIR_SORTED': pd.Categorical(['ArBr, RNH2'] * 3),
                'z-Score': [1.5, np.nan, 2.0],
            }
        )
        result = du.filter_data(source_df=df, reaction_types=['BH'])
        assert isinstance(result, pd.DataFrame)
        # Should not crash, NaN rows kept in dataframe


# ===========================================================================
# Plot edge cases
# ===========================================================================


class TestPlotEdgeCases:
    """Test plot functions with degenerate inputs."""

    def _make_df(self, n_cats: int, n_rows_per_cat: int = 5) -> pd.DataFrame:
        rows = []
        for i in range(n_cats):
            for j in range(n_rows_per_cat):
                rows.append(
                    {
                        'ELN_ID': f'E{i}',
                        'PLATENUMBER': 1,
                        'Coordinate': f'A{j}',
                        'AREA_TOTAL_REDUCED': 100.0,
                        'Reaction Type': 'BH',
                        'Catalyst': f'Cat_{i}',
                        'Solvent': 'DMF',
                        'Base': 'Cs2CO3',
                        'Ligand': 'XPhos',
                        'Additive': None,
                        'Coupling Reagent': None,
                        'Secondary Solvent': None,
                        'FG A': 'RNH2',
                        'FG B': 'ArBr',
                        'z-Score': float(i) + j * 0.1,
                    }
                )
        df = pd.DataFrame(rows)
        for col in ['Catalyst', 'Solvent', 'Base', 'Ligand', 'Reaction Type', 'ELN_ID', 'FG A', 'FG B']:
            df[col] = df[col].astype('category')
        return df

    def test_boxplot_single_category(self):
        df = self._make_df(1)
        fig, height = pu.create_boxplot(df, ['Catalyst'])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_boxplot_many_categories(self):
        df = self._make_df(50, n_rows_per_cat=3)
        fig, height = pu.create_boxplot(df, ['Catalyst'])
        assert isinstance(fig, go.Figure)
        assert height >= 50 * 110  # Adapts to category count

    def test_boxplot_single_data_point_per_category(self):
        df = self._make_df(5, n_rows_per_cat=1)
        fig, height = pu.create_boxplot(df, ['Catalyst'])
        assert isinstance(fig, go.Figure)

    def test_heatmap_single_y_category(self):
        df = self._make_df(1)
        df['Solvent'] = pd.Categorical(['DMF'] * len(df))
        # Can't really make a heatmap with 1 category on each axis
        # but it shouldn't crash
        try:
            fig, height = pu.create_heatmap(df, ['Catalyst', 'Solvent'])
            assert isinstance(fig, go.Figure)
        except (ValueError, KeyError):
            pass  # Expected for degenerate data

    def test_distribution_plot_few_points(self):
        df = self._make_df(1, n_rows_per_cat=3)
        fig, height = pu.create_distribution_plot(df)
        assert isinstance(fig, go.Figure)

    def test_qq_plot_few_points(self):
        df = self._make_df(1, n_rows_per_cat=5)
        fig, height = pu.create_qq_plot(df)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# Statistical function edge cases
# ===========================================================================


class TestStatisticalEdgeCases:
    def test_distribution_stats_single_group(self):
        df = pd.DataFrame(
            {
                'cat': ['A'] * 30,
                'z-Score': np.random.normal(0, 1, 30),
            }
        )
        result = du.compute_distribution_stats(df, 'cat', min_samples=5)
        assert len(result) == 1

    def test_distribution_stats_below_min_samples(self):
        df = pd.DataFrame(
            {
                'cat': ['A'] * 3,
                'z-Score': [1.0, 2.0, 3.0],
            }
        )
        result = du.compute_distribution_stats(df, 'cat', min_samples=20)
        assert len(result) == 0  # Filtered out

    def test_significance_tests_single_group(self):
        df = pd.DataFrame(
            {
                'cat': ['A'] * 10,
                'z-Score': range(10),
            }
        )
        result = du.compute_significance_tests(df, 'cat')
        assert result['n_groups'] <= 1
        assert pd.isna(result['kruskal_wallis']['statistic'])

    def test_significance_tests_two_groups(self):
        df = pd.DataFrame(
            {
                'cat': ['A'] * 10 + ['B'] * 10,
                'z-Score': list(range(10)) + list(range(5, 15)),
            }
        )
        result = du.compute_significance_tests(df, 'cat')
        assert result['n_groups'] == 2
        assert 'pairwise' in result
        assert len(result['pairwise']) == 1  # 1 pairwise comparison

    def test_distribution_summary_all_normal(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                'cat': ['A'] * 100 + ['B'] * 100,
                'z-Score': np.concatenate([rng.normal(0, 1, 100), rng.normal(0, 1, 100)]),
            }
        )
        summary = du.get_distribution_summary(df, 'cat')
        assert summary['n_groups'] == 2

    def test_distribution_summary_empty(self, empty_df):
        summary = du.get_distribution_summary(empty_df, 'Reaction Type')
        assert summary['n_groups'] == 0
        assert summary['n_normal'] == 0


# ===========================================================================
# Full pipeline edge cases with real data
# ===========================================================================


@pytest.mark.slow
class TestRealDataEdgeCases:
    """Edge cases using the real dataset."""

    def test_rare_reaction_type(self, full_dataset):
        """Reaction types with very few rows should still work."""
        rt_counts = full_dataset['Reaction Type'].value_counts()
        rare_rt = rt_counts[rt_counts < 100].index.tolist()
        for rt in rare_rt[:3]:  # Test first 3 rare types
            result = du.filter_data(
                source_df=full_dataset,
                reaction_types=[rt],
                reactant_types=['Catalyst'],
                min_eln=1,
            )
            assert isinstance(result, pd.DataFrame)

    def test_all_reaction_types_selected(self, full_dataset):
        """Selecting all reaction types at once."""
        all_rts = full_dataset['Reaction Type'].dropna().unique().tolist()
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=all_rts,
            reactant_types=['Catalyst'],
            min_eln=5,
            topn_zscore=3,
        )
        assert len(result) > 0

    def test_multiple_reaction_types(self, full_dataset):
        """Two reaction types combined."""
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig', 'Suzuki-Miyaura'],
            reactant_types=['Catalyst'],
            min_eln=5,
        )
        rts = result['Reaction Type'].unique().tolist()
        assert set(rts) <= {'Buchwald-Hartwig', 'Suzuki-Miyaura'}

    def test_max_components_equals_unique_count(self, full_dataset):
        """max_components exactly equals the number of unique categories."""
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=5,
            topn_zscore=5,
        )
        n_unique = dff['Catalyst'].nunique()
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=5,
            topn_zscore=5,
            max_components=n_unique,
        )
        assert result['Catalyst'].nunique() == n_unique

    def test_fg_a_all_plus_specific_fg_b(self, full_dataset):
        """FG A not specified, FG B specified."""
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            fg_b=['ArBr'],
            min_eln=5,
        )
        assert len(result) > 0
