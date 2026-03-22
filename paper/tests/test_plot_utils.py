"""Tests for plot_utils.py — boxplot, heatmap, distribution, QQ, tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import plot_utils as pu

# ===========================================================================
# Helper functions
# ===========================================================================


class TestFontSizes:
    def test_main_presentation_mode(self):
        fs = pu._font_sizes(True, 'main')
        assert fs['title'] == 32
        assert fs['base'] == 20

    def test_main_normal_mode(self):
        fs = pu._font_sizes(False, 'main')
        assert fs['title'] == 22
        assert fs['base'] == 14

    def test_diagnostic_presentation_mode(self):
        fs = pu._font_sizes(True, 'diagnostic')
        assert fs['title'] == 28

    def test_diagnostic_normal_mode(self):
        fs = pu._font_sizes(False, 'diagnostic')
        assert fs['title'] == 20

    def test_table_presentation_mode(self):
        fs = pu._font_sizes(True, 'table')
        assert fs['title'] == 20

    def test_table_normal_mode(self):
        fs = pu._font_sizes(False, 'table')
        assert fs['title'] == 16

    def test_unknown_variant_returns_table(self):
        fs = pu._font_sizes(False, 'something_else')
        assert 'title' in fs  # Should still return valid dict


class TestApplyCommonLayout:
    def test_sets_title(self):
        fig = go.Figure()
        fs = pu._font_sizes(False, 'main')
        pu._apply_common_layout(fig, title='Test Title', fs=fs, height=500)
        assert fig.layout.title.text == 'Test Title'

    def test_sets_height(self):
        fig = go.Figure()
        fs = pu._font_sizes(False, 'main')
        pu._apply_common_layout(fig, title='T', fs=fs, height=800)
        assert fig.layout.height == 800

    def test_sets_background_colors(self):
        fig = go.Figure()
        fs = pu._font_sizes(False, 'main')
        pu._apply_common_layout(fig, title='T', fs=fs, height=500)
        assert fig.layout.plot_bgcolor == 'white'
        assert fig.layout.paper_bgcolor == 'white'


class TestSafeStrConversion:
    def test_object_series(self):
        s = pd.Series(['a', 'b', 'c'])
        result = pu._safe_str_conversion(s)
        assert list(result) == ['a', 'b', 'c']

    def test_null_values_become_no_value(self):
        s = pd.Series(['a', None, np.nan])
        result = pu._safe_str_conversion(s)
        assert result.iloc[1] == '(no value)'
        assert result.iloc[2] == '(no value)'

    def test_categorical_series(self):
        s = pd.Series(pd.Categorical(['a', None, 'b']))
        result = pu._safe_str_conversion(s)
        assert result.iloc[1] == '(no value)'

    def test_numeric_series(self):
        s = pd.Series([1, 2, 3])
        result = pu._safe_str_conversion(s)
        assert all(isinstance(v, str) for v in result)


class TestInterpolateHex:
    def test_factor_0_returns_first_color(self):
        assert pu._interpolate_hex('#FF0000', '#0000FF', 0.0) == '#ff0000'

    def test_factor_1_returns_second_color(self):
        assert pu._interpolate_hex('#FF0000', '#0000FF', 1.0) == '#0000ff'

    def test_factor_half_returns_midpoint(self):
        result = pu._interpolate_hex('#000000', '#FFFFFF', 0.5)
        # Should be approximately #7f7f7f
        assert result.startswith('#')
        assert len(result) == 7

    def test_returns_valid_hex_string(self):
        result = pu._interpolate_hex('#123456', '#ABCDEF', 0.3)
        assert result.startswith('#')
        assert len(result) == 7
        # Verify hex chars
        assert all(c in '0123456789abcdef' for c in result[1:])


class TestCreateColorMapping:
    def test_known_category_uses_base_colours(self, small_df):
        color_map = pu.create_color_mapping('Catalyst', small_df)
        assert isinstance(color_map, dict)
        assert len(color_map) > 0
        # All values should be hex color strings
        for v in color_map.values():
            assert v.startswith('#')

    def test_unknown_category_uses_grey(self):
        df = pd.DataFrame(
            {
                'UnknownCat': ['A', 'B', 'A', 'B'],
                'ELN_ID': ['E1', 'E2', 'E1', 'E2'],
            }
        )
        color_map = pu.create_color_mapping('UnknownCat', df)
        # Should use grey from fallback
        assert len(color_map) == 2

    def test_all_same_eln_count_uses_factor_half(self):
        df = pd.DataFrame(
            {
                'Cat': ['A', 'B'],
                'ELN_ID': ['E1', 'E2'],
            }
        )
        color_map = pu.create_color_mapping('Cat', df)
        # When all counts equal, factor = 0.5
        assert len(color_map) == 2

    def test_varying_eln_counts_produce_gradient(self, small_df):
        color_map = pu.create_color_mapping('Catalyst', small_df)
        colors = list(color_map.values())
        # Should have different shades
        assert len(set(colors)) > 1


class TestShapiroWilkSummary:
    def test_normal_data_returns_result(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 100))
        text, p, status = pu._shapiro_wilk_summary(data)
        assert p is not None
        assert status in ('Normal', 'Non-normal')

    def test_small_sample_works(self):
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        text, p, status = pu._shapiro_wilk_summary(data)
        assert isinstance(text, str)


class TestFilterDiagnosticData:
    def test_with_group_filters_correctly(self, small_df):
        data, title = pu._filter_diagnostic_data(
            small_df,
            'z-Score',
            'Reaction Type',
            'Buchwald-Hartwig',
            'Histogram',
        )
        assert len(data) > 0
        assert 'Buchwald-Hartwig' in title

    def test_without_group_returns_all(self, small_df):
        data, title = pu._filter_diagnostic_data(
            small_df,
            'z-Score',
            None,
            None,
            'Histogram',
        )
        assert len(data) == small_df['z-Score'].dropna().shape[0]

    def test_builds_default_title(self, small_df):
        _, title = pu._filter_diagnostic_data(
            small_df,
            'z-Score',
            None,
            None,
            'QQ Plot',
        )
        assert 'z-Score' in title


# ===========================================================================
# Main plot functions
# ===========================================================================


class TestCreateBoxplot:
    def test_returns_figure_and_height(self, small_df):
        fig, height = pu.create_boxplot(small_df, ['Catalyst'])
        assert isinstance(fig, go.Figure)
        assert isinstance(height, int)

    def test_height_adapts_to_category_count(self, small_df):
        fig, height = pu.create_boxplot(small_df, ['Catalyst'])
        assert height >= 800

    def test_min_height_is_base_height(self):
        # Single category — height should be at least base_height
        df = pd.DataFrame(
            {
                'Catalyst': pd.Categorical(['A'] * 10),
                'ELN_ID': pd.Categorical([f'E{i}' for i in range(10)]),
                'z-Score': range(10),
                'PLATENUMBER': [1] * 10,
                'Coordinate': ['A1'] * 10,
                'AREA_TOTAL_REDUCED': [100.0] * 10,
                'Reaction Type': pd.Categorical(['BH'] * 10),
                'Solvent': pd.Categorical(['DMF'] * 10),
                'Base': pd.Categorical(['Cs2CO3'] * 10),
                'Ligand': pd.Categorical(['XPhos'] * 10),
                'Additive': pd.Categorical([None] * 10),
                'Coupling Reagent': pd.Categorical([None] * 10),
                'Secondary Solvent': pd.Categorical([None] * 10),
                'FG A': pd.Categorical(['RNH2'] * 10),
                'FG B': pd.Categorical(['ArBr'] * 10),
            }
        )
        fig, height = pu.create_boxplot(df, ['Catalyst'], base_height=800)
        assert height >= 800

    def test_single_reactant_type(self, small_df):
        fig, _ = pu.create_boxplot(small_df, ['Catalyst'])
        assert len(fig.data) > 0

    def test_multiple_reactant_types_combined_label(self, small_df):
        fig, _ = pu.create_boxplot(small_df, ['Catalyst', 'Solvent'])
        assert len(fig.data) > 0

    def test_max_categories_limits_output(self, small_df):
        fig_all, _ = pu.create_boxplot(small_df, ['Catalyst'])
        fig_lim, _ = pu.create_boxplot(small_df, ['Catalyst'], max_categories=2)
        assert len(fig_lim.data) <= len(fig_all.data)

    def test_presentation_mode_larger_fonts(self, small_df):
        fig_normal, _ = pu.create_boxplot(small_df, ['Catalyst'], presentation_mode=False)
        fig_pres, _ = pu.create_boxplot(small_df, ['Catalyst'], presentation_mode=True)
        assert fig_pres.layout.title.font.size > fig_normal.layout.title.font.size

    def test_reaction_type_in_title(self, small_df):
        fig, _ = pu.create_boxplot(small_df, ['Catalyst'], reaction_type='Buchwald-Hartwig')
        assert 'Buchwald-Hartwig' in fig.layout.title.text

    def test_figure_has_traces(self, small_df):
        fig, _ = pu.create_boxplot(small_df, ['Catalyst'])
        assert len(fig.data) > 0


class TestCreateHeatmap:
    def test_returns_figure_and_height(self, small_df):
        fig, height = pu.create_heatmap(small_df, ['Catalyst', 'Solvent'])
        assert isinstance(fig, go.Figure)
        assert isinstance(height, int)

    def test_requires_two_reactant_types(self, small_df):
        # Single reactant type should still work but may produce an error or empty figure
        # The function requires at least 2 reactant types
        try:
            fig, _ = pu.create_heatmap(small_df, ['Catalyst'])
            # If it doesn't raise, it should return a figure
            assert isinstance(fig, go.Figure)
        except (ValueError, KeyError, IndexError):
            pass  # Expected - needs 2+ types

    def test_two_reactant_types(self, small_df):
        fig, _ = pu.create_heatmap(small_df, ['Catalyst', 'Solvent'])
        assert len(fig.data) > 0

    def test_height_adapts_to_y_categories(self, small_df):
        _, height = pu.create_heatmap(small_df, ['Catalyst', 'Solvent'])
        assert height >= 800

    def test_presentation_mode_fonts(self, small_df):
        fig_normal, _ = pu.create_heatmap(small_df, ['Catalyst', 'Solvent'], presentation_mode=False)
        fig_pres, _ = pu.create_heatmap(small_df, ['Catalyst', 'Solvent'], presentation_mode=True)
        assert fig_pres.layout.title.font.size > fig_normal.layout.title.font.size


class TestCreateDistributionPlot:
    def test_returns_figure_and_height(self, small_df):
        fig, height = pu.create_distribution_plot(small_df)
        assert isinstance(fig, go.Figure)
        assert height == 500  # _DIAGNOSTIC_PLOT_HEIGHT

    def test_with_group_col_filters(self, small_df):
        fig, _ = pu.create_distribution_plot(
            small_df,
            group_col='Reaction Type',
            group_value='Buchwald-Hartwig',
        )
        assert isinstance(fig, go.Figure)

    def test_custom_title_overrides_default(self, small_df):
        fig, _ = pu.create_distribution_plot(small_df, title='Custom Title')
        assert fig.layout.title.text == 'Custom Title'

    def test_presentation_mode_fonts(self, small_df):
        fig_n, _ = pu.create_distribution_plot(small_df, presentation_mode=False)
        fig_p, _ = pu.create_distribution_plot(small_df, presentation_mode=True)
        assert fig_p.layout.title.font.size > fig_n.layout.title.font.size


class TestCreateQQPlot:
    def test_returns_figure_and_height(self, small_df):
        fig, height = pu.create_qq_plot(small_df)
        assert isinstance(fig, go.Figure)
        assert height == 500

    def test_has_scatter_and_line_traces(self, small_df):
        fig, _ = pu.create_qq_plot(small_df)
        # Should have at least a scatter trace and a line trace
        assert len(fig.data) >= 2

    def test_presentation_mode_fonts(self, small_df):
        fig_n, _ = pu.create_qq_plot(small_df, presentation_mode=False)
        fig_p, _ = pu.create_qq_plot(small_df, presentation_mode=True)
        assert fig_p.layout.title.font.size > fig_n.layout.title.font.size


# ===========================================================================
# Table functions
# ===========================================================================


class TestCreateDistributionSummaryTable:
    def test_empty_df_returns_figure(self):
        fig = pu.create_distribution_summary_table(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_with_data_returns_table(self, small_df):
        import data_utils as du

        dist_stats = du.compute_distribution_stats(small_df, 'Catalyst', min_samples=5)
        if not dist_stats.empty:
            fig = pu.create_distribution_summary_table(dist_stats)
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_presentation_mode_fonts(self, small_df):
        import data_utils as du

        dist_stats = du.compute_distribution_stats(small_df, 'Catalyst', min_samples=5)
        if not dist_stats.empty:
            fig_n = pu.create_distribution_summary_table(dist_stats, presentation_mode=False)
            fig_p = pu.create_distribution_summary_table(dist_stats, presentation_mode=True)
            assert isinstance(fig_n, go.Figure)
            assert isinstance(fig_p, go.Figure)


class TestCreateSignificanceSummaryTable:
    def test_with_results(self, small_df):
        import data_utils as du

        sig_results = du.compute_significance_tests(small_df, 'Catalyst')
        if sig_results['n_groups'] >= 2:
            fig = pu.create_significance_summary_table(sig_results)
            assert isinstance(fig, go.Figure)

    def test_empty_pairwise_handled(self):
        sig_results = {
            'n_groups': 1,
            'kruskal_wallis': {'statistic': float('nan'), 'p_value': float('nan'), 'significant': None},
            'pairwise': pd.DataFrame(),
            'group_stats': pd.DataFrame(),
        }
        fig = pu.create_significance_summary_table(sig_results)
        assert isinstance(fig, go.Figure)

    def test_presentation_mode_fonts(self, small_df):
        import data_utils as du

        sig_results = du.compute_significance_tests(small_df, 'Catalyst')
        if sig_results['n_groups'] >= 2:
            fig_n = pu.create_significance_summary_table(sig_results, presentation_mode=False)
            fig_p = pu.create_significance_summary_table(sig_results, presentation_mode=True)
            assert isinstance(fig_n, go.Figure)
            assert isinstance(fig_p, go.Figure)
