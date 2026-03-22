"""Tests for export_boxplots.py — sanitize, validate, save helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import export_boxplots as eb


class TestSanitizeFilename:
    def test_replaces_forward_slash(self):
        assert eb.sanitize_filename('a/b') == 'a-b'

    def test_replaces_backslash(self):
        assert eb.sanitize_filename('a\\b') == 'a-b'

    def test_replaces_colon(self):
        assert eb.sanitize_filename('a:b') == 'a-b'

    def test_replaces_spaces(self):
        assert eb.sanitize_filename('a b c') == 'a_b_c'

    def test_combined_replacements(self):
        assert eb.sanitize_filename('Amide coupling / test: 1') == 'Amide_coupling_-_test-_1'

    def test_clean_string_unchanged(self):
        assert eb.sanitize_filename('clean_name') == 'clean_name'


class TestApplyPublicationFonts:
    def test_sets_title_font_size_48(self):
        fig = go.Figure()
        eb.apply_publication_fonts(fig)
        assert fig.layout.title.font.size == 48

    def test_sets_base_font_size_32(self):
        fig = go.Figure()
        eb.apply_publication_fonts(fig)
        assert fig.layout.font.size == 32

    def test_sets_axis_font_sizes(self):
        fig = go.Figure()
        eb.apply_publication_fonts(fig)
        assert fig.layout.xaxis.title.font.size == 36
        assert fig.layout.yaxis.title.font.size == 36
        assert fig.layout.xaxis.tickfont.size == 28
        assert fig.layout.yaxis.tickfont.size == 28


class TestHasRealComponents:
    def test_none_returns_false(self):
        assert eb._has_real_components(None) is False

    def test_empty_series_returns_false(self):
        s = pd.Series([], dtype='object')
        assert eb._has_real_components(s) is False

    def test_nan_only_returns_false(self):
        s = pd.Series([np.nan, np.nan])
        assert eb._has_real_components(s) is False

    def test_na_string_returns_false(self):
        s = pd.Series(['<NA>', 'nan', 'NaN', ''])
        assert eb._has_real_components(s) is False

    def test_real_values_return_true(self):
        s = pd.Series(['Pd(OAc)2', 'CuI', 'Pd(PPh3)4'])
        assert eb._has_real_components(s) is True

    def test_mixed_values_return_true(self):
        s = pd.Series(['Pd(OAc)2', np.nan, ''])
        assert eb._has_real_components(s) is True


class TestValidateDataForPlot:
    def test_none_returns_false(self):
        assert eb.validate_data_for_plot(None, ['Catalyst']) is False

    def test_empty_df_returns_false(self):
        assert eb.validate_data_for_plot(pd.DataFrame(), ['Catalyst']) is False

    def test_no_real_components_returns_false(self):
        df = pd.DataFrame(
            {
                'Catalyst': [np.nan, np.nan, np.nan],
                'z-Score': [1.0, 2.0, 3.0],
            }
        )
        assert eb.validate_data_for_plot(df, ['Catalyst']) is False

    def test_below_min_unique_returns_false(self):
        df = pd.DataFrame(
            {
                'Catalyst': ['A', 'A', 'A'],
                'z-Score': [1.0, 2.0, 3.0],
            }
        )
        assert eb.validate_data_for_plot(df, ['Catalyst'], min_unique=5) is False

    def test_valid_data_returns_true(self):
        df = pd.DataFrame(
            {
                'Catalyst': [f'Cat{i}' for i in range(10)],
                'z-Score': range(10),
            }
        )
        assert eb.validate_data_for_plot(df, ['Catalyst'], min_unique=5) is True
