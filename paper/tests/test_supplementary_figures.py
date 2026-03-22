"""Tests for generate_supplementary_figures.py — filename, matching, column checks."""

from __future__ import annotations

import generate_supplementary_figures as gsf
import pandas as pd


class TestToSafeFilename:
    def test_spaces_to_underscores(self):
        assert gsf.to_safe_filename('Amide coupling') == 'Amide_coupling'

    def test_slashes_to_hyphens(self):
        assert gsf.to_safe_filename('C/H Activation') == 'C-H_Activation'

    def test_combined(self):
        assert gsf.to_safe_filename('C-N Coupling') == 'C-N_Coupling'


class TestMatchReactionTypesFromDirs:
    def test_exact_match(self):
        dirs = {'Buchwald-Hartwig'}
        available = {'Buchwald-Hartwig', 'Suzuki-Miyaura'}
        result = gsf.match_reaction_types_from_dirs(dirs, available)
        assert 'Buchwald-Hartwig' in result

    def test_underscore_to_space(self):
        dirs = {'Amide_coupling'}
        available = {'Amide coupling'}
        result = gsf.match_reaction_types_from_dirs(dirs, available)
        assert 'Amide coupling' in result

    def test_no_match_excluded(self):
        dirs = {'Nonexistent_Reaction'}
        available = {'Buchwald-Hartwig'}
        result = gsf.match_reaction_types_from_dirs(dirs, available)
        assert len(result) == 0

    def test_empty_sets(self):
        result = gsf.match_reaction_types_from_dirs(set(), set())
        assert result == []


class TestCheckAreaTotalReducedColumn:
    def test_column_present_returns_true(self):
        df = pd.DataFrame({'AREA_TOTAL_REDUCED': [1.0, 2.0]})
        assert gsf.check_area_total_reduced_column(df) is True

    def test_column_absent_returns_false(self):
        df = pd.DataFrame({'other_col': [1, 2]})
        assert gsf.check_area_total_reduced_column(df) is False
