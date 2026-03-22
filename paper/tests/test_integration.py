"""Integration tests using the full dataset and golden snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import data_utils as du
import plot_utils as pu

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'golden'


# ===========================================================================
# Golden snapshot tests
# ===========================================================================


def _load_golden(name: str) -> dict:
    """Load a golden snapshot JSON file."""
    path = GOLDEN_DIR / f'{name}.json'
    with open(path) as f:
        return json.load(f)


@pytest.mark.slow
@pytest.mark.golden
class TestGoldenSnapshots:
    @pytest.mark.parametrize(
        'golden_name',
        [
            'buchwald_hartwig_catalyst',
            'buchwald_hartwig_ligand',
            'suzuki_miyaura_catalyst',
            'all_reactions_base',
            'multi_category_catalyst_solvent',
            'no_filters',
        ],
    )
    def test_filter_output_matches_golden(self, full_dataset, golden_name):
        golden = _load_golden(golden_name)
        params = golden['params']

        result = du.filter_data(**params, source_df=full_dataset, return_stats=True)
        if isinstance(result, tuple):
            dff, _stats = result
        else:
            dff = result

        # Row count must match exactly
        assert (
            len(dff) == golden['row_count']
        ), f'Row count mismatch for {golden_name}: got {len(dff)}, expected {golden["row_count"]}'

        # ELN count must match exactly
        eln_count = int(dff['ELN_ID'].nunique())
        assert (
            eln_count == golden['eln_count']
        ), f'ELN count mismatch for {golden_name}: got {eln_count}, expected {golden["eln_count"]}'

        # z-Score statistics must match within tolerance
        z_stats = golden['z_score_stats']
        z_scores = dff['z-Score'].dropna()
        if z_stats['mean'] is not None:
            assert abs(z_scores.mean() - z_stats['mean']) < 0.01, f'Mean mismatch for {golden_name}'
            assert abs(z_scores.median() - z_stats['median']) < 0.01, f'Median mismatch for {golden_name}'

        # Category value counts must match
        if 'category_value_counts' in golden:
            for col, expected_counts in golden['category_value_counts'].items():
                if col in dff.columns:
                    actual_counts = dff[col].value_counts().to_dict()
                    actual_counts = {str(k): int(v) for k, v in actual_counts.items()}
                    assert actual_counts == expected_counts, f'Category counts mismatch for {col} in {golden_name}'


# ===========================================================================
# Full dataset filter chain
# ===========================================================================


@pytest.mark.slow
class TestFullDatasetFilterChain:
    def test_empty_filters_return_deduplicated_dataset(self, full_dataset):
        result = du.filter_data(source_df=full_dataset)
        # Dedup step 7 always runs, so result <= full dataset
        assert len(result) > 0
        assert len(result) <= len(full_dataset)

    def test_all_reaction_types_filter_without_error(self, full_dataset):
        reaction_types = full_dataset['Reaction Type'].dropna().unique().tolist()
        for rt in reaction_types:
            result = du.filter_data(
                source_df=full_dataset,
                reaction_types=[rt],
                reactant_types=['Catalyst'],
            )
            assert isinstance(result, pd.DataFrame)

    def test_deduplication_reduces_row_count(self, full_dataset):
        # Filter without dedup (dedup happens inside filter_data)
        bh = full_dataset[full_dataset['Reaction Type'] == 'Buchwald-Hartwig']
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
        )
        assert len(result) <= len(bh)

    def test_topn_limits_per_group(self, full_dataset):
        r1 = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            topn_zscore=1,
        )
        r5 = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            topn_zscore=5,
        )
        assert len(r1) <= len(r5)

    def test_max_components_limits_categories(self, full_dataset):
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            max_components=3,
        )
        assert result['Catalyst'].nunique() <= 3

    def test_min_eln_removes_sparse_groups(self, full_dataset):
        r1 = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=1,
        )
        r10 = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=10,
        )
        assert len(r10) <= len(r1)

    def test_cui_exclusion(self, full_dataset):
        result = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            exclude_cui=['exclude_cui'],
        )
        assert (result['Catalyst'] == 'CuI').sum() == 0

    def test_stats_dict_complete(self, full_dataset):
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

    def test_cache_consistency(self, full_dataset):
        params = dict(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=5,
        )
        r1 = du.filter_data(**params)
        r2 = du.filter_data(**params)
        pd.testing.assert_frame_equal(r1, r2)


# ===========================================================================
# Full dataset plot integration
# ===========================================================================


@pytest.mark.slow
class TestPlotIntegration:
    def test_boxplot_from_real_data(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            exclude_cui=['exclude_cui'],
            min_eln=5,
            topn_zscore=5,
            max_components=10,
        )
        fig, height = pu.create_boxplot(dff, ['Catalyst'], reaction_type='Buchwald-Hartwig')
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert height >= 800

    def test_heatmap_from_real_data(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst', 'Solvent'],
            min_eln=3,
            max_components=10,
        )
        fig, height = pu.create_heatmap(dff, ['Catalyst', 'Solvent'])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_distribution_plot_from_real_data(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
        )
        fig, height = pu.create_distribution_plot(dff)
        assert isinstance(fig, go.Figure)
        assert height == 500

    def test_qq_plot_from_real_data(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
        )
        fig, height = pu.create_qq_plot(dff)
        assert isinstance(fig, go.Figure)
        assert height == 500


# ===========================================================================
# Statistical functions with real data
# ===========================================================================


@pytest.mark.slow
class TestStatisticalFunctionsRealData:
    def test_distribution_stats(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
        )
        result = du.compute_distribution_stats(dff, 'Catalyst')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_significance_tests(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
            reactant_types=['Catalyst'],
            min_eln=5,
        )
        result = du.compute_significance_tests(dff, 'Catalyst')
        assert result['n_groups'] >= 2
        assert 'pairwise' in result

    def test_distribution_summary(self, full_dataset):
        dff = du.filter_data(
            source_df=full_dataset,
            reaction_types=['Buchwald-Hartwig'],
        )
        result = du.get_distribution_summary(dff, 'Catalyst')
        assert result['n_groups'] > 0
