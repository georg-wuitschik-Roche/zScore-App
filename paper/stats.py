#!/usr/bin/env python3
"""
stats.py
========
Statistical validation functions for the paper.

These functions compute Shapiro-Wilk normality tests, Kruskal-Wallis H-tests,
Mann-Whitney U pairwise comparisons, and permutation tests. They were used to
generate supplementary statistics for the publication and are NOT part of the
live React dashboard.

Usage:
    python paper/stats.py

Dependencies: scipy, pandas, numpy (see paper/requirements.txt)
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compute_distribution_stats(
    dff: pd.DataFrame,
    group_col: str = 'Reaction Type',
    value_col: str = 'z-Score',
    min_samples: int = 20,
) -> pd.DataFrame:
    """Compute distribution statistics for z-scores grouped by a category.

    Calculates skewness, kurtosis, and Shapiro-Wilk normality test p-value.

    Args:
        dff: DataFrame containing the data to analyze.
        group_col: Column to group by.
        value_col: Column containing the values to analyze.
        min_samples: Minimum number of samples required per group.

    Returns:
        DataFrame with columns: group, n, mean, std, skewness, kurtosis,
        shapiro_stat, shapiro_p, is_normal (at alpha=0.05).
    """
    results = []

    for name, group in dff.groupby(group_col):
        values = group[value_col].dropna()
        n = len(values)

        if n < min_samples:
            continue

        mean_val = values.mean()
        std_val = values.std()
        skewness = values.skew()
        kurtosis = values.kurtosis()

        sample_for_test = values.sample(min(5000, n), random_state=42) if n > 5000 else values
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample_for_test)
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan

        results.append(
            {
                'group': name,
                'n': n,
                'mean': round(mean_val, 4),
                'std': round(std_val, 4),
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'shapiro_stat': round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else np.nan,
                'shapiro_p': round(shapiro_p, 4) if not np.isnan(shapiro_p) else np.nan,
                'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
            }
        )

    return pd.DataFrame(results)


def compute_significance_tests(
    dff: pd.DataFrame,
    category_col: str,
    value_col: str = 'z-Score',
    top_n: int = 10,
    alpha: float = 0.05,
) -> dict:
    """Run Kruskal-Wallis H-test and pairwise Mann-Whitney U tests.

    Args:
        dff: DataFrame containing the data to analyze.
        category_col: Column containing the categories to compare.
        value_col: Column containing the values to compare.
        top_n: Number of top categories (by median) to include.
        alpha: Significance level.

    Returns:
        Dictionary with kruskal_wallis, pairwise, effect_sizes, group_stats.
    """
    medians = dff.groupby(category_col)[value_col].median().sort_values(ascending=False)
    top_categories = medians.head(top_n).index.tolist()
    dff_filtered = dff[dff[category_col].isin(top_categories)]

    groups = []
    group_names = []
    group_stats_list = []

    for cat in top_categories:
        values = dff_filtered[dff_filtered[category_col] == cat][value_col].dropna()
        if len(values) >= 5:
            groups.append(values.values)
            group_names.append(cat)
            group_stats_list.append(
                {
                    'category': cat,
                    'n': len(values),
                    'median': round(values.median(), 4),
                    'mean': round(values.mean(), 4),
                    'std': round(values.std(), 4),
                    'q25': round(values.quantile(0.25), 4),
                    'q75': round(values.quantile(0.75), 4),
                }
            )

    result: dict = {
        'n_groups': len(groups),
        'group_stats': pd.DataFrame(group_stats_list) if group_stats_list else pd.DataFrame(),
    }

    if len(groups) < 2:
        result['kruskal_wallis'] = {'statistic': np.nan, 'p_value': np.nan, 'significant': None}
        result['pairwise'] = pd.DataFrame()
        result['effect_sizes'] = pd.DataFrame()
        return result

    try:
        kw_stat, kw_p = stats.kruskal(*groups)
        result['kruskal_wallis'] = {'statistic': round(kw_stat, 4), 'p_value': kw_p, 'significant': kw_p < alpha}
    except Exception:
        result['kruskal_wallis'] = {'statistic': np.nan, 'p_value': np.nan, 'significant': None}

    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    alpha_corrected = alpha / n_comparisons if n_comparisons > 0 else alpha
    result['n_comparisons'] = n_comparisons
    result['alpha_corrected'] = alpha_corrected

    pairwise_results = []
    for i, j in combinations(range(len(groups)), 2):
        group_i, group_j = groups[i], groups[j]
        name_i, name_j = group_names[i], group_names[j]
        try:
            u_stat, p_value = stats.mannwhitneyu(group_i, group_j, alternative='two-sided')
            n1, n2 = len(group_i), len(group_j)
            r = 1 - (2 * u_stat) / (n1 * n2)
            pairwise_results.append(
                {
                    'group_1': name_i,
                    'group_2': name_j,
                    'u_statistic': round(u_stat, 2),
                    'p_value': p_value,
                    'significant': p_value < alpha_corrected,
                    'effect_size_r': round(r, 4),
                    'effect_magnitude': _interpret_effect_size(abs(r)),
                }
            )
        except Exception:
            pairwise_results.append(
                {
                    'group_1': name_i,
                    'group_2': name_j,
                    'u_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': None,
                    'effect_size_r': np.nan,
                    'effect_magnitude': 'N/A',
                }
            )

    result['pairwise'] = pd.DataFrame(pairwise_results)
    return result


def compute_permutation_test(
    dff: pd.DataFrame,
    category_col: str,
    value_col: str = 'z-Score',
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict:
    """Run permutation test for Kruskal-Wallis.

    Args:
        dff: DataFrame with filtered data.
        category_col: Column containing category labels to permute.
        value_col: Column with values to compare.
        n_permutations: Number of permutations.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with observed H, permuted H distribution, empirical p-value.
    """
    rng = np.random.default_rng(random_state)

    categories = dff[category_col].unique()
    groups = [dff[dff[category_col] == cat][value_col].values for cat in categories]
    observed_h, observed_p = stats.kruskal(*groups)

    permuted_h_values = []
    values = dff[value_col].values.copy()

    for _ in range(n_permutations):
        rng.shuffle(values)
        start_idx = 0
        shuffled_groups = []
        for cat in categories:
            n_cat = (dff[category_col] == cat).sum()
            shuffled_groups.append(values[start_idx : start_idx + n_cat])
            start_idx += n_cat
        h_perm, _ = stats.kruskal(*shuffled_groups)
        permuted_h_values.append(h_perm)

    permuted_h_values = np.array(permuted_h_values)
    empirical_p = (permuted_h_values >= observed_h).mean()

    return {
        'observed_h': observed_h,
        'standard_p': observed_p,
        'empirical_p': empirical_p,
        'n_permutations': n_permutations,
        'permuted_h_mean': permuted_h_values.mean(),
        'permuted_h_std': permuted_h_values.std(),
        'permuted_h_95th': np.percentile(permuted_h_values, 95),
        'significant_permutation': empirical_p < 0.05,
    }


def _interpret_effect_size(r: float) -> str:
    """Interpret rank-biserial correlation effect size."""
    if r < 0.1:
        return 'negligible'
    elif r < 0.3:
        return 'small'
    elif r < 0.5:
        return 'medium'
    else:
        return 'large'


def get_distribution_summary(
    dff: pd.DataFrame,
    group_col: str = 'Reaction Type',
    value_col: str = 'z-Score',
) -> dict:
    """Get a summary of distribution characteristics across all groups."""
    dist_stats = compute_distribution_stats(dff, group_col, value_col=value_col)

    if dist_stats.empty:
        return {
            'n_groups': 0,
            'n_normal': 0,
            'pct_normal': 0.0,
            'n_symmetric': 0,
            'pct_symmetric': 0.0,
            'n_moderate_skew': 0,
            'n_high_skew': 0,
            'median_skewness': np.nan,
            'median_kurtosis': np.nan,
        }

    n_groups = len(dist_stats)
    n_normal = dist_stats['is_normal'].sum() if 'is_normal' in dist_stats.columns else 0
    abs_skewness = dist_stats['skewness'].abs()

    return {
        'n_groups': n_groups,
        'n_normal': int(n_normal),
        'pct_normal': round(100 * n_normal / n_groups, 1) if n_groups > 0 else 0.0,
        'n_symmetric': int((abs_skewness < 0.5).sum()),
        'pct_symmetric': round(100 * (abs_skewness < 0.5).sum() / n_groups, 1) if n_groups > 0 else 0.0,
        'n_moderate_skew': int(((abs_skewness >= 0.5) & (abs_skewness < 1)).sum()),
        'n_high_skew': int((abs_skewness >= 1).sum()),
        'median_skewness': round(dist_stats['skewness'].median(), 4),
        'median_kurtosis': round(dist_stats['kurtosis'].median(), 4),
    }


if __name__ == '__main__':
    # Quick demo: run stats on the default dataset
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import data_utils as du

    print('Distribution summary for all reaction types:')
    summary = get_distribution_summary(du.DF)
    for k, v in summary.items():
        print(f'  {k}: {v}')
