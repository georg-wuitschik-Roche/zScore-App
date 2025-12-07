#!/usr/bin/env python3
"""
generate_supplementary_figures.py
==================================
Script to generate supplementary figures for the paper addressing 
reviewer concerns about statistical validity.

This script produces:
1. Raw data (AREA_TOTAL_REDUCED) distribution analysis:
   - Overall distribution histogram
   - Within-ELN distribution statistics (validates z-score transformation)
   - Summary of within-ELN skewness values
2. Z-score distribution analysis:
   - Histograms with normal overlays for major reaction types
   - Summary table of distribution statistics
3. Statistical significance test results (matching paper boxplot filters)

Usage:
    python generate_supplementary_figures.py

Output files are saved to the 'exports/supplementary/' directory.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

# Import project modules
import data_utils as du
import plot_utils as pu


def ensure_export_dir():
    """Create exports directory structure if it doesn't exist."""
    base_dir = Path("exports/supplementary")
    
    # Create subfolder structure
    subdirs = {
        'base': base_dir,
        'raw_data': base_dir / "raw_data",
        'histograms': base_dir / "histograms", 
        'distribution_stats': base_dir / "distribution_stats",
        'significance_tests': base_dir / "significance_tests",
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def generate_raw_data_distribution(df: pd.DataFrame, export_dirs: dict):
    """Generate distribution figures for AREA_TOTAL_REDUCED (raw data before z-transformation).
    
    This validates whether the z-score transformation is appropriate by examining
    the distribution of raw data within ELNs.
    """
    from scipy import stats
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    print("\n=== Generating Raw Data (AREA_TOTAL_REDUCED) Distribution ===\n")
    
    if 'AREA_TOTAL_REDUCED' not in df.columns:
        print("  WARNING: AREA_TOTAL_REDUCED column not found - skipping raw data analysis")
        return None
    
    raw_col = 'AREA_TOTAL_REDUCED'
    raw_data = df[raw_col].dropna()
    
    if len(raw_data) < 100:
        print(f"  WARNING: Insufficient raw data ({len(raw_data)} values)")
        return None
    
    print(f"  Analyzing {len(raw_data):,} raw data points")
    
    # 1. Overall AREA_TOTAL_REDUCED distribution
    print("\n  [1] Overall AREA_TOTAL_REDUCED distribution...")
    
    fig_overall = go.Figure()
    
    # Histogram
    fig_overall.add_trace(go.Histogram(
        x=raw_data,
        nbinsx=100,
        name='AREA_TOTAL_REDUCED',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    # Calculate statistics
    mean_val = raw_data.mean()
    std_val = raw_data.std()
    skew_val = stats.skew(raw_data)
    kurt_val = stats.kurtosis(raw_data)
    
    # Shapiro-Wilk on sample (max 5000)
    sample_size = min(5000, len(raw_data))
    sample = raw_data.sample(n=sample_size, random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    
    fig_overall.update_layout(
        title=dict(
            text=f"Distribution of AREA_TOTAL_REDUCED (Raw Data)<br>" +
                 f"<sup>n={len(raw_data):,} | skewness={skew_val:.2f} | kurtosis={kurt_val:.2f} | " +
                 f"Shapiro-Wilk p={shapiro_p:.2e}</sup>",
            font=dict(size=16)
        ),
        xaxis_title="AREA_TOTAL_REDUCED",
        yaxis_title="Count",
        template="plotly_white",
        width=1200,
        height=600
    )
    
    export_dir = export_dirs['raw_data']
    fig_overall.write_image(str(export_dir / "overall_histogram.png"), width=1200, height=600, scale=2)
    fig_overall.write_image(str(export_dir / "overall_histogram.svg"), width=1200, height=600)
    print(f"       Saved overall histogram (skewness={skew_val:.2f})")
    
    # 2. Within-ELN distribution analysis
    print("\n  [2] Within-ELN distribution analysis...")
    
    if 'ELN_ID' not in df.columns:
        print("       WARNING: ELN_ID column not found - skipping within-ELN analysis")
        return {'overall_skewness': skew_val, 'overall_kurtosis': kurt_val}
    
    # Calculate within-ELN statistics
    eln_stats = []
    elns_with_data = df.groupby('ELN_ID')[raw_col].filter(lambda x: x.dropna().shape[0] >= 20)
    unique_elns = df.loc[elns_with_data.index, 'ELN_ID'].unique()
    
    print(f"       Analyzing {len(unique_elns)} ELNs with >= 20 data points each")
    
    for eln_id in unique_elns:
        eln_data = df[df['ELN_ID'] == eln_id][raw_col].dropna()
        if len(eln_data) >= 20:
            eln_skew = stats.skew(eln_data)
            eln_kurt = stats.kurtosis(eln_data)
            
            # Shapiro-Wilk test
            sample_n = min(5000, len(eln_data))
            eln_sample = eln_data.sample(n=sample_n, random_state=42) if len(eln_data) > sample_n else eln_data
            try:
                _, eln_shapiro_p = stats.shapiro(eln_sample)
            except:
                eln_shapiro_p = np.nan
            
            eln_stats.append({
                'ELN_ID': eln_id,
                'n': len(eln_data),
                'mean': eln_data.mean(),
                'std': eln_data.std(),
                'skewness': eln_skew,
                'kurtosis': eln_kurt,
                'shapiro_p': eln_shapiro_p,
                'is_normal': eln_shapiro_p > 0.05 if not np.isnan(eln_shapiro_p) else False
            })
    
    eln_stats_df = pd.DataFrame(eln_stats)
    
    if len(eln_stats_df) > 0:
        # Save ELN-level statistics
        eln_stats_df.to_csv(export_dir / "within_eln_statistics.csv", index=False)
        
        # Summary statistics
        n_normal = eln_stats_df['is_normal'].sum()
        pct_normal = 100 * n_normal / len(eln_stats_df)
        median_skew = eln_stats_df['skewness'].median()
        n_symmetric = (eln_stats_df['skewness'].abs() < 0.5).sum()
        pct_symmetric = 100 * n_symmetric / len(eln_stats_df)
        
        print(f"       {len(eln_stats_df)} ELNs analyzed")
        print(f"       {pct_normal:.1f}% pass Shapiro-Wilk normality test")
        print(f"       {pct_symmetric:.1f}% have fairly symmetric distributions (|skew| < 0.5)")
        print(f"       Median within-ELN skewness: {median_skew:.3f}")
        
        # Create histogram of within-ELN skewness values
        fig_skew = go.Figure()
        fig_skew.add_trace(go.Histogram(
            x=eln_stats_df['skewness'],
            nbinsx=50,
            name='Within-ELN Skewness',
            marker_color='coral',
            opacity=0.7
        ))
        
        # Add vertical lines for reference
        fig_skew.add_vline(x=0, line_dash="dash", line_color="green", 
                          annotation_text="Normal (skew=0)")
        fig_skew.add_vline(x=-0.5, line_dash="dot", line_color="orange")
        fig_skew.add_vline(x=0.5, line_dash="dot", line_color="orange",
                          annotation_text="±0.5 threshold")
        
        fig_skew.update_layout(
            title=dict(
                text=f"Distribution of Within-ELN Skewness (AREA_TOTAL_REDUCED)<br>" +
                     f"<sup>{len(eln_stats_df)} ELNs | {pct_symmetric:.1f}% fairly symmetric | " +
                     f"median skewness={median_skew:.2f}</sup>",
                font=dict(size=16)
            ),
            xaxis_title="Skewness within ELN",
            yaxis_title="Number of ELNs",
            template="plotly_white",
            width=1000,
            height=500
        )
        
        fig_skew.write_image(str(export_dir / "within_eln_skewness.png"), width=1000, height=500, scale=2)
        fig_skew.write_image(str(export_dir / "within_eln_skewness.svg"), width=1000, height=500)
        print(f"       Saved within-ELN skewness distribution")
        
        # Save summary
        with open(export_dir / "summary.txt", 'w') as f:
            f.write("Distribution Summary for AREA_TOTAL_REDUCED (Raw Data)\n")
            f.write("=" * 60 + "\n\n")
            f.write("OVERALL DISTRIBUTION:\n")
            f.write(f"  Total data points: {len(raw_data):,}\n")
            f.write(f"  Mean: {mean_val:.4f}\n")
            f.write(f"  Std: {std_val:.4f}\n")
            f.write(f"  Skewness: {skew_val:.4f}\n")
            f.write(f"  Kurtosis: {kurt_val:.4f}\n")
            f.write(f"  Shapiro-Wilk p-value (n={sample_size} sample): {shapiro_p:.2e}\n\n")
            f.write("WITHIN-ELN DISTRIBUTIONS (validates z-score transformation):\n")
            f.write(f"  ELNs analyzed (n >= 20): {len(eln_stats_df)}\n")
            f.write(f"  ELNs passing Shapiro-Wilk (α=0.05): {n_normal} ({pct_normal:.1f}%)\n")
            f.write(f"  ELNs with |skewness| < 0.5: {n_symmetric} ({pct_symmetric:.1f}%)\n")
            f.write(f"  Median within-ELN skewness: {median_skew:.4f}\n")
            f.write(f"  Median within-ELN kurtosis: {eln_stats_df['kurtosis'].median():.4f}\n")
        
        print(f"       Saved raw data summary")
        
        return {
            'overall_skewness': skew_val,
            'overall_kurtosis': kurt_val,
            'n_elns': len(eln_stats_df),
            'pct_normal_elns': pct_normal,
            'pct_symmetric_elns': pct_symmetric,
            'median_within_eln_skewness': median_skew
        }
    
    return {'overall_skewness': skew_val, 'overall_kurtosis': kurt_val}


def generate_distribution_figures(df: pd.DataFrame, export_dirs: dict):
    """Generate histogram figures for major reaction types (z-scores)."""
    
    print("\n=== Generating Z-Score Distribution Figures ===\n")
    
    export_dir = export_dirs['histograms']
    
    # Get reaction types with sufficient data
    reaction_counts = df.groupby('Reaction Type').size()
    major_reactions = reaction_counts[reaction_counts >= 500].index.tolist()
    
    print(f"Found {len(major_reactions)} reaction types with >= 500 data points")
    
    for i, reaction_type in enumerate(major_reactions[:10], 1):  # Limit to top 10
        print(f"  [{i}] Processing: {reaction_type}")
        
        # Generate histogram
        try:
            fig_hist, _ = pu.create_distribution_plot(
                df,
                value_col='z-Score',
                group_col='Reaction Type',
                group_value=reaction_type,
                presentation_mode=True
            )
            
            # Save as PNG and SVG
            safe_name = reaction_type.replace(' ', '_').replace('/', '-')
            fig_hist.write_image(
                str(export_dir / f"{safe_name}.png"),
                width=1200, height=600, scale=2
            )
            fig_hist.write_image(
                str(export_dir / f"{safe_name}.svg"),
                width=1200, height=600
            )
            print(f"       Saved histogram for {reaction_type}")
        except Exception as e:
            print(f"       Error creating histogram: {e}")


def generate_distribution_summary(df: pd.DataFrame, export_dirs: dict):
    """Generate distribution statistics summary table."""
    
    print("\n=== Generating Distribution Summary ===\n")
    
    export_dir = export_dirs['distribution_stats']
    
    # Compute distribution statistics
    dist_stats = du.compute_distribution_stats(df, group_col='Reaction Type', min_samples=20)
    
    if dist_stats.empty:
        print("  No distribution statistics available")
        return
    
    # Save as CSV
    dist_stats.to_csv(export_dir / "zscore_statistics.csv", index=False)
    print(f"  Saved distribution statistics for {len(dist_stats)} reaction types")
    
    # Generate summary
    summary = du.get_distribution_summary(df, group_col='Reaction Type')
    
    # Save summary as text
    with open(export_dir / "zscore_summary.txt", 'w') as f:
        f.write("Distribution Summary for z-Score Data\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total groups analyzed: {summary['n_groups']}\n")
        f.write(f"Groups passing Shapiro-Wilk normality test (α=0.05): {summary['n_normal']} ({summary['pct_normal']:.1f}%)\n")
        f.write(f"Groups with fairly symmetric distribution (|skew| < 0.5): {summary['n_symmetric']} ({summary['pct_symmetric']:.1f}%)\n")
        f.write(f"Groups with moderate skewness (0.5 ≤ |skew| < 1): {summary['n_moderate_skew']}\n")
        f.write(f"Groups with high skewness (|skew| ≥ 1): {summary['n_high_skew']}\n")
        f.write(f"Median skewness across groups: {summary['median_skewness']:.4f}\n")
        f.write(f"Median kurtosis across groups: {summary['median_kurtosis']:.4f}\n")
    
    print(f"  Saved distribution summary")
    
    # Generate table figure
    try:
        fig_table = pu.create_distribution_summary_table(dist_stats, presentation_mode=True)
        fig_table.write_image(
            str(export_dir / "zscore_table.png"),
            width=1400, height=max(400, 50 + len(dist_stats) * 35), scale=2
        )
        print(f"  Saved distribution table figure")
    except Exception as e:
        print(f"  Error creating table figure: {e}")
    
    return summary


def generate_permutation_tests(df: pd.DataFrame, export_dirs: dict):
    """Generate permutation test results for key comparisons.
    
    We use permutation tests rather than Kruskal-Wallis/Mann-Whitney because:
    - Multiple observations come from the same ELN (violates independence)
    - Top-5 selection creates within-ELN correlations
    - Permutation tests respect the actual data structure
    
    Uses the SAME filters as the paper boxplots to ensure consistency.
    """
    
    print("\n=== Generating Permutation Tests ===\n")
    
    export_dir = export_dirs['significance_tests']
    
    # Define key comparisons - MUST match paper boxplot filters exactly!
    # See export_boxplots.py export_paper_boxplots() for reference
    # Settings from export_boxplots.py:
    #   DEFAULT_MIN_ELN = 10 (but generate_boxplot defaults to 5 for paper plots)
    #   topn_zscore = 5
    #   EXCLUDE_CUI = ['exclude_cui']
    #   EXCLUDE_SCALEUP = [True]
    #   INCLUDE_NULL_CATEGORIES = [True]
    
    key_analyses = [
        {
            'name': 'Buchwald-Hartwig_Ligands_R2NH_ArX',
            'description': 'Secondary amines + aryl halides (top 10)',
            'reaction_types': ['Buchwald-Hartwig'],
            'category': 'Ligand',
            'fg_a': ['R2NH'],  # Secondary amines
            'fg_b': ['ArBr', 'ArCl', 'ArI'],  # Aryl halides
            'top_n': 10,  # Match paper boxplot max_components
        },
        {
            'name': 'Buchwald-Hartwig_Catalysts_R2NH_ArX',
            'description': 'Secondary amines + aryl halides (top 10)',
            'reaction_types': ['Buchwald-Hartwig'],
            'category': 'Catalyst',
            'fg_a': ['R2NH'],
            'fg_b': ['ArBr', 'ArCl', 'ArI'],
            'top_n': 10,
        },
        {
            'name': 'Suzuki-Miyaura_Catalysts_ArX_ArB',
            'description': 'Aryl halides + aryl boronates (top 12)',
            'reaction_types': ['Suzuki-Miyaura'],
            'category': 'Catalyst',
            'fg_a': ['ArBr', 'ArCl', 'ArI'],  # Aryl halides
            'fg_b': ['ArB(OR)2', 'ArB(OH)2', 'ArBF3K'],  # Aryl boronates
            'top_n': 12,  # Match paper boxplot max_components
        },
        {
            'name': 'Suzuki-Miyaura_Solvent_Base_ArX_ArB',
            'description': 'Aryl halides + aryl boronates (top 10 Solvent/Base)',
            'reaction_types': ['Suzuki-Miyaura'],
            'category': ['Solvent', 'Base'],  # Combined category
            'fg_a': ['ArBr', 'ArCl', 'ArI'],  # Aryl halides
            'fg_b': ['ArB(OR)2', 'ArB(OH)2', 'ArBF3K'],  # Aryl boronates
            'top_n': 10,
        },
        # ========== ALL COMPONENTS (no top-N filter) ==========
        {
            'name': 'Buchwald-Hartwig_Ligands_R2NH_ArX_ALL',
            'description': 'Secondary amines + aryl halides (ALL ligands)',
            'reaction_types': ['Buchwald-Hartwig'],
            'category': 'Ligand',
            'fg_a': ['R2NH'],
            'fg_b': ['ArBr', 'ArCl', 'ArI'],
            'top_n': 100,  # High number to include all
        },
        {
            'name': 'Suzuki-Miyaura_Catalysts_ArX_ArB_ALL',
            'description': 'Aryl halides + aryl boronates (ALL catalysts)',
            'reaction_types': ['Suzuki-Miyaura'],
            'category': 'Catalyst',
            'fg_a': ['ArBr', 'ArCl', 'ArI'],
            'fg_b': ['ArB(OR)2', 'ArB(OH)2', 'ArBF3K'],
            'top_n': 100,
        },
        {
            'name': 'Suzuki-Miyaura_Solvent_Base_ArX_ArB_ALL',
            'description': 'Aryl halides + aryl boronates (ALL Solvent/Base)',
            'reaction_types': ['Suzuki-Miyaura'],
            'category': ['Solvent', 'Base'],
            'fg_a': ['ArBr', 'ArCl', 'ArI'],
            'fg_b': ['ArB(OR)2', 'ArB(OH)2', 'ArBF3K'],
            'top_n': 100,
        }
    ]
    
    # Common filter settings matching export_boxplots.py
    TOPN_ZSCORE = 5
    EXCLUDE_CUI = ['exclude_cui']
    EXCLUDE_SCALEUP = [True]
    INCLUDE_NULL_CATEGORIES = [True]
    
    for analysis in key_analyses:
        print(f"  Processing: {analysis['name']}")
        print(f"    Filters: {analysis['description']}")
        
        try:
            # Handle both single category and combined categories (e.g., ['Solvent', 'Base'])
            category = analysis['category']
            if isinstance(category, list):
                reactant_types = category
                category_col = 'Combined_Category'  # Will create this column
            else:
                reactant_types = [category]
                category_col = category
            
            # Use du.filter_data() with the SAME filters as paper boxplots
            # Settings match export_boxplots.py generate_boxplot() defaults
            MIN_ELN = 5  # Default in generate_boxplot
            
            # For top-N analyses, pass max_components to filter_data (like the boxplot does)
            # For ALL analyses, use None
            max_components = analysis.get('top_n') if analysis.get('top_n', 100) < 100 else None
            
            dff = du.filter_data(
                reactant_types=reactant_types,
                reaction_types=analysis['reaction_types'],
                fg_a=analysis.get('fg_a'),
                fg_b=analysis.get('fg_b'),
                exclude_cui=EXCLUDE_CUI,
                exclude_scaleup=EXCLUDE_SCALEUP,
                include_null_categories=INCLUDE_NULL_CATEGORIES,
                min_eln=MIN_ELN,
                topn_zscore=TOPN_ZSCORE,
                max_components=max_components,
            )
            
            # For combined categories, create the combined column (e.g., "Solvent | Base")
            if isinstance(analysis['category'], list) and len(analysis['category']) > 1:
                cols = analysis['category']
                # Check all columns exist
                missing = [c for c in cols if c not in dff.columns]
                if missing:
                    print(f"    Skipping - columns not found: {missing}")
                    continue
                # Create combined column like the boxplot does
                dff[category_col] = dff[cols[0]].astype(str) + ' | ' + dff[cols[1]].astype(str)
            elif category_col not in dff.columns:
                print(f"    Skipping - {category_col} column not found")
                continue
            
            print(f"    Filtered data: {len(dff)} rows")
            
            if len(dff) < 100:
                print(f"    Skipping - insufficient data ({len(dff)} rows)")
                continue
            
            # Run permutation test (valid even with non-independent observations)
            print(f"    Running permutation test (10,000 permutations)...")
            perm_results = du.compute_permutation_test(
                dff,
                category_col=category_col,
                n_permutations=10000
            )
            
            # Save permutation test results
            perm_df = pd.DataFrame([{
                'analysis': analysis['name'],
                'n_observations': len(dff),
                'n_categories': dff[category_col].nunique(),
                'observed_h': round(perm_results['observed_h'], 4),
                'empirical_p': round(perm_results['empirical_p'], 4),
                'n_permutations': perm_results['n_permutations'],
                'permuted_h_95th': round(perm_results['permuted_h_95th'], 4),
                'significant_at_0.05': perm_results['significant_permutation']
            }])
            perm_df.to_csv(
                export_dir / f"permutation_{analysis['name']}.csv",
                index=False
            )
            
            # Also save group summary statistics (medians, counts)
            group_stats = dff.groupby(category_col)['z-Score'].agg(['count', 'median', 'mean', 'std']).round(4)
            group_stats = group_stats.sort_values('median', ascending=False)
            group_stats.to_csv(export_dir / f"group_stats_{analysis['name']}.csv")
            
            print(f"    Permutation test: H={perm_results['observed_h']:.2f}, "
                  f"empirical p={perm_results['empirical_p']:.4f} "
                  f"({'significant' if perm_results['significant_permutation'] else 'not significant'})")
            print(f"    Saved permutation analysis for {analysis['name']}")
            
        except Exception as e:
            import traceback
            print(f"    Error: {e}")
            traceback.print_exc()


def main():
    """Main entry point for generating supplementary figures."""
    
    print("\n" + "=" * 60)
    print("  Generating Supplementary Figures for Statistical Validity")
    print("=" * 60)
    
    # Create export directory structure
    export_dirs = ensure_export_dir()
    print(f"\nExport directory: {export_dirs['base'].absolute()}")
    print(f"  Subfolders: raw_data/, histograms/, distribution_stats/, significance_tests/")
    
    # Load data
    print("\nLoading data...")
    df = du.DF
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    if 'z-Score' not in df.columns:
        print("ERROR: z-Score column not found in data")
        return
    
    # Check for required columns
    z_scores = df['z-Score'].dropna()
    print(f"Valid z-Score values: {len(z_scores):,}")
    
    if len(z_scores) < 100:
        print("ERROR: Insufficient data for analysis")
        return
    
    # Generate figures (each function uses appropriate subdirectory)
    raw_summary = generate_raw_data_distribution(df, export_dirs)
    generate_distribution_figures(df, export_dirs)
    zscore_summary = generate_distribution_summary(df, export_dirs)
    generate_permutation_tests(df, export_dirs)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    if raw_summary:
        print(f"\n  Raw Data (AREA_TOTAL_REDUCED) Analysis:")
        print(f"    - Overall skewness: {raw_summary.get('overall_skewness', 'N/A'):.3f}")
        if 'n_elns' in raw_summary:
            print(f"    - {raw_summary['n_elns']} ELNs analyzed for within-ELN distributions")
            print(f"    - {raw_summary['pct_normal_elns']:.1f}% of ELNs pass normality test")
            print(f"    - {raw_summary['pct_symmetric_elns']:.1f}% of ELNs have fairly symmetric distributions")
            print(f"    - Median within-ELN skewness: {raw_summary['median_within_eln_skewness']:.3f}")
    
    if zscore_summary:
        print(f"\n  Z-Score Distribution Analysis:")
        print(f"    - {zscore_summary['n_groups']} reaction types analyzed")
        print(f"    - {zscore_summary['pct_normal']:.1f}% pass Shapiro-Wilk normality test")
        print(f"    - {zscore_summary['pct_symmetric']:.1f}% have fairly symmetric distributions")
        print(f"    - Median skewness: {zscore_summary['median_skewness']:.3f}")
    
    print(f"\n  Files saved to: {export_dirs['base'].absolute()}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
