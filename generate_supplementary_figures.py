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
   - Histograms with normal overlays for all reaction types that have boxplots
   - Summary table of distribution statistics

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
    
    # 2. Within-Reaction Type distribution analysis
    print("\n  [2] Within-Reaction Type distribution analysis...")
    
    if 'Reaction Type' not in df.columns:
        print("       WARNING: Reaction Type column not found - skipping within-reaction type analysis")
        return {'overall_skewness': skew_val, 'overall_kurtosis': kurt_val}
    
    # Calculate within-Reaction Type statistics
    reaction_stats = []
    reactions_with_data = df.groupby('Reaction Type')[raw_col].filter(lambda x: x.dropna().shape[0] >= 20)
    unique_reactions = df.loc[reactions_with_data.index, 'Reaction Type'].unique()
    
    print(f"       Analyzing {len(unique_reactions)} reaction types with >= 20 data points each")
    
    for reaction_type in unique_reactions:
        reaction_data = df[df['Reaction Type'] == reaction_type][raw_col].dropna()
        if len(reaction_data) >= 20:
            reaction_skew = stats.skew(reaction_data)
            reaction_kurt = stats.kurtosis(reaction_data)
            
            # Shapiro-Wilk test
            sample_n = min(5000, len(reaction_data))
            reaction_sample = reaction_data.sample(n=sample_n, random_state=42) if len(reaction_data) > sample_n else reaction_data
            try:
                _, reaction_shapiro_p = stats.shapiro(reaction_sample)
            except:
                reaction_shapiro_p = np.nan
            
            reaction_stats.append({
                'Reaction Type': reaction_type,
                'n': len(reaction_data),
                'mean': reaction_data.mean(),
                'std': reaction_data.std(),
                'skewness': reaction_skew,
                'kurtosis': reaction_kurt,
                'shapiro_p': reaction_shapiro_p,
                'is_normal': reaction_shapiro_p > 0.05 if not np.isnan(reaction_shapiro_p) else False
            })
    
    reaction_stats_df = pd.DataFrame(reaction_stats)
    
    if len(reaction_stats_df) > 0:
        # Save reaction type-level statistics
        reaction_stats_df.to_csv(export_dir / "within_reaction_type_statistics.csv", index=False)
        
        # Summary statistics
        n_normal = reaction_stats_df['is_normal'].sum()
        pct_normal = 100 * n_normal / len(reaction_stats_df)
        median_skew = reaction_stats_df['skewness'].median()
        n_symmetric = (reaction_stats_df['skewness'].abs() < 0.5).sum()
        pct_symmetric = 100 * n_symmetric / len(reaction_stats_df)
        
        print(f"       {len(reaction_stats_df)} reaction types analyzed")
        print(f"       {pct_normal:.1f}% pass Shapiro-Wilk normality test")
        print(f"       {pct_symmetric:.1f}% have fairly symmetric distributions (|skew| < 0.5)")
        print(f"       Median within-reaction type skewness: {median_skew:.3f}")
        
        # Create histogram of within-reaction type skewness values
        fig_skew = go.Figure()
        fig_skew.add_trace(go.Histogram(
            x=reaction_stats_df['skewness'],
            nbinsx=50,
            name='Within-Reaction Type Skewness',
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
                text=f"Distribution of Within-Reaction Type Skewness (AREA_TOTAL_REDUCED)<br>" +
                     f"<sup>{len(reaction_stats_df)} reaction types | {pct_symmetric:.1f}% fairly symmetric | " +
                     f"median skewness={median_skew:.2f}</sup>",
                font=dict(size=16)
            ),
            xaxis_title="Skewness within Reaction Type",
            yaxis_title="Number of Reaction Types",
            template="plotly_white",
            width=1000,
            height=500
        )
        
        fig_skew.write_image(str(export_dir / "within_reaction_type_skewness.png"), width=1000, height=500, scale=2)
        fig_skew.write_image(str(export_dir / "within_reaction_type_skewness.svg"), width=1000, height=500)
        print(f"       Saved within-reaction type skewness distribution")
        
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
            f.write("WITHIN-REACTION TYPE DISTRIBUTIONS (validates z-score transformation):\n")
            f.write(f"  Reaction types analyzed (n >= 20): {len(reaction_stats_df)}\n")
            f.write(f"  Reaction types passing Shapiro-Wilk (α=0.05): {n_normal} ({pct_normal:.1f}%)\n")
            f.write(f"  Reaction types with |skewness| < 0.5: {n_symmetric} ({pct_symmetric:.1f}%)\n")
            f.write(f"  Median within-reaction type skewness: {median_skew:.4f}\n")
            f.write(f"  Median within-reaction type kurtosis: {reaction_stats_df['kurtosis'].median():.4f}\n")
        
        print(f"       Saved raw data summary")
        
        return {
            'overall_skewness': skew_val,
            'overall_kurtosis': kurt_val,
            'n_reaction_types': len(reaction_stats_df),
            'pct_normal_reactions': pct_normal,
            'pct_symmetric_reactions': pct_symmetric,
            'median_within_reaction_type_skewness': median_skew
        }
    
    return {'overall_skewness': skew_val, 'overall_kurtosis': kurt_val}


def generate_distribution_figures(df: pd.DataFrame, export_dirs: dict):
    """Generate histogram figures for all reaction types that have boxplots (AREA_TOTAL_REDUCED).
    
    This shows the underlying distribution of raw yield data (AREA_TOTAL_REDUCED) for each
    reaction type to address reviewer concerns about non-normal distributions.
    
    Reaction types are determined by checking which directories exist in exports/boxplots/.
    This ensures all reaction types with boxplots also have histograms.
    """
    
    print("\n=== Generating Raw Data (AREA_TOTAL_REDUCED) Distribution Figures ===\n")
    
    export_dir = export_dirs['histograms']
    
    # Check if AREA_TOTAL_REDUCED column exists
    if 'AREA_TOTAL_REDUCED' not in df.columns:
        print("  WARNING: AREA_TOTAL_REDUCED column not found - skipping histogram generation")
        return
    
    # Get reaction types from boxplots directory
    boxplots_dir = Path("exports/boxplots")
    boxplot_dir_names = set()
    
    if boxplots_dir.exists():
        # Get top-level directories (reaction types)
        for item in boxplots_dir.iterdir():
            if item.is_dir():
                boxplot_dir_names.add(item.name)
    
    # Also check what reaction types exist in the data
    available_reaction_types = set(df['Reaction Type'].dropna().unique())
    
    # Create mapping: convert boxplot directory names to data reaction type names
    # Directory names use underscores/hyphens, data uses spaces/slashes
    reaction_types_to_plot = []
    for dir_name in sorted(boxplot_dir_names):
        # Try to match directory name to data reaction type
        # First try exact match
        if dir_name in available_reaction_types:
            reaction_types_to_plot.append(dir_name)
        else:
            # Try converting: underscore -> space, hyphen -> slash
            converted = dir_name.replace('_', ' ').replace('-', '/')
            if converted in available_reaction_types:
                reaction_types_to_plot.append(converted)
            else:
                # Try reverse: space -> underscore, slash -> hyphen
                for rt in available_reaction_types:
                    rt_converted = rt.replace(' ', '_').replace('/', '-')
                    if rt_converted == dir_name:
                        reaction_types_to_plot.append(rt)
                        break
    
    # If no boxplots directory found, fall back to all reaction types in data
    if not reaction_types_to_plot:
        print("  WARNING: No boxplots directory found, generating histograms for all reaction types in data")
        reaction_types_to_plot = sorted(available_reaction_types)
    
    print(f"Found {len(reaction_types_to_plot)} reaction types with boxplots to process")
    
    for i, reaction_type in enumerate(reaction_types_to_plot, 1):
        print(f"  [{i}] Processing: {reaction_type}")
        
        # Check if we have data for this reaction type
        reaction_data = df[df['Reaction Type'] == reaction_type]['AREA_TOTAL_REDUCED'].dropna()
        if len(reaction_data) < 20:
            print(f"       Skipping - insufficient data ({len(reaction_data)} values)")
            continue
        
        # Generate histogram
        try:
            fig_hist, _ = pu.create_distribution_plot(
                df,
                value_col='AREA_TOTAL_REDUCED',
                group_col='Reaction Type',
                group_value=reaction_type,
                presentation_mode=True
            )
            
            # Save as PNG and SVG (use same filename format as boxplots)
            safe_name = reaction_type.replace(' ', '_').replace('/', '-')
            fig_hist.write_image(
                str(export_dir / f"{safe_name}.png"),
                width=1200, height=600, scale=2
            )
            fig_hist.write_image(
                str(export_dir / f"{safe_name}.svg"),
                width=1200, height=600
            )
            print(f"       Saved histogram for {reaction_type} ({len(reaction_data)} data points)")
        except Exception as e:
            print(f"       Error creating histogram: {e}")


def generate_distribution_summary(df: pd.DataFrame, export_dirs: dict):
    """Generate distribution statistics summary table for AREA_TOTAL_REDUCED (raw data)."""
    
    print("\n=== Generating Distribution Summary ===\n")
    
    export_dir = export_dirs['distribution_stats']
    
    # Check if AREA_TOTAL_REDUCED column exists
    if 'AREA_TOTAL_REDUCED' not in df.columns:
        print("  WARNING: AREA_TOTAL_REDUCED column not found - skipping distribution summary")
        return None
    
    # Compute distribution statistics for raw data
    dist_stats = du.compute_distribution_stats(df, group_col='Reaction Type', value_col='AREA_TOTAL_REDUCED', min_samples=20)
    
    if dist_stats.empty:
        print("  No distribution statistics available")
        return None
    
    # Save as CSV
    dist_stats.to_csv(export_dir / "raw_data_statistics.csv", index=False)
    print(f"  Saved distribution statistics for {len(dist_stats)} reaction types")
    
    # Generate summary
    summary = du.get_distribution_summary(df, group_col='Reaction Type', value_col='AREA_TOTAL_REDUCED')
    
    # Save summary as text
    with open(export_dir / "raw_data_summary.txt", 'w') as f:
        f.write("Distribution Summary for AREA_TOTAL_REDUCED (Raw Data)\n")
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
            str(export_dir / "raw_data_table.png"),
            width=1400, height=max(400, 50 + len(dist_stats) * 35), scale=2
        )
        print(f"  Saved distribution table figure")
    except Exception as e:
        print(f"  Error creating table figure: {e}")
    
    return summary


def main():
    """Main entry point for generating supplementary figures."""
    
    print("\n" + "=" * 60)
    print("  Generating Supplementary Figures for Statistical Validity")
    print("=" * 60)
    
    # Create export directory structure
    export_dirs = ensure_export_dir()
    print(f"\nExport directory: {export_dirs['base'].absolute()}")
    print(f"  Subfolders: raw_data/, histograms/, distribution_stats/")
    
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
    
    # Print final summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    if raw_summary:
        print(f"\n  Raw Data (AREA_TOTAL_REDUCED) Analysis:")
        print(f"    - Overall skewness: {raw_summary.get('overall_skewness', 'N/A'):.3f}")
        if 'n_reaction_types' in raw_summary:
            print(f"    - {raw_summary['n_reaction_types']} reaction types analyzed for within-reaction type distributions")
            print(f"    - {raw_summary['pct_normal_reactions']:.1f}% of reaction types pass normality test")
            print(f"    - {raw_summary['pct_symmetric_reactions']:.1f}% of reaction types have fairly symmetric distributions")
            print(f"    - Median within-reaction type skewness: {raw_summary['median_within_reaction_type_skewness']:.3f}")
    
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
