#!/usr/bin/env python3
"""
generate_supplementary_figures.py
==================================
Script to generate supplementary figures for the paper addressing
reviewer concerns about statistical validity.

This script produces:
1. Histograms for all reaction types that have boxplots
2. Distribution statistics CSV

Usage:
    python generate_supplementary_figures.py

Output files are saved to the 'exports/supplementary/' directory.
"""

import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

# Import project modules
import data_utils as du
import plot_utils as pu


def ensure_export_dir():
    """Create exports directory structure if it doesn't exist."""
    base_dir = Path('exports/supplementary')

    # Create subfolder structure
    subdirs = {
        'base': base_dir,
        'histograms': base_dir / 'histograms',
        'distribution_stats': base_dir / 'distribution_stats',
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    return subdirs


def check_area_total_reduced_column(df: pd.DataFrame) -> bool:
    """Check if AREA_TOTAL_REDUCED column exists in dataframe.

    Returns:
        True if column exists, False otherwise
    """
    return 'AREA_TOTAL_REDUCED' in df.columns


def to_safe_filename(name: str) -> str:
    """Convert reaction type name to safe filename format.

    Converts spaces to underscores and slashes to hyphens.

    Args:
        name: Reaction type name (e.g., "Amide coupling")

    Returns:
        Safe filename string (e.g., "Amide_coupling")
    """
    return name.replace(' ', '_').replace('/', '-')


def match_reaction_types_from_dirs(boxplot_dir_names: set, available_reaction_types: set) -> list:
    """Match boxplot directory names to data reaction type names.

    Handles conversion between directory naming (underscores/hyphens)
    and data naming (spaces/slashes).

    Args:
        boxplot_dir_names: Set of directory names from boxplots folder
        available_reaction_types: Set of reaction types in the data

    Returns:
        List of matched reaction type names from the data
    """
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
                    rt_converted = to_safe_filename(rt)
                    if rt_converted == dir_name:
                        reaction_types_to_plot.append(rt)
                        break

    return reaction_types_to_plot


def generate_distribution_figures(df: pd.DataFrame, export_dirs: dict):
    """Generate histogram figures for all reaction types that have boxplots (AREA_TOTAL_REDUCED).

    This shows the underlying distribution of raw yield data (AREA_TOTAL_REDUCED) for each
    reaction type to address reviewer concerns about non-normal distributions.

    Reaction types are determined by checking which directories exist in exports/boxplots/.
    This ensures all reaction types with boxplots also have histograms.
    """

    print('\n=== Generating Raw Data (AREA_TOTAL_REDUCED) Distribution Figures ===\n')

    export_dir = export_dirs['histograms']

    # Check if AREA_TOTAL_REDUCED column exists
    if not check_area_total_reduced_column(df):
        print('  WARNING: AREA_TOTAL_REDUCED column not found - skipping histogram generation')
        return

    # Get reaction types from boxplots directory
    boxplots_dir = Path('exports/boxplots')
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
    reaction_types_to_plot = match_reaction_types_from_dirs(boxplot_dir_names, available_reaction_types)

    # If no boxplots directory found, fall back to all reaction types in data
    if not reaction_types_to_plot:
        print('  WARNING: No boxplots directory found, generating histograms for all reaction types in data')
        reaction_types_to_plot = sorted(available_reaction_types)

    print(f'Found {len(reaction_types_to_plot)} reaction types with boxplots to process')

    for i, reaction_type in enumerate(reaction_types_to_plot, 1):
        print(f'  [{i}] Processing: {reaction_type}')

        # Check if we have data for this reaction type
        reaction_data = df[df['Reaction Type'] == reaction_type]['AREA_TOTAL_REDUCED'].dropna()
        if len(reaction_data) < 20:
            print(f'       Skipping - insufficient data ({len(reaction_data)} values)')
            continue

        # Generate histogram
        try:
            fig_hist, _ = pu.create_distribution_plot(
                df,
                value_col='AREA_TOTAL_REDUCED',
                group_col='Reaction Type',
                group_value=reaction_type,
                presentation_mode=True,
            )

            # Save as PNG and SVG with histogram_ prefix
            safe_name = to_safe_filename(reaction_type)
            fig_hist.write_image(str(export_dir / f'histogram_{safe_name}.png'), width=1200, height=600, scale=2)
            fig_hist.write_image(str(export_dir / f'histogram_{safe_name}.svg'), width=1200, height=600)
            print(f'       Saved histogram for {reaction_type} ({len(reaction_data)} data points)')
        except Exception as e:
            print(f'       Error creating histogram: {e}')


def generate_distribution_summary(df: pd.DataFrame, export_dirs: dict):
    """Generate distribution statistics CSV for AREA_TOTAL_REDUCED (raw data)."""

    print('\n=== Generating Distribution Summary ===\n')

    export_dir = export_dirs['distribution_stats']

    # Check if AREA_TOTAL_REDUCED column exists
    if not check_area_total_reduced_column(df):
        print('  WARNING: AREA_TOTAL_REDUCED column not found - skipping distribution summary')
        return None

    # Compute distribution statistics for raw data
    dist_stats = du.compute_distribution_stats(
        df, group_col='Reaction Type', value_col='AREA_TOTAL_REDUCED', min_samples=20
    )

    if dist_stats.empty:
        print('  No distribution statistics available')
        return None

    # Save as CSV
    dist_stats.to_csv(export_dir / 'raw_data_statistics.csv', index=False)
    print(f'  Saved distribution statistics for {len(dist_stats)} reaction types')

    # Generate summary for return value
    summary = du.get_distribution_summary(df, group_col='Reaction Type', value_col='AREA_TOTAL_REDUCED')

    return summary


def main():
    """Main entry point for generating supplementary figures."""

    print('\n' + '=' * 60)
    print('  Generating Supplementary Figures for Statistical Validity')
    print('=' * 60)

    # Create export directory structure
    export_dirs = ensure_export_dir()
    print(f'\nExport directory: {export_dirs["base"].absolute()}')
    print('  Subfolders: histograms/, distribution_stats/')

    # Load data
    print('\nLoading data...')
    df = du.DF
    print(f'Loaded {len(df):,} rows, {len(df.columns)} columns')

    if 'z-Score' not in df.columns:
        print('ERROR: z-Score column not found in data')
        return

    # Check for required columns
    z_scores = df['z-Score'].dropna()
    print(f'Valid z-Score values: {len(z_scores):,}')

    if len(z_scores) < 100:
        print('ERROR: Insufficient data for analysis')
        return

    # Generate figures (each function uses appropriate subdirectory)
    generate_distribution_figures(df, export_dirs)
    dist_summary = generate_distribution_summary(df, export_dirs)

    # Print final summary
    print('\n' + '=' * 60)
    print('  Summary')
    print('=' * 60)

    if dist_summary:
        print('\n  Distribution Analysis:')
        print(f'    - {dist_summary["n_groups"]} reaction types analyzed')
        print(f'    - {dist_summary["pct_normal"]:.1f}% pass Shapiro-Wilk normality test')
        print(f'    - {dist_summary["pct_symmetric"]:.1f}% have fairly symmetric distributions')
        print(f'    - Median skewness: {dist_summary["median_skewness"]:.3f}')

    print(f'\n  Files saved to: {export_dirs["base"].absolute()}')
    print('\n  Done!')


if __name__ == '__main__':
    main()
