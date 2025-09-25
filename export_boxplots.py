#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import List
from itertools import combinations

import pandas as pd

# Reuse app utilities
import data_utils as du
import plot_utils as pu

# Get all available reaction types and reactant categories dynamically
# All reaction types are derived from the dataset
# Reactant categories exclude functional group categories as they're not reactant types
REACTANT_CATEGORIES: List[str] = [
    cat for cat in du.CATEGORY_OPTIONS
    if cat not in ["Functional Group A", "Functional Group B"]
]

# Default filtering knobs to match dashboard sensibly
DEFAULT_MIN_ELN = 10
DEFAULT_TOPN_ZSCORE = 3
DEFAULT_MAX_COMPONENTS = None  # None => include all components
EXCLUDE_CUI = ['exclude_cui']
EXCLUDE_SCALEUP = [True]
INCLUDE_NULL_CATEGORIES = [True]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


def _has_real_components(series: pd.Series) -> bool:
    """Return True if there is at least one real (non-empty) component value.

    Treat NaN, empty strings, whitespace, and literal '<NA>' as empty.
    """
    if series is None:
        return False
    s = series.astype(str).str.strip()
    non_empty = ~s.isin(["", "nan", "NaN", "<NA>"])
    return s[non_empty].nunique(dropna=True) > 0


def _generate_category_combinations(categories: List[str]) -> List[List[str]]:
    """Generate all combinations of categories: singles and pairs only."""
    combinations_list = []

    # Add single categories
    for cat in categories:
        combinations_list.append([cat])

    # Add pairs of categories
    for pair in combinations(categories, 2):
        combinations_list.append(list(pair))

    return combinations_list


def export_boxplots(output_root: Path) -> None:
    # Clean up old exports first
    if output_root.exists():
        import shutil
        print(f"Cleaning up existing exports in {output_root}")
        shutil.rmtree(output_root)

    ensure_output_dir(output_root)

    # Determine reaction types dynamically from dataset
    reactions = [rt for rt in du.REACTION_TYPES if isinstance(rt, str) and rt.strip() != ""]
    if not reactions:
        print("No reaction types found in dataset. Aborting.")
        return

    # Generate all category combinations (singles + pairs)
    category_combinations = _generate_category_combinations(REACTANT_CATEGORIES)
    print(f"Will generate plots for {len(category_combinations)} category combinations")

    for reaction in reactions:
        # Prepare per-reaction path but do not create yet
        reaction_dir = output_root / sanitize_filename(reaction)
        reaction_dir_created = False

        # Adjust defaults per domain rule seen in callbacks: 10 for BH/SM else 5
        min_eln = 10 if reaction in ("Buchwald-Hartwig", "Suzuki-Miyaura") else 5

        for category_combo in category_combinations:
            combo_name = " + ".join(category_combo)
            print(f"Generating boxplot for reaction='{reaction}', categories='{combo_name}'")

            # Filter data the same way dashboard does
            dff = du.filter_data(
                reactant_types=category_combo,
                reaction_types=[reaction],
                fg_a=['All'],
                fg_b=['All'],
                exclude_cui=EXCLUDE_CUI,
                exclude_scaleup=EXCLUDE_SCALEUP,
                include_null_categories=INCLUDE_NULL_CATEGORIES,
                min_eln=min_eln,
                topn_zscore=DEFAULT_TOPN_ZSCORE,
                max_components=DEFAULT_MAX_COMPONENTS,
            )

            if dff is None or dff.empty:
                print(f"  Skipped (no data after filtering)")
                continue

            # Skip when only '(no value)' would be present for any category
            skip_combo = False
            for cat in category_combo:
                if cat not in dff.columns or not _has_real_components(dff[cat]):
                    print(f"  Skipped (only '(no value)' component for {cat})")
                    skip_combo = True
                    break

            if skip_combo:
                continue

            # Check if there are at least 3 unique combinations of the selected categories
            if len(category_combo) == 1:
                # For single category, count unique values in that category
                unique_combinations = dff[category_combo[0]].nunique()
            else:
                # For multiple categories, count unique combinations
                unique_combinations = dff.groupby(category_combo).size().shape[0]

            if unique_combinations < 3:
                print(f"  Skipped (only {unique_combinations} unique combinations, need at least 3)")
                continue

            try:
                fig, adaptive_height = pu.create_boxplot(dff, category_combo, presentation_mode=False, reaction_type=reaction)
            except Exception as e:
                print(f"  Failed to create figure: {e}")
                continue

            # Create reaction directory on first successful figure for this reaction
            if not reaction_dir_created:
                ensure_output_dir(reaction_dir)
                reaction_dir_created = True

            # Save PNG using kaleido
            combo_filename = sanitize_filename(" + ".join(category_combo))
            filename = f"boxplot__{sanitize_filename(reaction)}__{combo_filename}.png"
            out_path = reaction_dir / filename
            try:
                # Width similar to download callback; height from adaptive
                fig.write_image(str(out_path), format="png", width=1200, height=max(800, adaptive_height), scale=2)
                print(f"  Saved -> {out_path}")
            except Exception as e:
                print(f"  Failed to save image: {e}")


if __name__ == "__main__":
    out_dir = Path("exports") / "boxplots"
    export_boxplots(out_dir)
