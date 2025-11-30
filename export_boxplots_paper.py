#!/usr/bin/env python3
"""Export specific boxplots for paper figures.

This script exports four specific plots:
1. Boxplot of z-score by ligand for aryl bromides/aryl chlorides reacting with 
   secondary amines (top 10 ligands) - Buchwald-Hartwig
2. Boxplot of z-score by catalyst for aryl halides reacting with secondary amines 
   (top 10 catalysts) - Buchwald-Hartwig
3. Boxplot of z-score by catalyst for aryl halides reacting with aryl boronates 
   (top 12 catalysts) - Suzuki-Miyaura
4. Boxplot of z-score by solvent/base combinations for aryl halides reacting with 
   aryl boronates (top 10 combinations) - Suzuki-Miyaura
"""
from __future__ import annotations

from pathlib import Path

# Reuse app utilities
import data_utils as du
import plot_utils as pu
from export_boxplots import (
    ensure_output_dir,
    sanitize_filename,
    create_pdf_from_boxplots,
)

# Filtering defaults
MIN_ELN = 5
TOPN_ZSCORE = 5
EXCLUDE_CUI = ['exclude_cui']
EXCLUDE_SCALEUP = [True]
INCLUDE_NULL_CATEGORIES = [True]


def export_paper_boxplots(output_root: Path) -> None:
    """Export the two specific paper boxplots."""
    # Clean up old exports first
    if output_root.exists():
        import shutil
        print(f"Cleaning up existing exports in {output_root}")
        shutil.rmtree(output_root)

    ensure_output_dir(output_root)

    # Define the plots to generate
    # Naming convention: reactiontype_component_fga_fgb
    plots = [
        # Buchwald-Hartwig plots
        {
            "name": "buchwald_hartwig_ligand_R2NH_ArX",
            "title": "Buchwald-Hartwig - Secondary Amines + ArBr/ArCl",
            "description": "z-score by ligand for aryl bromides/chlorides reacting with secondary amines",
            "reaction_type": "Buchwald-Hartwig",
            "category": ["Ligand"],
            "fg_a": ["R2NH"],  # Secondary amines
            "fg_b": ["ArBr", "ArCl", "ArI"],  # Aryl bromides and chlorides
            "max_components": 10,
        },
        {
            "name": "buchwald_hartwig_catalyst_R2NH_ArX",
            "title": "Buchwald-Hartwig - Secondary Amines + Aryl Halides",
            "description": "z-score by catalyst for aryl halides reacting with secondary amines",
            "reaction_type": "Buchwald-Hartwig",
            "category": ["Catalyst"],
            "fg_a": ["R2NH"],  # Secondary amines
            "fg_b": ["ArBr", "ArCl", "ArI"],  # All aryl halides
            "max_components": 10,
        },
        # Suzuki-Miyaura plots
        {
            "name": "suzuki_miyaura_catalyst_ArX_ArB",
            "title": "Suzuki-Miyaura - Aryl Halides + Aryl Boronates",
            "description": "z-score by catalyst for aryl halides reacting with aryl boronates",
            "reaction_type": "Suzuki-Miyaura",
            "category": ["Catalyst"],
            "fg_a": ["ArBr", "ArCl", "ArI"],  # Aryl halides
            "fg_b": ["ArB(OR)2", "ArB(OH)2", "ArBF3K"],  # Aryl boronates
            "max_components": 12,
        },
        {
            "name": "suzuki_miyaura_solvent_base_ArX_ArB",
            "title": "Suzuki-Miyaura - Aryl Halides + Aryl Boronates",
            "description": "z-score by solvent/base combinations for aryl halides reacting with aryl boronates",
            "reaction_type": "Suzuki-Miyaura",
            "category": ["Solvent", "Base"],  # Combination of solvent and base
            "fg_a": ["ArBr", "ArCl", "ArI"],  # Aryl halides
            "fg_b": ["ArB(OR)2", "ArB(OH)2", "ArBF3K"],  # Aryl boronates
            "max_components": 10,
        },
    ]

    for plot_config in plots:
        print(f"\nGenerating: {plot_config['description']}")

        # Filter data
        dff = du.filter_data(
            reactant_types=plot_config["category"],
            reaction_types=[plot_config["reaction_type"]],
            fg_a=plot_config["fg_a"],
            fg_b=plot_config["fg_b"],
            exclude_cui=EXCLUDE_CUI,
            exclude_scaleup=EXCLUDE_SCALEUP,
            include_null_categories=INCLUDE_NULL_CATEGORIES,
            min_eln=MIN_ELN,
            topn_zscore=TOPN_ZSCORE,
            max_components=plot_config["max_components"],
        )

        if dff is None or dff.empty:
            print(f"  Skipped (no data after filtering)")
            continue

        # Check unique values
        if len(plot_config["category"]) == 1:
            category_col = plot_config["category"][0]
            unique_values = dff[category_col].nunique()
            print(f"  Found {unique_values} unique {category_col} values")
        else:
            # For combination categories, count unique combinations
            category_cols = plot_config["category"]
            unique_values = dff[category_cols].drop_duplicates().shape[0]
            print(f"  Found {unique_values} unique {' + '.join(category_cols)} combinations")

        if unique_values < 2:
            print(f"  Skipped (not enough unique values)")
            continue

        try:
            # Create the boxplot with presentation mode for larger fonts
            fig, adaptive_height = pu.create_boxplot(
                dff,
                plot_config["category"],
                presentation_mode=True,  # Larger fonts for paper
                reaction_type=plot_config["title"],
                max_categories=plot_config["max_components"],
            )
            
            # Further increase font sizes for paper publication (non-bold)
            fig.update_layout(
                title_font=dict(size=48, weight="normal"),
                font=dict(size=32, weight="normal"),
                xaxis=dict(
                    title_font=dict(size=36, weight="normal"),
                    tickfont=dict(size=28, weight="normal"),
                ),
                yaxis=dict(
                    title_font=dict(size=36, weight="normal"),
                    tickfont=dict(size=28, weight="normal"),
                ),
            )
        except Exception as e:
            print(f"  Failed to create figure: {e}")
            continue

        # Save PNG with high resolution for paper
        filename = f"{plot_config['name']}.png"
        out_path = output_root / filename
        try:
            fig.write_image(
                str(out_path),
                format="png",
                width=1600,  # Larger base size
                height=max(1000, int(adaptive_height * 1.25)),
                scale=4,  # Higher resolution
            )
            print(f"  Saved -> {out_path}")
        except Exception as e:
            print(f"  Failed to save image: {e}")

    # Generate PDF with all boxplots
    print("\nGenerating PDF with paper boxplots...")
    create_pdf_from_boxplots(output_root)


if __name__ == "__main__":
    out_dir = Path("exports") / "paper_boxplots"
    export_paper_boxplots(out_dir)

