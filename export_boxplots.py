#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

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
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def clean_directory(path: Path, message: Optional[str] = None) -> None:
    """Remove directory if it exists and print optional message."""
    if path.exists():
        if message:
            print(message)
        shutil.rmtree(path)


def sanitize_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    return (
        name.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


def apply_publication_fonts(fig: go.Figure) -> None:
    """Apply publication-quality font sizes to a figure."""
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


def save_figure(
    fig: go.Figure,
    output_dir: Path,
    base_filename: str,
    adaptive_height: float,
    indent: str = "  ",
) -> None:
    """Save a figure as both PNG and SVG.
    
    Args:
        fig: Plotly figure to save
        output_dir: Directory to save files in
        base_filename: Filename without extension
        adaptive_height: Height value from create_boxplot
        indent: Indentation for log messages
    """
    height = max(1000, int(adaptive_height * 1.25))
    
    # Save PNG with high resolution
    png_path = output_dir / f"{base_filename}.png"
    try:
        fig.write_image(
            str(png_path),
            format="png",
            width=1600,
            height=height,
            scale=4,
        )
        print(f"{indent}Saved -> {png_path}")
    except Exception as e:
        print(f"{indent}Failed to save PNG: {e}")

    # Save SVG for vector graphics
    svg_path = output_dir / f"{base_filename}.svg"
    try:
        fig.write_image(
            str(svg_path),
            format="svg",
            width=1600,
            height=height,
        )
        print(f"{indent}Saved -> {svg_path}")
    except Exception as e:
        print(f"{indent}Failed to save SVG: {e}")


def _has_real_components(series: pd.Series) -> bool:
    """Return True if there is at least one real (non-empty) component value.

    Treat NaN, empty strings, whitespace, and literal '<NA>' as empty.
    """
    if series is None:
        return False
    s = series.astype(str).str.strip()
    non_empty = ~s.isin(["", "nan", "NaN", "<NA>"])
    return s[non_empty].nunique(dropna=True) > 0


def validate_data_for_plot(
    dff: Optional[pd.DataFrame],
    categories: List[str],
    min_unique: int = 5,
    indent: str = "  ",
) -> bool:
    """Validate that filtered data is suitable for plotting.
    
    Args:
        dff: Filtered DataFrame
        categories: List of category columns
        min_unique: Minimum number of unique values required
        indent: Indentation for log messages
        
    Returns:
        True if data is valid for plotting, False otherwise
    """
    if dff is None or dff.empty:
        print(f"{indent}Skipped (no data after filtering)")
        return False

    # Check for real components in each category
    for cat in categories:
        if cat not in dff.columns or not _has_real_components(dff[cat]):
            print(f"{indent}Skipped (only '(no value)' component for {cat})")
            return False

    # Check minimum unique values
    unique_values = dff[categories[0]].nunique()
    if unique_values < min_unique:
        print(f"{indent}Skipped (only {unique_values} unique values, need at least {min_unique})")
        return False

    return True


def create_flat_export(tree_root: Path, flat_root: Path) -> None:
    """Create a flat export structure by copying all image files to a single directory.
    
    Args:
        tree_root: Path to the tree structure directory
        flat_root: Path to the flat export directory
    """
    clean_directory(flat_root, f"Cleaning up existing flat export in {flat_root}")
    ensure_output_dir(flat_root)
    
    # Find all PNG and SVG files in the tree structure
    image_files = list(tree_root.glob("**/*.png")) + list(tree_root.glob("**/*.svg"))
    
    if not image_files:
        print("No image files found to copy to flat export")
        return
    
    print(f"Copying {len(image_files)} image files to flat export...")
    
    for image_file in image_files:
        try:
            # Get relative path and convert to flat filename
            rel_path = image_file.relative_to(tree_root)
            flat_filename = str(rel_path).replace("/", "__").replace("\\", "__")
            flat_path = flat_root / flat_filename
            shutil.copy2(image_file, flat_path)
            print(f"  Copied: {rel_path} -> {flat_filename}")
        except Exception as e:
            print(f"  Error copying {image_file}: {e}")
    
    print(f"Flat export created with {len(image_files)} files in {flat_root}")


def create_supplementary_figure_list(flat_root: Path) -> None:
    """Create a list of supplementary figures from the flat export files.
    
    Args:
        flat_root: Path to the flat export directory
    """
    png_files = sorted(flat_root.glob("*.png"))
    
    if not png_files:
        print("No PNG files found to create supplementary figure list")
        return
    
    figure_list = []
    
    for i, png_file in enumerate(png_files, start=2):
        filename = png_file.stem
        parts = filename.split("__")
        
        if len(parts) >= 3:
            if parts[0] == "Buchwald-Hartwig" and len(parts) >= 4:
                reaction = "Buchwald-Hartwig"
                functional_group = parts[1].replace("_", " ")
                category = parts[3].replace("_", " ")
                description = f"{category} boxplot for {reaction} reactions for {functional_group} reacting with aryl halides"
            else:
                reaction = parts[0].replace("_", " ")
                category = parts[3].replace("_", " ")
                
                if reaction == "Suzuki-Miyaura":
                    description = f"{category} boxplot for {reaction} reactions for Aryl Groups"
                elif reaction == "Amide coupling" and "Coupling Reagent" in category:
                    description = f"{category} boxplot for {reaction} reactions"
                else:
                    description = f"{category} boxplot for {reaction} reactions"
            
            figure_list.append(f"Supplementary Figure {i}: {description}")
        else:
            figure_list.append(f"Supplementary Figure {i}: {filename.replace('_', ' ')}")
    
    # Write the list to a text file
    list_file = flat_root / "supplementary_figure_list.txt"
    with open(list_file, 'w') as f:
        for item in figure_list:
            f.write(item + "\n")
    
    print(f"Supplementary figure list created: {list_file}")
    print(f"Generated {len(figure_list)} figure descriptions")
    print("\nSupplementary Figure List:")
    for item in figure_list:
        print(item)


def generate_boxplot(
    categories: List[str],
    reaction: str,
    fg_a: List[str],
    fg_b: List[str],
    output_dir: Path,
    base_filename: str,
    title_reaction: str,
    min_eln: int = 5,
    topn_zscore: int = DEFAULT_TOPN_ZSCORE,
    max_components: Optional[int] = DEFAULT_MAX_COMPONENTS,
    min_unique: int = 5,
    indent: str = "  ",
) -> bool:
    """Generate and save a boxplot with the given parameters.
    
    Args:
        categories: List of category columns to plot
        reaction: Reaction type to filter by
        fg_a: Functional group A filter
        fg_b: Functional group B filter
        output_dir: Directory to save the figure
        base_filename: Base filename without extension
        title_reaction: Title to display on the plot
        min_eln: Minimum ELN count filter
        topn_zscore: Top N z-score filter
        max_components: Maximum components to include
        min_unique: Minimum unique values required
        indent: Indentation for log messages
        
    Returns:
        True if plot was generated successfully, False otherwise
    """
    # Filter data
    dff = du.filter_data(
        reactant_types=categories,
        reaction_types=[reaction],
        fg_a=fg_a,
        fg_b=fg_b,
        exclude_cui=EXCLUDE_CUI,
        exclude_scaleup=EXCLUDE_SCALEUP,
        include_null_categories=INCLUDE_NULL_CATEGORIES,
        min_eln=min_eln,
        topn_zscore=topn_zscore,
        max_components=max_components,
    )

    # Validate data
    if not validate_data_for_plot(dff, categories, min_unique=min_unique, indent=indent):
        return False

    try:
        fig, adaptive_height = pu.create_boxplot(
            dff,
            categories,
            presentation_mode=True,
            reaction_type=title_reaction,
            max_categories=max_components if max_components else 13,
        )
        apply_publication_fonts(fig)
    except Exception as e:
        print(f"{indent}Failed to create figure: {e}")
        return False

    save_figure(fig, output_dir, base_filename, adaptive_height, indent=indent)
    return True


def export_boxplots(output_root: Path) -> None:
    """Export all boxplots for supplementary materials."""
    clean_directory(output_root, f"Cleaning up existing exports in {output_root}")
    ensure_output_dir(output_root)

    allowed_reactions = [
        "Buchwald-Hartwig",
        "Suzuki-Miyaura", 
        "Amide coupling",
        "CO-Coupling",
        "CH-Activation",
        "CN-Coupling",
        "Arylation, acidic C-H"
    ]
    
    reactions = [rt for rt in du.REACTION_TYPES if isinstance(rt, str) and rt.strip() != "" and rt in allowed_reactions]
    if not reactions:
        print(f"No allowed reaction types found in dataset. Looking for: {allowed_reactions}")
        return

    allowed_categories = ["Solvent", "Base", "Catalyst", "Ligand"]
    single_categories = [[cat] for cat in allowed_categories]
    print(f"Will generate plots for {len(single_categories)} single categories only: {allowed_categories}")

    for reaction in reactions:
        reaction_dir = output_root / sanitize_filename(reaction)
        reaction_dir_created = False

        # Get categories for this reaction
        if reaction == "Amide coupling":
            reaction_categories = single_categories + [["Coupling Reagent", "Additive"]]
        elif reaction == "Buchwald-Hartwig":
            reaction_categories = []  # Handled separately
        else:
            reaction_categories = single_categories

        # Set functional group filters based on reaction type
        if reaction == "Suzuki-Miyaura":
            fg_a_filter = ['ArCl', 'ArI', 'ArBr']
            fg_b_filter = ['All']
        else:
            fg_a_filter = ['All']
            fg_b_filter = ['All']

        for categories in reaction_categories:
            category_name = " + ".join(categories)
            print(f"Generating boxplot for reaction='{reaction}', category='{category_name}'")

            title_reaction = f"{reaction} - Aryl Groups" if reaction == "Suzuki-Miyaura" else reaction
            
            if not reaction_dir_created:
                ensure_output_dir(reaction_dir)
                reaction_dir_created = True

            category_filename = sanitize_filename(" + ".join(categories))
            base_filename = f"boxplot__{sanitize_filename(reaction)}__{category_filename}"
            
            generate_boxplot(
                categories=categories,
                reaction=reaction,
                fg_a=fg_a_filter,
                fg_b=fg_b_filter,
                output_dir=reaction_dir,
                base_filename=base_filename,
                title_reaction=title_reaction,
            )

        # Special handling for Buchwald-Hartwig with functional group filtering
        if reaction == "Buchwald-Hartwig":
            if not reaction_dir_created:
                ensure_output_dir(reaction_dir)
                
            fg_groups = [
                (["RNH2"], "RNH2"),
                (["RNH2 a-branch"], "RNH2_a-branch"),
                (["R2NH"], "R2NH"),
                (["R2NH a-branch"], "R2NH_a-branch"),
                (["ArNH2"], "ArNH2"),
                (["ArNHR"], "ArNHR"),
                (["RCONH2", "RCONHR", "Lactam", "Urea"], "Amides")
            ]
            
            for fg_list, fg_name in fg_groups:
                print(f"Generating Buchwald-Hartwig plots for functional group: {fg_name}")
                fg_dir = reaction_dir / sanitize_filename(fg_name)
                ensure_output_dir(fg_dir)
                
                for categories in single_categories:
                    category_name = " + ".join(categories)
                    print(f"  Generating boxplot for category='{category_name}', FG='{fg_name}'")
                    
                    title_reaction = f"{reaction} - {fg_name}"
                    category_filename = sanitize_filename(" + ".join(categories))
                    base_filename = f"boxplot__{category_filename}"
                    
                    generate_boxplot(
                        categories=categories,
                        reaction=reaction,
                        fg_a=fg_list,
                        fg_b=['ArBr', 'ArI', 'ArCl'],
                        output_dir=fg_dir,
                        base_filename=base_filename,
                        title_reaction=title_reaction,
                        indent="    ",
                    )

    # Create flat export structure
    print("\nCreating flat export structure...")
    flat_export_dir = output_root.parent / "flat_export"
    create_flat_export(output_root, flat_export_dir)
    
    # Create supplementary figure list
    print("\nCreating supplementary figure list...")
    create_supplementary_figure_list(flat_export_dir)


def export_paper_boxplots(output_root: Path) -> None:
    """Export specific boxplots for paper figures.
    
    This function exports five specific plots:
    1. Boxplot of z-score by ligand for aryl bromides/aryl chlorides reacting with 
       secondary amines (top 10 ligands) - Buchwald-Hartwig
    2. Boxplot of z-score by catalyst for aryl halides reacting with secondary amines 
       (top 10 catalysts) - Buchwald-Hartwig
    3. Boxplot of z-score by catalyst for aryl halides reacting with aryl boronates 
       (top 12 catalysts) - Suzuki-Miyaura
    4. Boxplot of z-score by solvent/base combinations for aryl halides reacting with 
       aryl boronates (top 10 combinations) - Suzuki-Miyaura
    5. Boxplot of z-score by solvent/base/catalyst combinations for aryl halides reacting with 
       aryl boronates (top 10 combinations) - Suzuki-Miyaura
    """
    clean_directory(output_root, f"Cleaning up existing exports in {output_root}")
    ensure_output_dir(output_root)

    plots = [
        {
            "name": "buchwald_hartwig_ligand_R2NH_ArX",
            "title": "Buchwald-Hartwig - Secondary Amines + Aryl Halides",
            "description": "z-score by ligand for aryl bromides/chlorides reacting with secondary amines",
            "reaction_type": "Buchwald-Hartwig",
            "category": ["Ligand"],
            "fg_a": ["R2NH"],
            "fg_b": ["ArBr", "ArCl", "ArI"],
            "max_components": 10,
        },
        {
            "name": "buchwald_hartwig_catalyst_R2NH_ArX",
            "title": "Buchwald-Hartwig - Secondary Amines + Aryl Halides",
            "description": "z-score by catalyst for aryl halides reacting with secondary amines",
            "reaction_type": "Buchwald-Hartwig",
            "category": ["Catalyst"],
            "fg_a": ["R2NH"],
            "fg_b": ["ArBr", "ArCl", "ArI"],
            "max_components": 10,
        },
        {
            "name": "suzuki_miyaura_catalyst_ArX_ArB",
            "title": "Suzuki-Miyaura - Aryl Halides + Aryl Boronates",
            "description": "z-score by catalyst for aryl halides reacting with aryl boronates",
            "reaction_type": "Suzuki-Miyaura",
            "category": ["Catalyst"],
            "fg_a": ["ArBr", "ArCl", "ArI"],
            "fg_b": ["ArB(OR)2", "ArB(OH)2", "ArBF3K"],
            "max_components": 12,
        },
        {
            "name": "suzuki_miyaura_solvent_base_ArX_ArB",
            "title": "Suzuki-Miyaura - Aryl Halides + Aryl Boronates",
            "description": "z-score by solvent/base combinations for aryl halides reacting with aryl boronates",
            "reaction_type": "Suzuki-Miyaura",
            "category": ["Solvent", "Base"],
            "fg_a": ["ArBr", "ArCl", "ArI"],
            "fg_b": ["ArB(OR)2", "ArB(OH)2", "ArBF3K"],
            "max_components": 10,
        },
        {
            "name": "suzuki_miyaura_catalyst_solvent_base_ArX_ArB",
            "title": "Suzuki-Miyaura - Aryl Halides + Aryl Boronates",
            "description": "z-score by catalyst/solvent/base combinations for aryl halides reacting with aryl boronates",
            "reaction_type": "Suzuki-Miyaura",
            "category": ["Catalyst", "Solvent", "Base"],
            "fg_a": ["ArBr", "ArCl", "ArI"],
            "fg_b": ["ArB(OR)2", "ArB(OH)2", "ArBF3K"],
            "max_components": 10,
        },
    ]

    for plot_config in plots:
        print(f"\nGenerating: {plot_config['description']}")
        
        generate_boxplot(
            categories=plot_config["category"],
            reaction=plot_config["reaction_type"],
            fg_a=plot_config["fg_a"],
            fg_b=plot_config["fg_b"],
            output_dir=output_root,
            base_filename=plot_config["name"],
            title_reaction=plot_config["title"],
            topn_zscore=5,
            max_components=plot_config["max_components"],
            min_unique=2,
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--paper":
            out_dir = Path("exports") / "paper_boxplots"
            export_paper_boxplots(out_dir)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python export_boxplots.py [OPTIONS]")
            print()
            print("Options:")
            print("  (no args)    Export all boxplots to exports/boxplots/")
            print("  --paper      Export paper-specific boxplots to exports/paper_boxplots/")
            print("  --help, -h   Show this help message")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        out_dir = Path("exports") / "boxplots"
        export_boxplots(out_dir)
