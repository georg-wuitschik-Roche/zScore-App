#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import List
import glob

import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Spacer
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_CENTER

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


def create_flat_export(tree_root: Path, flat_root: Path) -> None:
    """Create a flat export structure by copying all PNG files to a single directory.
    
    Args:
        tree_root: Path to the tree structure directory
        flat_root: Path to the flat export directory
    """
    # Clean up old flat export first
    if flat_root.exists():
        import shutil
        print(f"Cleaning up existing flat export in {flat_root}")
        shutil.rmtree(flat_root)
    
    ensure_output_dir(flat_root)
    
    # Find all PNG files in the tree structure
    png_files = list(tree_root.glob("**/*.png"))
    
    if not png_files:
        print("No PNG files found to copy to flat export")
        return
    
    print(f"Copying {len(png_files)} PNG files to flat export...")
    
    for png_file in png_files:
        try:
            # Create a flat filename that includes the full path information
            # Get the relative path from the tree root
            rel_path = png_file.relative_to(tree_root)
            
            # Convert path separators to underscores and create a flat filename
            flat_filename = str(rel_path).replace("/", "__").replace("\\", "__")
            
            # Copy the file to the flat directory
            flat_path = flat_root / flat_filename
            import shutil
            shutil.copy2(png_file, flat_path)
            print(f"  Copied: {rel_path} -> {flat_filename}")
            
        except Exception as e:
            print(f"  Error copying {png_file}: {e}")
            continue
    
    print(f"Flat export created with {len(png_files)} files in {flat_root}")


def create_supplementary_figure_list(flat_root: Path) -> None:
    """Create a list of supplementary figures from the flat export files.
    
    Args:
        flat_root: Path to the flat export directory
    """
    # Find all PNG files in the flat export directory
    png_files = list(flat_root.glob("*.png"))
    
    if not png_files:
        print("No PNG files found to create supplementary figure list")
        return
    
    # Sort files alphabetically
    png_files.sort()
    
    # Create the supplementary figure list
    figure_list = []
    
    for i, png_file in enumerate(png_files, start=2):  # Start with number 2
        filename = png_file.stem  # Remove .png extension
        
        # Parse the filename to extract components
        # Format: [reaction]__boxplot__[reaction]__[category] for most reactions
        # Format: [reaction]__[functional_group]__boxplot__[category] for Buchwald-Hartwig
        
        parts = filename.split("__")
        
        if len(parts) >= 3:
            if parts[0] == "Buchwald-Hartwig" and len(parts) >= 4:
                # Buchwald-Hartwig format: Buchwald-Hartwig__[functional_group]__boxplot__[category]
                reaction = "Buchwald-Hartwig"
                functional_group = parts[1].replace("_", " ")
                category = parts[3].replace("_", " ")
                
                # Create descriptive text
                description = f"{category} boxplot for {reaction} reactions for {functional_group} reacting with aryl halides"
            else:
                # Other reactions format: [reaction]__boxplot__[reaction]__[category]
                reaction = parts[0].replace("_", " ")
                category = parts[3].replace("_", " ")
                
                # Handle special cases
                if reaction == "Suzuki-Miyaura":
                    description = f"{category} boxplot for {reaction} reactions for Aryl Groups"
                elif reaction == "Amide coupling" and "Coupling Reagent" in category:
                    description = f"{category} boxplot for {reaction} reactions"
                else:
                    description = f"{category} boxplot for {reaction} reactions"
            
            figure_list.append(f"Supplementary Figure {i}: {description}")
        else:
            # Fallback for unexpected format
            figure_list.append(f"Supplementary Figure {i}: {filename.replace('_', ' ')}")
    
    # Write the list to a text file
    list_file = flat_root / "supplementary_figure_list.txt"
    with open(list_file, 'w') as f:
        for item in figure_list:
            f.write(item + "\n")
    
    print(f"Supplementary figure list created: {list_file}")
    print(f"Generated {len(figure_list)} figure descriptions")
    
    # Also print to console
    print("\nSupplementary Figure List:")
    for item in figure_list:
        print(item)


def create_pdf_from_boxplots(output_root: Path) -> None:
    """Create a PDF document containing all generated boxplot images.
    
    Args:
        output_root: Path to the directory containing boxplot PNG files
    """
    # Find all PNG files in the output directory and subdirectories
    png_files = list(output_root.glob("**/*.png"))
    
    if not png_files:
        print("No PNG files found to include in PDF")
        return
    
    # Sort files for consistent ordering
    png_files.sort()
    
    # Create PDF filename
    pdf_path = output_root / "all_boxplots.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor='#1d1d1f'
    )
    
    # Add title page
    story.append(Paragraph("Boxplot Analysis Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Total plots: {len(png_files)}", styles['Normal']))
    story.append(PageBreak())
    
    # Add each image to the PDF
    for i, png_file in enumerate(png_files):
        try:
            # Create a title for the plot based on the filename
            plot_title = png_file.stem.replace("boxplot__", "").replace("__", " - ").replace("_", " ")
            
            # Add plot title
            story.append(Paragraph(f"Plot {i+1}: {plot_title}", styles['Heading2']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add the image
            # Scale image to fit page width while maintaining aspect ratio
            # First, get the actual image dimensions to calculate proper scaling
            from PIL import Image as PILImage
            with PILImage.open(png_file) as pil_img:
                img_width, img_height = pil_img.size
                aspect_ratio = img_height / img_width
                
                # Available space on page (considering margins)
                # A4 page is 8.27 x 11.69 inches, with margins we have about 7.5 x 10 inches
                max_width = 7.5 * inch
                max_height = 9.5 * inch
                
                # Calculate dimensions that fit within page bounds
                pdf_width = max_width
                pdf_height = pdf_width * aspect_ratio
                
                # If height exceeds maximum, scale down by height instead
                if pdf_height > max_height:
                    pdf_height = max_height
                    pdf_width = pdf_height / aspect_ratio
                
                img = Image(str(png_file), width=pdf_width, height=pdf_height)
                story.append(img)
            
            # Add page break after every plot except the last one
            if i < len(png_files) - 1:
                story.append(PageBreak())
                
        except Exception as e:
            print(f"Error adding {png_file} to PDF: {e}")
            continue
    
    # Build the PDF
    try:
        doc.build(story)
        print(f"PDF created successfully: {pdf_path}")
        print(f"Included {len(png_files)} boxplot images")
    except Exception as e:
        print(f"Error creating PDF: {e}")


def _has_real_components(series: pd.Series) -> bool:
    """Return True if there is at least one real (non-empty) component value.

    Treat NaN, empty strings, whitespace, and literal '<NA>' as empty.
    """
    if series is None:
        return False
    s = series.astype(str).str.strip()
    non_empty = ~s.isin(["", "nan", "NaN", "<NA>"])
    return s[non_empty].nunique(dropna=True) > 0


def export_boxplots(output_root: Path) -> None:
    # Clean up old exports first
    if output_root.exists():
        import shutil
        print(f"Cleaning up existing exports in {output_root}")
        shutil.rmtree(output_root)

    ensure_output_dir(output_root)

    # Filter to only specific reaction types as requested
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

    # Filter to only specific categories as requested
    allowed_categories = ["Solvent", "Base", "Catalyst", "Ligand"]
    
    # Generate only single category plots
    single_categories = [[cat] for cat in allowed_categories]
    print(f"Will generate plots for {len(single_categories)} single categories only: {allowed_categories}")

    for reaction in reactions:
        # Prepare per-reaction path but do not create yet
        reaction_dir = output_root / sanitize_filename(reaction)
        reaction_dir_created = False

        # Use a constant minimum ELN across all reactions for exports
        min_eln = 5

        # Get categories for this reaction
        if reaction == "Amide coupling":
            # For Amide coupling, include the combination plot
            reaction_categories = single_categories + [["Coupling Reagent", "Additive"]]
        elif reaction == "Buchwald-Hartwig":
            # For Buchwald-Hartwig, we'll handle this separately with functional group filtering
            reaction_categories = []
        else:
            # For all other reactions, use only single categories
            reaction_categories = single_categories

        for categories in reaction_categories:
            category_name = " + ".join(categories)  # Single category name except for Amide coupling
            print(f"Generating boxplot for reaction='{reaction}', category='{category_name}'")

            # Set functional group filters based on reaction type
            if reaction == "Suzuki-Miyaura":
                fg_a_filter = ['ArCl', 'ArI', 'ArBr']
                fg_b_filter = ['All']
            else:
                fg_a_filter = ['All']
                fg_b_filter = ['All']

            # Filter data the same way dashboard does
            dff = du.filter_data(
                reactant_types=categories,
                reaction_types=[reaction],
                fg_a=fg_a_filter,
                fg_b=fg_b_filter,
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
            skip_plot = False
            for cat in categories:
                if cat not in dff.columns or not _has_real_components(dff[cat]):
                    print(f"  Skipped (only '(no value)' component for {cat})")
                    skip_plot = True
                    break

            if skip_plot:
                continue

            # Check if there are at least 5 unique values in the category
            unique_values = dff[categories[0]].nunique()

            if unique_values < 5:
                print(f"  Skipped (only {unique_values} unique values, need at least 5)")
                continue

            try:
                # Customize title for Suzuki-Miyaura reactions
                if reaction == "Suzuki-Miyaura":
                    title_reaction = f"{reaction} - Aryl Groups"
                else:
                    title_reaction = reaction
                
                fig, adaptive_height = pu.create_boxplot(dff, categories, presentation_mode=True, reaction_type=title_reaction, max_categories=13)
                
                # Further increase font sizes for publication quality
                fig.update_layout(
                    title_font_size=48,
                    font_size=32,
                    xaxis_title_font_size=36,
                    yaxis_title_font_size=36,
                    xaxis_tickfont_size=28,
                    yaxis_tickfont_size=28,
                )
            except Exception as e:
                print(f"  Failed to create figure: {e}")
                continue

            # Create reaction directory on first successful figure for this reaction
            if not reaction_dir_created:
                ensure_output_dir(reaction_dir)
                reaction_dir_created = True

            # Save PNG with high resolution
            category_filename = sanitize_filename(" + ".join(categories))
            filename = f"boxplot__{sanitize_filename(reaction)}__{category_filename}.png"
            out_path = reaction_dir / filename
            try:
                fig.write_image(
                    str(out_path),
                    format="png",
                    width=1600,
                    height=max(1000, int(adaptive_height * 1.25)),
                    scale=4,
                )
                print(f"  Saved -> {out_path}")
            except Exception as e:
                print(f"  Failed to save image: {e}")

        # Special handling for Buchwald-Hartwig with functional group filtering
        if reaction == "Buchwald-Hartwig":
            # Define functional group sets for Buchwald-Hartwig
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
                
                # Create subfolder for this functional group
                fg_dir = reaction_dir / sanitize_filename(fg_name)
                ensure_output_dir(fg_dir)
                
                for categories in single_categories:
                    category_name = " + ".join(categories)  # Single category name
                    print(f"  Generating boxplot for category='{category_name}', FG='{fg_name}'")
                    
                    # Filter data for this specific functional group
                    # For Buchwald-Hartwig, limit FG B to only ArBr, ArI, and ArCl
                    dff = du.filter_data(
                        reactant_types=categories,
                        reaction_types=[reaction],
                        fg_a=fg_list,
                        fg_b=['ArBr', 'ArI', 'ArCl'],
                        exclude_cui=EXCLUDE_CUI,
                        exclude_scaleup=EXCLUDE_SCALEUP,
                        include_null_categories=INCLUDE_NULL_CATEGORIES,
                        min_eln=min_eln,
                        topn_zscore=DEFAULT_TOPN_ZSCORE,
                        max_components=DEFAULT_MAX_COMPONENTS,
                    )

                    if dff is None or dff.empty:
                        print(f"    Skipped (no data after filtering)")
                        continue

                    # Skip when only '(no value)' would be present for any category
                    skip_plot = False
                    for cat in categories:
                        if cat not in dff.columns or not _has_real_components(dff[cat]):
                            print(f"    Skipped (only '(no value)' component for {cat})")
                            skip_plot = True
                            break

                    if skip_plot:
                        continue

                    # Check if there are at least 5 unique values in the category
                    unique_values = dff[categories[0]].nunique()

                    if unique_values < 5:
                        print(f"    Skipped (only {unique_values} unique values, need at least 5)")
                        continue

                    try:
                        # Create custom title for Buchwald-Hartwig with functional group
                        title_reaction = f"{reaction} - {fg_name}"
                        
                        fig, adaptive_height = pu.create_boxplot(dff, categories, presentation_mode=True, reaction_type=title_reaction, max_categories=13)
                        
                        # Further increase font sizes for publication quality
                        fig.update_layout(
                            title_font_size=48,
                            font_size=32,
                            xaxis_title_font_size=36,
                            yaxis_title_font_size=36,
                            xaxis_tickfont_size=28,
                            yaxis_tickfont_size=28,
                        )
                    except Exception as e:
                        print(f"    Failed to create figure: {e}")
                        continue

                    # Save PNG with high resolution
                    category_filename = sanitize_filename(" + ".join(categories))
                    filename = f"boxplot__{category_filename}.png"
                    out_path = fg_dir / filename
                    try:
                        fig.write_image(
                            str(out_path),
                            format="png",
                            width=1600,
                            height=max(1000, int(adaptive_height * 1.25)),
                            scale=4,
                        )
                        print(f"    Saved -> {out_path}")
                    except Exception as e:
                        print(f"    Failed to save image: {e}")

    # Generate PDF with all boxplots after PNG export is complete
    print("\nGenerating PDF with all boxplots...")
    create_pdf_from_boxplots(output_root)
    
    # Create flat export structure
    print("\nCreating flat export structure...")
    flat_export_dir = output_root.parent / "flat_export"
    create_flat_export(output_root, flat_export_dir)
    
    # Generate PDF for flat export as well
    print("\nGenerating PDF for flat export...")
    create_pdf_from_boxplots(flat_export_dir)
    
    # Create supplementary figure list
    print("\nCreating supplementary figure list...")
    create_supplementary_figure_list(flat_export_dir)


def update_pdf_only(output_root: Path) -> None:
    """Update only the PDF file using existing PNG files.
    
    Args:
        output_root: Path to the directory containing boxplot PNG files
    """
    print("Updating PDF with existing boxplot images...")
    create_pdf_from_boxplots(output_root)


if __name__ == "__main__":
    import sys
    
    out_dir = Path("exports") / "boxplots"
    
    # Check if user wants to update PDF only
    if len(sys.argv) > 1 and sys.argv[1] == "--pdf-only":
        update_pdf_only(out_dir)
    else:
        export_boxplots(out_dir)
