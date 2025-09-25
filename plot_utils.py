from __future__ import annotations

"""plot_utils.py
================
Tiny helper module that keeps everything *visualisation*-related in one
place.  The public API purposefully mirrors the old inline functions so
existing callbacks can simply `import plot_utils as pu` and call
`pu.create_boxplot(...)`.
"""

from typing import Dict, Tuple

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

import data_utils as du


def _safe_str_conversion(series):
    """Convert a pandas Series to string while handling null values gracefully."""
    return series.fillna('(no value)').astype(str)

# ---------------------------------------------------------------------------
# 1. COLOUR MAPPING HELPER
# ---------------------------------------------------------------------------

# A human readable mapping from *chemical entity* to *base colour*.  The
# actual shade is then calculated via interpolation depending on the
# number of ELNs present for that particular entity.
BASE_COLOURS: Dict[str, Dict[str, str]] = {
    "Catalyst": {"light": "#89CFF1", "dark": "#003A6B"},  # blue shades
    "Solvent": {"light": "#90EE90", "dark": "#006400"},  # green shades
    "Base": {"light": "#FFB347", "dark": "#CC5500"},  # orange shades
    "Ligand": {"light": "#E6E6FA", "dark": "#4B0082"},  # purple shades
    "Additive": {"light": "#FFB6C1", "dark": "#8B0000"},  # red shades
    "Coupling Reagent": {"light": "#E6E6FA", "dark": "#191970"},
    "Functional Group A": {"light": "#FFC0CB", "dark": "#C71585"},
    "Functional Group B": {"light": "#87CEEB", "dark": "#006994"},
    "Secondary Solvent": {"light": "#98FB98", "dark": "#228B22"},
}


def _interpolate_hex(col1: str, col2: str, factor: float) -> str:
    """Linear interpolation between two hex colours (0 <= *factor* <= 1)."""

    def hex_to_rgb(hex_color: str):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#" + "".join(f"{c:02x}" for c in rgb)

    r1, g1, b1 = hex_to_rgb(col1)
    r2, g2, b2 = hex_to_rgb(col2)

    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)

    return rgb_to_hex((r, g, b))


def create_color_mapping(category: str, dff) -> Dict[str, str]:
    """Return a dict *category value -> colour*.

    The more ELNs a category value has the *darker* its colour becomes in
    the boxplot which gives the viewer a quick visual cue about data
    density.
    """

    base = BASE_COLOURS.get(category, {"light": "#D3D3D3", "dark": "#696969"})

    eln_counts = dff.groupby(category)["ELN_ID"].nunique()
    max_elns, min_elns = eln_counts.max(), eln_counts.min()

    colour_map: Dict[str, str] = {}
    for cat_val, cnt in eln_counts.items():
        factor = 0.5 if max_elns == min_elns else (cnt - min_elns) / (max_elns - min_elns)
        colour_map[cat_val] = _interpolate_hex(base["light"], base["dark"], factor)

    return colour_map


# ---------------------------------------------------------------------------
# 2. BOXPLOT CREATION
# ---------------------------------------------------------------------------


def create_boxplot(dff, reactant_types: list, base_height: int = 800, presentation_mode: bool = False, reaction_type: str = None) -> Tuple[go.Figure, int]:
    """Return `(figure, adaptive_height)` for the given dataframe.

    Args:
        dff: The filtered dataframe to plot
        reactant_types: List of selected reactant types (categories) to display
        base_height: Minimum height for the plot
        presentation_mode: Whether to use larger fonts for presentation
        reaction_type: Optional reaction type to include in title (for exports)

    The adaptive height makes sure the plot remains readable even with a
    large number of category values.
    """

    if not reactant_types or len(reactant_types) == 0:
        raise ValueError("At least one reactant type must be selected for boxplot")
    
    # Handle multiple reactant types by creating combined category labels
    if len(reactant_types) > 1:
        # Multiple reactant types selected - create combined category for y-axis
        dff = dff.copy()
        category_parts = []
        for reactant_type in reactant_types:
            if reactant_type in dff.columns:
                category_parts.append(_safe_str_conversion(dff[reactant_type]))
        
        if category_parts:
            # Combine the series element-wise with ' | ' separator
            if len(category_parts) == 1:
                dff['Combined_Category'] = category_parts[0]
            else:
                # Combine multiple series element-wise
                combined_values = category_parts[0].astype(str)
                for i in range(1, len(category_parts)):
                    combined_values = combined_values + ' | ' + category_parts[i].astype(str)
                dff['Combined_Category'] = combined_values
            y_category = 'Combined_Category'
            reactant_title = f"Boxplot of z-Score by {' | '.join(reactant_types)}"
            title = f"{reaction_type} - {reactant_title}" if reaction_type else reactant_title
        else:
            # Fallback to first available reactant type
            y_category = reactant_types[0]
            reactant_title = f"Boxplot of z-Score by {reactant_types[0]}"
            title = f"{reaction_type} - {reactant_title}" if reaction_type else reactant_title
            if dff[y_category].isnull().any():
                dff[y_category] = _safe_str_conversion(dff[y_category])
    else:
        # Single reactant type - handle null values for display
        dff = dff.copy()
        y_category = reactant_types[0]
        reactant_title = f"Boxplot of z-Score by {reactant_types[0]}"
        title = f"{reaction_type} - {reactant_title}" if reaction_type else reactant_title
        # Apply safe string conversion to handle null values gracefully
        if y_category in dff.columns and dff[y_category].isnull().any():
            dff[y_category] = _safe_str_conversion(dff[y_category])

    # 1. Ordering
    medians = dff.groupby(y_category)["z-Score"].median().sort_values(ascending=False)
    category_order = medians.index.tolist()

    # 2. Colour mapping (needs to run *before* adaptive height is computed because we call it anyway)
    colour_map = create_color_mapping(y_category, dff)

    # 3. Height calculation
    height = max(base_height, len(category_order) * 110)

    # Add custom data for hover template with all columns
    dff_hover = dff.copy()
    
    # Create comprehensive hover text with explicit HTML template
    def create_hover_text(row):
        # Helper function to clean values
        def clean_value(val):
            if pd.isna(val) or val == '<NA>' or val == '' or str(val).strip() == '':
                return ''
            return str(val)
        
        # Format numeric values appropriately
        z_score = f"{row['z-Score']:.3f}" if pd.notna(row['z-Score']) and row['z-Score'] != '<NA>' else ""
        area = f"{row['AREA_TOTAL_REDUCED']:.2f}%" if pd.notna(row['AREA_TOTAL_REDUCED']) and row['AREA_TOTAL_REDUCED'] != '<NA>' else ""
        
        # Build HTML template with all columns explicitly listed
        hover_html = f"""
        <b>Experiment Details:</b><br>
        ELN_ID: {clean_value(row.get('ELN_ID', ''))}<br>
        Plate: {clean_value(row.get('PLATENUMBER', ''))}<br>
        Coordinate: {clean_value(row.get('Coordinate', ''))}<br>
        <br>
        <b>Results:</b><br>
        z-Score: {z_score}<br>
        Area: {area}<br>
        <br>
        <b>Reaction:</b><br>
        Reaction Type: {clean_value(row.get('Reaction Type', ''))}<br>
        <br>
        <b>Reagents:</b><br>
        Catalyst: {clean_value(row.get('Catalyst', ''))}<br>
        Solvent: {clean_value(row.get('Solvent', ''))}<br>
        Base: {clean_value(row.get('Base', ''))}<br>
        Ligand: {clean_value(row.get('Ligand', ''))}<br>
        Additive: {clean_value(row.get('Additive', ''))}<br>
        Coupling Reagent: {clean_value(row.get('Coupling Reagent', ''))}<br>
        Functional Group A: {clean_value(row.get('FG A', ''))}<br>
        Functional Group B: {clean_value(row.get('FG B', ''))}<br>
        Secondary Solvent: {clean_value(row.get('Secondary Solvent', ''))}<br>
        """
        
        
        return hover_html
    
    dff_hover['hover_text'] = dff_hover.apply(create_hover_text, axis=1)
    
    # Calculate ELN count per category for tooltip
    eln_counts = dff_hover.groupby(y_category)['ELN_ID'].nunique()
    dff_hover['eln_count'] = dff_hover[y_category].map(eln_counts)
    
    fig = px.box(
        dff_hover,
        y=y_category,
        x="z-Score",
        color=y_category,
        points="all",
        title=title,
        category_orders={y_category: category_order},
        height=height,
        color_discrete_map=colour_map,
        custom_data=['hover_text', 'eln_count']
    )

    # ------------------------------------------------------------------
    # Styling tweaks – those are visual, not functional, so feel free to
    # adjust them for your branding.
    # ------------------------------------------------------------------
    # Create hover template for boxplot points with more detailed information
    hover_template = (
        "<b>%{y}</b><br>" +
        "z-Score: %{x:.2f}<br>" +
        "ELNs for this reactant type: %{customdata[1]}<br>" +
        "%{customdata[0]}" +
        "<extra></extra>"
    )
    
    fig.update_traces(
        hovertemplate=hover_template,
        hoverinfo="all"
    )

    # Adjust font sizes for presentation mode
    title_size = 32 if presentation_mode else 22
    base_font_size = 20 if presentation_mode else 14
    tick_font_size = 18 if presentation_mode else 14
    axis_title_size = 22 if presentation_mode else 16
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font_size=title_size,
        title_font_family="Helvetica Neue",
        title_font_color="#1d1d1f",
        margin=dict(l=60, r=60, t=100, b=60),
        font=dict(family="Helvetica Neue", size=base_font_size),
    )

    fig.update_xaxes(
        tickangle=0,
        showgrid=True,
        gridwidth=2,
        gridcolor="#d0d0d0",
        zeroline=False,
        showline=True,
        linewidth=3,
        linecolor="#cccccc",
        tickmode="auto",
        nticks=6,
        tickfont=dict(size=tick_font_size, weight="bold"),
        title_font=dict(size=axis_title_size, weight="bold"),
        # Make axis labels selectable
        ticktext=None,  # Use default tick text
        tickvals=None,  # Use default tick values
    )

    fig.update_yaxes(
        tickangle=0,
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=3,
        linecolor="#cccccc",
        tickfont=dict(size=tick_font_size, weight="bold"),
        title_font=dict(size=axis_title_size, weight="bold"),
        # Make axis labels selectable
        ticktext=None,  # Use default tick text
        tickvals=None,  # Use default tick values
    )

    return fig, height

# ---------------------------------------------------------------------------
# 3. HEATMAP CREATION
# ---------------------------------------------------------------------------


def create_heatmap(dff, reactant_types: list, base_height: int = 800, presentation_mode: bool = False) -> Tuple[go.Figure, int]:
    """Return `(figure, adaptive_height)` for a heatmap visualization.

    Args:
        dff: The filtered dataframe to plot
        reactant_types: List of selected reactant types (categories) to display
        base_height: Minimum height for the plot
        presentation_mode: Whether to use larger fonts for presentation

    Creates a heatmap with the first reactant type on y-axis and remaining types on x-axis.
    Requires at least two reactant types to be selected.
    """

    import plotly.graph_objects as go
    import numpy as np

    # Adjust font sizes for presentation mode (must be defined FIRST)
    title_size = 32 if presentation_mode else 22
    base_font_size = 20 if presentation_mode else 14
    tick_font_size = 18 if presentation_mode else 14
    axis_title_size = 22 if presentation_mode else 16
    colorbar_title_size = 18 if presentation_mode else 14
    colorbar_tick_size = 16 if presentation_mode else 12
    text_font_size = 14 if presentation_mode else 10

    # Require at least two reactant types for heatmap
    if not reactant_types or len(reactant_types) < 2:
        raise ValueError("Heatmap requires at least two reactant types to be selected")

    # Create hierarchical structure for x-axis
    y_category = reactant_types[0]  # First reactant type goes on y-axis
    
    if len(reactant_types) > 2:
        # Three or more reactant types selected - create hierarchical x-axis
        dff = dff.copy()
        # Create hierarchical labels for x-axis: reactant_type2 | reactant_type3 | ...
        x_parts = []
        for i in range(1, len(reactant_types)):
            if reactant_types[i] in dff.columns:
                x_parts.append(_safe_str_conversion(dff[reactant_types[i]]))
        
        if len(x_parts) > 1:
            # Combine the series element-wise with ' | ' separator
            combined_x_values = x_parts[0].astype(str)
            for i in range(1, len(x_parts)):
                combined_x_values = combined_x_values + ' | ' + x_parts[i].astype(str)
            dff['X_Category'] = combined_x_values
            x_category = 'X_Category'
            title = f'Heatmap: {reactant_types[0]} vs {" | ".join(reactant_types[1:])}'
        else:
            # Fallback to second reactant type only
            x_category = reactant_types[1]
            title = f'Heatmap: {reactant_types[0]} vs {reactant_types[1]}'
    else:
        # Two reactant types selected - second type on x-axis, first on y-axis
        x_category = reactant_types[1]
        title = f'Heatmap: {reactant_types[0]} vs {reactant_types[1]}'

    # Apply safe string conversion for y_category (single category case) if it has null values
    if not hasattr(dff, 'X_Category') and dff[y_category].isnull().any():
        dff[y_category] = _safe_str_conversion(dff[y_category])

    # Order y-axis categories by median z-Score (ascending) - best performing on top
    y_medians = dff.groupby(y_category)["z-Score"].median().sort_values(ascending=True)
    y_category_order = y_medians.index.tolist()

    if x_category:
        # Apply safe string conversion for x_category if it has null values and is not already combined
        if x_category != 'X_Category' and dff[x_category].isnull().any():
            dff[x_category] = _safe_str_conversion(dff[x_category])
        
        # Order x-axis categories by median z-Score (descending)
        x_medians = dff.groupby(x_category)["z-Score"].median().sort_values(ascending=False)
        x_category_order = x_medians.index.tolist()
        
        # Create 2D heatmap data: y_category vs x_category
        heatmap_data = []
        eln_counts = []  # Store ELN counts for tooltip
        for y_cat in y_category_order:
            row_data = []
            eln_row = []
            for x_cat in x_category_order:
                # Get data for this combination
                mask = (dff[y_category] == y_cat) & (dff[x_category] == x_cat)
                subset_data = dff[mask]["z-Score"]
                if len(subset_data) > 0:
                    # Use median z-score for this combination, excluding null values
                    valid_data = subset_data.dropna()
                    if len(valid_data) > 0:
                        row_data.append(valid_data.median())
                        # Count unique ELNs for this combination
                        eln_count = dff[mask]["ELN_ID"].nunique()
                        eln_row.append(eln_count)
                    else:
                        row_data.append(np.nan)  # All data was null
                        eln_row.append(0)
                else:
                    row_data.append(np.nan)  # No data for this combination
                    eln_row.append(0)
            heatmap_data.append(row_data)
            eln_counts.append(eln_row)
        
        heatmap_data = np.array(heatmap_data)
        eln_counts = np.array(eln_counts)
        
        # Flatten eln_counts for customdata (Plotly expects 1D array for heatmap customdata)
        # Create heatmap with categories on both axes
        # Calculate color scale bounds from valid data only
        valid_data = heatmap_data[~np.isnan(heatmap_data)]
        if len(valid_data) > 0:
            # Use percentiles for more robust color scaling
            zmin = np.percentile(valid_data, 5)  # 5th percentile
            zmax = np.percentile(valid_data, 95)  # 95th percentile
            zmid = np.median(valid_data)  # median as white point
            
            # Create dynamic color scale based on actual data range
            colorscale = [
                [0, 'blue'],
                [(zmid - zmin) / (zmax - zmin), 'white'],
                [1, 'red']
            ]
        else:
            zmin, zmax, zmid = 0, 1, 0.5
            colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]
            
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=x_category_order,
            y=y_category_order,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            text=[[f"{val:.2f}" if not np.isnan(val) else "" for _idx_val, val in enumerate(row)] for _idx_row, row in enumerate(heatmap_data)],
            texttemplate="%{text}",
            textfont={"size": text_font_size, "color": "black"},
            colorbar=dict(
                title=dict(
                    text="Median z-Score",
                    font=dict(size=colorbar_title_size, family="Helvetica Neue")
                ),
                tickfont=dict(size=colorbar_tick_size, family="Helvetica Neue")
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         '<b>%{x}</b><br>' +
                         'Median z-Score: %{z:.2f}<br>' +
                         'Number of ELNs: %{customdata[0]}<br>' +
                         '<extra></extra>',
            hoverongaps=False,
            customdata = eln_counts[..., None]          # shape (ny, nx, 1)

        ))
        
        # Update x-axis title
        if len(reactant_types) > 2:
            x_axis_title = " | ".join(reactant_types[1:])
        else:
            x_axis_title = reactant_types[1]

    # Height calculation
    num_y_categories = len(y_category_order)
    height = max(base_height, num_y_categories * 80)

    # Styling
    fig.update_layout(
        title=title,
        title_font_size=title_size,
        title_font_family="Helvetica Neue",
        title_font_color="#1d1d1f",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=100, b=60),
        font=dict(family="Helvetica Neue", size=base_font_size),
        height=height,
        xaxis=dict(
            title=dict(
                text=x_axis_title,
                font=dict(size=axis_title_size, weight="bold", family="Helvetica Neue")
            ),
            tickfont=dict(size=tick_font_size, weight="bold", family="Helvetica Neue"),
            showgrid=True,
            gridwidth=1,
            gridcolor="#d0d0d0",
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="#cccccc",
            side="top",  # Move x-axis labels to the top
            # Make axis labels selectable
            ticktext=None,  # Use default tick text
            tickvals=None,  # Use default tick values
        ),
        yaxis=dict(
            title=dict(
                text=reactant_types[0],
                font=dict(size=axis_title_size, weight="bold", family="Helvetica Neue")
            ),
            tickfont=dict(size=tick_font_size, weight="bold", family="Helvetica Neue"),
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="#cccccc",
            # Make axis labels selectable
            ticktext=None,  # Use default tick text
            tickvals=None,  # Use default tick values
        )
    )

    return fig, height
