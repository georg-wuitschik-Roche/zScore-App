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


def create_boxplot(dff, reactant_types: list, base_height: int = 800, presentation_mode: bool = False, reaction_type: str = None, max_categories: int = None) -> Tuple[go.Figure, int]:
    """Return `(figure, adaptive_height)` for the given dataframe.

    Args:
        dff: The filtered dataframe to plot
        reactant_types: List of selected reactant types (categories) to display
        base_height: Minimum height for the plot
        presentation_mode: Whether to use larger fonts for presentation
        reaction_type: Optional reaction type to include in title (for exports)
        max_categories: Optional maximum number of categories to display (by median z-Score)

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

    # 1b. Limit categories if max_categories is specified
    if max_categories is not None and len(category_order) > max_categories:
        category_order = category_order[:max_categories]
        # Filter dataframe to only include the top categories
        dff = dff[dff[y_category].isin(category_order)]

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
        <b>Reaction Conditions:</b><br>
        {clean_value(row.get('output_column', ''))}<br>
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


# ---------------------------------------------------------------------------
# 4. DISTRIBUTION DIAGNOSTIC PLOTS
# ---------------------------------------------------------------------------


def create_distribution_plot(
    dff: pd.DataFrame,
    value_col: str = 'z-Score',
    group_col: str = None,
    group_value: str = None,
    title: str = None,
    presentation_mode: bool = False
) -> Tuple[go.Figure, int]:
    """Create a histogram for distribution analysis.
    
    This visualization shows the distribution of the data values.
    
    Args:
        dff: DataFrame containing the data to plot
        value_col: Column containing the values to plot (default: 'z-Score')
        group_col: Optional column to filter by (e.g., 'Reaction Type')
        group_value: Value to filter group_col by
        title: Optional custom title
        presentation_mode: Whether to use larger fonts for presentation
        
    Returns:
        Tuple of (figure, height)
    """
    from scipy import stats
    
    # Filter data if group specified
    if group_col and group_value:
        data = dff[dff[group_col] == group_value][value_col].dropna()
        default_title = f'Distribution of {value_col} for {group_value}'
    else:
        data = dff[value_col].dropna()
        default_title = f'Distribution of {value_col}'
    
    title = title or default_title
    
    # Font sizes
    title_size = 28 if presentation_mode else 20
    base_font_size = 18 if presentation_mode else 14
    tick_font_size = 16 if presentation_mode else 12
    annotation_size = 14 if presentation_mode else 11
    
    # Create figure with histogram
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=50,
        name='Observed',
        opacity=0.7,
        marker_color='#4A90D9',
        histnorm='probability density'
    ))
    
    # Calculate statistics for annotation
    if len(data) > 10:
        skewness = data.skew()
        kurtosis = data.kurtosis()
        n = len(data)
        
        # Shapiro-Wilk test (sample if too large)
        sample_for_test = data.sample(min(5000, n), random_state=42) if n > 5000 else data
        try:
            _, shapiro_p = stats.shapiro(sample_for_test)
            shapiro_text = f'p={shapiro_p:.2e}' if shapiro_p < 0.001 else f'p={shapiro_p:.4f}'
        except Exception:
            shapiro_text = 'N/A'
        
        # Add annotation with statistics
        stats_text = (
            f"<b>Distribution Statistics</b><br>"
            f"n = {n:,}<br>"
            f"Skewness = {skewness:.3f}<br>"
            f"Kurtosis = {kurtosis:.3f}<br>"
            f"Shapiro-Wilk {shapiro_text}"
        )
        
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref='paper',
            yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=annotation_size, family='Helvetica Neue'),
            align='right',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#cccccc',
            borderwidth=1,
            borderpad=8
        )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size, family='Helvetica Neue')),
        xaxis_title=value_col,
        yaxis_title='Density',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Helvetica Neue', size=base_font_size),
        margin=dict(l=60, r=60, t=80, b=60),
        height=500
    )
    
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#e0e0e0',
        showline=True, linewidth=2, linecolor='#cccccc',
        tickfont=dict(size=tick_font_size)
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#e0e0e0',
        showline=True, linewidth=2, linecolor='#cccccc',
        tickfont=dict(size=tick_font_size)
    )
    
    return fig, 500


def create_qq_plot(
    dff: pd.DataFrame,
    value_col: str = 'z-Score',
    group_col: str = None,
    group_value: str = None,
    title: str = None,
    presentation_mode: bool = False
) -> Tuple[go.Figure, int]:
    """Create a Q-Q (quantile-quantile) plot to assess normality.
    
    A Q-Q plot compares the quantiles of the observed data against
    theoretical quantiles from a normal distribution. Points falling
    on the diagonal line indicate normal distribution.
    
    Args:
        dff: DataFrame containing the data to plot
        value_col: Column containing the values to plot (default: 'z-Score')
        group_col: Optional column to filter by (e.g., 'Reaction Type')
        group_value: Value to filter group_col by
        title: Optional custom title
        presentation_mode: Whether to use larger fonts for presentation
        
    Returns:
        Tuple of (figure, height)
    """
    from scipy import stats
    
    # Filter data if group specified
    if group_col and group_value:
        data = dff[dff[group_col] == group_value][value_col].dropna()
        default_title = f'Q-Q Plot of {value_col} for {group_value}'
    else:
        data = dff[value_col].dropna()
        default_title = f'Q-Q Plot of {value_col}'
    
    title = title or default_title
    
    # Font sizes
    title_size = 28 if presentation_mode else 20
    base_font_size = 18 if presentation_mode else 14
    tick_font_size = 16 if presentation_mode else 12
    annotation_size = 14 if presentation_mode else 11
    
    # Calculate theoretical quantiles
    data_sorted = np.sort(data)
    n = len(data_sorted)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=data_sorted,
        mode='markers',
        name='Data',
        marker=dict(color='#4A90D9', size=6, opacity=0.6)
    ))
    
    # Add reference line (y = x scaled to data)
    mu, std = data.mean(), data.std()
    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    line_y = mu + std * line_x
    
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name='Normal Reference',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # Calculate statistics for annotation
    skewness = data.skew()
    kurtosis = data.kurtosis()
    
    # Shapiro-Wilk test
    sample_for_test = data.sample(min(5000, n), random_state=42) if n > 5000 else data
    try:
        _, shapiro_p = stats.shapiro(sample_for_test)
        normality_status = "Normal" if shapiro_p > 0.05 else "Non-normal"
        shapiro_text = f'p={shapiro_p:.2e}' if shapiro_p < 0.001 else f'p={shapiro_p:.4f}'
    except Exception:
        normality_status = "Unknown"
        shapiro_text = 'N/A'
    
    # Add annotation
    stats_text = (
        f"<b>Normality Assessment</b><br>"
        f"n = {n:,}<br>"
        f"Skewness = {skewness:.3f}<br>"
        f"Kurtosis = {kurtosis:.3f}<br>"
        f"Shapiro-Wilk {shapiro_text}<br>"
        f"<b>Status: {normality_status}</b>"
    )
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=stats_text,
        showarrow=False,
        font=dict(size=annotation_size, family='Helvetica Neue'),
        align='left',
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#cccccc',
        borderwidth=1,
        borderpad=8
    )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size, family='Helvetica Neue')),
        xaxis_title='Theoretical Quantiles (Normal)',
        yaxis_title=f'Sample Quantiles ({value_col})',
        showlegend=True,
        legend=dict(x=0.7, y=0.15, bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Helvetica Neue', size=base_font_size),
        margin=dict(l=60, r=60, t=80, b=60),
        height=500
    )
    
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#e0e0e0',
        showline=True, linewidth=2, linecolor='#cccccc',
        tickfont=dict(size=tick_font_size),
        scaleanchor='y', scaleratio=1
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#e0e0e0',
        showline=True, linewidth=2, linecolor='#cccccc',
        tickfont=dict(size=tick_font_size)
    )
    
    return fig, 500


def create_distribution_summary_table(
    dist_stats: pd.DataFrame,
    presentation_mode: bool = False
) -> go.Figure:
    """Create a formatted table showing distribution statistics for multiple groups.
    
    Args:
        dist_stats: DataFrame from compute_distribution_stats()
        presentation_mode: Whether to use larger fonts
        
    Returns:
        Plotly figure containing the table
    """
    if dist_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No distribution statistics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Format the dataframe for display
    display_df = dist_stats.copy()
    
    # Format p-values
    if 'shapiro_p' in display_df.columns:
        display_df['shapiro_p'] = display_df['shapiro_p'].apply(
            lambda x: f'{x:.2e}' if pd.notna(x) and x < 0.001 else (f'{x:.4f}' if pd.notna(x) else 'N/A')
        )
    
    # Format is_normal
    if 'is_normal' in display_df.columns:
        display_df['is_normal'] = display_df['is_normal'].apply(
            lambda x: '✓' if x == True else ('✗' if x == False else '?')
        )
    
    # Create color coding for cells based on values
    font_size = 14 if presentation_mode else 11
    header_font_size = 16 if presentation_mode else 12
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>' + col.replace('_', ' ').title() + '</b>' for col in display_df.columns],
            fill_color='#4A90D9',
            font=dict(color='white', size=header_font_size, family='Helvetica Neue'),
            align='center',
            height=40
        ),
        cells=dict(
            values=[display_df[col] for col in display_df.columns],
            fill_color=[['white', '#f9f9f9'] * (len(display_df) // 2 + 1)][:len(display_df)],
            font=dict(size=font_size, family='Helvetica Neue'),
            align='center',
            height=30
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='Distribution Statistics by Reaction Type',
            font=dict(size=20 if presentation_mode else 16, family='Helvetica Neue')
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=max(300, 50 + len(display_df) * 35)
    )
    
    return fig


def create_significance_summary_table(
    sig_results: dict,
    presentation_mode: bool = False
) -> go.Figure:
    """Create a formatted table showing statistical significance test results.
    
    Args:
        sig_results: Dictionary from compute_significance_tests()
        presentation_mode: Whether to use larger fonts
        
    Returns:
        Plotly figure containing the results
    """
    from plotly.subplots import make_subplots
    
    font_size = 14 if presentation_mode else 11
    header_font_size = 16 if presentation_mode else 12
    title_size = 20 if presentation_mode else 16
    
    # Create subplots for different sections
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.15, 0.35, 0.5],
        specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "table"}]],
        vertical_spacing=0.08,
        subplot_titles=[
            'Overall Test (Kruskal-Wallis)',
            'Group Statistics',
            'Pairwise Comparisons (Mann-Whitney U with Bonferroni correction)'
        ]
    )
    
    # 1. Kruskal-Wallis summary
    kw = sig_results.get('kruskal_wallis', {})
    kw_data = pd.DataFrame([{
        'Test': 'Kruskal-Wallis H',
        'Statistic': f"{kw.get('statistic', 'N/A'):.2f}" if isinstance(kw.get('statistic'), (int, float)) else 'N/A',
        'p-value': f"{kw.get('p_value', 'N/A'):.2e}" if isinstance(kw.get('p_value'), (int, float)) and kw.get('p_value', 1) < 0.001 else (f"{kw.get('p_value', 'N/A'):.4f}" if isinstance(kw.get('p_value'), (int, float)) else 'N/A'),
        'Significant': '✓ Yes' if kw.get('significant') else '✗ No',
        'Groups': sig_results.get('n_groups', 'N/A'),
        'α (corrected)': f"{sig_results.get('alpha_corrected', 0.05):.4f}"
    }])
    
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>' + col + '</b>' for col in kw_data.columns],
            fill_color='#4A90D9',
            font=dict(color='white', size=header_font_size, family='Helvetica Neue'),
            align='center'
        ),
        cells=dict(
            values=[kw_data[col] for col in kw_data.columns],
            fill_color='white',
            font=dict(size=font_size, family='Helvetica Neue'),
            align='center'
        )
    ), row=1, col=1)
    
    # 2. Group statistics
    group_stats = sig_results.get('group_stats', pd.DataFrame())
    if not group_stats.empty:
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>' + col.replace('_', ' ').title() + '</b>' for col in group_stats.columns],
                fill_color='#4A90D9',
                font=dict(color='white', size=header_font_size, family='Helvetica Neue'),
                align='center'
            ),
            cells=dict(
                values=[group_stats[col] for col in group_stats.columns],
                fill_color=[['white', '#f9f9f9'] * (len(group_stats) // 2 + 1)][:len(group_stats)],
                font=dict(size=font_size, family='Helvetica Neue'),
                align='center'
            )
        ), row=2, col=1)
    
    # 3. Pairwise comparisons (show top 10 most significant)
    pairwise = sig_results.get('pairwise', pd.DataFrame())
    if not pairwise.empty:
        # Sort by p-value and take top entries
        pairwise_display = pairwise.sort_values('p_value').head(15)
        display_cols = ['group_1', 'group_2', 'p_value_formatted', 'significant', 'effect_size_r', 'effect_magnitude']
        display_cols = [c for c in display_cols if c in pairwise_display.columns]
        
        # Format significant column
        pairwise_display = pairwise_display.copy()
        if 'significant' in pairwise_display.columns:
            pairwise_display['significant'] = pairwise_display['significant'].apply(
                lambda x: '✓' if x else '✗'
            )
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>' + col.replace('_', ' ').title() + '</b>' for col in display_cols],
                fill_color='#4A90D9',
                font=dict(color='white', size=header_font_size, family='Helvetica Neue'),
                align='center'
            ),
            cells=dict(
                values=[pairwise_display[col] for col in display_cols],
                fill_color=[['white', '#f9f9f9'] * (len(pairwise_display) // 2 + 1)][:len(pairwise_display)],
                font=dict(size=font_size, family='Helvetica Neue'),
                align='center'
            )
        ), row=3, col=1)
    
    fig.update_layout(
        title=dict(
            text='Statistical Significance Analysis',
            font=dict(size=title_size, family='Helvetica Neue')
        ),
        height=800,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig
