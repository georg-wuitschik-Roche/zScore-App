from __future__ import annotations

"""plot_utils.py
================
Tiny helper module that keeps everything *visualisation*-related in one
place.  The public API purposefully mirrors the old inline functions so
existing callbacks can simply `import plot_utils as pu` and call
`pu.create_boxplot(...)`.
"""

from scipy import stats as sp_stats

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_FONT_FAMILY = 'Helvetica Neue'
_PLOT_BG = 'white'
_GRID_COLOR = '#d0d0d0'
_LINE_COLOR = '#cccccc'
_TITLE_COLOR = '#1d1d1f'
_ANNOTATION_BG = 'rgba(255, 255, 255, 0.9)'
_HISTOGRAM_BINS = 50
_DIAGNOSTIC_PLOT_HEIGHT = 500
_SHAPIRO_MAX_SAMPLES = 5000
_MAX_PAIRWISE_DISPLAY = 15
_TABLE_HEADER_COLOR = '#4A90D9'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _font_sizes(presentation_mode: bool, variant: str = 'main') -> dict:
    """Return a dict of font sizes for *variant* (main | diagnostic | table).

    Centralises the repeated ternary expressions that were scattered
    across every plot function.
    """
    if variant == 'main':
        return {
            'title': 32 if presentation_mode else 22,
            'base': 20 if presentation_mode else 14,
            'tick': 18 if presentation_mode else 14,
            'axis_title': 22 if presentation_mode else 16,
            'colorbar_title': 18 if presentation_mode else 14,
            'colorbar_tick': 16 if presentation_mode else 12,
            'text': 14 if presentation_mode else 10,
        }
    if variant == 'diagnostic':
        return {
            'title': 28 if presentation_mode else 20,
            'base': 18 if presentation_mode else 14,
            'tick': 16 if presentation_mode else 12,
            'annotation': 14 if presentation_mode else 11,
        }
    # table
    return {
        'title': 20 if presentation_mode else 16,
        'font': 14 if presentation_mode else 11,
        'header': 16 if presentation_mode else 12,
    }


def _apply_common_layout(fig: go.Figure, *, title: str, fs: dict, height: int,
                          margin: dict | None = None) -> None:
    """Apply the shared layout styling (background, title, font)."""
    fig.update_layout(
        title=dict(text=title, font=dict(
            size=fs['title'], family=_FONT_FAMILY, color=_TITLE_COLOR)),
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PLOT_BG,
        font=dict(family=_FONT_FAMILY, size=fs.get('base', fs.get('font', 14))),
        margin=margin or dict(l=60, r=60, t=100, b=60),
        height=height,
    )


def _style_diagnostic_axes(fig: go.Figure, fs: dict, **x_extra) -> None:
    """Apply shared axis styling for diagnostic plots (histogram, QQ)."""
    common = dict(showgrid=True, gridwidth=1, gridcolor='#e0e0e0',
                  showline=True, linewidth=2, linecolor=_LINE_COLOR,
                  tickfont=dict(size=fs['tick']))
    fig.update_xaxes(**common, **x_extra)
    fig.update_yaxes(**common)


def _filter_diagnostic_data(
    dff: pd.DataFrame, value_col: str, group_col: str | None,
    group_value: str | None, plot_label: str,
) -> tuple[pd.Series, str]:
    """Filter data and build a default title for diagnostic plots."""
    if group_col and group_value:
        data = dff[dff[group_col] == group_value][value_col].dropna()
        default_title = f'{plot_label} of {value_col} for {group_value}'
    else:
        data = dff[value_col].dropna()
        default_title = f'{plot_label} of {value_col}'
    return data, default_title


def _shapiro_wilk_summary(data: pd.Series) -> tuple[str, float | None, str | None]:
    """Run Shapiro-Wilk test, return (formatted_text, p_value, status)."""
    n = len(data)
    sample = data.sample(min(_SHAPIRO_MAX_SAMPLES, n), random_state=42) if n > _SHAPIRO_MAX_SAMPLES else data
    try:
        _, p = sp_stats.shapiro(sample)
        text = f'p={p:.2e}' if p < 0.001 else f'p={p:.4f}'
        status = 'Normal' if p > 0.05 else 'Non-normal'
        return text, p, status
    except Exception:
        return 'N/A', None, 'Unknown'


def _add_stats_annotation(fig: go.Figure, text: str, fs: dict,
                           x: float = 0.98, align: str = 'right') -> None:
    """Add a bordered annotation box with statistics text."""
    fig.add_annotation(
        x=x, y=0.98, xref='paper', yref='paper',
        text=text, showarrow=False,
        font=dict(size=fs['annotation'], family=_FONT_FAMILY),
        align=align,
        bgcolor=_ANNOTATION_BG,
        bordercolor=_LINE_COLOR, borderwidth=1, borderpad=8,
    )


def _safe_str_conversion(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to string while handling null values gracefully."""
    if hasattr(series, 'cat'):
        series = series.astype('object')
    return series.fillna('(no value)').astype(str)

# ---------------------------------------------------------------------------
# 1. COLOUR MAPPING HELPER
# ---------------------------------------------------------------------------

# A human readable mapping from *chemical entity* to *base colour*.  The
# actual shade is then calculated via interpolation depending on the
# number of ELNs present for that particular entity.
BASE_COLOURS: dict[str, dict[str, str]] = {
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


def create_color_mapping(category: str, dff) -> dict[str, str]:
    """Return a dict *category value -> colour*.

    The more ELNs a category value has the *darker* its colour becomes in
    the boxplot which gives the viewer a quick visual cue about data
    density.
    """

    base = BASE_COLOURS.get(category, {"light": "#D3D3D3", "dark": "#696969"})

    eln_counts = dff.groupby(category)["ELN_ID"].nunique()
    max_elns, min_elns = eln_counts.max(), eln_counts.min()

    colour_map: dict[str, str] = {}
    for cat_val, cnt in eln_counts.items():
        factor = 0.5 if max_elns == min_elns else (cnt - min_elns) / (max_elns - min_elns)
        colour_map[cat_val] = _interpolate_hex(base["light"], base["dark"], factor)

    return colour_map


# ---------------------------------------------------------------------------
# 2. BOXPLOT CREATION
# ---------------------------------------------------------------------------


def create_boxplot(dff, reactant_types: list, base_height: int = 800, presentation_mode: bool = False, reaction_type: str = None, max_categories: int = None) -> tuple[go.Figure, int]:
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

    # Build hover text using vectorized string operations (avoids row-by-row apply)
    def _col_str(col: str) -> pd.Series:
        """Return a cleaned string Series for *col*, '' for missing values."""
        if col not in dff_hover.columns:
            return pd.Series('', index=dff_hover.index)
        s = dff_hover[col]
        if hasattr(s, 'cat'):
            s = s.astype('object')
        if s.dtype == 'object':
            return s.fillna('').astype(str).str.strip()
        return s.fillna('').astype(str)

    z_score_s = dff_hover['z-Score'].map(
        lambda v: f'{v:.3f}' if pd.notna(v) else '', na_action=None,
    )
    area_s = dff_hover['AREA_TOTAL_REDUCED'].map(
        lambda v: f'{v:.2f}%' if pd.notna(v) else '', na_action=None,
    )

    dff_hover['hover_text'] = (
        '<b>Experiment Details:</b><br>'
        + 'ELN_ID: ' + _col_str('ELN_ID') + '<br>'
        + 'Plate: ' + _col_str('PLATENUMBER') + '<br>'
        + 'Coordinate: ' + _col_str('Coordinate') + '<br>'
        + '<br>'
        + '<b>Results:</b><br>'
        + 'z-Score: ' + z_score_s + '<br>'
        + 'Area: ' + area_s + '<br>'
        + '<br>'
        + '<b>Reaction:</b><br>'
        + 'Reaction Type: ' + _col_str('Reaction Type') + '<br>'
        + '<br>'
        + '<b>Reaction Conditions:</b><br>'
        + _col_str('output_column') + '<br>'
        + '<br>'
        + '<b>Reagents:</b><br>'
        + 'Catalyst: ' + _col_str('Catalyst') + '<br>'
        + 'Solvent: ' + _col_str('Solvent') + '<br>'
        + 'Base: ' + _col_str('Base') + '<br>'
        + 'Ligand: ' + _col_str('Ligand') + '<br>'
        + 'Additive: ' + _col_str('Additive') + '<br>'
        + 'Coupling Reagent: ' + _col_str('Coupling Reagent') + '<br>'
        + 'Functional Group A: ' + _col_str('FG A') + '<br>'
        + 'Functional Group B: ' + _col_str('FG B') + '<br>'
        + 'Secondary Solvent: ' + _col_str('Secondary Solvent') + '<br>'
    )
    
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

    fs = _font_sizes(presentation_mode, 'main')

    _apply_common_layout(fig, title=title, fs=fs, height=height)
    fig.update_layout(showlegend=False)

    fig.update_xaxes(
        tickangle=0, showgrid=True, gridwidth=2, gridcolor=_GRID_COLOR,
        zeroline=False, showline=True, linewidth=3, linecolor=_LINE_COLOR,
        tickmode="auto", nticks=6,
        tickfont=dict(size=fs['tick'], weight="bold"),
        title_font=dict(size=fs['axis_title'], weight="bold"),
    )
    fig.update_yaxes(
        tickangle=0, showgrid=False, zeroline=False,
        showline=True, linewidth=3, linecolor=_LINE_COLOR,
        tickfont=dict(size=fs['tick'], weight="bold"),
        title_font=dict(size=fs['axis_title'], weight="bold"),
    )

    return fig, height

# ---------------------------------------------------------------------------
# 3. HEATMAP CREATION
# ---------------------------------------------------------------------------


def create_heatmap(dff, reactant_types: list, base_height: int = 800, presentation_mode: bool = False) -> tuple[go.Figure, int]:
    """Return `(figure, adaptive_height)` for a heatmap visualization.

    Args:
        dff: The filtered dataframe to plot
        reactant_types: List of selected reactant types (categories) to display
        base_height: Minimum height for the plot
        presentation_mode: Whether to use larger fonts for presentation

    Creates a heatmap with the first reactant type on y-axis and remaining types on x-axis.
    Requires at least two reactant types to be selected.
    """

    fs = _font_sizes(presentation_mode, 'main')

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
        
        # Build 2D matrices via pivot_table (replaces O(Y*X) nested loops)
        heatmap_df = dff.pivot_table(
            index=y_category, columns=x_category,
            values='z-Score', aggfunc='median',
        )
        eln_df = dff.pivot_table(
            index=y_category, columns=x_category,
            values='ELN_ID', aggfunc='nunique',
        )
        # Reindex to the desired order (fills missing combos with NaN / 0)
        heatmap_df = heatmap_df.reindex(index=y_category_order, columns=x_category_order)
        eln_df = eln_df.reindex(index=y_category_order, columns=x_category_order).fillna(0).astype(int)

        heatmap_data = heatmap_df.values
        eln_counts = eln_df.values
        
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
            text=[[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": fs['text'], "color": "black"},
            colorbar=dict(
                title=dict(
                    text="Median z-Score",
                    font=dict(size=fs['colorbar_title'], family=_FONT_FAMILY)
                ),
                tickfont=dict(size=fs['colorbar_tick'], family=_FONT_FAMILY)
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

    _apply_common_layout(fig, title=title, fs=fs, height=height)
    fig.update_layout(
        xaxis=dict(
            title=dict(text=x_axis_title,
                       font=dict(size=fs['axis_title'], weight="bold", family=_FONT_FAMILY)),
            tickfont=dict(size=fs['tick'], weight="bold", family=_FONT_FAMILY),
            showgrid=True, gridwidth=1, gridcolor=_GRID_COLOR,
            zeroline=False, showline=True, linewidth=2, linecolor=_LINE_COLOR,
            side="top",
        ),
        yaxis=dict(
            title=dict(text=reactant_types[0],
                       font=dict(size=fs['axis_title'], weight="bold", family=_FONT_FAMILY)),
            tickfont=dict(size=fs['tick'], weight="bold", family=_FONT_FAMILY),
            showgrid=False, zeroline=False,
            showline=True, linewidth=2, linecolor=_LINE_COLOR,
        ),
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
) -> tuple[go.Figure, int]:
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
    data, default_title = _filter_diagnostic_data(
        dff, value_col, group_col, group_value, 'Distribution')
    title = title or default_title
    fs = _font_sizes(presentation_mode, 'diagnostic')

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=_HISTOGRAM_BINS, name='Observed',
        opacity=0.7, marker_color='#4A90D9', histnorm='probability density',
    ))

    if len(data) > 10:
        n = len(data)
        shapiro_text, _, _ = _shapiro_wilk_summary(data)
        stats_text = (
            f"<b>Distribution Statistics</b><br>"
            f"n = {n:,}<br>"
            f"Skewness = {data.skew():.3f}<br>"
            f"Kurtosis = {data.kurtosis():.3f}<br>"
            f"Shapiro-Wilk {shapiro_text}"
        )
        _add_stats_annotation(fig, stats_text, fs, x=0.98, align='right')

    _apply_common_layout(fig, title=title, fs=fs, height=_DIAGNOSTIC_PLOT_HEIGHT,
                          margin=dict(l=60, r=60, t=80, b=60))
    fig.update_layout(
        xaxis_title=value_col, yaxis_title='Density',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
    )
    _style_diagnostic_axes(fig, fs)

    return fig, _DIAGNOSTIC_PLOT_HEIGHT


def create_qq_plot(
    dff: pd.DataFrame,
    value_col: str = 'z-Score',
    group_col: str = None,
    group_value: str = None,
    title: str = None,
    presentation_mode: bool = False
) -> tuple[go.Figure, int]:
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
    data, default_title = _filter_diagnostic_data(
        dff, value_col, group_col, group_value, 'Q-Q Plot')
    title = title or default_title
    fs = _font_sizes(presentation_mode, 'diagnostic')

    data_sorted = np.sort(data)
    n = len(data_sorted)
    theoretical_quantiles = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles, y=data_sorted,
        mode='markers', name='Data',
        marker=dict(color='#4A90D9', size=6, opacity=0.6),
    ))

    mu, std = data.mean(), data.std()
    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    fig.add_trace(go.Scatter(
        x=line_x, y=mu + std * line_x,
        mode='lines', name='Normal Reference',
        line=dict(color='#E74C3C', width=2, dash='dash'),
    ))

    shapiro_text, _, normality_status = _shapiro_wilk_summary(data)
    stats_text = (
        f"<b>Normality Assessment</b><br>"
        f"n = {n:,}<br>"
        f"Skewness = {data.skew():.3f}<br>"
        f"Kurtosis = {data.kurtosis():.3f}<br>"
        f"Shapiro-Wilk {shapiro_text}<br>"
        f"<b>Status: {normality_status}</b>"
    )
    _add_stats_annotation(fig, stats_text, fs, x=0.02, align='left')

    _apply_common_layout(fig, title=title, fs=fs, height=_DIAGNOSTIC_PLOT_HEIGHT,
                          margin=dict(l=60, r=60, t=80, b=60))
    fig.update_layout(
        xaxis_title='Theoretical Quantiles (Normal)',
        yaxis_title=f'Sample Quantiles ({value_col})',
        showlegend=True,
        legend=dict(x=0.7, y=0.15, bgcolor='rgba(255,255,255,0.8)'),
    )
    _style_diagnostic_axes(fig, fs, scaleanchor='y', scaleratio=1)

    return fig, _DIAGNOSTIC_PLOT_HEIGHT


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
    
    fs = _font_sizes(presentation_mode, 'table')

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>' + col.replace('_', ' ').title() + '</b>' for col in display_df.columns],
            fill_color=_TABLE_HEADER_COLOR,
            font=dict(color='white', size=fs['header'], family=_FONT_FAMILY),
            align='center', height=40,
        ),
        cells=dict(
            values=[display_df[col] for col in display_df.columns],
            fill_color=[['white', '#f9f9f9'] * (len(display_df) // 2 + 1)][:len(display_df)],
            font=dict(size=fs['font'], family=_FONT_FAMILY),
            align='center', height=30,
        ),
    )])

    height = max(300, 50 + len(display_df) * 35)
    _apply_common_layout(fig, title='Distribution Statistics by Reaction Type',
                          fs=fs, height=height, margin=dict(l=20, r=20, t=60, b=20))
    
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

    fs = _font_sizes(presentation_mode, 'table')
    
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
    
    _hdr = dict(fill_color=_TABLE_HEADER_COLOR,
                font=dict(color='white', size=fs['header'], family=_FONT_FAMILY),
                align='center')
    _cell_font = dict(size=fs['font'], family=_FONT_FAMILY)

    def _stripe(n: int):
        return [['white', '#f9f9f9'] * (n // 2 + 1)][:n]

    fig.add_trace(go.Table(
        header=dict(values=['<b>' + c + '</b>' for c in kw_data.columns], **_hdr),
        cells=dict(values=[kw_data[c] for c in kw_data.columns],
                   fill_color='white', font=_cell_font, align='center'),
    ), row=1, col=1)

    # 2. Group statistics
    group_stats = sig_results.get('group_stats', pd.DataFrame())
    if not group_stats.empty:
        fig.add_trace(go.Table(
            header=dict(values=['<b>' + c.replace('_', ' ').title() + '</b>'
                                for c in group_stats.columns], **_hdr),
            cells=dict(values=[group_stats[c] for c in group_stats.columns],
                       fill_color=_stripe(len(group_stats)),
                       font=_cell_font, align='center'),
        ), row=2, col=1)

    # 3. Pairwise comparisons
    pairwise = sig_results.get('pairwise', pd.DataFrame())
    if not pairwise.empty:
        pairwise_display = pairwise.sort_values('p_value').head(_MAX_PAIRWISE_DISPLAY)
        display_cols = [c for c in [
            'group_1', 'group_2', 'p_value_formatted', 'significant',
            'effect_size_r', 'effect_magnitude',
        ] if c in pairwise_display.columns]

        pairwise_display = pairwise_display.copy()
        if 'significant' in pairwise_display.columns:
            pairwise_display['significant'] = pairwise_display['significant'].apply(
                lambda x: '✓' if x else '✗')

        fig.add_trace(go.Table(
            header=dict(values=['<b>' + c.replace('_', ' ').title() + '</b>'
                                for c in display_cols], **_hdr),
            cells=dict(values=[pairwise_display[c] for c in display_cols],
                       fill_color=_stripe(len(pairwise_display)),
                       font=_cell_font, align='center'),
        ), row=3, col=1)

    _apply_common_layout(fig, title='Statistical Significance Analysis',
                          fs=fs, height=800, margin=dict(l=20, r=20, t=80, b=20))

    return fig
