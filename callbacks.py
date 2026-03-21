from __future__ import annotations

"""callbacks.py
===============
All Dash callbacks live here so the rest of the application logic can
remain agnostic of the Dash decorator magic.  The file exposes a single
public function – :pyfunc:`register` – which wires the callbacks to a
passed in `dash.Dash` instance.

Keeping the callbacks in their own file avoids *cyclical* imports
(category options depend on data filtering depends on callback inputs…)
by having a clean, top-level entry point.
"""

import io
import base64
import logging

import pandas as pd
from dash import Input, Output, State, callback_context, dcc, html, no_update

import data_utils as du
import plot_utils as pu

logger = logging.getLogger(__name__)

# Required columns for uploaded datasets
REQUIRED_COLUMNS = [
    'ELN_ID', 'PLATENUMBER', 'Coordinate', 'AREA_TOTAL_REDUCED',
    'Base', 'Catalyst', 'Solvent', 'Ligand', 'Reaction Type',
    'FG A', 'FG B', 'FG_sorted', 'z-Score'
]

# Default filter values — single source of truth
DEFAULT_REACTION_TYPES = ['Buchwald-Hartwig']
DEFAULT_FG_A = ['RNH2 a-branch', 'RNH2']
DEFAULT_FG_B = ['ArBr', 'ArCl']
DEFAULT_REACTANT_FALLBACK = ['Ligand']
DEFAULT_MIN_ELN = 5
DEFAULT_TOPN_ZSCORE = 5
DEFAULT_MAX_COMPONENTS = 10

# Maximum upload size in bytes (50 MB)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Shared filter inputs used by multiple callbacks
FILTER_INPUTS = [
    Input('reactant-types-dropdown', 'value'),
    Input('reaction-type-dropdown', 'value'),
    Input('functional-group-a-dropdown', 'value'),
    Input('functional-group-b-dropdown', 'value'),
    Input('exclude-cui-checkbox', 'value'),
    Input('include-scaleup-checkbox', 'value'),
    Input('include-null-categories-checkbox', 'value'),
    Input('min-eln-input', 'value'),
    Input('topn-zscore-input', 'value'),
    Input('max-components-input', 'value'),
    Input('uploaded-data-store', 'data'),
]

# Same inputs but as State (for download callbacks)
FILTER_STATES = [
    State('reactant-types-dropdown', 'value'),
    State('reaction-type-dropdown', 'value'),
    State('functional-group-a-dropdown', 'value'),
    State('functional-group-b-dropdown', 'value'),
    State('exclude-cui-checkbox', 'value'),
    State('include-scaleup-checkbox', 'value'),
    State('include-null-categories-checkbox', 'value'),
    State('min-eln-input', 'value'),
    State('topn-zscore-input', 'value'),
    State('max-components-input', 'value'),
    State('uploaded-data-store', 'data'),
]

_FILTER_KEYS = [
    'reactant_types', 'reaction_types', 'fg_a', 'fg_b',
    'exclude_cui', 'exclude_scaleup', 'include_null_categories',
    'min_eln', 'topn_zscore', 'max_components', 'uploaded_data',
]


def _parse_filter_args(args: tuple) -> dict:
    """Convert positional filter args into a kwargs dict for filter_data.

    Returns a dict with keys matching ``du.filter_data`` parameters plus
    ``'uploaded_data'`` (the raw JSON string from ``dcc.Store``).
    """
    return dict(zip(_FILTER_KEYS, args))


def _call_filter_data(fkw: dict, **overrides) -> pd.DataFrame:
    """Call ``du.filter_data`` using a dict from ``_parse_filter_args``.

    *overrides* are forwarded to ``du.filter_data`` and take precedence
    (e.g. ``return_stats=True``, ``min_eln=None``).
    """
    uploaded_data = fkw['uploaded_data']
    source_df = du.parse_uploaded_data(uploaded_data) if uploaded_data else None
    kwargs = {k: fkw[k] for k in _FILTER_KEYS if k != 'uploaded_data'}
    kwargs['source_df'] = source_df
    kwargs.update(overrides)
    return du.filter_data(**kwargs)


def _is_tutorial_step_satisfied(
    step_idx: int,
    *,
    reaction_types,
    reactant_types,
    fg_a_vals,
    fg_b_vals,
    filter_panel_style,
    min_eln,
    topn,
    max_comp,
    exclude_cui_val,
    tabs_value,
) -> bool:
    """Check whether a tutorial step's gating condition is met."""
    try:
        if step_idx == 0:
            return bool(reaction_types)
        if step_idx == 1:
            return bool(reactant_types)
        if step_idx == 2:
            return bool(fg_a_vals)
        if step_idx == 3:
            return bool(fg_b_vals)
        if step_idx == 4:
            if not filter_panel_style:
                return False
            return (filter_panel_style.get('maxHeight') != '0px'
                    and filter_panel_style.get('display') != 'none')
        if step_idx == 5:
            return min_eln is not None and min_eln != 10
        if step_idx == 6:
            return topn is not None and topn != 3
        if step_idx == 7:
            return max_comp is not None and max_comp != 10
        if step_idx == 8:
            return isinstance(exclude_cui_val, list) and ('exclude_cui' not in exclude_cui_val)
        if step_idx == 9:
            return tabs_value == 'tab-heatmap'
        if step_idx == 10:
            return True
    except Exception:
        return False
    return False

## ---------------------------------------------------------------------------
# Public – callback registration
# ---------------------------------------------------------------------------

def register(app):  # noqa: C901 – complexity is mostly decorator noise
    """Attach **all** callbacks to *app*.

    The function does not return anything – mutating *app* is enough for
    Dash to pick up the callbacks.
    """

    # ------------------------------------------------------------------
    # Presentation mode callbacks ------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('presentation-mode-store', 'data'),
         Output('presentation-mode-toggle', 'children'),
         Output('presentation-mode-toggle', 'className'),
         Output('main-container', 'className')],
        [Input('presentation-mode-toggle', 'n_clicks')],
        [State('presentation-mode-store', 'data')]
    )
    def _toggle_presentation_mode(n_clicks, is_presentation_mode):
        """Toggle presentation mode on and off."""
        if n_clicks is None or n_clicks == 0:
            return False, "Presentation Mode", "presentation-toggle-btn", "app-container"
        
        # Toggle the mode
        new_mode = not is_presentation_mode
        
        if new_mode:
            # Entering presentation mode
            button_text = "Exit Presentation"
            button_class = "presentation-toggle-btn active"
            container_class = "app-container presentation-mode"
        else:
            # Exiting presentation mode
            button_text = "Presentation Mode"
            button_class = "presentation-toggle-btn"
            container_class = "app-container"
        
        return new_mode, button_text, button_class, container_class


    # ------------------------------------------------------------------
    # Dataset upload processing -----------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('uploaded-data-store', 'data'),
         Output('upload-status-indicator', 'children'),
         Output('upload-status-indicator', 'className'),
         Output('upload-error-modal', 'style'),
         Output('upload-error-content', 'children')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')],
        prevent_initial_call=True
    )
    def _process_uploaded_data(contents, filename):
        """Parse uploaded CSV file, validate schema, and store in browser session."""
        if contents is None:
            return no_update, no_update, no_update, no_update, no_update
        
        modal_hidden = {'display': 'none'}
        modal_visible = {'display': 'flex'}
        
        try:
            # Decode the base64 content
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            # Enforce upload size limit
            if len(decoded) > MAX_UPLOAD_BYTES:
                size_mb = len(decoded) / (1024 * 1024)
                error_content = html.Div([
                    html.P([html.Strong('File: '), filename]),
                    html.P([
                        html.Strong('Error: '),
                        f'File size ({size_mb:.1f} MB) exceeds the 50 MB limit.'
                    ]),
                ])
                return (
                    None,
                    html.Span('File too large'),
                    'upload-status error',
                    modal_visible,
                    error_content,
                )

            # Try to read CSV with encoding and delimiter detection
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            delimiters_to_try = [',', ';', '\t']
            df = None
            
            for encoding in encodings_to_try:
                for delimiter in delimiters_to_try:
                    try:
                        df = pd.read_csv(io.BytesIO(decoded), encoding=encoding, sep=delimiter)
                        # Check if we got a reasonable number of columns (more than 1)
                        if len(df.columns) > 1:
                            break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) <= 1:
                error_content = html.Div([
                    html.P([
                        html.Strong('File: '), filename
                    ]),
                    html.P([
                        html.Strong('Error: '), 
                        'Could not decode the file. The file encoding is not supported.'
                    ]),
                    html.P('Please ensure your CSV file is saved with UTF-8 encoding.')
                ])
                return (
                    None,
                    html.Span('Upload failed'),
                    'upload-status error',
                    modal_visible,
                    error_content
                )
            
            # Validate required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                # Show columns found in the file
                found_cols = list(df.columns)
                error_content = html.Div([
                    html.P([
                        html.Strong('File: '), filename
                    ]),
                    html.P([
                        html.Strong('Error: '), 
                        f'Missing required columns: {", ".join(missing_cols)}'
                    ]),
                    html.P([
                        html.Strong('Columns found in your file:')
                    ]),
                    html.Code(
                        ', '.join(found_cols[:20]) + ('...' if len(found_cols) > 20 else ''),
                        style={'display': 'block', 'padding': '8px', 'background': '#f5f5f5', 
                               'borderRadius': '4px', 'fontSize': '12px', 'marginTop': '4px'}
                    ),
                ])
                return (
                    None,
                    html.Span('Missing columns'),
                    'upload-status error',
                    modal_visible,
                    error_content
                )
            
            # Clean the data similar to default dataset processing
            df["z-Score"] = (
                df["z-Score"].astype(str).str.replace(",", ".").str.strip().pipe(pd.to_numeric, errors="coerce")
            )
            
            # Check if z-Score has valid numeric values
            valid_zscores = df["z-Score"].notna().sum()
            if valid_zscores == 0:
                error_content = html.Div([
                    html.P([
                        html.Strong('File: '), filename
                    ]),
                    html.P([
                        html.Strong('Error: '), 
                        'The z-Score column contains no valid numeric values.'
                    ]),
                    html.P('Please ensure the z-Score column contains numeric values (decimals can use . or , as separator).')
                ])
                return (
                    None,
                    html.Span('Invalid z-Score'),
                    'upload-status error',
                    modal_visible,
                    error_content
                )
            
            if "AREA_TOTAL_REDUCED" in df.columns:
                df["AREA_TOTAL_REDUCED"] = (
                    df["AREA_TOTAL_REDUCED"].astype(str).str.replace(",", ".").str.strip().pipe(pd.to_numeric, errors="coerce")
                )
            
            # Create FG_PAIR_SORTED if FG A and FG B exist
            if "FG A" in df.columns and "FG B" in df.columns:
                if "FG_sorted" in df.columns:
                    df["FG_PAIR_SORTED"] = df["FG_sorted"]
                else:
                    df["FG_PAIR_SORTED"] = df.apply(
                        lambda r: ", ".join(sorted([str(r["FG A"]), str(r["FG B"])])), axis=1
                    )
            
            # Convert DataFrame to JSON for storage
            data_json = df.to_json(date_format='iso', orient='split')
            
            row_count = len(df)
            status_text = f'Using: {filename} ({row_count:,} rows)'
            
            return (
                data_json,
                html.Span([
                    html.I(className='fas fa-check-circle', style={'marginRight': '4px'}),
                    status_text
                ]),
                'upload-status success',
                modal_hidden,
                []
            )
            
        except Exception:
            logger.exception("Upload processing failed for file: %s", filename)
            error_content = html.Div([
                html.P([
                    html.Strong('File: '), filename if filename else 'Unknown'
                ]),
                html.P([
                    html.Strong('Error: '),
                    'An unexpected error occurred while processing your file.'
                ]),
                html.P('Please check that your file is a valid CSV file with the required columns.')
            ])
            return (
                None,
                html.Span('Upload failed'),
                'upload-status error',
                modal_visible,
                error_content
            )

    # ------------------------------------------------------------------
    # Close upload error modal ------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('upload-error-modal', 'style', allow_duplicate=True),
        [Input('upload-error-close', 'n_clicks'),
         Input('upload-error-close-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def _close_upload_error_modal(close_x, close_btn):
        """Close the upload error modal."""
        return {'display': 'none'}

    # ------------------------------------------------------------------
    # Update reaction type dropdown options on data upload --------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('reaction-type-dropdown', 'options'),
        [Input('uploaded-data-store', 'data')]
    )
    def _update_reaction_type_options(uploaded_data):
        """Update reaction type dropdown options based on uploaded or default data."""
        source_df = du.get_active_dataframe(uploaded_data)
        reaction_types = du.get_reaction_types_from_data(source_df)
        return [{"label": rt, "value": rt} for rt in reaction_types]

    # ------------------------------------------------------------------
    # Central compute: filtered data + stats ---------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('filter-stats-store', 'data'),
        FILTER_INPUTS,
    )
    def _compute_stats(*args):
        """Compute filtered dataset and stats, return only stats to client."""
        fkw = _parse_filter_args(args)
        _dff, stats = _call_filter_data(fkw, return_stats=True)
        return stats

    # ------------------------------------------------------------------
    # Reactant types dropdown dependencies ------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('reactant-types-dropdown', 'options'),
        [Input('reaction-type-dropdown', 'value'),
         Input('uploaded-data-store', 'data')]
    )
    def _update_reactant_types_options(reaction_types, uploaded_data):
        """Update reactant types dropdown options based on selected reaction types."""
        # Get source DataFrame (uploaded or default)
        source_df = du.get_active_dataframe(uploaded_data)
        
        if not reaction_types or len(reaction_types) == 0:
            # Always exclude Functional Group A and Functional Group B
            filtered_categories = [c for c in du.CATEGORY_OPTIONS if c not in ['Functional Group A', 'Functional Group B']]
            return [{'label': c, 'value': c} for c in filtered_categories]
        
        # Filter data for the selected reaction types
        dff = source_df[source_df['Reaction Type'].isin(reaction_types)]
        
        # Check which categories have non-null values for these reaction types
        # Always exclude Functional Group A and Functional Group B
        available_categories = []
        for category in du.CATEGORY_OPTIONS:
            if (category in dff.columns and 
                dff[category].notna().any() and 
                category not in ['Functional Group A', 'Functional Group B']):
                available_categories.append(category)
        
        return [{'label': c, 'value': c} for c in available_categories]

    # ------------------------------------------------------------------
    # Functional-group A (formerly "aryl halide") -----------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('functional-group-a-dropdown', 'options'),
        [Input('reaction-type-dropdown', 'value'),
         Input('uploaded-data-store', 'data')]
    )
    def _update_fg_a_options(reaction_types, uploaded_data):
        """Populate *Functional Group A* dropdown based on chosen reaction types."""

        if not reaction_types:
            return [{'label': 'All', 'value': 'All'}]

        # Get source DataFrame (uploaded or default)
        source_df = du.get_active_dataframe(uploaded_data)
        dff = source_df[source_df['Reaction Type'].isin(reaction_types)]

        if {'FG A', 'FG B'}.issubset(dff.columns):
            fg_values = pd.Series(pd.concat([dff['FG A'], dff['FG B']])).dropna().unique()
            options = ['All'] + sorted(fg_values.tolist())
            return [{'label': v, 'value': v} for v in options]

        # Fallback – column missing (should not happen)
        return [{'label': 'All', 'value': 'All'}]

    # ------------------------------------------------------------------
    # Functional-group B (formerly "N-nucleophile/boronate") -------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('functional-group-b-dropdown', 'options'),
        [Input('reaction-type-dropdown', 'value'),
         Input('functional-group-a-dropdown', 'value'),
         Input('uploaded-data-store', 'data')]
    )
    def _update_fg_b_options(reaction_types, fg_a, uploaded_data):
        """Return Functional-group B options conditioned on *FG A* selection."""

        if not reaction_types:
            return [{'label': 'All', 'value': 'All'}]

        # Get source DataFrame (uploaded or default)
        source_df = du.get_active_dataframe(uploaded_data)
        dff = source_df[source_df['Reaction Type'].isin(reaction_types)]

        if {'FG A', 'FG B'}.issubset(dff.columns):
            # Handle multi-selection for FG A
            if fg_a and 'All' not in fg_a:
                # If specific FG A values are selected, find all groups that co-occur with any of them
                all_other_fgs = []
                
                # Convert to list if single string (for backward compatibility)
                fg_a_list = fg_a if isinstance(fg_a, list) else [fg_a]
                
                for fg_a_val in fg_a_list:
                    mask = (dff['FG A'] == fg_a_val) | (dff['FG B'] == fg_a_val)
                    dff_sub = dff[mask]

                    other_fgs = []
                    other_fgs.extend(dff_sub.loc[dff_sub['FG A'] == fg_a_val, 'FG B'])
                    other_fgs.extend(dff_sub.loc[dff_sub['FG B'] == fg_a_val, 'FG A'])
                    all_other_fgs.extend(other_fgs)

                fg_values = pd.Series(all_other_fgs).dropna().unique()
            else:
                # If "All" is selected or nothing specific, show all available functional groups
                fg_values = pd.Series(pd.concat([dff['FG A'], dff['FG B']])).dropna().unique()

            options = ['All'] + sorted(fg_values.tolist())
            return [{'label': v, 'value': v} for v in options]

        # Fallback
        return [{'label': 'All', 'value': 'All'}]

    # ------------------------------------------------------------------
    # Reset functional groups when reaction types change ---------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('functional-group-a-dropdown', 'value'),
         Output('functional-group-b-dropdown', 'value')],
        [Input('reaction-type-dropdown', 'value')],
        [State('functional-group-a-dropdown', 'options'),
         State('functional-group-b-dropdown', 'options')],
        prevent_initial_call=True
    )
    def _reset_functional_groups_on_reaction_change(reaction_types, fg_a_options, fg_b_options):
        """Clear both functional group dropdowns when reaction types change."""
        # Clear functional groups to 'All' when reaction types change
        return ['All'], ['All']

    # ------------------------------------------------------------------
    # Set initial functional group A values on page load ---------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('functional-group-a-dropdown', 'value', allow_duplicate=True),
        [Input('functional-group-a-dropdown', 'options')],
        [State('functional-group-a-dropdown', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def _set_initial_fg_a_values(fg_a_options, current_fg_a):
        """Set initial values for functional group A dropdown when options are populated."""
        desired_fg_a = DEFAULT_FG_A

        # Only set defaults if current values are truly empty (None or []), not if explicitly set to ['All']
        if not current_fg_a:  # This covers None, [], or empty list
            if fg_a_options:
                available_values = [opt['value'] for opt in fg_a_options]
                valid_defaults = [val for val in desired_fg_a if val in available_values]
                if valid_defaults:
                    return valid_defaults
                # If no defaults available, return 'All'
                return ['All']
        
        return current_fg_a

    # ------------------------------------------------------------------
    # Set initial functional group B values on page load ---------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('functional-group-b-dropdown', 'value', allow_duplicate=True),
        [Input('functional-group-b-dropdown', 'options')],
        [State('functional-group-b-dropdown', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def _set_initial_fg_b_values(fg_b_options, current_fg_b):
        """Set initial values for functional group B dropdown when options are populated."""
        desired_fg_b = DEFAULT_FG_B

        # Only set defaults if current values are truly empty (None or []), not if explicitly set to ['All']
        if not current_fg_b:  # This covers None, [], or empty list
            if fg_b_options:
                available_values = [opt['value'] for opt in fg_b_options]
                valid_defaults = [val for val in desired_fg_b if val in available_values]
                if valid_defaults:
                    return valid_defaults
                # If no defaults available, return 'All'
                return ['All']
        
        return current_fg_b

    # ------------------------------------------------------------------
    # Reset filters callback -------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('reaction-type-dropdown', 'value'),
         Output('reactant-types-dropdown', 'value'),
         Output('functional-group-a-dropdown', 'value', allow_duplicate=True),
         Output('functional-group-b-dropdown', 'value', allow_duplicate=True),
         Output('exclude-cui-checkbox', 'value'),
         Output('include-scaleup-checkbox', 'value'),
         Output('include-null-categories-checkbox', 'value'),
         Output('min-eln-input', 'value'),
         Output('topn-zscore-input', 'value'),
         Output('max-components-input', 'value')],
        [Input('reset-btn', 'n_clicks'),
         Input('reaction-type-dropdown', 'value')],
        [State('reactant-types-dropdown', 'value'),
         State('functional-group-a-dropdown', 'options'),
         State('functional-group-b-dropdown', 'options'),
         State('uploaded-data-store', 'data')],
        prevent_initial_call=True
    )
    def _reset_filters_and_update_min_eln(n_clicks, reaction_types, current_reactant_types, fg_a_options, fg_b_options, uploaded_data):
        """Reset filters or update min ELN based on trigger."""
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Get source DataFrame (uploaded or default)
        source_df = du.get_active_dataframe(uploaded_data)
        
        if triggered_id == 'reset-btn':
            # Reset all filters
            # Get available categories for the current reaction types
            if reaction_types and len(reaction_types) > 0:
                dff = source_df[source_df['Reaction Type'].isin(reaction_types)]
                available_categories = []
                for category in du.CATEGORY_OPTIONS:
                    if (category in dff.columns and 
                        dff[category].notna().any() and 
                        category not in ['Functional Group A', 'Functional Group B']):
                        available_categories.append(category)
                
                # Select default reactant types from available ones
                if 'Catalyst' in available_categories:
                    default_reactant_types = ['Catalyst']
                elif available_categories:
                    default_reactant_types = [available_categories[0]]
                else:
                    # Fallback to first non-excluded category
                    fallback_categories = [c for c in du.CATEGORY_OPTIONS if c not in ['Functional Group A', 'Functional Group B']]
                    default_reactant_types = [fallback_categories[0]] if fallback_categories else [du.CATEGORY_OPTIONS[0]]
            else:
                # Fallback to first non-excluded category
                fallback_categories = [c for c in du.CATEGORY_OPTIONS if c not in ['Functional Group A', 'Functional Group B']]
                default_reactant_types = ['Catalyst'] if 'Catalyst' in fallback_categories else [fallback_categories[0]]
            
            default_exclude_cui = ['exclude_cui']
            default_exclude_scaleup = [True]
            default_include_null_categories = [True]
            default_min_eln = DEFAULT_MIN_ELN
            default_topn_zscore = DEFAULT_TOPN_ZSCORE
            default_max_components = DEFAULT_MAX_COMPONENTS

            desired_fg_a = DEFAULT_FG_A
            desired_fg_b = DEFAULT_FG_B
            
            # Reset FG A
            default_fg_a = ['All']  # Default fallback
            if fg_a_options:
                available_fg_a = [opt['value'] for opt in fg_a_options]
                valid_fg_a = [val for val in desired_fg_a if val in available_fg_a]
                if valid_fg_a:
                    default_fg_a = valid_fg_a
            
            # Reset FG B  
            default_fg_b = ['All']  # Default fallback
            if fg_b_options:
                available_fg_b = [opt['value'] for opt in fg_b_options]
                valid_fg_b = [val for val in desired_fg_b if val in available_fg_b]
                if valid_fg_b:
                    default_fg_b = valid_fg_b
            
            default_reaction_types = DEFAULT_REACTION_TYPES
            
            return (default_reaction_types, default_reactant_types, default_fg_a, default_fg_b,
                    default_exclude_cui, default_exclude_scaleup, default_include_null_categories, default_min_eln, default_topn_zscore, default_max_components)
        
        elif triggered_id == 'reaction-type-dropdown':
            # Handle reaction type change - keep reactant types the same
            # Keep reactant types unchanged when reaction types change
            return (no_update, current_reactant_types, no_update, no_update,
                    no_update, no_update, no_update, no_update, no_update, no_update)
        
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # ------------------------------------------------------------------
    # Update min ELN based on functional group selections ---------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('min-eln-input', 'value', allow_duplicate=True),
        [Input('functional-group-a-dropdown', 'value'),
         Input('functional-group-b-dropdown', 'value')],
        [State('reaction-type-dropdown', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def _update_min_eln_on_fg_change(fg_a_values, fg_b_values, reaction_types):
        """Update min ELN when functional group selections change."""
        return DEFAULT_MIN_ELN

    # ------------------------------------------------------------------
    # Filter panel toggle ----------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('filter-panel-container', 'style'),
        [Input('toggle-filters-btn', 'n_clicks')],
        [State('filter-panel-container', 'style')],
        prevent_initial_call=True
    )
    def _toggle_filter_panel(n_clicks, current_style):
        """Toggle the filter panel visibility with smooth animation."""
        if n_clicks is None:
            return no_update
        
        if current_style is None:
            current_style = {}
        
        # Toggle visibility with smooth animation
        if current_style.get('maxHeight') == '0px' or current_style.get('display') == 'none':
            # Show panel
            current_style.update({
                'display': 'block',
                'maxHeight': '200px',  # Adjust based on content height
                'padding': '20px',
                'margin': '0',
                'border': '1px solid #e0e0e0',
                'borderTop': 'none',
                'borderRadius': '0 0 8px 8px',
                'background': '#fafbfc',
                'overflow': 'hidden',
                'transition': 'max-height 0.3s ease-in-out, padding 0.3s ease-in-out'
            })
        else:
            # Hide panel
            current_style.update({
                'maxHeight': '0px',
                'padding': '0 20px',
                'overflow': 'hidden'
            })
        
        return current_style

    # ------------------------------------------------------------------
    # Tabs management --------------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('analysis-tabs', 'children'),
         Output('analysis-tabs', 'value')],
        [Input('reactant-types-dropdown', 'value')],
        [State('analysis-tabs', 'value')]
    )
    def _update_tabs_based_on_categories(reactant_types, current_tab):
        """Dynamically show/hide heatmap tab based on number of selected reactant types."""
        
        # Start with boxplot tab
        tabs = [
            dcc.Tab(
                label='Boxplot',
                value='tab-graph',
                children=[
                    html.Div(
                        id='boxplot-container',
                        style={'height': '800px', 'width': '100%'},
                        children=[
                            dcc.Graph(
                                id='boxplot',
                                clear_on_unhover=True,
                                style={'height': '100%'}
                            )
                        ]
                    )
                ]
            )
        ]
        
        # Only include heatmap tab if two or more reactant types are selected
        if reactant_types and len(reactant_types) >= 2:  # If two or more reactant types are selected
            tabs.append(
                dcc.Tab(
                    label='Heatmap',
                    value='tab-heatmap',
                    children=[
                        html.Div(
                            id='heatmap-container',
                            style={'height': '800px', 'width': '100%'},
                            children=[
                                dcc.Graph(
                                    id='heatmap',
                                    clear_on_unhover=True,
                                    style={'height': '100%'}
                                )
                            ]
                        )
                    ]
                )
            )
        
        # Always append statistics tab LAST
        tabs.append(
            dcc.Tab(
                label='Statistics',
                value='tab-stats',
                children=[
                    html.Div(
                        id='stats-container',
                        style={'height': 'auto', 'width': '100%', 'padding': '12px 0'},
                        children=[
                            html.Div(id='stats-content')
                        ]
                    )
                ]
            )
        )

        # If current tab is heatmap but heatmap is no longer available, switch to boxplot
        if current_tab == 'tab-heatmap' and (not reactant_types or len(reactant_types) < 2):
            return tabs, 'tab-graph'
        
        # If current tab is none of the available tabs, default to boxplot
        available_values = [t.value for t in tabs]
        if current_tab not in available_values:
            return tabs, 'tab-graph'
        
        return tabs, current_tab

    # ------------------------------------------------------------------
    # Descriptive statistics tab ---------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('stats-content', 'children'),
        FILTER_INPUTS + [Input('analysis-tabs', 'value')],
    )
    def _update_stats_table(*args):
        # Last arg is active_tab (appended after FILTER_INPUTS)
        active_tab = args[-1]
        if active_tab != 'tab-stats':
            return no_update

        fkw = _parse_filter_args(args[:len(FILTER_INPUTS)])
        dff = _call_filter_data(fkw, min_eln=None, max_components=None)
        
        if dff is None or dff.empty:
            return html.Div('No data available for the current filters.', style={'color': '#6c757d'})
        
        # Select relevant numeric columns for descriptive stats
        numeric_cols = []
        for col in ['z-Score', 'AREA_TOTAL_REDUCED']:
            if col in dff.columns:
                numeric_cols.append(col)
        # Add any other numeric columns automatically
        for col in dff.columns:
            if col not in numeric_cols and pd.api.types.is_numeric_dtype(dff[col]):
                numeric_cols.append(col)
        
        if not numeric_cols:
            return html.Div('No numeric columns found for statistics.', style={'color': '#6c757d'})
        
        # Compute describe
        desc = dff[numeric_cols].describe().T.reset_index().rename(columns={'index': 'Metric'})
        # Pretty format counts as ints
        if 'count' in desc.columns:
            desc['count'] = desc['count'].round(0).astype(int)
        
        # Round floats for readability
        for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            if col in desc.columns:
                desc[col] = desc[col].round(3)
        
        # Add ELN count overall
        eln_count = dff['ELN_ID'].nunique() if 'ELN_ID' in dff.columns else len(dff)
        header = html.Div(f"Rows: {len(dff):,} | Unique ELNs: {eln_count:,}", style={'marginBottom': '8px', 'color': '#6c757d'})
        
        # Render as simple HTML table to avoid extra dependencies
        table_header = html.Tr([html.Th(col) for col in desc.columns])
        table_rows = [html.Tr([html.Td(desc.iloc[i][col]) for col in desc.columns]) for i in range(len(desc))]
        table = html.Table([
            html.Thead(table_header),
            html.Tbody(table_rows)
        ], style={'width': '100%', 'borderCollapse': 'collapse'}, className='stats-table')
        
        return html.Div([header, table])

    # ------------------------------------------------------------------
    # Max components slider update -------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('max-components-input', 'max'),
         Output('max-components-input', 'marks'),
         Output('max-components-input', 'value', allow_duplicate=True)],
        [Input('filter-stats-store', 'data')],
        [State('max-components-input', 'value'),
         State('max-components-input', 'max')],
        prevent_initial_call='initial_duplicate'
    )
    def _update_max_components_slider(stats, current_value, previous_max):
        """Dynamically update the max components slider based on available data.

        Uses the precomputed cap from the stats store to avoid recomputation.
        """
        if not stats or 'max_components_cap' not in stats:
            # Preserve current value (or default 10) while marks are minimal
            return 10, {1: '1', 5: '5', 10: '10'}, (current_value if current_value else 10)

        max_value = max(1, int(stats['max_components_cap']))
        marks = {1: '1'}
        if max_value <= 20:
            # For smaller values, use 5-step increments
            for i in range(5, max_value + 1, 5):
                marks[i] = str(i)
        else:
            # For larger values, use 10-step increments to prevent overlap
            for i in range(10, max_value + 1, 10):
                marks[i] = str(i)
        
        # If max_value is not a multiple of the step size, add it as the final mark
        step_size = 10 if max_value > 20 else 5
        if max_value % step_size != 0:
            marks[max_value] = str(max_value)
        # Determine a sensible slider value:
        # - If unset: use 10 or the new max, whichever is smaller
        # - If previously clamped to a smaller max (detected via previous_max) and capacity increased,
        #   bump back up to default (10 or new max)
        # - Otherwise: clamp the existing value to the new range
        default_target = min(10, max_value)
        if current_value is None:
            proposed_value = default_target
        elif previous_max is not None and isinstance(previous_max, (int, float)) and max_value > previous_max and current_value == previous_max:
            # Previously clamped to the old cap – now that capacity grew, bump up to default
            proposed_value = default_target
        elif previous_max is not None and isinstance(previous_max, (int, float)) and max_value > previous_max and current_value < default_target:
            # Capacity grew and current value is below the recommended default – raise to default
            proposed_value = default_target
        elif current_value <= 1 and max_value > 1:
            # Common case when we had only 1 component; raise when more are available
            proposed_value = default_target
        else:
            proposed_value = current_value

        proposed_value = max(1, min(proposed_value, max_value))
        return max_value, marks, proposed_value

    # ------------------------------------------------------------------
    # CSV download ------------------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        *FILTER_STATES,
        prevent_initial_call=True,
    )
    def _download_csv(n_clicks, *args):
        if not n_clicks:
            return no_update
        try:
            fkw = _parse_filter_args(args)
            dff = _call_filter_data(fkw)
            return dcc.send_data_frame(dff.to_csv, "filtered_data.csv", index=False)
        except Exception:
            logger.exception("CSV download failed")
            return no_update

    # ------------------------------------------------------------------
    # PNG download ------------------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output("download-png", "data"),
        Input("download-png-btn", "n_clicks"),
        *FILTER_STATES,
        State("analysis-tabs", "value"),
        State('presentation-mode-store', 'data'),
        prevent_initial_call=True,
    )
    def _download_png(n_clicks, *args):
        if not n_clicks:
            return no_update

        fkw = _parse_filter_args(args[:len(FILTER_STATES)])
        active_tab = args[len(FILTER_STATES)]
        presentation_mode = args[len(FILTER_STATES) + 1]

        reactant_types = fkw['reactant_types'] or DEFAULT_REACTANT_FALLBACK

        if active_tab == 'tab-stats':
            return no_update

        try:
            fkw['reactant_types'] = reactant_types
            dff = _call_filter_data(fkw)
            if active_tab == 'tab-heatmap':
                fig, adaptive_height = pu.create_heatmap(dff, reactant_types, presentation_mode=presentation_mode)
                filename = 'heatmap.png'
            else:
                fig, adaptive_height = pu.create_boxplot(dff, reactant_types, presentation_mode=presentation_mode)
                filename = 'boxplot.png'
            buf = io.BytesIO()
            fig.write_image(buf, format="png", width=1200, height=adaptive_height, scale=2)
            buf.seek(0)
            return dcc.send_bytes(buf.getvalue(), filename)
        except Exception:
            logger.exception("PNG download failed")
            return no_update

    # ------------------------------------------------------------------
    # Statistics updates -----------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('whole-dataset-stats', 'style'),
         Output('whole-dataset-content', 'children'),
         Output('functional-group-a-stats', 'style'),
         Output('functional-group-a-stats-content', 'children'),
         Output('functional-group-b-stats', 'style'),
         Output('functional-group-b-stats-content', 'children')],
        [Input('filter-stats-store', 'data'),
         Input('reactant-types-dropdown', 'value')]
    )
    def _update_filtering_statistics(stats, reactant_types):
        """Update the filtering statistics display."""
        # Check if we have the minimum required filters
        if not stats or not reactant_types:
            # Hide all stat containers
            hidden_style = {
                'display': 'none'
            }
            return (hidden_style, [], hidden_style, [], hidden_style, [])
        
        # Base style for visible containers
        visible_style = {
            'display': 'block'
        }
        
        # Whole dataset stats
        whole_dataset_style = visible_style.copy()
        whole_dataset_content = []
        if 'whole_dataset' in stats:
            stage_stats = stats['whole_dataset']
            whole_dataset_content = [
                html.Div(f"ELNs: {stage_stats['elns']:,}")
            ]
        
        # Functional Group A stats
        fg_a_style = visible_style.copy()
        fg_a_content = []
        if 'after_fg_a' in stats:
            stage_stats = stats['after_fg_a']
            fg_a_content = [
                html.Div(f"ELNs: {stage_stats['elns']:,}")
            ]
        else:
            fg_a_content = [
                html.Div("No selection", style={'marginBottom': '2px', 'color': '#adb5bd', 'fontStyle': 'italic'}),
                html.Div("", style={'color': '#adb5bd'})
            ]
        
        # Functional Group B stats
        fg_b_style = visible_style.copy()
        fg_b_content = []
        if 'after_fg_b' in stats:
            stage_stats = stats['after_fg_b']
            fg_b_content = [
                html.Div(f"ELNs: {stage_stats['elns']:,}")
            ]
        else:
            fg_b_content = [
                html.Div("No selection", style={'marginBottom': '2px', 'color': '#adb5bd', 'fontStyle': 'italic'}),
                html.Div("", style={'color': '#adb5bd'})
            ]
        
        return (whole_dataset_style, whole_dataset_content,
                fg_a_style, fg_a_content,
                fg_b_style, fg_b_content)

    # ------------------------------------------------------------------
    # Reactive boxplot --------------------------------------------------
    # ------------------------------------------------------------------
    def _update_plot(plot_fn, expected_tab, *args):
        """Shared logic for boxplot and heatmap callbacks."""
        fkw = _parse_filter_args(args[:len(FILTER_INPUTS)])
        active_tab = args[len(FILTER_INPUTS)]
        presentation_mode = args[len(FILTER_INPUTS) + 1]

        if active_tab != expected_tab:
            return no_update, no_update

        reactant_types = fkw['reactant_types'] or DEFAULT_REACTANT_FALLBACK
        fkw['reactant_types'] = reactant_types

        dff = _call_filter_data(fkw)
        fig, adaptive_height = plot_fn(dff, reactant_types, presentation_mode=presentation_mode)

        container_height = max(800, adaptive_height + 100)
        container_style = {
            'height': f'{container_height}px',
            'width': '100%'
        }
        return fig, container_style

    @app.callback(
        [Output('boxplot', 'figure'),
         Output('boxplot-container', 'style')],
        FILTER_INPUTS + [
            Input('analysis-tabs', 'value'),
            Input('presentation-mode-store', 'data'),
        ],
    )
    def _update_boxplot(*args):
        return _update_plot(pu.create_boxplot, 'tab-graph', *args)

    # ------------------------------------------------------------------
    # Reactive heatmap -------------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        [Output('heatmap', 'figure'),
         Output('heatmap-container', 'style')],
        FILTER_INPUTS + [
            Input('analysis-tabs', 'value'),
            Input('presentation-mode-store', 'data'),
        ],
    )
    def _update_heatmap(*args):
        return _update_plot(pu.create_heatmap, 'tab-heatmap', *args)






    # ------------------------------------------------------------------
    # Interactive tutorial ----------------------------------------------
    # ------------------------------------------------------------------
    @app.callback(
        Output('tutorial-store', 'data'),
        [Input('start-tutorial-btn', 'n_clicks'),
         Input('tutorial-next', 'n_clicks'),
         Input('tutorial-back', 'n_clicks'),
         Input('tutorial-skip', 'n_clicks'),
         # Auto-advance triggers
         Input('reaction-type-dropdown', 'value'),
         Input('reactant-types-dropdown', 'value'),
         Input('functional-group-a-dropdown', 'value'),
         Input('functional-group-b-dropdown', 'value'),
         Input('filter-panel-container', 'style'),
         Input('min-eln-input', 'value'),
         Input('topn-zscore-input', 'value'),
         Input('max-components-input', 'value'),
         Input('exclude-cui-checkbox', 'value'),
         Input('analysis-tabs', 'value')],
        [State('tutorial-store', 'data')]
    )
    def _tutorial_state(start_clicks, next_clicks, back_clicks, skip_clicks,
                        reaction_types, reactant_types, fg_a_vals, fg_b_vals, filter_panel_style,
                        min_eln, topn, max_comp, exclude_cui_val, tabs_value,
                        data):
        """Update tutorial active flag and step based on controls and gating."""
        data = data or {'active': False, 'step': 0}
        ctx = callback_context
        if not ctx.triggered:
            return data
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        steps_count = 11  # 0..10

        _step_kw = dict(
            reaction_types=reaction_types, reactant_types=reactant_types,
            fg_a_vals=fg_a_vals, fg_b_vals=fg_b_vals,
            filter_panel_style=filter_panel_style, min_eln=min_eln,
            topn=topn, max_comp=max_comp, exclude_cui_val=exclude_cui_val,
            tabs_value=tabs_value,
        )

        if trigger == 'start-tutorial-btn':
            return {'active': True, 'step': 0}
        if trigger == 'tutorial-skip':
            return {'active': False, 'step': 0}
        # Ignore controls when tutorial is not active
        if not data.get('active'):
            return data
        if trigger == 'tutorial-back':
            prev_step = max(0, (data.get('step') or 0) - 1)
            return {**data, 'step': prev_step}
        if trigger == 'tutorial-next':
            current_step = data.get('step') or 0
            # If last step, finish
            if current_step >= steps_count - 1:
                return {'active': False, 'step': 0}
            # Allow continuing even if not satisfied (acts as skip)
            return {**data, 'step': current_step + 1}

        # Auto-advance only when the triggering input corresponds to the current step
        if data.get('active'):
            current_step = data.get('step') or 0
            step_to_triggers = {
                0: ['reaction-type-dropdown'],
                1: ['reactant-types-dropdown'],
                2: ['functional-group-a-dropdown'],
                3: ['functional-group-b-dropdown'],
                4: ['filter-panel-container'],
                5: ['min-eln-input'],
                6: ['topn-zscore-input'],
                7: ['max-components-input'],
                8: ['exclude-cui-checkbox'],
                9: ['analysis-tabs'],
            }
            # Extract triggering component id (without property)
            trigger_id = trigger.split('.')[0]
            if trigger_id in step_to_triggers.get(current_step, []):
                if _is_tutorial_step_satisfied(current_step, **_step_kw):
                    # If last step reached, finish, otherwise advance
                    if current_step >= steps_count - 1:
                        return {'active': False, 'step': 0}
                    return {**data, 'step': current_step + 1}

        return data

    @app.callback(
        [Output('tutorial-overlay', 'style'),
         Output('tutorial-title', 'children'),
         Output('tutorial-body', 'children'),
         Output('tutorial-next', 'disabled'),
         Output('tutorial-next', 'children'),
         Output('reaction-type-dropdown', 'className'),
         Output('reactant-types-dropdown', 'className'),
         Output('functional-group-a-dropdown', 'className'),
         Output('functional-group-b-dropdown', 'className'),
         Output('toggle-filters-btn', 'className'),
         Output('min-eln-input', 'className'),
         Output('topn-zscore-input', 'className'),
         Output('max-components-input', 'className'),
         Output('exclude-cui-checkbox', 'className'),
         Output('include-scaleup-checkbox', 'className'),
         Output('include-null-categories-checkbox', 'className'),
         Output('analysis-tabs', 'className')],
        [Input('tutorial-store', 'data'),
         Input('filter-panel-container', 'style'),
         Input('min-eln-input', 'value'),
         Input('topn-zscore-input', 'value'),
         Input('max-components-input', 'value'),
         Input('exclude-cui-checkbox', 'value'),
         Input('include-scaleup-checkbox', 'value'),
         Input('include-null-categories-checkbox', 'value'),
         Input('reaction-type-dropdown', 'value'),
         Input('reactant-types-dropdown', 'value'),
         Input('functional-group-a-dropdown', 'value'),
         Input('functional-group-b-dropdown', 'value'),
         Input('analysis-tabs', 'value')]
    )
    def _tutorial_present(data, filter_panel_style, min_eln, topn, max_comp,
                          exclude_cui_val, include_scaleup_val, include_null_val,
                          reaction_types, reactant_types, fg_a_vals, fg_b_vals, tabs_value):
        """Render tutorial overlay and highlight the active target."""
        data = data or {'active': False, 'step': 0}
        active = data.get('active', False)
        step = data.get('step', 0)

        # Define steps metadata
        steps = [
            {"id": "reaction-type-dropdown", "title": "Select Reaction Type(s)", "body": "Pick one or more reaction classes to focus the analysis."},
            {"id": "reactant-types-dropdown", "title": "Select Reactant Type(s)", "body": "Choose the reagent categories to analyze (e.g., Catalyst, Base)."},
            {"id": "functional-group-a-dropdown", "title": "Choose Functional Group A", "body": "Optionally filter by reacting functional groups (side A)."},
            {"id": "functional-group-b-dropdown", "title": "Choose Functional Group B", "body": "Optionally filter by reacting functional groups (side B)."},
            {"id": "toggle-filters-btn", "title": "Open Options", "body": "Click Options to reveal advanced filters."},
            {"id": "min-eln-input", "title": "Minimum ELNs", "body": "Drag to require a minimum number of ELNs (data points) per selection."},
            {"id": "topn-zscore-input", "title": "Top-N z-Score", "body": "Limit to the top-N z-scores per ELN and selected reactant type(s)."},
            {"id": "max-components-input", "title": "Max Components", "body": "Cap how many components are displayed in plots."},
            {"id": "exclude-cui-checkbox", "title": "Exclude CuI as Catalyst", "body": "Toggle to include/exclude CuI catalyst entries."},
            {"id": "analysis-tabs", "title": "Explore Results", "body": "Switch tabs to view Heatmap or Boxplot."},
            {"id": None, "title": "You're all set!", "body": "That’s the tour. You can restart anytime via Start Tutorial."},
        ]

        # Gating – determine if action complete for current step
        _step_kw = dict(
            reaction_types=reaction_types, reactant_types=reactant_types,
            fg_a_vals=fg_a_vals, fg_b_vals=fg_b_vals,
            filter_panel_style=filter_panel_style, min_eln=min_eln,
            topn=topn, max_comp=max_comp, exclude_cui_val=exclude_cui_val,
            tabs_value=tabs_value,
        )

        # Next is always enabled; label reflects completion state
        disabled_next = False
        if step >= len(steps) - 1:
            next_label = 'Finish'
        else:
            next_label = 'Next' if _is_tutorial_step_satisfied(step, **_step_kw) else 'Skip'

        # Clamp step to valid range to avoid indexing errors
        if step < 0:
            step = 0
        if step >= len(steps):
            step = len(steps) - 1

        # Overlay visibility
        overlay_style = {'display': 'block'} if active else {'display': 'none'}

        # Highlight assignment
        target_id = steps[step]['id'] if 0 <= step < len(steps) else None
        highlight = 'tutorial-highlight'
        none_cls = ''

        def cls_for(cid: str) -> str:
            return highlight if (active and target_id == cid) else none_cls

        title = steps[step]['title'] if 0 <= step < len(steps) else ''
        body = steps[step]['body'] if 0 <= step < len(steps) else ''

        return (
            overlay_style,
            title,
            body,
            disabled_next,
            next_label,
            cls_for('reaction-type-dropdown'),
            cls_for('reactant-types-dropdown'),
            cls_for('functional-group-a-dropdown'),
            cls_for('functional-group-b-dropdown'),
            cls_for('toggle-filters-btn'),
            cls_for('min-eln-input'),
            cls_for('topn-zscore-input'),
            cls_for('max-components-input'),
            cls_for('exclude-cui-checkbox'),
            cls_for('include-scaleup-checkbox'),
            cls_for('include-null-categories-checkbox'),
            cls_for('analysis-tabs'),
        )

    @app.callback(
        Output('start-tutorial-btn', 'children'),
        [Input('tutorial-store', 'data')]
    )
    def _tutorial_label(data):
        active = bool(data and data.get('active'))
        return 'Restart Tutorial' if active else 'Start Tutorial'


