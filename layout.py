from __future__ import annotations

"""layout.py
===============
Single source of truth for the **visual layout** of the Dash application.

Why keep the layout in its own module?
-------------------------------------
1. **Separation of concerns** -- the layout describes *what* the user sees
   while callbacks describe *how* the app reacts to user input.  Having
   them in different files helps future maintainers reason about the
   codebase.
2. **Reusability** -- a clearly scoped `serve_layout()` function can be
   imported from notebooks or unit tests to render components in
   isolation.
3. **Avoid circular imports** -- the layout needs *no* callback
   decorators so it can be imported *before* callbacks are registered.

The function names are chosen to read nicely inside :pyfile:`app.py`:
"""

from dash import dcc, html

import data_utils as du

# ---------------------------------------------------------------------------
# 1. CONVENIENCE -- drop-down option helpers
# ---------------------------------------------------------------------------

# Pre-compute **static** option lists (the reactive options are handled via
# callbacks in :pyfile:`callbacks.py`).
REACTION_TYPE_OPTIONS = [{"label": rt, "value": rt} for rt in du.REACTION_TYPES]
CATEGORY_OPTIONS = [{"label": c, "value": c} for c in du.CATEGORY_OPTIONS]


# ---------------------------------------------------------------------------
# 2. COMPONENT HELPERS
# ---------------------------------------------------------------------------

def _stats_badge(prefix: str) -> html.Div:
    """Return a hidden stats badge component pair for a dropdown."""
    return html.Div(
        id=f'{prefix}-stats',
        className='stats-badge',
        children=[
            html.Div(id=f'{prefix}-stats-content', className='stats-badge-content')
        ],
    )


# ---------------------------------------------------------------------------
# 3. PUBLIC API
# ---------------------------------------------------------------------------

def serve_layout() -> html.Div:  # noqa: D401 (imperative mood is fine here)
    """Return the *root* Dash component (called by Dash on page load).

    Styles live in ``assets/app.css``.
    """

    return html.Div(
        id="main-container",
        className="app-container",
        children=[
            # In-memory store for lightweight stats (filtered data no longer stored client-side)
            dcc.Store(id='filter-stats-store'),
            # Store for presentation mode state
            dcc.Store(id='presentation-mode-store', data=False),
            # Store for interactive tutorial state
            dcc.Store(id='tutorial-store', data={'active': False, 'step': 0}),
            # Store for user-uploaded dataset (memory storage - no size limit)
            dcc.Store(id='uploaded-data-store', storage_type='memory'),
            # Store for upload status messages
            dcc.Store(id='upload-status-store', storage_type='memory'),
            # Store for upload error modal visibility
            dcc.Store(id='upload-error-store', storage_type='memory'),
            # --------------------------------------------------------------
            # HEADER -- logo & title
            # --------------------------------------------------------------
            html.Div(
                className="header",
                children=[
                    html.Img(src="assets/logo.png", className="logo"),
                    html.H1(
                        "Data-Driven Reagent Selection for Empirical Chemical Discovery",
                        className="title",
                    ),
                    html.Div(
                        className="presentation-toggle-container",
                        children=[
                            html.Div(
                                className="upload-container",
                                children=[
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Button(
                                            "Upload Dataset",
                                            id="upload-btn",
                                            className="upload-btn",
                                        ),
                                        accept='.csv',
                                        max_size=50 * 1024 * 1024,
                                    ),
                                    html.Div(
                                        id='upload-status-indicator',
                                        className='upload-status',
                                        children=[],
                                    ),
                                ],
                            ),
                            html.Button("Reset", id="reset-btn"),
                            html.Button(
                                "Presentation Mode",
                                id="presentation-mode-toggle",
                                className="presentation-toggle-btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "Start Tutorial",
                                id="start-tutorial-btn",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
            # --------------------------------------------------------------
            # DROPDOWN ROW -- primary filters
            # --------------------------------------------------------------
            html.Div(
                className="dropdown-row",
                children=[
                    html.Div([
                        html.Label('Reaction Type(s):'),
                        dcc.Dropdown(
                            id='reaction-type-dropdown',
                            options=REACTION_TYPE_OPTIONS,
                            value=['Buchwald-Hartwig'] if 'Buchwald-Hartwig' in du.REACTION_TYPES else [du.REACTION_TYPES[0]],
                            multi=True,
                            placeholder='Select one or more reaction types...',
                        ),
                        _stats_badge('whole-dataset'),
                    ], className='dropdown-col'),
                ],
            ),

            # --------------------------------------------------------------
            # FUNCTIONAL GROUP SELECTION ROW
            # --------------------------------------------------------------
            html.Div(
                className='functional-group-row',
                children=[
                    html.Div([
                        html.Label('Reacting Functional Group(s) A:'),
                        dcc.Dropdown(
                            id='functional-group-a-dropdown',
                            options=[{'label': 'All', 'value': 'All'}],
                            value=['RNH2 a-branch', 'RNH2'],
                            multi=True,
                            className='fg-dropdown',
                            placeholder='Select functional groups...',
                        ),
                        _stats_badge('functional-group-a'),
                    ]),
                    html.Div([
                        html.Label('Reacting Functional Group(s) B:'),
                        dcc.Dropdown(
                            id='functional-group-b-dropdown',
                            options=[{'label': 'All', 'value': 'All'}],
                            value=['ArBr', 'ArCl'],
                            multi=True,
                            className='fg-dropdown',
                            placeholder='Select functional groups...',
                        ),
                        _stats_badge('functional-group-b'),
                    ]),
                ],
            ),

            # --------------------------------------------------------------
            # REACTANT TYPE SELECTION ROW
            # --------------------------------------------------------------
            html.Div(
                className="dropdown-row",
                children=[
                    html.Div([
                        html.Label('Reactant Type(s):'),
                        dcc.Dropdown(
                            id='reactant-types-dropdown',
                            options=CATEGORY_OPTIONS,
                            value=['Catalyst'] if 'Catalyst' in du.CATEGORY_OPTIONS else [du.CATEGORY_OPTIONS[0]],
                            multi=True,
                            placeholder='Select one or more reactant types...',
                        ),
                    ], className='dropdown-col'),
                ],
            ),

            # --------------------------------------------------------------
            # OPTIONS TOGGLE
            # --------------------------------------------------------------
            html.Div(
                id='filter-toggle-container',
                className='filter-toggle-container',
                children=[
                    html.Div(id='filter-toggle-line', className='filter-toggle-line'),
                    html.Button(
                        id='toggle-filters-btn',
                        children=[
                            html.I(className='fas fa-filter btn-icon'),
                            html.Span('Options'),
                        ],
                    ),
                ],
            ),

            # --------------------------------------------------------------
            # OPTIONS PANEL (collapsible)
            # --------------------------------------------------------------
            html.Div(
                id='filter-panel-container',
                className='filter-panel',
                children=[
                    html.Div(className='filter-options-row sliders', children=[
                        html.Label('Minimum Number of ELNs:'),
                        html.Div(
                            dcc.Slider(
                                id='min-eln-input',
                                min=1, max=20, step=1, value=5,
                                marks={i: str(i) for i in [1, 5, 10, 15, 20]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True, persistence_type='local',
                            ),
                            className='slider-wrap min-eln',
                        ),
                        html.Label('Top-N z-Score per (ELN_ID, selected reactant type(s)):'),
                        html.Div(
                            dcc.Slider(
                                id='topn-zscore-input',
                                min=1, max=10, step=1, value=5,
                                marks={i: str(i) for i in [1, 3, 5, 7, 10]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True, persistence_type='local',
                            ),
                            className='slider-wrap topn',
                        ),
                        html.Label('Max Components to Display:'),
                        html.Div(
                            dcc.Slider(
                                id='max-components-input',
                                min=1, max=10, step=1, value=10,
                                marks={1: '1', 5: '5', 10: '10'},
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True, persistence_type='local',
                            ),
                            className='slider-wrap max-comp',
                        ),
                    ]),
                    html.Div(className='filter-options-row', children=[
                        dcc.Checklist(
                            id='exclude-cui-checkbox',
                            options=[{'label': 'Exclude CuI as Catalyst', 'value': 'exclude_cui'}],
                            value=['exclude_cui'],
                            inline=True,
                            className='checklist-item',
                            persistence=True, persistence_type='local',
                        ),
                        dcc.Checklist(
                            id='include-scaleup-checkbox',
                            options=[{'label': 'Exclude Scale-Up Plates', 'value': True}],
                            value=[True],
                            inline=True,
                            className='checklist-item',
                            persistence=True, persistence_type='local',
                        ),
                        dcc.Checklist(
                            id='include-null-categories-checkbox',
                            options=[{'label': 'Include combinations with null reactant types', 'value': True}],
                            value=[True],
                            inline=True,
                            persistence=True, persistence_type='local',
                        ),
                    ]),
                    html.Div(className='filter-options-row downloads', children=[
                        html.Button('Download CSV', id='download-csv-btn', className='download-btn-gap'),
                        dcc.Download(id='download-csv'),
                        html.Button('Download PNG', id='download-png-btn'),
                        dcc.Download(id='download-png'),
                    ]),
                ],
            ),

            # --------------------------------------------------------------
            # ANALYSIS TABS
            # --------------------------------------------------------------
            dcc.Tabs(
                id='analysis-tabs',
                value='tab-graph',
                children=[
                    dcc.Tab(
                        label='Boxplot',
                        value='tab-graph',
                        children=[
                            html.Div(
                                id='boxplot-container',
                                className='plot-container',
                                children=[
                                    dcc.Loading(
                                        id="boxplot-loading",
                                        type="default",
                                        children=dcc.Graph(
                                            id='boxplot',
                                            clear_on_unhover=True,
                                            style={'height': '100%'},
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label='Heatmap',
                        value='tab-heatmap',
                        children=[
                            html.Div(
                                id='heatmap-container',
                                className='plot-container',
                                children=[
                                    dcc.Graph(
                                        id='heatmap',
                                        clear_on_unhover=True,
                                        style={'height': '100%'},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label='Statistics',
                        value='tab-stats',
                        children=[
                            html.Div(
                                id='stats-container',
                                className='stats-container',
                                children=[
                                    dcc.Loading(
                                        id='stats-loading',
                                        type='default',
                                        children=html.Div(id='stats-content'),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # --------------------------------------------------------------
            # UPLOAD ERROR MODAL
            # --------------------------------------------------------------
            html.Div(
                id='upload-error-modal',
                className='upload-error-modal',
                style={'display': 'none'},
                children=[
                    html.Div(
                        className='upload-error-panel',
                        children=[
                            html.Div(
                                className='upload-error-header',
                                children=[
                                    html.H3('Upload Error'),
                                    html.Button(
                                        '\u00d7',
                                        id='upload-error-close',
                                        className='upload-error-close-btn',
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            html.Div(
                                id='upload-error-content',
                                className='upload-error-body',
                                children=[],
                            ),
                            html.Div(
                                className='upload-error-footer',
                                children=[
                                    html.H4('Required Columns:'),
                                    html.Code(
                                        'ELN_ID, PLATENUMBER, Coordinate, AREA_TOTAL_REDUCED, Base, Catalyst, '
                                        'Solvent, Ligand, Reaction Type, FG A, FG B, FG_sorted, z-Score',
                                        className='required-columns-code',
                                    ),
                                    html.Button(
                                        'Close',
                                        id='upload-error-close-btn',
                                        className='close-btn-full',
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # --------------------------------------------------------------
            # TUTORIAL OVERLAY
            # --------------------------------------------------------------
            html.Div(
                id='tutorial-overlay',
                className='tutorial-overlay',
                children=[
                    html.Div(
                        id='tutorial-panel',
                        className='tutorial-panel',
                        children=[
                            html.H3(id='tutorial-title', children='Welcome'),
                            html.Div(id='tutorial-body', children="Let's take a quick tour of the app."),
                            html.Div(
                                className='tutorial-btn-row',
                                children=[
                                    html.Button('Back', id='tutorial-back', n_clicks=0),
                                    html.Button('Skip', id='tutorial-skip', n_clicks=0),
                                    html.Button('Next', id='tutorial-next', n_clicks=0),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
