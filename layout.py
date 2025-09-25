from __future__ import annotations

"""layout.py
===============
Single source of truth for the **visual layout** of the Dash application.

Why keep the layout in its own module?
-------------------------------------
1. **Separation of concerns** – the layout describes *what* the user sees
   while callbacks describe *how* the app reacts to user input.  Having
   them in different files helps future maintainers reason about the
   codebase.
2. **Reusability** – a clearly scoped `serve_layout()` function can be
   imported from notebooks or unit tests to render components in
   isolation.
3. **Avoid circular imports** – the layout needs *no* callback
   decorators so it can be imported *before* callbacks are registered.

The function names are chosen to read nicely inside :pyfile:`app.py`:
"""

from dash import dcc, html

import data_utils as du

# ---------------------------------------------------------------------------
# 1. CONVENIENCE – drop-down option helpers
# ---------------------------------------------------------------------------

# Pre-compute **static** option lists (the reactive options are handled via
# callbacks in :pyfile:`callbacks.py`).
REACTION_TYPE_OPTIONS = [{"label": rt, "value": rt} for rt in du.REACTION_TYPES]
CATEGORY_OPTIONS = [{"label": c, "value": c} for c in du.CATEGORY_OPTIONS]




# ---------------------------------------------------------------------------
# 2. PUBLIC API
# ---------------------------------------------------------------------------

def serve_layout() -> html.Div:  # noqa: D401 (imperative mood is fine here)
    """Return the *root* Dash component (called by Dash on page load)."""

    # The outermost container allows us to apply a single class for global
    # styling (see ``assets/apple.css``).
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
            # --------------------------------------------------------------
            # 2.1  HEADER – logo & title
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
                            html.Button(
                                "Reset",
                                id="reset-btn"
                            ),
                            html.Button(
                                "Presentation Mode",
                                id="presentation-mode-toggle",
                                className="presentation-toggle-btn",
                                n_clicks=0
                            ),
                            html.Button(
                                "Start Tutorial",
                                id="start-tutorial-btn",
                                n_clicks=0
                            )
                        ]
                    )
                ],
            ),
            # --------------------------------------------------------------
            # 2.2  DROPDOWN ROW – primary filters
            # --------------------------------------------------------------
            html.Div(
                className="dropdown-row",
                children=[
                    # Reaction type ------------------------------------------------
                    html.Div([
                        html.Label('Reaction Type(s):'),
                        dcc.Dropdown(
                            id='reaction-type-dropdown',
                            options=REACTION_TYPE_OPTIONS,
                            value=['Buchwald-Hartwig'] if 'Buchwald-Hartwig' in du.REACTION_TYPES else [du.REACTION_TYPES[0]],
                            multi=True,
                            placeholder='Select one or more reaction types...',
                        ),
                        html.Div(
                            id='whole-dataset-stats',
                            style={'display': 'none', 'marginTop': '4px', 'textAlign': 'center'},
                            children=[
                                html.Div(id='whole-dataset-content', style={'fontSize': '11px', 'color': '#6c757d'})
                            ]
                        )
                    ], className='dropdown-col'),
                ],
            ),

            # --------------------------------------------------------------
            # 2.2.1  FUNCTIONAL GROUP SELECTION ROW
            # --------------------------------------------------------------
            html.Div(
                className='functional-group-row',
                style={'display': 'flex', 'justifyContent': 'flex-start', 'margin': '16px 0', 'gap': '16px'},
                children=[
                    html.Div([
                        html.Label('Reacting Functional Group(s) A:'),
                        dcc.Dropdown(
                            id='functional-group-a-dropdown',
                            options=[{'label': 'All', 'value': 'All'}],
                            value=['RNH2 a-branch', 'RNH2'],
                            multi=True,
                            style={'minWidth': '200px'},
                            placeholder='Select functional groups...',
                        ),
                        html.Div(
                            id='functional-group-a-stats',
                            style={'display': 'none', 'marginTop': '4px', 'textAlign': 'center'},
                            children=[
                                html.Div(id='functional-group-a-stats-content', style={'fontSize': '11px', 'color': '#6c757d'})
                            ]
                        )
                    ]),
                    html.Div([
                        html.Label('Reacting Functional Group(s) B:'),
                        dcc.Dropdown(
                            id='functional-group-b-dropdown',
                            options=[{'label': 'All', 'value': 'All'}],
                            value=['ArBr', 'ArCl'],
                            multi=True,
                            style={'minWidth': '200px'},
                            placeholder='Select functional groups...',
                        ),
                        html.Div(
                            id='functional-group-b-stats',
                            style={'display': 'none', 'marginTop': '4px', 'textAlign': 'center'},
                            children=[
                                html.Div(id='functional-group-b-stats-content', style={'fontSize': '11px', 'color': '#6c757d'})
                            ]
                        )
                    ]),
                ],
            ),

            # --------------------------------------------------------------
            # 2.3  REACTANT TYPE SELECTION ROW
            # --------------------------------------------------------------
            html.Div(
                className="dropdown-row",
                children=[
                    # Reactant types (combined categories) -----------------------
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
            # 2.6  OPTIONS TOGGLE (moved above charts)
            # --------------------------------------------------------------
            html.Div(
                id='filter-toggle-container',
                style={'position': 'relative', 'margin': '32px 0 24px 0'},
                children=[
                    html.Div(
                        id='filter-toggle-line',
                        style={
                            'position': 'absolute',
                            'left': '0',
                            'top': '50%',
                            'width': '100%',
                            'height': '2px',
                            'backgroundColor': '#d0d0d0',
                            'zIndex': '1'
                        }
                    ),
                    html.Button(
                        id='toggle-filters-btn',
                        children=[
                            html.I(className='fas fa-filter', style={'marginRight': '8px'}),
                            html.Span('Options', style={'textAlign': 'center'})
                        ],
                        style={
                            'position': 'absolute',
                            'left': '20px',
                            'top': '50%',
                            'transform': 'translateY(-50%)',
                            'backgroundColor': 'white',
                            'border': '2px solid #d0d0d0',
                            'borderRadius': '20px',
                            'padding': '8px 16px',
                            'cursor': 'pointer',
                            'zIndex': '2',
                            'fontSize': '14px',
                            'fontWeight': '500',
                            'color': '#666',
                            'display': 'flex',
                            'alignItems': 'center'
                        }
                    )
                ]
            ),

            # --------------------------------------------------------------
            # 2.6.1  OPTIONS PANEL (appears below button when toggled)
            # --------------------------------------------------------------
            html.Div(
                id='filter-panel-container',
                className='filter-panel',
                style={
                    'margin': '0',
                    'padding': '20px',
                    'border': '1px solid #e0e0e0',
                    'borderTop': 'none',
                    'borderRadius': '0 0 8px 8px',
                    'background': '#fafbfc',
                    'display': 'none',
                    'maxHeight': '0',
                    'overflow': 'hidden',
                    'transition': 'max-height 0.3s ease-in-out, padding 0.3s ease-in-out'
                },
                children=[
                    html.Div([
                        html.Label('Minimum Number of ELNs:', style={'marginRight': '8px'}),
                        html.Div(
                            dcc.Slider(
                                id='min-eln-input',
                                min=1,
                                max=20,
                                step=1,
                                value=10,
                                marks={i: str(i) for i in [1, 5, 10, 15, 20]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True,
                                persistence_type='local',
                            ),
                            style={'width': '150px', 'marginRight': '32px'}
                        ),
                        html.Label('Top-N z-Score per (ELN_ID, selected reactant type(s)):', style={'marginRight': '8px'}),
                        html.Div(
                            dcc.Slider(
                                id='topn-zscore-input',
                                min=1,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in [1, 3, 5, 7, 10]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True,
                                persistence_type='local',
                            ),
                            style={'width': '120px', 'marginRight': '32px'}
                        ),
                        html.Label('Max Components to Display:', style={'marginRight': '8px'}),
                        html.Div(
                            dcc.Slider(
                                id='max-components-input',
                                min=1,
                                max=10,
                                step=1,
                                value=10,
                                marks={1: '1', 5: '5', 10: '10'},
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True,
                                persistence_type='local',
                            ),
                            style={'width': '150px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'marginBottom': '16px', 'justifyContent': 'flex-start'}),
                    html.Div([
                        dcc.Checklist(
                            id='exclude-cui-checkbox',
                            options=[{'label': 'Exclude CuI as Catalyst', 'value': 'exclude_cui'}],
                            value=['exclude_cui'],
                            inline=True,
                            style={'marginRight': '32px'},
                            persistence=True,
                            persistence_type='local',
                        ),
                        dcc.Checklist(
                            id='include-scaleup-checkbox',
                            options=[{'label': 'Exclude Scale-Up Plates', 'value': True}],
                            value=[True],
                            inline=True,
                            style={'marginRight': '32px'},
                            persistence=True,
                            persistence_type='local',
                        ),
                        dcc.Checklist(
                            id='include-null-categories-checkbox',
                            options=[{'label': 'Include combinations with null reactant types', 'value': True}],
                            value=[True],
                            inline=True,
                            persistence=True,
                            persistence_type='local',
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'justifyContent': 'flex-start'}),
                    html.Div([
                        html.Button('Download CSV', id='download-csv-btn', style={'marginRight': '16px'}),
                        dcc.Download(id='download-csv'),
                        html.Button('Download PNG', id='download-png-btn', style={'marginLeft': '16px'}),
                        dcc.Download(id='download-png'),
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start', 'marginTop': '16px'}),
                ]
            ),

            # --------------------------------------------------------------
            # 2.7  ANALYSIS TABS
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
                                style={'height': '800px', 'width': '100%'},
                                children=[
                                    dcc.Loading(
                                        id="boxplot-loading",
                                        type="default",
                                        children=dcc.Graph(
                                            id='boxplot',
                                            clear_on_unhover=True,
                                            style={'height': '100%'}
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
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
                    ),
                    dcc.Tab(
                        label='Statistics',
                        value='tab-stats',
                        children=[
                            html.Div(
                                id='stats-container',
                                style={'width': '100%', 'padding': '12px 0'},
                                children=[
                                    dcc.Loading(
                                        id='stats-loading',
                                        type='default',
                                        children=html.Div(id='stats-content')
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),

            # --------------------------------------------------------------
            # 2.8  TUTORIAL OVERLAY
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
                            html.Div(id='tutorial-body', children='Let\'s take a quick tour of the app.'),
                            html.Div(
                                style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '8px', 'marginTop': '12px'},
                                children=[
                                    html.Button('Back', id='tutorial-back', n_clicks=0),
                                    html.Button('Skip', id='tutorial-skip', n_clicks=0),
                                    html.Button('Next', id='tutorial-next', n_clicks=0),
                                ]
                            )
                        ]
                    )
                ]
            ),

        ],
    )