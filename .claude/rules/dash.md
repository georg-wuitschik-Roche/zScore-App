---
paths:
  - "layout.py"
  - "callbacks.py"
  - "app.py"
---

# Dash Conventions

## Layout
- `serve_layout()` is a callable — Dash calls it on each page load for dynamic content.
- All UI components live in `layout.py`. Never define components in `callbacks.py`.
- Use `dcc.Store` for client-side state. Avoid global mutable state.

```python
# Good - layout.py
def serve_layout() -> html.Div:
    return html.Div([
        dcc.Store(id="filter-stats-store", storage_type="memory"),
        html.Div(id="main-content"),
    ])

# Bad - defining layout in callbacks
@app.callback(...)
def update():
    return html.Div([html.H1("Title"), ...])  # Layout belongs in layout.py
```

## Callbacks
- All callbacks registered in `callbacks.py` via `register_callbacks(app)`.
- Use `prevent_initial_call=True` unless the callback must fire on page load.
- Prefix callback functions with underscore: `_toggle_presentation_mode`.
- Group related outputs in a single callback to reduce round-trips.

```python
# Good
def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        [Output("boxplot", "figure"), Output("stats", "children")],
        [Input("reaction-type", "value"), Input("min-elns", "value")],
        [State("uploaded-data-store", "data")],
        prevent_initial_call=True,
    )
    def _update_boxplot(reaction_types, min_elns, uploaded_data):
        ...

# Bad - no prevent_initial_call, no underscore prefix
@app.callback(Output("boxplot", "figure"), Input("btn", "n_clicks"))
def update_boxplot(n):
    ...
```

## Component IDs
Use kebab-case for component IDs. Be descriptive.

```python
# Good
html.Div(id="reaction-type-dropdown")
dcc.Store(id="uploaded-data-store")

# Bad
html.Div(id="rtdd")
dcc.Store(id="store1")
```

## State Management
- `dcc.Store(storage_type="memory")` for session-scoped data (uploaded datasets)
- `dcc.Store(storage_type="session")` for persistence across refreshes (rarely needed)
- Server-side cache in `data_utils.py` for expensive computations (LRU with MD5 keys)

## No Inline Styles (Prefer CSS)
Use `assets/app.css` for styling. Only use inline styles for truly dynamic values.

```python
# Good - class-based styling
html.Div(className="filter-panel")

# Acceptable - dynamic value
html.Div(style={"height": f"{computed_height}px"})

# Bad - static inline styles
html.Div(style={"backgroundColor": "#f5f5f7", "borderRadius": "8px"})
```
