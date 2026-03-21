from __future__ import annotations

"""app.py
===========
Entry-point that *bootstraps* the Dash application by wiring together the
modular `layout.py` and `callbacks.py` helpers.  Keeping this file tiny makes
it immediately clear **where** the application starts and avoids duplicate
logic (all heavy lifting lives in the dedicated modules).
"""

import os

from dash import Dash

import layout  # visual components
import callbacks  # interactivity/callbacks


def _create_dash_app() -> Dash:  # noqa: D401 (imperative mood is fine here)
    """Return a fully configured :class:`dash.Dash` instance."""

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="Z-Score Dashboard",
        update_title=None,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap",
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css",
        ],
    )
    app.layout = layout.serve_layout  # callable – Dash calls it on every page load

    # Register ALL callbacks in a single line – keeps import order clean
    callbacks.register(app)

    return app


# Expose *app* so `gunicorn app:app` just works ------------------------------
app: Dash = _create_dash_app()
server = app.server  # Required for Plotly Cloud deployment


if __name__ == "__main__":
    # Local development – enable hot-reloading
    app.run(debug=os.environ.get("DASH_DEBUG", "1") == "1") 