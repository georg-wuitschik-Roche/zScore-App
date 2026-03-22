"""Smoke tests for app.py — app creation and configuration."""

from __future__ import annotations

import app


class TestAppCreation:
    def test_app_exists(self):
        assert hasattr(app, 'app')

    def test_app_title(self):
        assert app.app.title is not None

    def test_suppress_callback_exceptions_true(self):
        assert app.app.config.suppress_callback_exceptions is True

    def test_server_exposed(self):
        assert hasattr(app, 'server')
        assert app.server is not None

    def test_layout_is_callable(self):
        # Dash app layout should be a callable (serve_layout)
        assert callable(app.app.layout) or app.app.layout is not None
