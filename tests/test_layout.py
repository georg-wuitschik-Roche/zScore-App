"""Tests for layout.py — component structure, IDs, and default values."""

from __future__ import annotations

import pytest
from dash import html

import layout

# ===========================================================================
# Component tree walking helper
# ===========================================================================


def _collect_ids(component, ids=None):
    """Recursively collect all component IDs from a Dash component tree."""
    if ids is None:
        ids = set()
    if hasattr(component, 'id') and component.id:
        ids.add(component.id)
    if hasattr(component, 'children'):
        children = component.children
        if isinstance(children, list | tuple):
            for child in children:
                _collect_ids(child, ids)
        elif children is not None:
            _collect_ids(children, ids)
    return ids


# ===========================================================================
# serve_layout
# ===========================================================================


class TestServeLayout:
    @pytest.fixture(autouse=True)
    def _layout(self):
        self.root = layout.serve_layout()

    def test_returns_html_div(self):
        assert isinstance(self.root, html.Div)

    def test_contains_url_location(self):
        ids = _collect_ids(self.root)
        assert 'url' in ids

    def test_contains_all_stores(self):
        ids = _collect_ids(self.root)
        expected_stores = [
            'filter-stats-store',
            'filter-trigger-store',
            'presentation-mode-store',
            'tutorial-store',
            'uploaded-data-store',
            'url-restore-flag',
        ]
        for store_id in expected_stores:
            assert store_id in ids, f'Missing store: {store_id}'

    def test_contains_landing_page(self):
        ids = _collect_ids(self.root)
        assert 'landing-page' in ids

    def test_contains_dashboard_page(self):
        ids = _collect_ids(self.root)
        assert 'dashboard-page' in ids

    def test_contains_upload_error_modal(self):
        ids = _collect_ids(self.root)
        assert 'upload-error-modal' in ids

    def test_contains_tutorial_overlay(self):
        ids = _collect_ids(self.root)
        assert 'tutorial-overlay' in ids


# ===========================================================================
# Dashboard layout
# ===========================================================================


class TestDashboardLayout:
    @pytest.fixture(autouse=True)
    def _layout(self):
        self.root = layout.serve_layout()
        self.ids = _collect_ids(self.root)

    def test_contains_dashboard_logo(self):
        assert 'dashboard-logo' in self.ids

    def test_contains_reaction_type_dropdown(self):
        assert 'reaction-type-dropdown' in self.ids

    def test_contains_reactant_types_dropdown(self):
        assert 'reactant-types-dropdown' in self.ids

    def test_contains_fg_a_dropdown(self):
        assert 'functional-group-a-dropdown' in self.ids

    def test_contains_fg_b_dropdown(self):
        assert 'functional-group-b-dropdown' in self.ids

    def test_contains_filter_panel(self):
        assert 'filter-panel-container' in self.ids

    def test_contains_analysis_tabs(self):
        assert 'analysis-tabs' in self.ids

    def test_contains_min_eln_slider(self):
        assert 'min-eln-input' in self.ids

    def test_contains_topn_slider(self):
        assert 'topn-zscore-input' in self.ids

    def test_contains_max_components_slider(self):
        assert 'max-components-input' in self.ids

    def test_contains_exclude_cui_checkbox(self):
        assert 'exclude-cui-checkbox' in self.ids

    def test_contains_scaleup_checkbox(self):
        assert 'include-scaleup-checkbox' in self.ids

    def test_contains_null_categories_checkbox(self):
        assert 'include-null-categories-checkbox' in self.ids

    def test_contains_download_buttons(self):
        assert 'download-csv-btn' in self.ids
        assert 'download-png-btn' in self.ids

    def test_contains_upload_component(self):
        assert 'upload-data' in self.ids

    def test_contains_settings_toggle(self):
        assert 'settings-toggle' in self.ids

    def test_contains_presentation_mode_toggle(self):
        assert 'presentation-mode-toggle' in self.ids

    def test_contains_reset_button(self):
        assert 'reset-btn' in self.ids

    def test_contains_boxplot(self):
        assert 'boxplot' in self.ids

    def test_contains_heatmap(self):
        assert 'heatmap' in self.ids

    def test_contains_stats_content(self):
        assert 'stats-content' in self.ids

    def test_contains_download_components(self):
        assert 'download-csv' in self.ids
        assert 'download-png' in self.ids


# ===========================================================================
# Component IDs validation
# ===========================================================================


class TestComponentIds:
    def test_all_filter_input_ids_present(self):
        """Verify all IDs used in FILTER_INPUTS exist in the layout."""
        import callbacks as cb

        root = layout.serve_layout()
        ids = _collect_ids(root)

        for inp in cb.FILTER_INPUTS:
            assert inp.component_id in ids, f'FILTER_INPUT component {inp.component_id} not found in layout'

    def test_all_filter_state_ids_present(self):
        """Verify all IDs used in FILTER_STATES exist in the layout."""
        import callbacks as cb

        root = layout.serve_layout()
        ids = _collect_ids(root)

        for state in cb.FILTER_STATES:
            assert state.component_id in ids, f'FILTER_STATE component {state.component_id} not found in layout'

    def test_stats_badge_ids_present(self):
        root = layout.serve_layout()
        ids = _collect_ids(root)
        for prefix in ['whole-dataset', 'functional-group-a', 'functional-group-b']:
            assert f'{prefix}-stats' in ids
            assert f'{prefix}-stats-content' in ids


# ===========================================================================
# Default values
# ===========================================================================


class TestDefaultValues:
    def test_initial_fg_options_computed(self):
        # layout module precomputes FG options at import time
        assert hasattr(layout, 'INITIAL_FG_OPTIONS')
        assert isinstance(layout.INITIAL_FG_OPTIONS, list)
        assert len(layout.INITIAL_FG_OPTIONS) > 0
