"""E2E tests for dashboard filter controls."""

from __future__ import annotations

import pytest

pytest.importorskip('playwright')

from playwright.sync_api import sync_playwright


@pytest.fixture
def dashboard_page(app_url):
    """Navigate directly to the dashboard page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.goto(f'{app_url}/dashboard?rt=Buchwald-Hartwig')
        pg.wait_for_load_state('networkidle')
        # Wait for the boxplot to render
        pg.wait_for_selector('#boxplot', timeout=15000)
        yield pg
        browser.close()


@pytest.mark.e2e
class TestDashboardFilters:
    def test_dashboard_loads(self, dashboard_page):
        assert dashboard_page.locator('#dashboard-page').is_visible()

    def test_boxplot_visible(self, dashboard_page):
        assert dashboard_page.locator('#boxplot').is_visible()

    def test_reaction_type_dropdown_visible(self, dashboard_page):
        assert dashboard_page.locator('#reaction-type-dropdown').is_visible()

    def test_reactant_types_dropdown_visible(self, dashboard_page):
        assert dashboard_page.locator('#reactant-types-dropdown').is_visible()

    def test_fg_a_dropdown_visible(self, dashboard_page):
        assert dashboard_page.locator('#functional-group-a-dropdown').is_visible()

    def test_fg_b_dropdown_visible(self, dashboard_page):
        assert dashboard_page.locator('#functional-group-b-dropdown').is_visible()

    def test_reset_button_visible(self, dashboard_page):
        assert dashboard_page.locator('#reset-btn').is_visible()

    def test_download_buttons_in_options_panel(self, dashboard_page):
        # Expand options panel first
        toggle = dashboard_page.locator('#toggle-filters-btn')
        if toggle.is_visible():
            toggle.click()
            dashboard_page.wait_for_timeout(500)
        assert dashboard_page.locator('#download-csv-btn').is_visible()
        assert dashboard_page.locator('#download-png-btn').is_visible()

    def test_stats_tab_exists(self, dashboard_page):
        tabs = dashboard_page.locator('#analysis-tabs')
        assert tabs.is_visible()
