"""E2E tests for URL state management."""

from __future__ import annotations

import pytest

pytest.importorskip('playwright')

from playwright.sync_api import sync_playwright


@pytest.fixture
def page(app_url):
    """Create a Playwright browser page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        yield pg, app_url
        browser.close()


@pytest.mark.e2e
class TestUrlState:
    def test_direct_url_with_params_loads_dashboard(self, page):
        pg, app_url = page
        pg.goto(f'{app_url}/dashboard?rt=Buchwald-Hartwig')
        pg.wait_for_load_state('networkidle')
        assert pg.locator('#dashboard-page').is_visible()

    def test_url_without_params_shows_landing(self, page):
        pg, app_url = page
        pg.goto(app_url)
        pg.wait_for_load_state('networkidle')
        assert pg.locator('#landing-page').is_visible()

    def test_root_url_shows_landing_page(self, page):
        pg, app_url = page
        pg.goto(f'{app_url}/')
        pg.wait_for_load_state('networkidle')
        assert pg.locator('#landing-page').is_visible()
