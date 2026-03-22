"""E2E tests for the landing page."""

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
        pg.goto(app_url)
        pg.wait_for_load_state('networkidle')
        yield pg
        browser.close()


@pytest.mark.e2e
class TestLandingPage:
    def test_page_loads(self, page):
        assert page.title() is not None

    def test_logo_visible(self, page):
        logo = page.locator('#landing-logo')
        assert logo.is_visible()

    def test_explore_button_visible(self, page):
        btn = page.locator('#explore-btn')
        assert btn.is_visible()

    def test_selecting_reaction_navigates_to_dashboard(self, page, app_url):
        # Select a reaction type and click explore
        dropdown = page.locator('#landing-reaction-dropdown')
        if dropdown.is_visible():
            dropdown.click()
            # Select first option
            page.locator('.Select-option').first.click()

        explore = page.locator('#explore-btn')
        explore.click()
        page.wait_for_url(f'{app_url}/dashboard*', timeout=10000)
        assert '/dashboard' in page.url
