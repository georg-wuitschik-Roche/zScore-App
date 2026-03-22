"""Playwright fixtures for E2E tests."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

# App startup timeout (seconds)
_STARTUP_TIMEOUT = 30
_APP_PORT = 8051  # Use non-default port to avoid conflicts


@pytest.fixture(scope='session')
def app_url():
    """Start the Dash app and return its base URL.

    The app runs in a subprocess for the entire test session.
    """
    app_path = Path(__file__).resolve().parent.parent.parent / 'app.py'
    env = {
        'DASH_DEBUG': 'false',
        'PORT': str(_APP_PORT),
        'PATH': '/usr/bin:/usr/local/bin',
    }

    # Inherit parent env and override
    import os

    full_env = os.environ.copy()
    full_env.update(env)

    proc = subprocess.Popen(
        [sys.executable, str(app_path)],
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f'http://localhost:{_APP_PORT}'

    # Wait for server to be ready
    import urllib.request

    deadline = time.time() + _STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base_url, timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode() if proc.stdout else ''
                stderr = proc.stderr.read().decode() if proc.stderr else ''
                pytest.fail(f'App process exited early.\nstdout: {stdout}\nstderr: {stderr}')
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail(f'App did not start within {_STARTUP_TIMEOUT}s')

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
