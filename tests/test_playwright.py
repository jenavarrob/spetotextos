import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def chabela_base_url() -> Iterator[str]:
    """Use CHABELA_BASE_URL or start a local Ollama-only Chabela server."""
    configured_url = os.getenv("CHABELA_BASE_URL")
    if configured_url:
        yield configured_url.rstrip("/")
        return

    port = os.getenv("CHABELA_TEST_PORT", "8765")
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", port],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if requests.get(f"{base_url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.25)
        else:
            pytest.fail("Chabela test server did not start within 20 seconds")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.playwright
def test_check_ollama_button_shows_connected_status(chabela_base_url: str):
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    playwright_error = playwright_api.Error
    expect = playwright_api.expect

    try:
        with sync_playwright() as playwright:
            # Prefer an already-installed Edge/Chrome channel so the test does
            # not require downloading Playwright's bundled Chromium binary.
            browser_channel = os.getenv("PLAYWRIGHT_BROWSER_CHANNEL", "msedge")
            debug_mode = os.getenv("PWDEBUG", "0") == "1"
            headless = os.getenv("PW_HEADLESS", "0" if debug_mode else "1") != "0"
            slow_mo = int(os.getenv("PW_SLOW_MO", "250" if debug_mode else "0"))
            browser = playwright.chromium.launch(
                channel=browser_channel,
                headless=headless,
                slow_mo=slow_mo,
            )
            context = browser.new_context(
                http_credentials={
                    "username": os.getenv("APP_USERNAME", "admin"),
                    "password": os.getenv("APP_PASSWORD", "changeme"),
                }
            )
            page = context.new_page()
            page.on("console", lambda message: print(f"[browser:{message.type}] {message.text}"))
            page.on("pageerror", lambda error: print(f"[page error] {error}"))
            try:
                page.goto(chabela_base_url, wait_until="domcontentloaded")
                assert page.get_by_text("Text provider").is_visible()
                if os.getenv("PW_PAUSE", "0") == "1":
                    page.pause()
                page.get_by_role("button", name="Check Ollama").click()
                status = page.locator("#ollamaStatus")
                expect(status).to_contain_text("Ollama connected", timeout=15000)
            except AssertionError:
                screenshot_path = Path(os.getenv("PW_SCREENSHOT", "test-results/ollama-check-failure.png"))
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Playwright failure screenshot: {screenshot_path.resolve()}")
                print(f"Page URL: {page.url}")
                print(f"Ollama status: {page.locator('#ollamaStatus').inner_text()}")
                raise
            browser.close()
    except playwright_error as exc:
        pytest.skip(f"Playwright Chromium is not installed or cannot launch: {exc}")
