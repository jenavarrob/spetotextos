import os
from pathlib import Path

import pytest

from test_playwright import chabela_base_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "Business-card.jpg"


@pytest.mark.playwright
def test_tesseract_ocr_uploads_business_card(chabela_base_url: str):
    """Select the default local OCR mode and upload the repository test image."""
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    playwright_error = playwright_api.Error
    expect = playwright_api.expect

    if not IMAGE_PATH.is_file():
        pytest.fail(f"OCR test image is missing: {IMAGE_PATH}")

    try:
        with sync_playwright() as playwright:
            browser_channel = os.getenv("PLAYWRIGHT_BROWSER_CHANNEL", "msedge")
            browser = playwright.chromium.launch(
                channel=browser_channel,
                headless=os.getenv("PW_HEADLESS", "1") != "0",
            )
            context = browser.new_context(
                http_credentials={
                    "username": os.getenv("APP_USERNAME", "admin"),
                    "password": os.getenv("APP_PASSWORD", "changeme"),
                }
            )
            page = context.new_page()
            try:
                page.goto(chabela_base_url, wait_until="domcontentloaded")

                mode = page.locator("#imageAnalysisMode")
                expect(mode).to_have_value("tesseract")

                # Playwright's set_input_files() is preferable to automating
                # the native file chooser and is reliable in headless CI.
                page.locator("#imageInput").set_input_files(str(IMAGE_PATH))
                with page.expect_response(
                    lambda response: (
                        "/image-to-text/" in response.url
                        and response.request.method == "POST"
                    ),
                    timeout=120_000,
                ) as response_info:
                    page.get_by_role("button", name="📷 Upload Image").click()

                response = response_info.value
                assert response.ok, f"Tesseract upload failed: {response.status} {response.text()}"
                payload = response.json()
                assert payload["mode"] == "tesseract"
                assert isinstance(payload["reply"], str)
                reply = payload["reply"]
                for expected_text in ("Adventure", "Works", "Cycles"):
                    assert expected_text in reply, (
                        f"Tesseract response did not include {expected_text!r}: {reply!r}"
                    )
            finally:
                browser.close()
    except playwright_error as exc:
        pytest.skip(f"Playwright browser is not available: {exc}")
