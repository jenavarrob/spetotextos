import os
import re

import pytest

from test_playwright import chabela_base_url


@pytest.mark.playwright
def test_ollama_answers_capital_of_spain(chabela_base_url: str):
    """Submit a text question through the UI and verify the Ollama answer."""
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    playwright_error = playwright_api.Error
    expect = playwright_api.expect

    try:
        with sync_playwright() as playwright:
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

                provider = page.locator("#textProvider")
                if provider.count() == 0:
                    provider = page.get_by_label("Text provider")
                expect(provider).to_have_value("ollama")

                message_input = page.locator("#messageInput")
                if message_input.count() == 0:
                    # There are separate normal-chat and RAG text areas. The
                    # first textarea is the normal text-only chat.
                    message_input = page.locator("textarea").first
                message_input.fill("What is the capital of Spain?")

                send_button = page.get_by_role(
                    "button",
                    name=re.compile(r"send|ask", re.IGNORECASE),
                ).first
                with page.expect_response(
                    lambda response: (
                        "/chat/" in response.url
                        and "rag_" not in response.url
                        and response.request.method == "POST"
                    ),
                    timeout=120_000,
                ) as chat_response_info:
                    send_button.click()

                chat_response = chat_response_info.value
                if not chat_response.ok:
                    pytest.fail(
                        "Chat request failed: "
                        f"{chat_response.status} {chat_response.status_text} "
                        f"at {chat_response.url}\n"
                        f"Response body: {chat_response.text()}"
                    )

                expect(
                    page.get_by_text(re.compile(r"\bMadrid\b", re.IGNORECASE)).last
                ).to_be_visible(timeout=120_000)
            finally:
                browser.close()
    except playwright_error as exc:
        pytest.skip(f"Playwright browser is not available: {exc}")
