"""
NetworkSniffer — Playwright-based browser traffic interceptor.

Launches a persistent Chromium browser, hooks all XHR/fetch requests,
and provides heuristics to identify the DeepSeek chat completion endpoint.
"""
import importlib.util
import warnings
from pathlib import Path
from typing import Optional

from rich.console import Console

from .constants import BROWSER_USER_AGENT, PROFILE_DIR
from .models import CapturedRequest

try:
    from playwright.async_api import (
        async_playwright,
        BrowserContext,
        Page,
        Request,
        Response,
    )
except ImportError:
    async_playwright = None  # type: ignore[assignment]
    BrowserContext = Page = Request = Response = object  # type: ignore[misc,assignment]

# Detect availability without importing playwright-stealth at module import time.
# v1.0.6 imports deprecated pkg_resources, and this project intentionally treats
# warnings as errors during tests. The optional dependency is loaded only when a
# real browser starts, inside a tightly scoped warning filter below.
HAS_STEALTH = importlib.util.find_spec("playwright_stealth") is not None

console = Console()


async def _apply_stealth(page: Page) -> None:
    if not HAS_STEALTH:
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=DeprecationWarning,
            )
            from playwright_stealth import stealth_async
        await stealth_async(page)
    except ImportError:
        return


class NetworkSniffer:
    """Playwright traffic interceptor with an optional isolated profile."""

    def __init__(
        self,
        headless: bool = False,
        profile_dir: Optional[Path] = None,
    ) -> None:
        self.headless = headless
        self.profile_dir = profile_dir or PROFILE_DIR
        self.captured: list[CapturedRequest] = []
        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self, target_url: str) -> Page:
        """Launch browser, attach interceptors, navigate to target."""
        if async_playwright is None:
            raise RuntimeError(
                "playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        self.playwright = await async_playwright().start()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            user_agent=BROWSER_USER_AGENT,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-dev-shm-usage",
            ],
            ignore_default_args=["--enable-automation"],
        )
        self.page = await self.context.new_page()

        await _apply_stealth(self.page)

        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)

        await self.page.goto(target_url, wait_until="domcontentloaded")
        return self.page

    async def _on_request(self, request: Request) -> None:
        """Intercept request and store metadata for XHR/fetch calls."""
        if request.resource_type not in ("xhr", "fetch"):
            return
        post_data = ""
        try:
            raw_bytes = getattr(request, "post_data_buffer", None)
            if raw_bytes:
                post_data = raw_bytes.decode("utf-8", errors="replace")
            else:
                data = getattr(request, "post_data", None)
                if data:
                    post_data = (
                        data
                        if isinstance(data, str)
                        else data.decode("utf-8", errors="replace")
                    )
        except Exception:
            post_data = ""
        self.captured.append(
            CapturedRequest(
                url=request.url,
                method=request.method,
                request_headers=dict(request.headers),
                request_body=post_data,
                response_status=0,
                response_headers={},
                response_body="",
            )
        )

    async def _on_response(self, response: Response) -> None:
        """Match a response to the most recent unresolved captured request."""
        try:
            request = response.request
        except Exception:
            return
        for captured in reversed(self.captured):
            if captured.url == request.url and captured.response_status == 0:
                try:
                    body = await response.body()
                    captured.response_body = body.decode("utf-8", errors="replace")
                except Exception:
                    captured.response_body = "[binary or failed]"
                captured.response_status = response.status
                captured.response_headers = dict(response.headers)
                break

    async def stop(self) -> None:
        """Clean shutdown and tolerate already-closed browser objects."""
        for obj, method in [
            (self.page, "close"),
            (self.context, "close"),
            (self.playwright, "stop"),
        ]:
            if obj:
                try:
                    await getattr(obj, method)()
                except Exception:
                    pass

    def detect_chat_api(self) -> Optional[CapturedRequest]:
        """Return the most likely chat/completion request from captured traffic."""
        candidates = []
        for captured in self.captured:
            if captured.method != "POST":
                continue
            body = captured.response_body.lower()
            score = 0
            if "choices" in body:
                score += 10
            if "delta" in body or 'data: {"choices"' in body:
                score += 8
            if '"v":' in body or '"p":' in body:
                score += 9
            if "fragments" in body or "response_message_id" in body:
                score += 7
            if "message" in body or "content" in body:
                score += 5
            if captured.response_status == 200:
                score += 3
            if "/chat" in captured.url.lower() or "/completion" in captured.url.lower():
                score += 7
            if score >= 8:
                candidates.append((score, captured))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] if candidates else None

    def list_candidates(self) -> list[CapturedRequest]:
        """Return non-trivial POST XHR/fetch requests for manual review."""
        return [
            captured
            for captured in self.captured
            if captured.method == "POST" and captured.response_body
        ]
