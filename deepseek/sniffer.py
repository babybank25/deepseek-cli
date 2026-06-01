"""
NetworkSniffer — Playwright-based browser traffic interceptor.

Launches a persistent Chromium browser, hooks all XHR/fetch requests,
and provides heuristics to identify the DeepSeek chat completion endpoint.
"""
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
    try:
        from playwright_stealth import stealth_async
        HAS_STEALTH = True
    except ImportError:
        HAS_STEALTH = False
except ImportError:
    async_playwright = None  # type: ignore[assignment]
    HAS_STEALTH = False

console = Console()


class NetworkSniffer:
    """Playwright-based traffic interceptor with heuristic API detection."""

    def __init__(self, headless: bool = False) -> None:
        self.headless = headless
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
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
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

        if HAS_STEALTH:
            await stealth_async(self.page)

        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)

        await self.page.goto(target_url, wait_until="domcontentloaded")
        return self.page

    async def _on_request(self, request: Request) -> None:
        """Intercept request — store metadata for XHR/fetch calls."""
        if request.resource_type not in ("xhr", "fetch"):
            return
        post_data = ""
        # Newer Playwright: post_data_buffer is bytes; older: post_data is str
        try:
            raw_bytes = getattr(request, "post_data_buffer", None)
            if raw_bytes:
                post_data = raw_bytes.decode("utf-8", errors="replace")
            else:
                pd = getattr(request, "post_data", None)
                if pd:
                    post_data = pd if isinstance(pd, str) else pd.decode("utf-8", errors="replace")
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
        """Match response to existing captured request, fill in body."""
        try:
            req = response.request
        except Exception:
            return
        for cap in reversed(self.captured):
            if cap.url == req.url and cap.response_status == 0:
                try:
                    body = await response.body()
                    cap.response_body = body.decode("utf-8", errors="replace")
                except Exception:
                    cap.response_body = "[binary or failed]"
                cap.response_status = response.status
                cap.response_headers = dict(response.headers)
                break

    async def stop(self) -> None:
        """Clean shutdown — tolerates already-closed browser/page/context."""
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
        """Heuristic: find the most likely chat/completion endpoint.

        Scores POST requests whose response contains streaming AI content.
        Supports both OpenAI-style and DeepSeek JSON-patch SSE formats.
        """
        candidates = []
        for cap in self.captured:
            if cap.method != "POST":
                continue
            body = cap.response_body.lower()
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
            if cap.response_status == 200:
                score += 3
            if "/chat" in cap.url.lower() or "/completion" in cap.url.lower():
                score += 7
            if score >= 8:
                candidates.append((score, cap))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else None

    def list_candidates(self) -> list[CapturedRequest]:
        """Return all non-trivial POST XHR/fetch requests for manual review."""
        return [c for c in self.captured if c.method == "POST" and c.response_body]
