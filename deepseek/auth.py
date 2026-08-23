"""Self-healing DeepSeek Web authentication lifecycle."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from .constants import PROFILE_DIR
from .metrics import metrics
from .models import APIConfig
from .protocol import probe_protocol
from .sniffer import NetworkSniffer

PersistCallback = Callable[[APIConfig], None]
_AUTH_RECOVERY_TIMEOUT = float(os.environ.get("DEEPSEEK_AUTH_RECOVERY_TIMEOUT", "90"))


class AuthManager:
    """Share validated credentials and one browser-recovery lock per account."""

    def __init__(
        self,
        config: APIConfig,
        *,
        persist: Optional[PersistCallback] = None,
        profile_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.persist = persist
        self.profile_dir = profile_dir or PROFILE_DIR
        self._lock = asyncio.Lock()
        self._validated = False

    def invalidate(self) -> None:
        self._validated = False

    async def ensure_valid(self, *, force: bool = False) -> bool:
        """Probe saved auth, then open a visible browser only if recovery is needed."""
        if self._validated and not force:
            return True
        async with self._lock:
            if self._validated and not force:
                return True
            capabilities = await probe_protocol(self.config)
            if capabilities.auth_ok and capabilities.pow_ok:
                self._validated = True
                return True
            return await self._recover_with_browser()

    async def recover(self) -> bool:
        self.invalidate()
        return await self.ensure_valid(force=True)

    async def _recover_with_browser(self) -> bool:
        metrics.incr("auth_refresh_total")
        sniffer = NetworkSniffer(headless=False, profile_dir=self.profile_dir)
        try:
            page = await sniffer.start(self.config.target_url)
            await self._hydrate_saved_state(sniffer, page)
            deadline = time.monotonic() + _AUTH_RECOVERY_TIMEOUT
            while time.monotonic() < deadline:
                candidate = await self._candidate_from_browser(sniffer, page)
                if candidate is not None:
                    capabilities = await probe_protocol(candidate)
                    if capabilities.auth_ok and capabilities.pow_ok:
                        self.config.auth_token = candidate.auth_token
                        self.config.cookies = dict(candidate.cookies)
                        if self.persist is not None:
                            self.persist(self.config)
                        self._validated = True
                        metrics.incr("auth_refresh_success_total")
                        return True
                await asyncio.sleep(1.5)
        except Exception:
            metrics.incr("auth_refresh_failed_total")
            return False
        finally:
            with contextlib.suppress(Exception):
                await sniffer.stop()
        metrics.incr("auth_refresh_failed_total")
        return False

    async def _hydrate_saved_state(self, sniffer: NetworkSniffer, page) -> None:
        context = sniffer.context
        if context is None:
            return
        origin = self.config.target_url.rstrip("/")
        cookies = [
            {
                "name": name,
                "value": value,
                "url": origin,
            }
            for name, value in self.config.cookies.items()
            if value
        ]
        if cookies:
            with contextlib.suppress(Exception):
                await context.add_cookies(cookies)
        if self.config.auth_token:
            token = self.config.auth_token
            with contextlib.suppress(Exception):
                await page.evaluate(
                    "token => localStorage.setItem('userToken', JSON.stringify({value: token}))",
                    token,
                )
        with contextlib.suppress(Exception):
            await page.reload(wait_until="domcontentloaded")

    async def _candidate_from_browser(
        self,
        sniffer: NetworkSniffer,
        page,
    ) -> Optional[APIConfig]:
        token = ""
        try:
            raw = await page.evaluate("() => localStorage.getItem('userToken') || ''")
            if isinstance(raw, str) and raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = raw
                if isinstance(parsed, dict):
                    value = parsed.get("value")
                    token = value if isinstance(value, str) else ""
                elif isinstance(parsed, str):
                    token = parsed
        except Exception:
            token = ""
        if not token:
            return None

        context = sniffer.context
        if context is None:
            return None
        try:
            browser_cookies = await context.cookies([self.config.target_url])
        except Exception:
            return None
        cookies = {
            cookie["name"]: cookie["value"]
            for cookie in browser_cookies
            if cookie.get("name") and cookie.get("value")
        }
        if not cookies:
            return None
        return replace(self.config, auth_token=token, cookies=cookies)
