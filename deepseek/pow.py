"""Proof-of-Work solvers for DeepSeekHashV1 and current web workers.

Order is controlled by ``APIClient``: embedded WASM is the fast path, the
current browser worker is compatibility fallback, and Node is the final local
fallback for the known DeepSeekHashV1 algorithm.
"""
from __future__ import annotations

import base64
import contextlib
import json
import logging
import subprocess
from typing import Optional

from .constants import PACKAGE_DIR

logger = logging.getLogger(__name__)
_WASM_BYTES_CACHE: dict[str, bytes] = {}


def _load_wasm_bytes() -> bytes:
    path = PACKAGE_DIR / "assets" / "wasm_b64.txt"
    cache_key = str(path)
    cached = _WASM_BYTES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        raise FileNotFoundError(f"WASM base64 file not found: {path}")
    decoded = base64.b64decode(path.read_text(encoding="utf-8").strip())
    _WASM_BYTES_CACHE[cache_key] = decoded
    return decoded


class DeepSeekHash:
    """Run the reverse-engineered DeepSeekHashV1 algorithm via embedded WASM."""

    def __init__(self) -> None:
        self.instance = None
        self.memory = None
        self.store = None

    def init(self) -> "DeepSeekHash":
        import wasmtime

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, _load_wasm_bytes())
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]
        return self

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](
            self.store, length, 1
        )
        view = self.memory.data_ptr(self.store)
        try:
            import ctypes

            ctypes.memmove(ctypes.addressof(view.contents) + ptr, encoded, length)
        except (TypeError, AttributeError):
            for index, byte in enumerate(encoded):
                view[ptr + index] = byte
        return ptr, length

    def calculate_hash(
        self,
        algorithm: str,
        challenge: str,
        salt: str,
        difficulty: int,
        expire_at: int,
    ) -> int:
        import struct

        prefix = f"{salt}_{expire_at}_"
        retptr = self.instance.exports(self.store)[
            "__wbindgen_add_to_stack_pointer"
        ](self.store, -16)
        try:
            challenge_ptr, challenge_len = self._write_to_memory(challenge)
            prefix_ptr, prefix_len = self._write_to_memory(prefix)
            self.instance.exports(self.store)["wasm_solve"](
                self.store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(difficulty),
            )
            view = self.memory.data_ptr(self.store)
            status = int.from_bytes(
                bytes(view[retptr : retptr + 4]),
                byteorder="little",
                signed=True,
            )
            if status == 0:
                return 0
            value = struct.unpack("<d", bytes(view[retptr + 8 : retptr + 16]))[0]
            return int(value)
        finally:
            self.instance.exports(self.store)[
                "__wbindgen_add_to_stack_pointer"
            ](self.store, 16)


def solve_pow_node(biz_data: dict) -> Optional[int]:
    """Solve known PoW via the bundled Node helper when available."""
    solver_js = PACKAGE_DIR / "assets" / "pow_solver.js"
    if not solver_js.exists():
        logger.debug("Node.js PoW solver not found: %s", solver_js)
        return None
    try:
        result = subprocess.run(
            ["node", str(solver_js), json.dumps(biz_data)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.debug("Node solver error: %s", result.stderr[:200])
            return None
        decoded = json.loads(base64.b64decode(result.stdout.strip()).decode())
        return int(decoded["answer"])
    except FileNotFoundError:
        logger.debug("Node.js not found")
        return None
    except Exception as error:
        logger.debug("Node solver unexpected error: %s", error)
        return None


async def solve_pow_browser(
    biz_data: dict,
    *,
    target_path: str,
    target_url: str,
    worker_url: str,
    timeout_ms: int = 120_000,
) -> Optional[int]:
    """Run DeepSeek's configured current web worker in an ephemeral Chromium.

    The browser is a compatibility fallback only. Normal requests stay on the
    embedded WASM fast path and do not pay Chromium startup cost.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return None

    playwright = None
    browser = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(
            target_url,
            wait_until="domcontentloaded",
            timeout=min(timeout_ms, 60_000),
        )
        answer = await page.evaluate(
            """async ({workerUrl, challenge, targetPath, timeoutMs}) => {
                const response = await fetch(workerUrl);
                if (!response.ok) throw new Error(`worker fetch failed: ${response.status}`);
                const source = await response.text();
                const objectUrl = URL.createObjectURL(
                    new Blob([source], {type: 'application/javascript'})
                );
                try {
                    return await new Promise((resolve, reject) => {
                        const worker = new Worker(objectUrl);
                        const timer = setTimeout(() => {
                            worker.terminate();
                            reject(new Error('pow worker timeout'));
                        }, timeoutMs);
                        worker.onmessage = event => {
                            clearTimeout(timer);
                            worker.terminate();
                            const data = event.data || {};
                            const value = data.answer || {};
                            if (data.type === 'pow-answer' && Number.isFinite(Number(value.answer))) {
                                resolve(Number(value.answer));
                            } else {
                                reject(new Error('unexpected pow worker response'));
                            }
                        };
                        worker.onerror = event => {
                            clearTimeout(timer);
                            worker.terminate();
                            reject(new Error(event.message || 'pow worker error'));
                        };
                        worker.postMessage({
                            type: 'pow-challenge',
                            challenge: {
                                algorithm: challenge.algorithm,
                                challenge: challenge.challenge,
                                salt: challenge.salt,
                                difficulty: challenge.difficulty,
                                signature: challenge.signature,
                                expireAt: challenge.expire_at,
                                targetPath,
                            },
                        });
                    });
                } finally {
                    URL.revokeObjectURL(objectUrl);
                }
            }""",
            {
                "workerUrl": worker_url,
                "challenge": biz_data,
                "targetPath": target_path,
                "timeoutMs": timeout_ms,
            },
        )
        return int(answer) if answer is not None else None
    except Exception as error:
        logger.debug("Browser PoW fallback failed: %s", error)
        return None
    finally:
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        if playwright is not None:
            with contextlib.suppress(Exception):
                await playwright.stop()
