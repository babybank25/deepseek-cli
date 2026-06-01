"""
Proof-of-Work solver for DeepSeek's DeepSeekHashV1 algorithm.

The WASM binary (sha3_wasm_bg.wasm) is loaded from assets/wasm_b64.txt at
runtime rather than embedded in source, keeping this file readable.

Fallback chain:
  1. WASM via wasmtime  (fastest, exact match with browser)
  2. Node.js subprocess (if wasmtime unavailable)
  3. Return None        (caller proceeds without PoW token)
"""
import base64
import json
import logging
import subprocess
from typing import Optional

from .constants import PACKAGE_DIR

logger = logging.getLogger(__name__)

# ── WASM binary loading ───────────────────────────────────────

_WASM_BYTES_CACHE: dict = {}


def _load_wasm_bytes() -> bytes:
    """Load the WASM binary from assets/wasm_b64.txt (cached per-path)."""
    wasm_b64_path = PACKAGE_DIR / "assets" / "wasm_b64.txt"
    cache_key = str(wasm_b64_path)
    cached = _WASM_BYTES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not wasm_b64_path.exists():
        raise FileNotFoundError(f"WASM base64 file not found: {wasm_b64_path}")
    decoded = base64.b64decode(wasm_b64_path.read_text(encoding="utf-8").strip())
    _WASM_BYTES_CACHE[cache_key] = decoded
    return decoded


# ── WASM solver ───────────────────────────────────────────────

class DeepSeekHash:
    """Runs the DeepSeekHashV1 PoW algorithm via the embedded WASM module.

    Usage::

        hasher = DeepSeekHash()
        hasher.init()
        answer = hasher.calculate_hash(algorithm, challenge, salt, difficulty, expire_at)
    """

    def __init__(self) -> None:
        self.instance = None
        self.memory = None
        self.store = None

    def init(self) -> "DeepSeekHash":
        """Load and instantiate the WASM module. Returns self for chaining."""
        import wasmtime  # optional dependency — ImportError handled by caller

        engine = wasmtime.Engine()
        wasm_bytes = _load_wasm_bytes()
        module = wasmtime.Module(engine, wasm_bytes)

        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()

        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]
        return self

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        """Allocate WASM memory and write a UTF-8 string into it (bulk copy)."""
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](
            self.store, length, 1
        )
        # ctypes-style memmove via slicing; falls back to per-byte if not supported
        view = self.memory.data_ptr(self.store)
        try:
            import ctypes
            ctypes.memmove(
                ctypes.addressof(view.contents) + ptr,
                encoded,
                length,
            )
        except (TypeError, AttributeError):
            for i, byte in enumerate(encoded):
                view[ptr + i] = byte
        return ptr, length

    def calculate_hash(
        self,
        algorithm: str,
        challenge: str,
        salt: str,
        difficulty: int,
        expire_at: int,
    ) -> int:
        """Solve the PoW challenge and return the answer integer (0 = failure)."""
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
                bytes(view[retptr : retptr + 4]), byteorder="little", signed=True
            )
            if status == 0:
                return 0

            value = struct.unpack("<d", bytes(view[retptr + 8 : retptr + 16]))[0]
            return int(value)
        finally:
            self.instance.exports(self.store)[
                "__wbindgen_add_to_stack_pointer"
            ](self.store, 16)


# ── Node.js fallback ──────────────────────────────────────────

def solve_pow_node(biz_data: dict) -> Optional[int]:
    """Solve PoW via Node.js + assets/pow_solver.js.

    Returns the answer integer, or None on failure.
    """
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
        logger.debug("Node.js not found — install Node.js for PoW fallback")
        return None
    except Exception as e:
        logger.debug("Node solver unexpected error: %s", e)
        return None
