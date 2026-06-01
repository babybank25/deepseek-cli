"""
Session persistence — save/load APIConfig to/from disk.
"""
import json
import logging
from typing import Optional

from .constants import CONFIG_DIR, CONFIG_FILE
from .models import APIConfig

logger = logging.getLogger(__name__)


class SessionManager:
    """Save / load API config and full auth state."""

    @staticmethod
    def save_config(config: APIConfig) -> None:
        """Atomically write config to disk (write tmp → rename)."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(CONFIG_FILE)

    @staticmethod
    def load_config() -> Optional[APIConfig]:
        """Load config from disk, returning None if missing or corrupted."""
        if not CONFIG_FILE.exists():
            return None
        try:
            raw = CONFIG_FILE.read_bytes()
            # Try UTF-8 first, then fall back to common encodings
            # (old deepseek_cli.py saved with system encoding on Windows = cp874/cp1252)
            text = None
            for enc in ("utf-8", "utf-8-sig", "cp874", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            if text is None:
                logger.warning("Config file has unknown encoding; re-run --discover")
                return None
            data = json.loads(text)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Config file corrupted (%s); re-run --discover", e)
            return None
        try:
            config = APIConfig.from_dict(data)
        except (TypeError, KeyError) as e:
            logger.warning("Config schema mismatch (%s); re-run --discover", e)
            return None
        # Migrate: if file is not valid UTF-8 JSON, re-save (one-time)
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                SessionManager.save_config(config)
            except Exception:
                pass
        return config

    @staticmethod
    def config_exists() -> bool:
        return CONFIG_FILE.exists()
