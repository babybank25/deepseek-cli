"""Session persistence — save/load APIConfig to/from disk."""
import json
import logging
from typing import Optional

from .constants import CONFIG_DIR, CONFIG_FILE
from .models import APIConfig

logger = logging.getLogger(__name__)
CURRENT_CONFIG_VERSION = 2


class SessionManager:
    """Save/load API config and migrate old snapshots without losing auth."""

    @staticmethod
    def save_config(config: APIConfig) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(CONFIG_FILE)

    @staticmethod
    def load_config() -> Optional[APIConfig]:
        if not CONFIG_FILE.exists():
            return None
        try:
            raw = CONFIG_FILE.read_bytes()
            text = None
            for encoding in ("utf-8", "utf-8-sig", "cp874", "cp1252", "latin-1"):
                try:
                    text = raw.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            if text is None:
                logger.warning("Config file has unknown encoding; run --repair")
                return None
            data = json.loads(text)
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("Config file corrupted (%s); run --repair", error)
            return None
        try:
            config = APIConfig.from_dict(data)
        except (TypeError, KeyError) as error:
            logger.warning("Config schema mismatch (%s); run --repair", error)
            return None

        needs_migration = False
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            needs_migration = True
        if config.config_version < CURRENT_CONFIG_VERSION:
            config.config_version = CURRENT_CONFIG_VERSION
            needs_migration = True
        if "pow_worker_url" not in data:
            needs_migration = True
        if needs_migration:
            try:
                SessionManager.save_config(config)
            except OSError:
                logger.warning("Could not persist migrated config")
        return config

    @staticmethod
    def config_exists() -> bool:
        return CONFIG_FILE.exists()
