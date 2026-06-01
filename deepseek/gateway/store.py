"""Multi-account config storage.

Accounts are stored as:
  ~/.deepseek_cli/accounts/account_0.json
  ~/.deepseek_cli/accounts/account_1.json
  ...

The legacy ~/.deepseek_cli/config.json is treated as account_0 if no
accounts/ directory exists (backward compatibility).
"""
from __future__ import annotations
import json
from typing import Optional

from ..constants import CONFIG_DIR, CONFIG_FILE
from ..models import APIConfig


ACCOUNTS_DIR = CONFIG_DIR / "accounts"


class AccountStore:
    """Persist and load multiple account configs."""

    @staticmethod
    def list_accounts() -> list[str]:
        """Return sorted list of account names."""
        if not ACCOUNTS_DIR.exists():
            # Backward compat: if legacy config exists, treat as account_0
            if CONFIG_FILE.exists():
                return ["account_0"]
            return []
        names = sorted(
            p.stem for p in ACCOUNTS_DIR.glob("*.json")
        )
        return names

    @staticmethod
    def load(name: str) -> Optional[APIConfig]:
        """Load a single account config by name."""
        # Backward compat: account_0 may be the legacy config.json
        path = ACCOUNTS_DIR / f"{name}.json"
        if not path.exists() and name == "account_0" and CONFIG_FILE.exists():
            path = CONFIG_FILE
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            for enc in ("utf-8", "utf-8-sig", "cp874", "cp1252", "latin-1"):
                try:
                    data = json.loads(raw.decode(enc))
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            else:
                return None
            return APIConfig.from_dict(data)
        except Exception:
            return None

    @staticmethod
    def save(name: str, config: APIConfig) -> None:
        """Save an account config."""
        ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ACCOUNTS_DIR / f"{name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    @staticmethod
    def delete(name: str) -> bool:
        path = ACCOUNTS_DIR / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def next_name() -> str:
        """Return the next available account name (account_0, account_1, ...)."""
        existing = AccountStore.list_accounts()
        for i in range(1000):
            name = f"account_{i}"
            if name not in existing:
                return name
        return f"account_{len(existing)}"
