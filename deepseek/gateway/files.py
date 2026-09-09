"""Persistent file-to-account affinity for multi-account uploads."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..constants import CONFIG_DIR

FILE_BINDINGS_FILE = CONFIG_DIR / "file_bindings.json"
FILE_BINDING_TTL_SECONDS = 7 * 24 * 60 * 60


class FileAffinityError(ValueError):
    pass


class FileAffinityStore:
    def __init__(
        self,
        path: Path = FILE_BINDINGS_FILE,
        ttl_seconds: float = FILE_BINDING_TTL_SECONDS,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, dict] = {}
        self.reload()

    def reload(self, valid_accounts: Optional[set[str]] = None) -> None:
        self._items = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        changed = False
        for file_id, value in raw.items():
            if not isinstance(file_id, str) or not isinstance(value, dict):
                changed = True
                continue
            account = value.get("account_name")
            try:
                updated_at = float(value.get("updated_at") or 0)
            except (TypeError, ValueError):
                changed = True
                continue
            if not isinstance(account, str):
                changed = True
                continue
            if valid_accounts is not None and account not in valid_accounts:
                changed = True
                continue
            if updated_at and now - updated_at > self.ttl_seconds:
                changed = True
                continue
            self._items[file_id] = {
                "account_name": account,
                "updated_at": updated_at,
            }
        if changed:
            self._save()

    def bind(self, file_id: str, account_name: str) -> None:
        self._items[file_id] = {
            "account_name": account_name,
            "updated_at": time.time(),
        }
        self._save()

    def account_for(
        self,
        file_ids: Optional[list[str]],
        *,
        require_all_known: bool = False,
    ) -> Optional[str]:
        if not file_ids:
            return None

        unknown = [file_id for file_id in file_ids if file_id not in self._items]
        if require_all_known and unknown:
            raise FileAffinityError(f"unknown file ID(s): {unknown}")

        accounts = {
            item["account_name"]
            for file_id in file_ids
            if (item := self._items.get(file_id)) is not None
        }
        if len(accounts) > 1:
            raise FileAffinityError(
                "Files in one request belong to different DeepSeek accounts"
            )
        return next(iter(accounts)) if accounts else None

    def delete_account(self, account_name: str) -> None:
        removed = [
            file_id
            for file_id, item in self._items.items()
            if item.get("account_name") == account_name
        ]
        for file_id in removed:
            self._items.pop(file_id, None)
        if removed:
            self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError:
            pass
