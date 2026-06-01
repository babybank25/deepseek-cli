"""Tests for deepseek.session — SessionManager."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.models import APIConfig
from deepseek.session import SessionManager


def _make_config(**overrides) -> APIConfig:
    base = {
        "target_url": "https://chat.deepseek.com",
        "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
        "method": "POST",
        "headers": {"x-custom": "value"},
        "body_template": {},
        "auth_token": "tok123",
        "cookies": {"session": "abc"},
    }
    base.update(overrides)
    return APIConfig.from_dict(base)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", tmp_path / "config.json")

    cfg = _make_config()
    SessionManager.save_config(cfg)

    assert (tmp_path / "config.json").exists()
    loaded = SessionManager.load_config()
    assert loaded is not None
    assert loaded.auth_token == "tok123"
    assert loaded.cookies == {"session": "abc"}
    assert loaded.headers == {"x-custom": "value"}


def test_atomic_write_no_tmp_left(tmp_path, monkeypatch):
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", tmp_path / "config.json")

    SessionManager.save_config(_make_config())
    assert not (tmp_path / "config.tmp").exists()


def test_load_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", tmp_path / "config.json")

    assert SessionManager.load_config() is None


def test_load_returns_none_on_corrupt_json(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    config_file.write_text("{ invalid json }", encoding="utf-8")
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", config_file)

    result = SessionManager.load_config()
    assert result is None


def test_load_returns_none_on_schema_mismatch(tmp_path, monkeypatch):
    """If schema mismatch (e.g. unknown required field type), return None gracefully."""
    config_file = tmp_path / "config.json"
    # Valid JSON but missing required fields should fail from_dict
    config_file.write_text('{"target_url": 123}', encoding="utf-8")
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", config_file)

    result = SessionManager.load_config()
    assert result is None


def test_config_exists(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", config_file)

    assert not SessionManager.config_exists()
    SessionManager.save_config(_make_config())
    assert SessionManager.config_exists()


def test_save_preserves_custom_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("deepseek.session.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("deepseek.session.CONFIG_FILE", tmp_path / "config.json")

    cfg = _make_config()
    cfg.completion_path = "/api/v1/chat/completion"
    SessionManager.save_config(cfg)

    loaded = SessionManager.load_config()
    assert loaded.completion_path == "/api/v1/chat/completion"
