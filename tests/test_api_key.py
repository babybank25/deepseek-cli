"""Tests for secure local compatibility API key provisioning."""
from __future__ import annotations

from deepseek import api_key as api_key_module
from deepseek.api_key import extract_api_key, is_api_key_authorized, load_or_create_api_key


def test_explicit_key_has_highest_priority(monkeypatch, tmp_path):
    monkeypatch.setattr(api_key_module, "API_KEY_FILE", tmp_path / "key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    info = load_or_create_api_key("cli-key")
    assert info.key == "cli-key"
    assert info.source == "cli"


def test_environment_key_used_when_no_cli_key(monkeypatch, tmp_path):
    monkeypatch.setattr(api_key_module, "API_KEY_FILE", tmp_path / "key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    info = load_or_create_api_key()
    assert info.key == "env-key"
    assert info.source == "env"


def test_generated_key_is_persistent(monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    monkeypatch.setattr(api_key_module, "API_KEY_FILE", key_file)
    monkeypatch.setattr(api_key_module, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    first = load_or_create_api_key()
    second = load_or_create_api_key()
    assert first.source == "generated"
    assert first.key.startswith("sk-ds-")
    assert second.source == "file"
    assert second.key == first.key


def test_extract_accepts_bearer_and_x_api_key():
    assert extract_api_key("Bearer abc", None) == "abc"
    assert extract_api_key(None, "xyz") == "xyz"


def test_authorization_uses_constant_time_comparison_contract():
    assert is_api_key_authorized("Bearer expected", None, "expected") is True
    assert is_api_key_authorized("Bearer wrong", None, "expected") is False
