"""Tests for module-level helpers extracted from serve_mode().

These functions are pure (no closure dependency on the running server)
and therefore unit-testable in isolation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.client import APIClient
from deepseek.models import APIConfig
from deepseek.server import (
    _apply_preset,
    _is_fresh_conversation,
    _messages_to_prompt,
    _model_to_preset,
    _resolve_auto_compact_override,
)


def _cfg() -> APIConfig:
    return APIConfig.from_dict({
        "target_url": "https://chat.deepseek.com",
        "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
        "method": "POST",
        "headers": {},
        "body_template": {},
    })


# ── _model_to_preset ──────────────────────────────────────────

class TestModelToPreset:
    def test_expert_model(self):
        p = _model_to_preset("deepseek-reasoner")
        assert p["model_type"] == "expert"
        assert p["thinking_enabled"] is True
        assert p["search_enabled"] is False

    def test_r1_alias(self):
        assert _model_to_preset("model-r1")["thinking_enabled"] is True

    def test_think_alias(self):
        assert _model_to_preset("think-mode")["thinking_enabled"] is True

    def test_explicit_expert(self):
        # 'expert' model name → Expert preset with R1 ON (matches the
        # official web UI).
        p = _model_to_preset("deepseek-expert")
        assert p["model_type"] == "expert"
        assert p["thinking_enabled"] is True

    def test_default_falls_back_to_fast(self):
        p = _model_to_preset("deepseek-chat")
        assert p["model_type"] == "default"
        assert p["thinking_enabled"] is False
        assert p["search_enabled"] is True

    def test_unknown_model_falls_back_to_fast(self):
        p = _model_to_preset("anything-else")
        assert p["model_type"] == "default"

    def test_empty_string_safe(self):
        p = _model_to_preset("")
        assert p["model_type"] == "default"


# ── _apply_preset ─────────────────────────────────────────────

class TestApplyPreset:
    def test_applies_known_keys(self):
        c = APIClient(_cfg())
        _apply_preset(c, {
            "model_type": "expert",
            "thinking_enabled": True,
            "search_enabled": False,
            "auto_compact_threshold": 5,
        })
        assert c.model_type == "expert"
        assert c.thinking_enabled is True
        assert c.search_enabled is False
        assert c.auto_compact_threshold == 5

    def test_partial_preset(self):
        c = APIClient(_cfg())
        c.model_type = "default"
        _apply_preset(c, {"thinking_enabled": True})
        # Only thinking changed; model_type untouched
        assert c.model_type == "default"
        assert c.thinking_enabled is True

    def test_unknown_keys_ignored(self):
        c = APIClient(_cfg())
        _apply_preset(c, {"unknown_field": "ignored", "model_type": "expert"})
        assert c.model_type == "expert"


# ── _resolve_auto_compact_override ────────────────────────────

class TestResolveAutoCompactOverride:
    def test_absent_returns_none(self):
        assert _resolve_auto_compact_override({}) is None

    def test_false_disables(self):
        assert _resolve_auto_compact_override({"auto_compact": False}) == 0

    def test_true_means_use_default(self):
        assert _resolve_auto_compact_override({"auto_compact": True}) is None

    def test_int_used_as_threshold(self):
        assert _resolve_auto_compact_override({"auto_compact": 10}) == 10

    def test_zero_int_disables(self):
        assert _resolve_auto_compact_override({"auto_compact": 0}) == 0

    def test_negative_clamped_to_zero(self):
        assert _resolve_auto_compact_override({"auto_compact": -5}) == 0

    def test_non_numeric_ignored(self):
        assert _resolve_auto_compact_override({"auto_compact": "yes"}) is None


# ── _messages_to_prompt ───────────────────────────────────────

class TestMessagesToPrompt:
    def test_empty(self):
        assert _messages_to_prompt([]) == ""

    def test_no_user_role(self):
        assert _messages_to_prompt([{"role": "system", "content": "x"}]) == ""

    def test_returns_last_user(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "last"},
        ]
        # Multi-turn: system not injected
        assert _messages_to_prompt(msgs) == "last"

    def test_first_turn_includes_system(self):
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        out = _messages_to_prompt(msgs)
        assert "[System]" in out
        assert "be helpful" in out
        assert "[User]" in out
        assert "hi" in out


# ── _is_fresh_conversation ────────────────────────────────────

class TestIsFreshConversation:
    def test_one_user_msg_is_fresh(self):
        assert _is_fresh_conversation([{"role": "user", "content": "x"}]) is True

    def test_no_users_is_fresh(self):
        assert _is_fresh_conversation([{"role": "system", "content": "x"}]) is True

    def test_two_users_is_not_fresh(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        assert _is_fresh_conversation(msgs) is False


# ── ConversationPool len/get ──────────────────────────────────

class TestConversationPoolPublicAPI:
    def test_len_zero(self):
        from deepseek.server import ConversationPool
        from deepseek.models import APIConfig
        cfg = APIConfig.from_dict({
            "target_url": "https://x", "api_endpoint": "https://x/a",
            "method": "POST", "headers": {}, "body_template": {},
        })
        p = ConversationPool(cfg)
        assert len(p) == 0
        assert p.get("missing") is None
