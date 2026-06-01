"""Tests for deepseek.cli — helper functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.cli import export_conversation


# ── export_conversation ───────────────────────────────────────

class TestExportConversation:
    def test_creates_file(self, tmp_path):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        out = tmp_path / "chat.md"
        result = export_conversation(history, str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Hello" in content
        assert "Hi there!" in content
        assert "**You**" in content
        assert "**AI**" in content
        assert result == out

    def test_auto_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        path = export_conversation(history)
        assert path.exists()
        assert path.name.startswith("deepseek_chat_")

    def test_empty_history_writes_empty_export(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = export_conversation([])
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "DeepSeek Chat Export" in content

    def test_multiple_exchanges(self, tmp_path):
        history = []
        for i in range(5):
            history.append({"role": "user", "content": f"Q{i}"})
            history.append({"role": "assistant", "content": f"A{i}"})
        out = tmp_path / "multi.md"
        export_conversation(history, str(out))
        content = out.read_text(encoding="utf-8")
        for i in range(5):
            assert f"Q{i}" in content
            assert f"A{i}" in content
