"""Tests for deepseek.pow — PoW solver utilities."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.pow import DeepSeekHash, solve_pow_node


# ── solve_pow_node ────────────────────────────────────────────

class TestSolvePowNode:
    def test_returns_none_when_solver_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepseek.pow.PACKAGE_DIR", tmp_path)
        result = solve_pow_node({"challenge": "abc"})
        assert result is None

    def test_returns_none_when_node_not_found(self, tmp_path, monkeypatch):
        # Create a fake solver.js so the path check passes
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "pow_solver.js").write_text("// fake")
        monkeypatch.setattr("deepseek.pow.PACKAGE_DIR", tmp_path)

        with patch("deepseek.pow.subprocess.run", side_effect=FileNotFoundError):
            result = solve_pow_node({"challenge": "abc"})
        assert result is None

    def test_returns_none_on_nonzero_exit(self, tmp_path, monkeypatch):
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "pow_solver.js").write_text("// fake")
        monkeypatch.setattr("deepseek.pow.PACKAGE_DIR", tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("deepseek.pow.subprocess.run", return_value=mock_result):
            result = solve_pow_node({"challenge": "abc"})
        assert result is None

    def test_parses_valid_node_output(self, tmp_path, monkeypatch):
        import base64

        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "pow_solver.js").write_text("// fake")
        monkeypatch.setattr("deepseek.pow.PACKAGE_DIR", tmp_path)

        answer_json = json.dumps({"answer": 42})
        b64_output = base64.b64encode(answer_json.encode()).decode()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b64_output + "\n"
        with patch("deepseek.pow.subprocess.run", return_value=mock_result):
            result = solve_pow_node({"challenge": "abc"})
        assert result == 42


# ── DeepSeekHash ──────────────────────────────────────────────

class TestDeepSeekHash:
    def test_init_fails_gracefully_without_wasmtime(self, monkeypatch):
        """If wasmtime is not installed, init() should raise ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wasmtime":
                raise ImportError("No module named 'wasmtime'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        hasher = DeepSeekHash()
        with pytest.raises(ImportError):
            hasher.init()

    def test_init_fails_gracefully_without_wasm_file(self, tmp_path, monkeypatch):
        """If wasm_b64.txt is missing, init() should raise FileNotFoundError."""
        monkeypatch.setattr("deepseek.pow.PACKAGE_DIR", tmp_path)
        hasher = DeepSeekHash()
        with pytest.raises(FileNotFoundError):
            hasher.init()
