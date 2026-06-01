"""Tests for deepseek.constants — verify all constants are sane."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek import constants


def test_version_format():
    parts = constants.VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_config_paths_under_home():
    assert constants.CONFIG_DIR.is_absolute()
    assert str(constants.CONFIG_DIR).endswith(".deepseek_cli")
    assert constants.CONFIG_FILE.parent == constants.CONFIG_DIR


def test_package_dir_exists():
    assert constants.PACKAGE_DIR.exists()
    assert (constants.PACKAGE_DIR / "__init__.py").exists()


def test_retry_backoff_ascending():
    b = constants.RETRY_BACKOFF
    assert len(b) >= 2
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))


def test_max_retries_positive():
    assert constants.MAX_RETRIES > 0


def test_pow_retry_limit_positive():
    assert constants.POW_RETRY_LIMIT > 0


def test_request_timeout_positive():
    assert constants.REQUEST_TIMEOUT > 0


def test_history_limit_non_negative():
    assert constants.HISTORY_LIMIT >= 0


def test_api_paths_start_with_slash():
    assert constants.COMPLETION_PATH.startswith("/")
    assert constants.SESSION_PATH.startswith("/")
    assert constants.POW_CHALLENGE_PATH.startswith("/")


def test_default_target_url_is_https():
    assert constants.DEFAULT_TARGET_URL.startswith("https://")


def test_browser_user_agent_contains_chrome():
    assert "Chrome" in constants.BROWSER_USER_AGENT


def test_help_text_contains_key_commands():
    for cmd in ["/exit", "/new", "/reauth", "/help", "/export", "/status"]:
        assert cmd in constants.HELP_TEXT
