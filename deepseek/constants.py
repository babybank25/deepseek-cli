"""
Global constants and configuration paths.
All tuneable values live here — change once, applies everywhere.
"""
from pathlib import Path

# ── Filesystem paths ──────────────────────────────────────────
CONFIG_DIR = Path.home() / ".deepseek_cli"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROFILE_DIR = CONFIG_DIR / "browser_profile"
ACCOUNT_PROFILE_DIR = CONFIG_DIR / "browser_accounts"

# ── Package root (for locating assets/) ──────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent

# ── Version ───────────────────────────────────────────────────
VERSION = "2.1.0"

# ── DeepSeek API paths ────────────────────────────────────────
DEFAULT_TARGET_URL = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"
SESSION_PATH = "/api/v0/chat_session/create"
POW_CHALLENGE_PATH = "/api/v0/chat/create_pow_challenge"
DEFAULT_POW_WORKER_URL = "https://fe-static.deepseek.com/chat/static/76608.8f2a9fa413.js"

# ── Network resilience ────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]
POW_RETRY_LIMIT = 2
REQUEST_TIMEOUT = 90.0

# ── Auto-compact ─────────────────────────────────────────────
AUTO_COMPACT_THRESHOLD = 20

# ── Chat UX ───────────────────────────────────────────────────
HISTORY_LIMIT = 100

# ── Browser ───────────────────────────────────────────────────
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# ── Default system prompt ─────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """\
You are an expert software engineer with deep knowledge of system design, \
algorithms, data structures, and software best practices. \
You write clean, efficient, well-documented code. \
You explain technical concepts clearly and concisely. \
When writing code, prefer readability and correctness over cleverness. \
Always consider edge cases and error handling.\
"""
HELP_TEXT = """\
[bold]Commands:[/]
  [cyan]/exit[/]              Quit
  [cyan]/new[/]               Start a new conversation (clears history)
  [cyan]/compact[/]           Summarize → start new session keeping context
  [cyan]/reauth[/]            Re-open browser to refresh session
  [cyan]/status[/]            Show session info
  [cyan]/think[/]             Toggle DeepThink R1 mode
  [cyan]/search[/]            Toggle web search
  [cyan]/model expert|fast[/] Switch model
  [cyan]/export [file][/]     Save conversation to Markdown file
  [cyan]/help[/]              Show this help
"""
