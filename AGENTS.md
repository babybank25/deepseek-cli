# Repository Guidelines

## Project Structure & Module Organization
`deepseek/` contains the main package. `__main__.py` wires the CLI, `cli.py` handles the
terminal UI, `server.py` exposes the FastAPI API, `client.py` is the core DeepSeek session
client, and `gateway/` holds multi-account routing. `tests/` mirrors that surface with
unit tests for client, server helpers, gateway, parsing, and persistence. `run.py` is the
top-level launcher. `deepseek/assets/` stores runtime files such as the PoW WASM and JS
fallback.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime and test dependencies.
- `playwright install chromium` prepares the browser used for discovery.
- `python -m deepseek --discover` captures a logged-in browser session and saves config.
- `python -m deepseek --chat` starts the Rich terminal chat.
- `python -m deepseek --serve --port 8000` runs the OpenAI-compatible API server.
- `python -m pytest tests/` runs the full test suite.
- `python -m pyflakes deepseek/ tests/` is the lightweight lint check mentioned in the README.

## Coding Style & Naming Conventions
Use standard Python style: 4-space indentation, `snake_case` for functions and modules,
`PascalCase` for classes, and explicit, descriptive names. Keep helpers small and module-
level when they are pure and testable. Follow the existing pattern of short docstrings only
where they add clarity; do not add decorative comments or unused abstractions.

## Testing Guidelines
Pytest is the test runner, with `asyncio_mode = "strict"` configured in `pyproject.toml`.
Keep tests in `tests/` and name them `test_*.py`. Prefer focused unit tests for helpers and
parser logic; add regression coverage whenever you touch session handling, SSE parsing, or
gateway routing.

## Commit & Pull Request Guidelines
This checkout does not include local git history, so use concise imperative commit subjects
such as `fix SSE fragment parsing`. In pull requests, describe the behavior change, list the
commands used to verify it, and call out any user-facing changes to CLI flags, server
endpoints, or browser discovery. Include screenshots only when the UI actually changes.

## Security & Configuration Tips
Do not commit captured sessions, account tokens, or `.deepseek_cli/` runtime data. If you
change discovery or auth handling, verify the target browser flow and keep config loading
compatible with existing saved sessions.
