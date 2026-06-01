# Contributing Guide

## Development Setup

1. Install Python 3.11+.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Install browser runtime for discovery:
   - `playwright install chromium`

## Run Locally

- CLI chat: `python -m deepseek --chat`
- API server: `python -m deepseek --serve --port 8000`
- Guided launcher: `python run.py`

## Tests and Checks

- Run tests: `python -m pytest tests/`
- Lint sanity check: `python -m pyflakes deepseek/ tests/`

Please add or update tests for behavior changes, especially in:
- `deepseek/client.py` (SSE parsing, retry/session behavior)
- `deepseek/server.py` (OpenAI compatibility and request handling)
- `deepseek/gateway/` (multi-account routing and cooldown logic)

## Pull Requests

Open PRs with:
- Clear problem statement
- What changed and why
- Verification commands run and results
- Any user-visible API/CLI behavior changes

Keep PRs focused and small. Avoid unrelated refactors in the same change.

## Security

Read `SECURITY.md` before submitting. Do not include real tokens/cookies in
tests, docs, logs, screenshots, or commit history.

