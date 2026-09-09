# DeepSeek Web CLI

OpenAI-compatible client and HTTP API for `chat.deepseek.com`. Built by reverse-engineering the web app, no paid API key required. Works as a Terminal chat, an OpenAI-compatible HTTP server (Claude Code / LiteLLM / LangChain / OpenAI SDK), with a multi-account gateway, prompt-emulated tool calling, rolling-summary auto-compact, and full observability.

```
                    ┌──────────── Terminal Chat ─────────────┐
                    │  /think /search /model /compact ...   │
                    └────────────┬───────────────────────────┘
                                 │
        ┌────────────────────────┴───────────────────────────┐
        │                  APIClient (deepseek/client.py)    │
        │  ── session lifecycle  PoW solver  SSE parser  ── │
        │   auto-reset on flag change   auto-compact loop   │
        └────┬──────────────────────────────────┬───────────┘
             │ single                           │ multi
             │                                  │
   ┌─────────▼────────┐               ┌─────────▼─────────┐
   │ Default client   │               │ AccountPool        │
   │ (single session) │               │ score-based pick   │
   │                  │               │ exp. cooldown      │
   │                  │               │ per-account locks  │
   └─────────┬────────┘               └─────────┬──────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │ FastAPI server    │   /v1/chat/completions
                  │ (deepseek/server) │   /v1/files
                  │ headers + metrics │   /v1/conversations[/...]
                  │ admin endpoints   │   /admin/* /metrics /health
                  └───────────────────┘
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [File-by-File Map](#file-by-file-map)
4. [Terminal Chat](#terminal-chat)
5. [API Server](#api-server)
6. [Tool Calling (Emulated)](#tool-calling-emulated)
7. [Auto-Compact (Rolling Summary)](#auto-compact-rolling-summary)
8. [Multi-Account Gateway](#multi-account-gateway)
9. [Observability — Headers, Metrics, Admin](#observability--headers-metrics-admin)
10. [How DeepSeek's Web Schema Works](#how-deepseeks-web-schema-works)
11. [Reverse-Engineering Tricks](#reverse-engineering-tricks)
12. [Development Guide](#development-guide)
13. [Testing](#testing)
14. [Common Tasks (where to edit)](#common-tasks-where-to-edit)
15. [Known Pitfalls / Schema Drift](#known-pitfalls--schema-drift)
16. [Limitations](#limitations)

---

## Quick Start

```powershell
python -m pip install .
playwright install chromium

# 1. Capture your browser session (login + send a message in the popup window)
deepseek --discover

# 2a. Terminal chat
deepseek --chat
#    or just:
python run.py                # interactive menu

# 2b. OpenAI-compatible HTTP server
deepseek --serve --port 8000
```

`python -m deepseek ...` remains equivalent when running directly from a source checkout.

Point any OpenAI-compatible client at `http://127.0.0.1:8000/v1`. `/v1/*` routes require the local DeepSeek CLI API key generated at `~/.deepseek_cli/api_key`, unless `DEEPSEEK_API_KEY` or `--api-key` overrides it.

---

## Architecture

The whole system is a thin layered API around a single state machine — `APIClient` — which talks to DeepSeek's web endpoints. Every consumer (CLI, HTTP server, gateway pool, tests) calls the same `send_message_stream()` method, so a fix made there propagates everywhere.

### Layers

```
┌──────────────────────────────────────────────┐
│  Entry points                                 │
│   run.py            interactive menu          │
│   deepseek/__main__ CLI argparse              │
│   deepseek/cli.py   Rich terminal chat        │
│   deepseek/server.py FastAPI HTTP             │
└──────────────────────┬───────────────────────┘
                       │
          ┌────────────▼────────────┐
          │ Orchestration            │
          │  AccountPool (gateway)  │
          │  ConversationPool       │
          │  Conversation           │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │ Core APIClient           │
          │  ── session create       │
          │  ── PoW solve            │
          │  ── SSE parse            │
          │  ── auto-compact loop    │
          │  ── retry / failover     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │ Persistence              │
          │  SessionManager          │
          │  AccountStore            │
          │  Pool stats.json         │
          └──────────────────────────┘
```

### Request flow (single client, non-streaming)

1. `server.chat_completions()` accepts JSON, validates, builds `prompt`.
2. `_apply_preset()` sets `model_type`, `thinking_enabled`, `search_enabled` on the chosen `APIClient`.
3. `APIClient.send_message_stream(prompt)` is called. Wrapped in `_strip_think_tags()` middleware.
4. `_stream_completion()` runs:
   * Auto-compact check (turn count vs threshold).
   * Session-flag check (resets DeepSeek session if any of the three flags changed since the last session was created).
   * `_create_session()` if no session_id (with retries).
   * `_generate_pow_token()` (WASM → Node.js → empty fallback).
   * POST → SSE → `_do_stream()` parses fragment-aware patches.
5. Tokens (`("text", str)` / `("thinking", str)`) bubble back up.
6. Server collects, optionally extracts `<tool_call>...` blocks via `tools.extract_tool_calls()`, returns OpenAI-shape JSON.

### Request flow (gateway, streaming, with tools)

1. Server determines `tools_active=True` → flatten conversation into a single prompt via `tools.compose_tools_prompt()`.
2. Server builds the preset including `auto_compact_threshold` override (if any).
3. `AccountPool.send_message_stream(prompt, model_preset=…, on_account_chosen=cb)` picks the best account using `_score()` (idle, low errors, oldest, jitter), grabs its lock, applies preset under the lock, runs the inner stream.
4. Pool retries on quota errors (exponential cooldown, persisted) by excluding the bad account and re-picking.
5. Server buffers tokens (tools mode requires the full reply to extract `<tool_call>` blocks).
6. After `[DONE]`, server parses tool calls, emits a final SSE chunk with `delta.tool_calls` + `finish_reason: tool_calls`.

---

## File-by-File Map

```
deepseek-cli/
├── README.md                      ← you are here
├── requirements.txt
├── run.py                         entry: interactive menu (chat / serve / discover / pool)
└── deepseek/
    ├── __init__.py                package metadata, exposes VERSION
    ├── __main__.py                argparse CLI (--discover --chat --serve --gateway ...)
    ├── constants.py               filesystem paths, retry knobs, default system prompt,
    │                              AUTO_COMPACT_THRESHOLD
    ├── exceptions.py              AuthExpiredError (public),
    │                              _PowExpiredError / _SessionNotFoundError (private)
    ├── models.py                  APIConfig + CapturedRequest dataclasses,
    │                              JWT-aware token_expires_in / cookie_expires_soon
    ├── session.py                 SessionManager — atomic config save/load
    ├── pow.py                     DeepSeekHashV1 solver: WASM (wasmtime) + Node.js fallback
    ├── sniffer.py                 Playwright-based traffic interceptor
    ├── discover.py                browser flow: capture session, detect API paths, save config
    ├── client.py                  ★ CORE ★  APIClient
    │                              session create + PoW + SSE parse + auto-compact +
    │                              upload_file + retry/backoff + <think> stripper
    ├── tools.py                   OpenAI tool-calling emulator
    │                              build_tools_system_prompt / messages_to_prompt /
    │                              extract_tool_calls / compose_tools_prompt
    ├── metrics.py                 thread-safe counters + Prometheus exporter
    ├── cli.py                     Rich terminal chat (commands + history)
    ├── server.py                  FastAPI app, ConversationPool, all HTTP endpoints,
    │                              module-level helpers (_model_to_preset, _apply_preset,
    │                              _resolve_auto_compact_override, _classify_upstream_error)
    ├── gateway/
    │   ├── __init__.py            re-exports
    │   ├── account.py             Account dataclass (per-account lock, exp. cooldown)
    │   ├── store.py               disk persistence: ~/.deepseek_cli/accounts/*.json
    │   └── pool.py                AccountPool: score-based picking, failover,
    │                              flush_stats, on_account_chosen callback
    └── assets/
        ├── pow_solver.js          Node.js fallback for PoW
        ├── sha3_wasm_bg.wasm      DeepSeek's hash WASM
        └── wasm_b64.txt           base64 of the wasm (loaded at runtime)

tests/
├── conftest.py
├── test_cli.py                    export_conversation
├── test_client.py                 APIClient SSE parser, R1 fragment schema,
│                                  <think> tag stripper, auto-compact, mode reset,
│                                  upload_file, set_pending_files
├── test_constants.py              constant sanity checks
├── test_discover.py               _detect_api_paths, _extract_auth
├── test_exceptions.py             exception identity
├── test_gateway.py                AccountStore, Account cooldown, AccountPool
│                                  scoring/failover/persistence/callback
├── test_metrics.py                Prometheus counters
├── test_models.py                 APIConfig + JWT helpers
├── test_pow.py                    Node fallback + WASM init guards
├── test_server_helpers.py         pure helpers (preset / fresh-conv / override / errors)
├── test_session.py                SessionManager corruption + schema mismatch
├── test_sniffer.py                detect_chat_api scoring
└── test_tools.py                  tool-call shim parsing + composition
```

---

## Terminal Chat

```bash
python -m deepseek --chat
# or
python run.py    # menu → 1
```

### Commands

| Command | Behaviour |
|---|---|
| `/exit` | quit |
| `/new` | start a new DeepSeek session (clears local history too) |
| `/compact` | summarize the current session, start a fresh one with the summary as preamble |
| `/think` | toggle DeepThink R1 (resets session — mode is bound to the session) |
| `/search` | toggle Web Search (resets session) |
| `/model expert\|fast` | preset (resets session): `expert` ⇒ R1 ON + Search OFF, `fast` ⇒ R1 OFF + Search ON |
| `/attach <path>` | upload file → file_id queued for the next message |
| `/export [filename]` | save conversation to Markdown |
| `/status` | session id, model, R1, search, msg count, turn count, token expiry |
| `/help` | command list |

### Flags

| Flag | Effect |
|---|---|
| `--auto-compact N` | override the rolling-summary threshold (0 disables) |
| `--target URL` | switch target (default `https://chat.deepseek.com`) |
| `--headless` | run discovery without a visible browser window |

### Why R1 mode resets the session

DeepSeek binds `model_type`, `thinking_enabled`, and `search_enabled` to the **session** at creation. Toggling them mid-conversation is a no-op on the server. `APIClient._stream_completion()` snapshots these three values into `_session_flags`; if they differ on the next call, it drops the session and creates a new one before sending. The CLI also clears local history to keep the user's view consistent with the upstream session.

---

## API Server

```powershell
# Single account (default)
deepseek --serve --port 8000

# Multi-account gateway (load every saved account)
deepseek --serve --gateway

# With auth + admin
deepseek --serve --api-key SECRET --admin-token ADMIN --auto-compact 30
```

Base URL: `http://127.0.0.1:8000/v1`

The server always requires a **local API key** for `/v1/*`. By default it creates and reuses `~/.deepseek_cli/api_key`. `DEEPSEEK_API_KEY` and `--api-key` override that value. This local key authenticates clients to this server; it is **not** the DeepSeek Web browser token or cookie used upstream.

### CLIProxyAPI

Start the gateway with:

```powershell
deepseek --serve --gateway
```

Configure CLIProxyAPI with an **OpenAI-compatible upstream** using:

- Base URL: `http://127.0.0.1:8000/v1`
- API key: the contents of `~/.deepseek_cli/api_key`, or the same `DEEPSEEK_API_KEY` / `--api-key` override used to start this server
- Models: `deepseek-chat`, `deepseek-fast`, `deepseek-reasoner`

DeepSeek Web credentials remain private to DeepSeek CLI and must not be used as the local CLIProxyAPI API key.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | basic info + pool status |
| `GET` | `/metrics` | Prometheus text exposition |
| `GET` | `/v1/models` | OpenAI model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (stream/non-stream, tools, file_ids, conversation_id, auto_compact override) |
| `POST` | `/v1/responses` | OpenAI Responses-compatible input/output, including `previous_response_id` tool continuation |
| `GET` | `/v1/conversations` | list active multi-turn conversations |
| `DELETE` | `/v1/conversations/{id}` | end a conversation, free its session |
| `POST` | `/v1/conversations/{id}/compact` | force rolling-summary on a conversation |
| `POST` | `/v1/files` | multipart upload → returns DeepSeek `file_id` |
| `GET` | `/admin/pool` | gateway: pool status + cooldown ETAs (X-Admin-Token) |
| `POST` | `/admin/pool/{name}/unblock` | gateway: clear cooldown on one account (persisted) |
| `POST` | `/admin/compact` | force compact across default + all gateway clients |

### Request body extensions

The chat endpoint accepts standard OpenAI fields plus a few of our own:

| Field | Type | Effect |
|---|---|---|
| `conversation_id` | str | bind this turn to a stateful conversation (kept ~30 min idle) |
| `file_ids` | list[str] | DeepSeek file ids to attach to the next message |
| `auto_compact` | int / bool | override the server's compact threshold for this request only (0 / False to disable) |
| `tools` | list | OpenAI function-calling (emulated, see below) |
| `tool_choice` | str | `"none"` disables tool emission even when `tools` is set |

### Response headers

| Header | Meaning |
|---|---|
| `X-Mode` | `single` or `gateway` |
| `X-Request-Id` | trace id used in logs |
| `X-Conversation-Id` | server conversation identifier used for resumable history |
| `X-Account-Used` | gateway: which account served the request |
| `Retry-After` | gateway: seconds until soonest cooldown ends (on 503) |

---

## Tool Calling (Emulated)

DeepSeek's web API has **no native function-calling protocol**. We emulate it via prompt injection (in `deepseek/tools.py`):

1. `build_tools_system_prompt(tools)` describes available tools and the **exact** wire format the model must use:
   ```
   <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
   ```
2. `compose_tools_prompt(messages, tools)` flattens the entire OpenAI message list (including `role: "tool"` results) into one plaintext prompt, prepending the tool instruction.
3. The server forces a fresh DeepSeek session for tools-mode (stateless — every turn re-sends the full conversation).
4. `extract_tool_calls(model_text)` finds every `<tool_call>` block, validates JSON, returns OpenAI-format `tool_calls` array. Failures (`invalid_json`, `no_name`) are logged and counted in metrics.

For the client: just use the OpenAI SDK's `tools` parameter; it works.

```python
from pathlib import Path
from openai import OpenAI

api_key = (Path.home() / ".deepseek_cli" / "api_key").read_text().strip()
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What's the weather in Bangkok?"}],
    tools=[{"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
)
print(resp.choices[0].message.tool_calls)
```

### Limitations

* Streaming for tools mode buffers the entire reply, then emits a single delta with `tool_calls`. This is unavoidable: we need the whole text to detect `<tool_call>` blocks.
* Tool mode uses a fresh upstream DeepSeek session. OpenAI conversation continuity is preserved by persisting and replaying canonical message/tool history rather than by reusing the upstream session.

---

## Auto-Compact (Rolling Summary)

When a session reaches `auto_compact_threshold` turns (default `20`, set in `constants.py`), the next call:

1. Sends `SUMMARY_PROMPT` to the **current** session, captures the reply.
2. Resets the session.
3. Stashes the summary in `_compact_summary`.
4. The first message of the fresh session prepends:
   ```
   [Previous conversation summary]
   <summary text>

   <user's actual message>
   ```

So the model keeps long-term context without keeping ever-growing token windows.

### Where this is configured

| Layer | How to set |
|---|---|
| Library default | `constants.AUTO_COMPACT_THRESHOLD = 20` |
| CLI / Server flag | `--auto-compact N` (0 disables) |
| Per-request (HTTP) | request body `"auto_compact": false \| true \| <int>` |
| Per-conversation (HTTP) | `POST /v1/conversations/{id}/compact` |
| Admin (HTTP) | `POST /admin/compact` (X-Admin-Token) |

### Internal flag and metric

* `APIClient._just_compacted` flips to `True` once after a compact fires; the server reads it to set `X-Auto-Compact-Triggered`, then clears it.
* `metrics.deepseek_auto_compacts_total` counter increments each time.

---

## Multi-Account Gateway

`AccountPool` (in `deepseek/gateway/pool.py`) routes a request to the best available account:

* **Scoring**: `(busy?, consecutive_quota_hits, last_used, random_jitter)` — lower is better. Historical `total_errors` remains observable but does not permanently penalize a recovered account. Selecting under per-account locks means parallel requests can use *different* accounts simultaneously while DeepSeek's stateful session remains single-writer.
* **Exponential cooldown**: 60s → 120s → 240s → ... capped at 1h per account. Reset to 0 on any successful request.
* **Failover**: on quota-class errors (HTTP 429, codes `40400/40401/40402`, message contains `quota`/`rate limit`/...) the account is excluded from this request and the pool retries with the next-best one — *as long as we haven't streamed any tokens to the caller yet* (no duplicate output).
* **Persistence**: cooldowns + counters land in `~/.deepseek_cli/pool_stats.json` so a restart doesn't lose track.
* **Race-safe**: presets and `file_ids` are applied **inside** the chosen account's lock via `model_preset` and `file_ids` parameters of `send_message_stream`.

### Adding accounts

```bash
python run.py            # menu → 4 → b   (opens a browser, captures, saves account_N)
```

Or programmatically:

```python
from deepseek.gateway import AccountPool, AccountStore
from deepseek.discover import discover_mode

config = await discover_mode("https://chat.deepseek.com")
await AccountPool().add_account(AccountStore.next_name(), config)
```

---

## Observability — Headers, Metrics, Admin

### Prometheus metrics (`GET /metrics`)

```
deepseek_requests_total
deepseek_streamed_requests_total
deepseek_failed_requests_total
deepseek_auth_errors_total
deepseek_pool_no_account_total
deepseek_tool_calls_total
deepseek_tool_calls_invalid_json_total
deepseek_tool_calls_no_name_total
deepseek_file_uploads_total
deepseek_file_uploads_failed_total
deepseek_auto_compacts_total
deepseek_avg_latency_ms
deepseek_uptime_seconds
```

The registry is a process-wide singleton (`metrics.metrics`). Reset on restart — no persistence by design.

### Structured logging

Every request gets `request_id = r<ms>_<6hex>`. Use `logging` (Python stdlib) — `deepseek.gateway.pool` and `deepseek.tools` already log warnings for quota events and malformed tool calls. Set the root logger level to `INFO` to see account routing decisions.

### Admin endpoints (require `--admin-token`)

```bash
# Inspect pool
curl -H "X-Admin-Token: $ADMIN" http://127.0.0.1:8000/admin/pool

# Force-clear cooldown after manual review (persisted to disk)
curl -X POST -H "X-Admin-Token: $ADMIN" \
     http://127.0.0.1:8000/admin/pool/account_0/unblock

# Force compact on every client (default + all accounts)
curl -X POST -H "X-Admin-Token: $ADMIN" \
     http://127.0.0.1:8000/admin/compact
```

---

## How DeepSeek's Web Schema Works

The biggest source of bugs is DeepSeek changing its SSE JSON-patch format. The current schema (May 2026) the parser handles is:

1. Initial **snapshot** announces the response object and the *first* fragment:
   ```json
   {"v":{"response":{"message_id":2,"fragments":[
     {"id":2,"type":"THINK","content":"We"}
   ]}}}
   ```
2. The **first** content patch carries an explicit path:
   ```json
   {"p":"response/fragments/-1/content","o":"APPEND","v":" need"}
   ```
3. **Subsequent patches drop the `p` field entirely** — they implicitly target the previous path:
   ```json
   {"v":" to"}
   {"v":" respond"}
   ...
   ```
4. A **new fragment** is announced with a list APPEND:
   ```json
   {"p":"response/fragments","o":"APPEND",
    "v":[{"id":3,"type":"RESPONSE","content":"Hi"}]}
   ```
5. Then the cycle repeats: one explicit-path patch, many bare `{"v": ...}` rows.

### What the parser tracks

| Variable | Purpose |
|---|---|
| `last_path` | reused when a patch has no `p` field |
| `active_fragment_type` | "THINK" / "RESPONSE" / etc. — routes content patches to thinking vs text channel even when the path itself is generic |
| `streamed_text` | for snapshot reconciliation (REPLACE ops) |
| `new_message_id` | becomes `last_message_id` for the next turn |

### Defense layers against R1 chain-of-thought leak

1. **Path-based**: `_classify_path_for_thinking()` matches `think`, `reasoning`, `chain_of_thought`, `cot`.
2. **Fragment-type-based**: when the path is just `response/fragments/-1/content` (no `think` keyword), the parser checks the active fragment's declared type.
3. **`<think>` tag stripper**: `_strip_think_tags()` is applied as a stateful middleware around all text tokens. Catches reasoning that leaks as raw `<think>...</think>` text, even when the tag is split across SSE chunks.

Set `DEEPSEEK_DUMP_SSE=1` to mirror raw lines to `~/.deepseek_cli/sse_dump.log`. Use this if the parser stops working — DeepSeek likely changed schema again.

---

## Reverse-Engineering Tricks

### 1. Browser-driven session capture

`sniffer.NetworkSniffer` launches Chromium with `playwright`, hooks every XHR/fetch, lets the user log in normally, then heuristically picks the chat completion endpoint from captured traffic. Auth token, cookies, and full request headers are dumped to `config.json`. `playwright_stealth` (optional) reduces automation fingerprints.

### 2. Proof-of-Work

DeepSeek's `x-ds-pow-response` header is mandatory. The algorithm is `DeepSeekHashV1` (a Keccak/SHA-3 variant) implemented as a WASM module shipped with the web client.

* **Layer 1 — wasmtime**: load `assets/wasm_b64.txt`, decode, instantiate, call `wasm_solve`. Sub-millisecond.
* **Layer 2 — Node.js**: spawn `node assets/pow_solver.js <challenge_json>`. Used when wasmtime is unavailable.
* **Layer 3 — empty token**: send the request without the header. Often works for low-difficulty endpoints.

A failed PoW response (codes `40300`/`40301`) triggers a re-solve and one retry. After that the client raises `AuthExpiredError` — usually means the auth token is stale, not the PoW.

### 3. Two-step session creation

Every chat is rooted in a `chat_session_id` from `POST /api/v0/chat_session/create`. Subsequent messages reference `parent_message_id` to thread correctly. We recreate the session whenever:

* User runs `/new` or `/compact`.
* `model_type` / `thinking_enabled` / `search_enabled` change (these are fixed at session creation server-side).
* The server returns codes `40200`/`40201` (session not found).

### 4. JSON-patch SSE → text stream

DeepSeek does not stream OpenAI-shaped chunks. It streams a sequence of JSON patches against an evolving `response` object. The parser maintains a tiny state machine — see "How DeepSeek's Web Schema Works" above.

### 5. Resilience features

| What | Where | Notes |
|---|---|---|
| Network retries | `client._stream_completion`, `_create_session` | exponential backoff with jitter via `_backoff_seconds()` |
| 5xx → NetworkError | `_do_stream` | mapped so the retry loop catches it |
| String error codes | `_raise_for_code` | accepts `"40300"` as well as `40300` |
| Buffer cap | `_do_stream` | 1 MiB JSON buffer to avoid OOM on malformed SSE |
| SSE comment skip | `_do_stream` | ignores `: keepalive` |
| Pool stats persist | `pool._persist_stats` | survives restart |
| Fresh-session detection | `_session_flags` snapshot | avoids stale-mode bugs |
| Tool failure logging | `tools.extract_tool_calls` | counts `invalid_json` / `no_name` |
| File-id eager consume | `client._stream_completion` | 4xx bounce can't leak file_ids into the next request |

---

## Development Guide

### Layout rules

* **Pure helpers** live at module scope (`server._model_to_preset`, `server._apply_preset`, `tools.extract_tool_calls`). They have no closure dependencies and are unit-testable in isolation.
* **Stateful types** (`APIClient`, `Account`, `AccountPool`, `ConversationPool`) hide internal mutables behind public methods (`set_pending_files`, `flush_stats`, `accounts` property, `__len__`, `get(id)`).
* **Public exceptions** (`AuthExpiredError`) propagate to callers; `_PowExpiredError` and `_SessionNotFoundError` are internal signals consumed by the retry loop and never escape `APIClient`.

### Adding a new HTTP endpoint

1. Open `deepseek/server.py`, find `# ── Routes ──`.
2. Define a coroutine decorated with `@app.<method>(...)`.
3. Call `_check_auth(authorization)` (or `_check_admin` for admin routes).
4. Use the public APIs of `pool`, `gateway_pool`, `_default_client` — never reach into `_accounts` / `_conversations` from here.
5. Add a unit test in `tests/test_server_helpers.py` for any new pure helpers.
6. Add an end-to-end smoke test by writing a temporary `smoke_*.py` (delete after validating).

### Adding a new SSE patch path

1. Run with `DEEPSEEK_DUMP_SSE=1` and reproduce the case.
2. Inspect `~/.deepseek_cli/sse_dump.log`.
3. Extend `_classify_path_for_thinking()` (for reasoning paths) or the format-detection block in `_do_stream()` (for new content shapes).
4. Add a regression test in `tests/test_client.py` (see `TestR1FragmentSchema` for an example using mocked SSE lines).

### Adding a new field to APIConfig

1. Edit the dataclass in `deepseek/models.py`.
2. Set a sensible default — `from_dict()` already ignores unknown keys, so old configs stay loadable.
3. Update `tests/test_models.py` with a roundtrip test.
4. Discovery (`discover.py`) populates the field by parsing captured browser traffic.

### Adding a new metric

1. Add a field to the `_State` dataclass in `deepseek/metrics.py`.
2. Add a reader in `Metrics.snapshot()`.
3. Add HELP/TYPE lines in `Metrics.to_prometheus()`.
4. Increment from the right call sites — usually `server.py`.
5. Test in `tests/test_metrics.py`.

### Where flags propagate

When you change defaults or add a knob, make sure every consumer is consistent:

| Surface | File | Symbol |
|---|---|---|
| Library default | `constants.py` | `AUTO_COMPACT_THRESHOLD` etc. |
| `APIClient` init | `client.py` `__init__` | reads from `constants` |
| CLI flag | `__main__.py` | `argparse.add_argument` |
| CLI consumption | `cli.py` `chat_mode` / `server.py` `serve_mode` | argument |
| Server preset map | `server._model_to_preset` | maps OpenAI model name → preset dict |
| Pool override | `gateway/pool.py` `send_message_stream` `model_preset=` | applied under lock |

---

## Testing

```powershell
# Full suite (~25 s)
python -m pytest tests/

# Lint
python -m pyflakes deepseek/ tests/
```

228 tests covering the parser, retry loops, session behaviour, gateway scoring, tool emulation, metrics, JWT helpers, server helpers, and persistence corners.

### Writing parser tests

Use `_collect_tokens(client, lines)` from `tests/test_client.py` — it mocks `httpx.AsyncClient.stream` with a list of SSE lines and collects every `(type, text)` tuple. See `TestDoStream`, `TestR1FragmentSchema` for examples covering all five SSE shapes.

### Writing gateway tests

`pool._replace_accounts_for_test([Account, ...])` is the white-box hook for installing a custom account list without going through disk. See `TestAccountPoolPresetIsolation` for an example that verifies preset application is race-safe.

---

## Common Tasks (where to edit)

| Task | File(s) | Notes |
|---|---|---|
| Add a chat command | `cli.py` `COMMANDS` + `chat_mode` body | also `constants.HELP_TEXT` |
| Add a model alias | `server._model_to_preset` | substring match on `model` string |
| Change cooldown ladder | `gateway/account.py` `COOLDOWN_BASE_SECONDS` / `COOLDOWN_MAX_SECONDS` | applies pool-wide |
| Change request timeout | `constants.REQUEST_TIMEOUT` | per-request lock acquire ceiling |
| Change auto-compact default | `constants.AUTO_COMPACT_THRESHOLD` | propagates to all clients |
| Change SSE buffer cap | `client._do_stream` `MAX_JSON_BUF` | currently 1 MiB |
| Add a new SSE fragment type | `client._do_stream` fragment-list APPEND branch + tests | handle the `type` field |
| Add an upstream error class | `_classify_upstream_error` in `server.py` + test | maps to HTTP status |
| Tweak tool-call output format | `tools.build_tools_system_prompt` + `extract_tool_calls` regex | regex is `_TOOL_CALL_RE` |
| Reset all account stats | delete `~/.deepseek_cli/pool_stats.json` | next start re-loads zeros |

---

## Known Pitfalls / Schema Drift

These are the places that have broken in the past — patch carefully.

1. **Patches without `p` field are NOT new content streams** — they implicitly target the previous path. Track `last_path` in `_do_stream`. Removing this code re-introduces the R1 chain-of-thought leak instantly.
2. **Fragment type `THINK` vs `RESPONSE` is in fragment metadata, not the path.** A path of `response/fragments/-1/content` could be either. Use `active_fragment_type` to disambiguate.
3. **Mode flags are session-bound.** Toggling `model_type` / `thinking_enabled` / `search_enabled` after session creation is silently ignored by DeepSeek. Always reset.
4. **`_pending_file_ids` is consumed eagerly on send**, not on success. Otherwise a 4xx upstream bounces leaves them dangling for the next, unrelated request.
5. **`asyncio.Task` from `_cleanup_expired` must be tracked** — pure fire-and-forget gets garbage-collected mid-flight. We hold them in `_cleanup_tasks` set with `add_done_callback(discard)`.
6. **Per-account lock release order** matters. `pool._acquire_account` re-checks availability after acquiring; if the account just became exhausted, we release-and-skip rather than holding.
7. **Streamed responses cannot include `X-Auto-Compact-Triggered`** — the value is unknown when headers are sent. We expose `X-Auto-Compact-Available: 1` instead and rely on the next non-stream request or `metrics.deepseek_auto_compacts_total`.
8. **Tool-call streaming buffers the full reply.** Don't yield SSE chunks while in tools mode until the upstream is done; otherwise `<tool_call>` blocks split across emissions are unrecoverable.
9. **`r.raise_for_status()` was removed from `_create_session`** — it raised `httpx.HTTPStatusError` on 5xx, which the retry loop didn't catch. We classify status codes manually now.
10. **Setting `WriteTimeout` on uvicorn breaks SSE.** uvicorn defaults are correct; don't add timeout middleware to streaming routes.

---

## Limitations

* **Heuristic token estimates** — the server reports `usage.prompt_tokens = len(prompt) // 4`; this is a crude approximation, not a real tokenizer.
* **Synthetic streaming for tool calls** — the OpenAI standard streams `delta.tool_calls` incrementally; we buffer and emit one consolidated chunk.
* **No Anthropic `/v1/messages` endpoint** — only OpenAI-compatible. Use Claude Code in OpenAI-mode (`OPENAI_BASE_URL` env var).
* **Metrics reset on restart** — by design. The state file `pool_stats.json` exists for cooldown info, not metrics.
* **Foreground server, no daemon mode** — the server runs in the foreground. Use `nssm` / `systemd` if you want it as a service.
* **Interactive recovery can still be required** — saved auth is probed automatically. When it is stale, `AuthManager` attempts serialized browser recovery and persists only validated credentials. Use `--repair` when automatic recovery cannot restore the session.

---

## License

MIT License. See [LICENSE](LICENSE).

This project interacts with third-party services. Users are responsible for
complying with the relevant Terms of Service and local laws.

---

## Open Source Release Safety Checklist

Before publishing code or opening a pull request:

1. Do not commit secrets or runtime auth state:
   - `~/.deepseek_cli/config.json`
   - `~/.deepseek_cli/accounts/*.json`
   - `~/.deepseek_cli/sse_dump.log`
2. Run verification commands:
   - `python -m pytest tests/`
   - `python -m pyflakes deepseek/ tests/`
3. Review diffs for leaked tokens/cookies:
   - `Authorization: Bearer ...`
   - `Cookie: ...`
   - Raw captured request headers
4. Follow project contribution and disclosure policies:
   - `CONTRIBUTING.md`
   - `SECURITY.md`
   - `CODE_OF_CONDUCT.md`
