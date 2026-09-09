# DeepSeek CLI Hardening Design

**Date:** 2026-09-09

## Goal

Harden the existing `deepseek-cli` implementation across four areas without rewriting working subsystems:

- A. Reliability and recovery
- B. Coding-agent and CLIProxyAPI compatibility
- C. Multi-account gateway correctness and resilience
- D. CLI, packaging, security guidance, and documentation

The implementation must reuse current modules whenever they already own the required behavior, keep execution flow easy to follow, avoid unnecessary abstraction, and preserve performance, security, and stability.

## Compatibility Decision

Compatibility level is **B**:

- Preserve the existing `/v1/*` HTTP API surface and behavior unless fixing a demonstrated bug.
- Preserve existing `~/.deepseek_cli` config and account files.
- CLI commands and wording may change when that makes the tool simpler or clearer.
- `python -m deepseek` must continue to work after packaging improvements.

## Design Principles

1. **Harden in place.** Existing modules remain the owners of their current responsibilities.
2. **Reuse before adding.** Do not create a new subsystem when the existing owner can be fixed or extended directly.
3. **Traceable flow.** Request, routing, recovery, and response paths should be readable top-to-bottom without unnecessary indirection.
4. **Root-cause fixes only.** Do not add retries, fallbacks, or abstractions without a failing test, observed defect, or clear operational need.
5. **Bounded recovery.** Every automatic recovery path has a finite retry budget.
6. **Never replay after output.** Once response content has reached a caller, the request must not be automatically replayed.
7. **Fail closed when routing is ambiguous.** Multi-account file or conversation lineage must not silently move between accounts.
8. **No unrelated refactor.** Improvements stay inside the approved A/B/C/D scope.
9. **No feature-count optimization.** Existing behavior that already works should be locked with tests rather than rewritten.
10. **Small commits.** Each change should be independently reviewable and testable.

## Current Architecture

The project already has the required foundation:

```text
DeepSeek Web
    |
    v
protocol.py / auth.py / pow.py / sse.py
    |
    v
client.py
    |
    +----------------------+----------------------+
    |                                             |
    v                                             v
gateway/*                                  single-account path
    |                                             |
    +----------------------+----------------------+
                           |
                           v
              server.py / openai_compat.py / tools.py
                           |
                           v
              OpenAI-compatible clients
                           |
                           v
               CLIProxyAPI / coding agents
```

The design keeps this structure.

## Existing Components to Reuse

### Authentication

`deepseek/auth.py::AuthManager`

Continue using it for:

- saved credential validation
- serialized browser recovery via `asyncio.Lock`
- browser state hydration
- candidate credential validation
- persistence only after validation

Do not create another authentication manager or service.

### Protocol probing

`deepseek/protocol.py`

Continue using `ProtocolCapabilities` and `probe_protocol()` for side-effect-free checks of:

- reachability
- auth validity
- PoW availability
- PoW algorithm
- known API paths

Extend only if an implementation task proves the current capability model is insufficient.

### Client resilience shell

`deepseek/client.py`

Continue using it as the narrow compatibility layer around the proven v2 client core. It owns:

- self-healing auth integration
- PoW solver ordering
- framed SSE transport normalization
- no-retry-after-output behavior

### Legacy protocol core

`deepseek/_client_legacy.py`

Keep this file and reuse it. It contains mature behavior for:

- session lifecycle
- retry logic
- semantic SSE handling
- uploads
- compaction
- DeepSeek-specific schema handling

Do not rewrite or merge it merely because of the `_legacy` name. Modify it only if a failing test demonstrates that the root cause lives there.

### Tool handling

`deepseek/tools.py` and `deepseek/_tools_legacy.py`

Keep the existing split:

- `_tools_legacy.py`: established parser and prompt-generation core
- `tools.py`: narrow resilience additions such as one-shot malformed-response recovery

Do not create a new tool protocol subsystem.

### OpenAI compatibility

`deepseek/openai_compat.py`

Continue using it for pure request/response normalization between Chat Completions and Responses API semantics. It must not become responsible for networking, routing, persistence, or retries.

### Conversation lineage

`deepseek/conversation.py`

Continue using it as the source of truth for:

- `conversation_id`
- `previous_response_id`
- history fingerprints
- resumable session lineage

Do not add a second history manager.

### Multi-account gateway

Reuse:

- `deepseek/gateway/account.py`
- `deepseek/gateway/pool.py`
- `deepseek/gateway/bindings.py`
- `deepseek/gateway/files.py`
- `deepseek/gateway/store.py`

These modules already represent the correct boundaries for account state, scheduling, conversation affinity, file affinity, and persistence.

### Metrics

`deepseek/metrics.py`

Keep the existing process-wide metrics registry. Add counters only when they answer a useful operational question.

### HTTP server and CLI

Reuse:

- `deepseek/server.py`
- `deepseek/cli.py`
- `deepseek/__main__.py`

Do not introduce a web-service layer, controller layer, CLI framework, or command framework unless the current implementation becomes demonstrably insufficient.

## Baseline Evidence

The current full test run produced:

```text
272 passed
3 failed
1 ResourceWarning
```

The three failures directly identify work in A, B, and C:

1. `tests/test_hybrid_client.py::test_missing_pow_challenge_with_stale_auth_raises_auth_error`
   - stale auth combined with a missing PoW challenge currently escapes as `RuntimeError` instead of `AuthExpiredError`
2. `tests/test_server_integration.py::test_previous_response_replays_stateless_tool_history`
   - `previous_response_id` continuation loses the `function_call_output`
3. `tests/test_file_affinity.py::test_unknown_file_id_fails_closed_with_multiple_accounts`
   - an unknown file ID can still route in multi-account mode instead of failing closed

The warning reports an unclosed event loop and must be investigated rather than hidden.

---

# A. Reliability and Recovery

## A1. Error ownership

The component closest to the failure should classify it. The layer with enough state to recover it should perform the recovery.

| Failure | Classification owner | Recovery owner | Policy |
|---|---|---|---|
| network timeout/connect error | existing client core | existing client core | use current bounded backoff |
| auth expired | client/protocol | `AuthManager` | one browser recovery attempt before output |
| PoW rejected | existing client core | existing PoW retry path | obtain fresh challenge within current retry budget |
| no PoW challenge + stale auth | `client.py` using `probe_protocol()` | `AuthManager` | raise `AuthExpiredError`, recover once |
| no PoW challenge + valid auth | `client.py` | none | explicit PoW/protocol failure |
| stale session | existing client core | existing session recreation path | current bounded behavior |
| quota/rate limit | `AccountPool` | `AccountPool` | cooldown account; stateless failover if safe |
| pinned route unavailable | `AccountPool` | none | fail closed |
| malformed tool reply | `tools.py` | existing recovery prompt | one retry |
| error after output begins | caller | none | propagate immediately |
| invalid client request | server/client validation | none | return client error, no retry |

## A2. Retry invariant

The most important invariant is:

```text
no output yet
    -> retry may be allowed when the operation is known to be safe

output already delivered
    -> never replay automatically
```

`APIClient.send_message_stream()` already tracks whether output has been yielded. This existing behavior remains the primary guard and must not be duplicated in another retry manager.

## A3. Missing PoW challenge classification

Current incorrect behavior:

```text
_fetch_pow_challenge()
    -> None
    -> RuntimeError("DeepSeek PoW challenge is unavailable")
```

Desired behavior:

```text
challenge missing
    -> probe_protocol(config)
       -> auth_ok == false
          -> AuthExpiredError
          -> existing AuthManager recovery
       -> reachable == false
          -> network/protocol failure
       -> auth_ok == true and pow_ok == false
          -> explicit PoW/protocol failure
```

Reuse `probe_protocol()`. Do not create a second probe implementation.

## A4. Browser recovery policy

The current `AuthManager` flow is retained:

```text
saved auth
    -> probe
       -> valid: reuse
       -> invalid: acquire recovery lock
           -> open visible browser
           -> hydrate saved state
           -> capture candidate token/cookies
           -> probe candidate
           -> persist valid candidate
           -> close browser resources
```

Requirements:

- concurrent failures for the same account must not launch multiple recovery browsers
- browser resources must close on success, timeout, and exception
- old credentials must not be treated as repaired if candidate validation fails
- persisted credentials must be written only after validation succeeds
- recovery failure must surface clearly rather than silently continuing with stale credentials

## A5. Retry budget audit

Before adding any retry, inspect the current `_client_legacy.py` behavior.

The intended ownership is:

```text
network failure       -> legacy client retry
PoW rejection         -> legacy PoW retry
expired auth          -> APIClient wrapper recovery once
quota                  -> AccountPool failover
malformed tool output -> tools recovery once
```

Do not stack independent server, gateway, client, and auth retries around the same failure class.

## A6. Resource cleanup

Audit and lock behavior for:

- `APIClient.close()`
- `Account.close()`
- `AccountPool.close_all()`
- browser sniffer shutdown
- stream context exit
- conversation client cleanup

Investigate the existing ResourceWarning at its source. Do not suppress it simply to make CI green.

---

# B. Coding-Agent and CLIProxyAPI Compatibility

## B1. Supported compatibility surface

The current implementation already supports:

- `/v1/chat/completions`
- `/v1/responses`
- tools
- tool results
- `previous_response_id`
- `conversation_id`
- streaming

The work in this phase hardens that surface. It does not introduce another protocol family.

## B2. Responses tool continuation

The demonstrated bug is:

```text
response 1
    -> model emits tool call
    -> caller executes tool
response 2
    -> previous_response_id
    -> function_call_output
    -> current prompt loses tool output
```

The continuation must preserve semantic history:

```text
user request
assistant tool call with call ID and arguments
tool result matching that call ID
next model turn
```

Reuse:

- `ConversationIndex`
- `responses_request_to_chat()` and related OpenAI compatibility helpers
- `messages_to_prompt()` / `compose_tools_prompt()`

Trace the loss first. Fix only the module that owns the lost state.

## B3. Tool-loop contract

The implementation must correctly handle:

- user -> normal text response
- user -> tool call
- tool call -> tool result -> model continuation
- multiple tool calls with stable call IDs
- malformed tool syntax with one recovery attempt
- normal text containing angle brackets without false tool parsing
- JSON tool result content without semantic loss
- `tool_choice=none`
- system/developer instructions
- streaming and non-streaming semantic equivalence

Do not rewrite the parser if existing tests show it already satisfies the contract.

## B4. Chat Completions and Responses parity

Equivalent logical requests through `/v1/chat/completions` and `/v1/responses` should preserve the same semantics for:

- system/developer instructions
- user messages
- assistant messages
- tool calls
- tool results
- conversation continuity
- model selection

The response schemas remain different because they are different APIs.

## B5. Streaming policy

Continue using `deepseek/sse.py` and the framed transport wrapper in `deepseek/client.py`.

Add or retain tests for:

- multi-line SSE events
- chunk boundaries splitting frames
- `[DONE]`
- reasoning/text separation
- disconnects
- partial output followed by failure

Keep current buffered tool-call streaming. Do not implement incremental synthetic tool-call streaming in this project phase because DeepSeek may split `<tool_call>` blocks across emissions and correctness is more important than fake real-time behavior.

## B6. Anthropic API scope

Do not add `/v1/messages` in this phase. CLIProxyAPI can consume the existing OpenAI-compatible upstream, so improving the current API is higher value and lower risk.

---

# C. Multi-Account Gateway

## C1. File affinity policy

File routing must follow this policy:

```text
known file ID
    -> route to recorded account

unknown file ID + one configured account
    -> allow the only possible account for backward compatibility

unknown file ID + multiple accounts
    -> fail closed because ownership is ambiguous

file IDs mapped to different accounts
    -> fail closed
```

`gateway/files.py` remains the owner of file/account affinity truth. `AccountPool` should consume that result, not reconstruct file ownership itself.

## C2. Conversation affinity

An established conversation remains pinned to its recorded account.

The following must not silently move it:

- quota exhaustion
- auth problems
- temporary account busy state
- scheduler preference

If the pinned account cannot safely serve the request, return an explicit unavailable error.

## C3. Stateless failover

Stateless requests may fail over only when:

- the request is not pinned by conversation or file affinity
- no output has been delivered
- the failure class is safe to retry, such as quota exhaustion

Once an account has emitted output, do not switch accounts for the same request.

## C4. Scheduling audit

Current score inputs include:

- busy state
- consecutive quota hits
- total errors
- last-used time
- random tie breaking

Do not redesign the scheduler without evidence.

Audit `total_errors` specifically because it is a lifetime counter. If tests demonstrate that old historical failures permanently disadvantage a recovered account, simplify scoring using already available current-state signals such as:

- availability
- busy state
- consecutive quota hits
- last-used time

Do not create a new health-score engine.

## C5. Persistence

Keep existing persisted formats:

- `~/.deepseek_cli/accounts/*.json`
- `pool_stats.json`
- conversation bindings
- file affinity state

Continue atomic temp-file-then-replace writes where already used.

No config format migration is part of this work.

## C6. Gateway metrics

Add only counters that materially help diagnose routing, for example if needed:

- stateless failovers
- ambiguous file-affinity rejects

Do not add metrics for every branch.

---

# D. CLI, Packaging, Security Guidance, and Documentation

## D1. Packaging

`pyproject.toml` currently only contains pytest configuration. Extend it to support standard installation with:

- `[build-system]`
- `[project]`
- `[project.scripts]`

Target behavior:

```text
pip install .
deepseek --version
deepseek --serve
python -m deepseek --version
```

Both the installed console command and module invocation must work.

## D2. CLI

Keep `argparse`; do not add Click, Typer, Fire, or another CLI dependency.

Existing flags such as:

- `--serve`
- `--chat`
- `--probe`
- `--discover`
- `--repair`

are still simple enough. Improve wording, help text, status output, and recovery guidance before considering a subcommand migration.

## D3. Security

Reuse the current security design:

- persistent/generated local API key
- Bearer and `x-api-key` support
- constant-time key comparison
- optional admin token
- `127.0.0.1` default bind address

If binding to a non-loopback address such as `0.0.0.0`, warn the operator that the service is being exposed beyond localhost. Do not block the configuration because LAN/container use cases may be legitimate.

Add or retain tests ensuring status/log paths do not reveal:

- DeepSeek auth tokens
- cookies
- local API key
- admin token

## D4. Dependencies

A/B/C should add **no runtime dependencies**.

Continue using the existing stack:

- httpx
- playwright
- rich
- wasmtime
- fastapi
- uvicorn
- pytest
- pytest-asyncio

Keep `requirements.txt` for existing workflows. Packaging metadata in `pyproject.toml` should remain consistent with it.

Do not add generic libraries for retries, settings, logging, or CLI parsing when the standard library and current dependencies already satisfy the need.

## D5. CI

Reuse the existing CI workflow and extend it only as needed to prove installation works.

The release-quality CI path should cover:

- install dependencies/package
- `pip check`
- `compileall`
- full pytest suite
- installed `deepseek --version` smoke test

Keep the existing Windows/Linux and Python 3.11/3.12 matrix unless a concrete packaging constraint requires change.

## D6. README alignment

Documentation must describe current behavior, not development history.

Update sections covering:

- installation
- auth recovery
- manual `--repair` fallback
- local API key
- gateway mode
- OpenAI Chat Completions support
- Responses API support
- CLIProxyAPI usage
- real limitations

Remove stale statements such as the current claim that automatic re-authentication does not exist.

Do not write README content in a historical form such as "we added", "previously", or "removed". Describe the resulting system as it exists.

---

# File Impact

## Expected production files

### Reliability

- `deepseek/client.py`
- `deepseek/auth.py` only if tests require a fix
- `deepseek/protocol.py` only if tests prove capability classification is insufficient
- `deepseek/_client_legacy.py` only if root-cause evidence points there

### Agent compatibility

- `deepseek/server.py`
- `deepseek/openai_compat.py` only for normalization defects
- `deepseek/conversation.py` only for lineage defects
- `deepseek/tools.py` only for resilience defects
- `deepseek/_tools_legacy.py` only if parser tests prove the core is wrong

### Gateway

- `deepseek/gateway/files.py`
- `deepseek/gateway/pool.py`
- `deepseek/gateway/account.py` only if routing tests require it
- `deepseek/metrics.py` only for operationally useful counters

### Distribution

- `pyproject.toml`
- `deepseek/__main__.py`
- `deepseek/cli.py` only for real UX issues
- `.github/workflows/ci.yml`
- `README.md`

## Expected tests

- `tests/test_hybrid_client.py`
- `tests/test_auth_protocol.py`
- `tests/test_client.py`
- `tests/test_pow.py`
- `tests/test_sse.py`
- `tests/test_session.py`
- `tests/test_server_integration.py`
- `tests/test_openai_compat.py`
- `tests/test_conversation_index.py`
- `tests/test_tools.py`
- `tests/test_file_affinity.py`
- `tests/test_gateway.py`
- `tests/test_gateway_affinity.py`

Do not create new test directories when an existing file already owns the behavior.

---

# Task Boundaries

## Task 1: Fix reliability baseline

Primary target: the existing stale-auth-plus-missing-PoW failure.

Expected files:

- `deepseek/client.py`
- `tests/test_hybrid_client.py`
- possibly `tests/test_auth_protocol.py`

Acceptance:

- stale auth is classified as `AuthExpiredError`
- valid-auth PoW failure remains a PoW/protocol failure
- existing solver ordering remains unchanged

## Task 2: Auth and resource recovery hardening

Audit:

- concurrent auth recovery
- browser cleanup
- client cleanup
- recovery failure
- partial-output retry protection

A successful audit may produce only tests or no production diff. Do not force code changes.

## Task 3: Fix Responses tool continuation

Trace the current loss of `function_call_output` across `previous_response_id`.

Candidate files:

- `deepseek/server.py`
- `deepseek/openai_compat.py`
- `deepseek/conversation.py`

Acceptance:

- tool result reaches the second model prompt
- call identity and tool name remain intact
- ordinary previous-response continuation still works

## Task 4: Lock agent compatibility contract

Use existing tests or add focused regression tests for:

- Chat Completions
- Responses API
- normal text
- tool calls
- multiple tool calls
- developer/system messages
- malformed tool recovery
- streaming

Do not modify production code if current behavior already passes the contract.

## Task 5: Fix gateway file affinity

Primary files:

- `deepseek/gateway/files.py`
- `deepseek/gateway/pool.py`
- `tests/test_file_affinity.py`

Acceptance:

- known file IDs stay pinned
- conflicting file ownership is rejected
- unknown file + multiple accounts fails closed
- single-account backward-compatible behavior remains possible

## Task 6: Gateway routing and failover audit

Inspect scheduling and safe failover behavior, especially lifetime `total_errors` in `_score()`.

Modify scoring only if a focused test demonstrates undesirable behavior.

Acceptance:

- stateless quota failover works before output
- pinned routes never silently move accounts
- output-producing requests never switch accounts afterward

## Task 7: Clean baseline gate

Before packaging work:

- all tests pass
- the existing ResourceWarning is root-caused and resolved or shown to be external/unavoidable with evidence
- no new warnings are hidden

## Task 8: Package installation

Primary files:

- `pyproject.toml`
- `deepseek/__main__.py`
- `.github/workflows/ci.yml`

Acceptance:

```text
pip install .
deepseek --version
python -m deepseek --version
```

all succeed.

## Task 9: CLI and documentation alignment

Primary files:

- `deepseek/cli.py` only where necessary
- `deepseek/__main__.py` only where necessary
- `README.md`

Acceptance:

- CLI guidance matches actual recovery behavior
- README no longer contains stale automatic-auth statements
- docs describe current system behavior rather than development history

## Task 10: Release verification

Run the release-quality verification set:

- package installation smoke test
- `pip check`
- `compileall`
- full pytest suite
- installed CLI smoke tests
- compatibility review for `/v1/*`
- config/account backward-compatibility review
- secret-leakage review
- final Git diff review

---

# Explicit Non-Goals

This project phase does not include:

- Python-to-Go rewrite
- Anthropic `/v1/messages`
- GUI
- daemon/service manager
- web dashboard
- custom retry framework
- generic provider architecture
- repository/service/factory layers
- rewrite of `_client_legacy.py` for style
- rewrite of `_tools_legacy.py` for style
- exact tokenizer implementation
- incremental synthetic tool-call streaming
- database adoption
- new config format
- automatic account-file migration format

---

# Test Strategy

Use the existing test stack:

- pytest
- pytest-asyncio
- FastAPI TestClient
- monkeypatch/fakes

Testing layers:

```text
unit
 -> auth / protocol / tools / affinity / OpenAI conversion

component
 -> APIClient / AccountPool / ConversationIndex

route integration
 -> FastAPI TestClient

full regression
 -> pytest tests/
```

Do not perform real logged-in browser recovery in CI. It depends on a real third-party account and credentials and would be fragile. Browser recovery logic should be covered with mocks/fakes in CI and a manual smoke test before release.

Every implementation task follows TDD when changing behavior:

1. reproduce or add the failing regression test
2. confirm it fails for the intended reason
3. make the smallest production change
4. run the focused tests
5. run relevant neighboring tests
6. commit the isolated change

---

# Acceptance Criteria

## A. Reliability

- stale auth during PoW acquisition routes into auth recovery
- automatic auth recovery occurs at most once per safe request attempt
- concurrent recovery is serialized per account
- no automatic replay after output is delivered
- browser/client/session resources are cleaned up correctly
- no retry layer duplicates an existing retry owner

## B. Agent compatibility

- Chat Completions and Responses preserve tool-loop semantics
- `previous_response_id` continuation keeps tool calls and tool results
- call IDs remain consistent
- system/developer semantics remain intact
- streaming does not duplicate output or replay failed partial responses

## C. Gateway

- conversation affinity remains stable
- file affinity remains stable
- ambiguous file ownership fails closed
- stateless requests can fail over safely before output
- pinned requests do not silently migrate
- scheduling changes, if any, are justified by focused tests

## D. Distribution

- `pip install .` works
- installed `deepseek` console command works
- `python -m deepseek` still works
- CI verifies package installation
- README matches actual current behavior

## Compatibility

- existing `/v1/*` clients remain supported
- existing config/account files remain readable
- API/admin authentication is not weakened

## Quality

- full test suite passes
- no new warnings are suppressed
- existing ResourceWarning is investigated at root cause
- no secrets appear in status, logs, committed fixtures, or docs
- request/recovery/routing flows remain readable without unnecessary abstraction

---

# Definition of Done

A request should still be understandable through a short flow:

```text
request
  -> server.py
  -> OpenAI normalization / tools
  -> APIClient or AccountPool
  -> auth / PoW / SSE as needed
  -> DeepSeek
  -> normalized response
```

Recovery should remain equally direct:

```text
error
  -> classify at the owner
  -> recover only at the layer with enough state
  -> bounded safe retry
  -> success or explicit failure
```

The hardening effort is complete when A, B, C, and D meet their acceptance criteria without introducing unnecessary service, manager, adapter, repository, factory, scheduler, or retry layers.