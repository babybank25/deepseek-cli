# DeepSeek CLI Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing DeepSeek CLI for reliable long-running use with CLIProxyAPI/coding agents, safer multi-account routing, installable CLI packaging, and documentation that matches current behavior.

**Architecture:** Keep the current module ownership and harden in place. `server.py` remains request orchestration, `client.py` remains the resilience shell over `_client_legacy.py`, `AuthManager` remains the only auth-recovery owner, and `AccountPool`/affinity stores remain the gateway owners. No new service/repository/factory/retry layers are introduced.

**Tech Stack:** Python 3.11/3.12, asyncio, httpx 0.28.1, Playwright 1.50.0, Rich 14.0.0, Wasmtime 28.0.0, FastAPI 0.128.8, Uvicorn 0.40.0, pytest 8.3.5, pytest-asyncio 0.25.3.

**Spec:** `docs/superpowers/specs/2026-09-09-deepseek-cli-hardening-design.md`

## Global Constraints

- Preserve the existing `/v1/*` API surface unless fixing a demonstrated bug.
- Preserve existing `~/.deepseek_cli` config/account formats.
- `python -m deepseek` must continue to work.
- CLI wording/flags may change only when it makes the tool simpler or clearer.
- Reuse current modules before adding new code or files.
- Do not rewrite `_client_legacy.py` or `_tools_legacy.py` for style.
- Do not add runtime dependencies for A/B/C hardening.
- Do not add Anthropic `/v1/messages`, GUI, daemon/service manager, dashboard, database, generic provider architecture, or custom retry framework.
- Automatic retry is allowed only before output has reached the caller.
- Ambiguous conversation/file routing must fail closed.
- Every behavior change follows TDD and ends with focused tests plus a small explicit-path commit.
- Do not stage, restore, delete, or modify unrelated files.

## Current Baseline

Plan written against `main` after:

- `5e14692 fix(gateway): resolve unknown file affinity, stale pow auth, and tool replay`
- `4142510 fix(ci): filter upstream anyio deprecation warning and update gitignore`
- `8331d6f docs: add deepseek cli hardening design`

Current verification:

```text
python -m pytest -q
275 passed, 1 ResourceWarning
```

The original three failing regressions are already fixed. Do not re-implement those fixes. The remaining plan extends coverage and closes hardening gaps around those paths.

---

## File Structure / Ownership Lock

| File | Responsibility in this plan |
|---|---|
| `deepseek/client.py` | Auth/PoW failure classification around the proven client core |
| `deepseek/auth.py` | Single-account auth validation and serialized browser recovery |
| `deepseek/protocol.py` | Side-effect-free protocol/auth/PoW capability probe, reused as-is unless tests prove otherwise |
| `deepseek/_client_legacy.py` | Proven session/retry/SSE/upload/compaction core; no planned edit |
| `deepseek/server.py` | HTTP orchestration, Responses/Chat continuity, stream presentation, non-loopback bind warning |
| `deepseek/openai_compat.py` | Pure Responses ↔ Chat normalization; test first, no planned edit unless normalization fails |
| `deepseek/tools.py` / `_tools_legacy.py` | Existing tool prompt/parser/recovery behavior; contract-test first |
| `deepseek/conversation.py` | Persisted `conversation_id` / `previous_response_id` lineage; test first |
| `deepseek/gateway/files.py` | File/account ownership truth, including unknown-file validation |
| `deepseek/gateway/pool.py` | Routing, locks, affinity use, cooldown/failover, scheduling score |
| `deepseek/gateway/account.py` | Existing account state; no planned edit |
| `deepseek/metrics.py` | Existing metrics; no planned edit unless a concrete routing counter is needed |
| `deepseek/__main__.py` | CLI parser plus installable synchronous console entry point |
| `pyproject.toml` | Package metadata, dependencies, script entry point, package assets, pytest config |
| `.github/workflows/ci.yml` | Install/package smoke plus existing compile/test matrix |
| `README.md` | Current-state installation, auth recovery, API, gateway, CLIProxyAPI, limitations |

---

### Task 1: Finish PoW/Auth Failure Classification

**Files:**
- Modify: `deepseek/client.py:150-178`
- Test: `tests/test_hybrid_client.py:130-184`

**Interfaces:**
- Consumes: `probe_protocol(config: APIConfig) -> ProtocolCapabilities`
- Produces: no new public interface; `_generate_pow_token_for_path()` keeps returning a token or raising the existing public/internal error types.

- [ ] **Step 1: Add regression tests for unreachable and valid-auth missing-challenge cases**

Append to `tests/test_hybrid_client.py` before `_async_value`:

```python
@pytest.mark.asyncio
async def test_missing_pow_challenge_when_protocol_unreachable_is_not_auth_error(monkeypatch):
    client = APIClient(_config())

    async def no_challenge(target_path=None):
        return None

    client._fetch_pow_challenge = no_challenge  # type: ignore[assignment]
    unreachable = ProtocolCapabilities(
        reachable=False,
        auth_ok=False,
        pow_ok=False,
        pow_algorithm=None,
        completion_path="/api/v0/chat/completion",
        session_path="/api/v0/chat_session/create",
        pow_challenge_path="/api/v0/chat/create_pow_challenge",
        status_code=None,
        error="connect timeout",
    )
    monkeypatch.setattr("deepseek.protocol.probe_protocol", _async_value(unreachable))

    with pytest.raises(RuntimeError, match="protocol probe failed"):
        await client._generate_pow_token_for_path("/api/v0/chat/completion")


@pytest.mark.asyncio
async def test_missing_pow_challenge_with_valid_auth_stays_pow_error(monkeypatch):
    client = APIClient(_config())

    async def no_challenge(target_path=None):
        return None

    client._fetch_pow_challenge = no_challenge  # type: ignore[assignment]
    valid_auth = ProtocolCapabilities(
        reachable=True,
        auth_ok=True,
        pow_ok=False,
        pow_algorithm=None,
        completion_path="/api/v0/chat/completion",
        session_path="/api/v0/chat_session/create",
        pow_challenge_path="/api/v0/chat/create_pow_challenge",
        status_code=200,
        error="challenge unavailable",
    )
    monkeypatch.setattr("deepseek.protocol.probe_protocol", _async_value(valid_auth))

    with pytest.raises(RuntimeError, match="PoW challenge is unavailable"):
        await client._generate_pow_token_for_path("/api/v0/chat/completion")
```

- [ ] **Step 2: Run focused tests and verify the unreachable case fails for the intended reason**

Run:

```powershell
python -m pytest tests/test_hybrid_client.py::test_missing_pow_challenge_with_stale_auth_raises_auth_error tests/test_hybrid_client.py::test_missing_pow_challenge_when_protocol_unreachable_is_not_auth_error tests/test_hybrid_client.py::test_missing_pow_challenge_with_valid_auth_stays_pow_error -q
```

Expected before implementation:

```text
stale-auth test: PASS
unreachable test: FAIL because current code treats reachable=False/auth_ok=False as AuthExpiredError
valid-auth test: PASS
```

- [ ] **Step 3: Make the smallest classification change in `client.py`**

Replace the current missing-challenge probe block with:

```python
        challenge = await self._fetch_pow_challenge(target_path=target_path)
        if not challenge:
            metrics.incr("pow_failed_total")
            capabilities = None
            try:
                from .protocol import probe_protocol

                capabilities = await probe_protocol(self.config)
            except Exception:
                pass

            if capabilities is not None:
                if capabilities.reachable and (
                    not capabilities.auth_ok or capabilities.status_code in (401, 403)
                ):
                    raise AuthExpiredError(
                        "DeepSeek authentication expired or invalid: "
                        f"{capabilities.error or 'auth failed'}"
                    )
                if not capabilities.reachable:
                    raise RuntimeError(
                        "DeepSeek protocol probe failed: "
                        f"{capabilities.error or 'upstream unreachable'}"
                    )

            raise RuntimeError("DeepSeek PoW challenge is unavailable")
```

Do not change solver ordering or `_client_legacy.py`.

- [ ] **Step 4: Run the focused reliability suite**

Run:

```powershell
python -m pytest tests/test_hybrid_client.py tests/test_auth_protocol.py tests/test_pow.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only the reliability change**

```powershell
git add deepseek/client.py tests/test_hybrid_client.py
git commit -m "fix(auth): distinguish unreachable pow probe from expired auth"
```

---

### Task 2: Deduplicate Concurrent Auth Recovery

**Files:**
- Modify: `deepseek/auth.py:23-58`
- Test: `tests/test_auth_protocol.py:1-106`

**Interfaces:**
- Consumes: existing `AuthManager.ensure_valid(force: bool = False) -> bool`
- Produces: same public signatures; concurrent `recover()` calls for one account share the first successful recovery instead of opening a second browser.

- [ ] **Step 1: Add true concurrent recovery and force-revalidation tests**

Add `import asyncio` to `tests/test_auth_protocol.py`, then add:

```python
@pytest.mark.asyncio
async def test_concurrent_recoveries_share_one_browser_refresh(monkeypatch):
    manager = AuthManager(_config())
    bad = ProtocolCapabilities(
        reachable=True,
        auth_ok=False,
        pow_ok=False,
        pow_algorithm=None,
        completion_path="/c",
        session_path="/s",
        pow_challenge_path="/p",
        status_code=401,
    )
    monkeypatch.setattr("deepseek.auth.probe_protocol", AsyncMock(return_value=bad))

    started = asyncio.Event()
    release = asyncio.Event()
    recovery_calls = 0

    async def recover_once():
        nonlocal recovery_calls
        recovery_calls += 1
        started.set()
        await release.wait()
        manager._validated = True
        return True

    monkeypatch.setattr(manager, "_recover_with_browser", recover_once)

    first = asyncio.create_task(manager.recover())
    await started.wait()
    second = asyncio.create_task(manager.recover())
    release.set()

    assert await asyncio.gather(first, second) == [True, True]
    assert recovery_calls == 1


@pytest.mark.asyncio
async def test_force_revalidates_when_auth_was_already_valid(monkeypatch):
    manager = AuthManager(_config())
    manager._validated = True
    good = ProtocolCapabilities(
        reachable=True,
        auth_ok=True,
        pow_ok=True,
        pow_algorithm="DeepSeekHashV1",
        completion_path="/c",
        session_path="/s",
        pow_challenge_path="/p",
        status_code=200,
    )
    probe = AsyncMock(return_value=good)
    monkeypatch.setattr("deepseek.auth.probe_protocol", probe)

    assert await manager.ensure_valid(force=True) is True
    probe.assert_awaited_once()
```

- [ ] **Step 2: Run both tests and verify only the concurrency test fails**

```powershell
python -m pytest tests/test_auth_protocol.py::test_concurrent_recoveries_share_one_browser_refresh tests/test_auth_protocol.py::test_force_revalidates_when_auth_was_already_valid -q
```

Expected before implementation: concurrent test fails with `recovery_calls == 2`; force-revalidation test passes.

- [ ] **Step 3: Preserve force semantics while allowing a waiter to reuse another caller's success**

Update `AuthManager.ensure_valid()`:

```python
    async def ensure_valid(self, *, force: bool = False) -> bool:
        """Probe saved auth, then open a visible browser only if recovery is needed."""
        validated_before_wait = self._validated
        if validated_before_wait and not force:
            return True

        async with self._lock:
            if self._validated and (not force or not validated_before_wait):
                return True

            capabilities = await probe_protocol(self.config)
            if capabilities.auth_ok and capabilities.pow_ok:
                self._validated = True
                return True
            return await self._recover_with_browser()
```

Do not add a second lock, recovery queue, generation class, or manager.

- [ ] **Step 4: Add browser-cleanup contract coverage without changing production code**

Add a test that exercises the existing `finally: await sniffer.stop()` path using a fake sniffer:

```python
@pytest.mark.asyncio
async def test_browser_recovery_stops_sniffer_after_start_failure(monkeypatch):
    manager = AuthManager(_config())
    stopped = 0

    class FakeSniffer:
        context = None

        def __init__(self, *args, **kwargs):
            pass

        async def start(self, _url):
            raise RuntimeError("browser failed")

        async def stop(self):
            nonlocal stopped
            stopped += 1

    monkeypatch.setattr("deepseek.auth.NetworkSniffer", FakeSniffer)

    assert await manager._recover_with_browser() is False
    assert stopped == 1
```

- [ ] **Step 5: Run the complete auth/recovery neighborhood**

```powershell
python -m pytest tests/test_auth_protocol.py tests/test_hybrid_client.py tests/test_discover.py tests/test_sniffer.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add deepseek/auth.py tests/test_auth_protocol.py
git commit -m "fix(auth): coalesce concurrent browser recovery"
```

---

### Task 3: Lock Responses Tool-Continuation Semantics

**Files:**
- Test: `tests/test_openai_compat.py:1-84`
- Test: `tests/test_server_integration.py:70-168`
- No production edit expected if current `5e14692` behavior holds.

**Interfaces:**
- Consumes: `responses_request_to_chat(payload: dict) -> dict`, `ConversationIndex`, `messages_to_prompt()`
- Produces: regression coverage proving `previous_response_id`, tool-call IDs, developer instructions, and multiple tool results survive normalization and replay.

- [ ] **Step 1: Add normalization tests for developer instructions and tool results**

Append to `tests/test_openai_compat.py`:

```python
def test_responses_developer_message_normalizes_to_system():
    chat = responses_request_to_chat(
        {
            "input": [
                {"type": "message", "role": "developer", "content": "follow project rules"},
                {"type": "message", "role": "user", "content": "hello"},
            ]
        }
    )
    assert chat["messages"][0] == {"role": "system", "content": "follow project rules"}
    assert chat["messages"][1] == {"role": "user", "content": "hello"}


def test_responses_function_call_output_preserves_call_id_and_content():
    chat = responses_request_to_chat(
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-weather",
                    "output": "sunny 32C",
                }
            ]
        }
    )
    assert chat["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "call-weather",
            "content": "sunny 32C",
        }
    ]
```

- [ ] **Step 2: Add an integration test for two tool calls resumed by `previous_response_id`**

Append to `tests/test_server_integration.py`:

```python
@pytest.mark.asyncio
async def test_previous_response_replays_multiple_tool_outputs(monkeypatch, tmp_path):
    prompts = []

    async def fake_stream(self, message):
        prompts.append(message)
        if len(prompts) == 1:
            yield (
                "text",
                '<tool_call>{"name":"weather","arguments":{"city":"Bangkok"}}</tool_call>'
                '<tool_call>{"name":"time","arguments":{"city":"Bangkok"}}</tool_call>',
            )
        else:
            yield ("text", "done")

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        first = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "input": "Plan my morning",
                "tools": [
                    {"type": "function", "name": "weather", "parameters": {"type": "object"}},
                    {"type": "function", "name": "time", "parameters": {"type": "object"}},
                ],
            },
        )
        calls = [item for item in first.json()["output"] if item["type"] == "function_call"]
        assert len(calls) == 2

        second = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "previous_response_id": first.json()["id"],
                "input": [
                    {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "sunny"},
                    {"type": "function_call_output", "call_id": calls[1]["call_id"], "output": "08:00"},
                ],
            },
        )

    assert second.status_code == 200
    assert "weather" in prompts[1]
    assert "time" in prompts[1]
    assert "sunny" in prompts[1]
    assert "08:00" in prompts[1]
```

- [ ] **Step 3: Run the new contract tests**

```powershell
python -m pytest tests/test_openai_compat.py tests/test_server_integration.py -q
```

Expected against the current baseline: PASS. If either new test fails, stop this task and trace whether the loss occurs in `openai_compat.py`, `_merge_prior_history()`, or `ConversationIndex.remember()` before editing production code. Do not add a new history layer.

- [ ] **Step 4: Commit the contract coverage**

```powershell
git add tests/test_openai_compat.py tests/test_server_integration.py
git commit -m "test(api): lock responses tool continuation semantics"
```

---

### Task 4: Lock Streaming Agent Contracts

**Files:**
- Test: `tests/test_server_integration.py:1-220`
- Test: `tests/test_hybrid_client.py:1-210`
- No production edit expected.

**Interfaces:**
- Consumes: existing FastAPI `/v1/chat/completions`, `/v1/responses`, `APIClient.send_message_stream()`
- Produces: regression coverage that normal text streams incrementally, tool streams are buffered until a complete tool call exists, and partial auth failures are not replayed.

- [ ] **Step 1: Add Responses SSE event-order coverage**

Add to `tests/test_server_integration.py`:

```python
@pytest.mark.asyncio
async def test_responses_stream_emits_created_delta_and_completed(monkeypatch, tmp_path):
    async def fake_stream(self, _message):
        yield ("text", "hello ")
        yield ("text", "world")

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={"model": "deepseek-chat", "input": "hello", "stream": True},
        ) as response:
            events = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert response.status_code == 200
    joined = "\n".join(events)
    assert "response.created" in joined
    assert "response.output_text.delta" in joined
    assert "hello " in joined
    assert "world" in joined
    assert "response.completed" in joined
```

- [ ] **Step 2: Add split tool-tag buffering coverage**

Add:

```python
@pytest.mark.asyncio
async def test_responses_tool_stream_buffers_split_tool_block(monkeypatch, tmp_path):
    async def fake_stream(self, _message):
        yield ("text", '<tool_call>{"name":"look')
        yield ("text", 'up","arguments":{"q":1}}</tool_call>')

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "input": "lookup",
                "stream": True,
                "tools": [
                    {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
                ],
            },
        ) as response:
            events = [line for line in response.iter_lines() if line.startswith("data: ")]

    joined = "\n".join(events)
    assert "response.output_text.delta" not in joined
    assert "response.output_item.added" in joined
    assert '"type": "function_call"' in joined
    assert '"name": "lookup"' in joined
```

- [ ] **Step 3: Re-run the existing no-replay-after-output test together with the new route tests**

```powershell
python -m pytest tests/test_hybrid_client.py::test_auth_error_after_partial_output_is_not_retried tests/test_server_integration.py -q
```

Expected: PASS. No production change is expected.

- [ ] **Step 4: Commit test-only streaming guarantees**

```powershell
git add tests/test_server_integration.py
git commit -m "test(stream): lock agent streaming behavior"
```

---

### Task 5: Make Multi-Account File Affinity Fully Fail-Closed

**Files:**
- Modify: `deepseek/gateway/files.py:62-82`
- Modify: `deepseek/gateway/pool.py:248-264`
- Test: `tests/test_file_affinity.py:1-110`

**Interfaces:**
- Changes internal signature to `FileAffinityStore.account_for(file_ids: Optional[list[str]], *, require_all_known: bool = False) -> Optional[str]`.
- Existing callers without the keyword retain current behavior.
- `AccountPool` sets `require_all_known=True` only when more than one account is configured.

- [ ] **Step 1: Add store-level and pool-level tests for mixed known/unknown files**

Add to `tests/test_file_affinity.py`:

```python
def test_file_affinity_can_require_every_file_to_be_known(tmp_path):
    store = FileAffinityStore(path=tmp_path / "files.json")
    store.bind("known", "a0")

    assert store.account_for(["known", "unknown"]) == "a0"
    with pytest.raises(FileAffinityError, match="unknown file"):
        store.account_for(["known", "unknown"], require_all_known=True)


@pytest.mark.asyncio
async def test_mixed_known_and_unknown_file_fails_closed_with_multiple_accounts(tmp_path, monkeypatch):
    files = FileAffinityStore(path=tmp_path / "files.json")
    bindings = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=bindings, file_store=files)
    pool._replace_accounts_for_test([Account("a0", _config()), Account("a1", _config())])
    files.bind("known", "a0")

    async def stream(self, _message):
        yield ("text", "should-not-route")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", stream)
    with pytest.raises(NoAccountAvailableError, match="unknown file"):
        async for _ in pool.send_message_stream(
            "hello",
            file_ids=["known", "unknown"],
        ):
            pass
```

- [ ] **Step 2: Verify the new cases fail with the current implementation**

```powershell
python -m pytest tests/test_file_affinity.py -q
```

Expected before implementation: the new `require_all_known` call fails because the keyword does not exist; mixed known/unknown pool routing also does not fail closed.

- [ ] **Step 3: Centralize unknown-file knowledge in `FileAffinityStore`**

Change `account_for()` to:

```python
    def account_for(
        self,
        file_ids: Optional[list[str]],
        *,
        require_all_known: bool = False,
    ) -> Optional[str]:
        if not file_ids:
            return None

        unknown = [file_id for file_id in file_ids if file_id not in self._items]
        if require_all_known and unknown:
            raise FileAffinityError(f"unknown file ID(s): {unknown}")

        accounts = {
            item["account_name"]
            for file_id in file_ids
            if (item := self._items.get(file_id)) is not None
        }
        if len(accounts) > 1:
            raise FileAffinityError(
                "Files in one request belong to different DeepSeek accounts"
            )
        return next(iter(accounts)) if accounts else None
```

- [ ] **Step 4: Make `AccountPool` use the store policy instead of duplicating the unknown-file check**

Replace:

```python
            file_account = self._files.account_for(file_ids)
```

with:

```python
            file_account = self._files.account_for(
                file_ids,
                require_all_known=len(self._accounts) > 1,
            )
```

Then delete the now-redundant block:

```python
        if file_ids and file_account is None and len(self._accounts) > 1:
            raise NoAccountAvailableError(...)
```

The existing `except FileAffinityError -> NoAccountAvailableError` remains the only translation point.

- [ ] **Step 5: Run all file/gateway affinity tests**

```powershell
python -m pytest tests/test_file_affinity.py tests/test_gateway_affinity.py tests/test_gateway.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add deepseek/gateway/files.py deepseek/gateway/pool.py tests/test_file_affinity.py
git commit -m "fix(gateway): reject partially unknown file affinity"
```

---

### Task 6: Remove Permanent Lifetime-Error Bias From Routing

**Files:**
- Modify: `deepseek/gateway/pool.py:133-142`
- Test: `tests/test_gateway.py:195-230`
- Test: `tests/test_gateway_affinity.py:80-166`

**Interfaces:**
- Consumes existing `Account` state: `lock`, `consecutive_quota_hits`, `last_used`.
- Produces no new interface; `_score()` stops using lifetime `total_errors` as a scheduling input while status/persistence keep recording it.

- [ ] **Step 1: Replace the old lifetime-error scheduling expectation with a current-health expectation**

Replace `test_prefers_idle_with_fewer_errors` with:

```python
    def test_prefers_least_recently_used_when_current_health_is_equal(self):
        pool = AccountPool()
        recovered = _make_account("recovered")
        recovered.total_errors = 100
        recovered.consecutive_quota_hits = 0
        recovered.last_used = 1.0

        recent = _make_account("recent")
        recent.total_errors = 0
        recent.consecutive_quota_hits = 0
        recent.last_used = 2.0

        pool._replace_accounts_for_test([recovered, recent])
        assert pool._next_account().name == "recovered"
```

- [ ] **Step 2: Add stateless quota failover coverage**

Add to `tests/test_gateway.py`:

```python
@pytest.mark.asyncio
async def test_stateless_quota_failure_fails_over_before_output(monkeypatch):
    pool = AccountPool()
    a0 = _make_account("a0")
    a1 = _make_account("a1")
    a0.last_used = 0.0
    a1.last_used = 1.0
    pool._replace_accounts_for_test([a0, a1])

    async def stream(self, _message):
        if self is a0.client:
            raise RuntimeError("quota exceeded")
        yield ("text", "from-a1")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", stream)

    output = []
    async for token in pool.send_message_stream("hello"):
        output.append(token)

    assert output == [("text", "from-a1")]
    assert a0.consecutive_quota_hits == 1
```

- [ ] **Step 3: Run the routing tests and verify the current-health test fails before implementation**

```powershell
python -m pytest tests/test_gateway.py::TestAccountPoolNextAccount tests/test_gateway.py::test_stateless_quota_failure_fails_over_before_output tests/test_gateway_affinity.py -q
```

Expected before implementation: least-recently-used test fails because `total_errors` currently precedes `last_used` in `_score()`; failover test should pass or expose a genuine failover defect.

- [ ] **Step 4: Simplify `_score()` using only current routing state**

Change:

```python
    def _score(self, account: Account) -> tuple:
        return (
            1 if account.lock.locked() else 0,
            account.consecutive_quota_hits,
            account.total_errors,
            account.last_used,
            random.random(),
        )
```

To:

```python
    def _score(self, account: Account) -> tuple:
        return (
            1 if account.lock.locked() else 0,
            account.consecutive_quota_hits,
            account.last_used,
            random.random(),
        )
```

Keep `total_errors` in status and persisted stats. Do not add a health-score class or new counter field.

- [ ] **Step 5: Run all gateway tests**

```powershell
python -m pytest tests/test_gateway.py tests/test_gateway_affinity.py tests/test_file_affinity.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add deepseek/gateway/pool.py tests/test_gateway.py
git commit -m "fix(gateway): avoid permanent routing penalty from old errors"
```

---

### Task 7: Eliminate the Remaining ResourceWarning Without Suppression

**Files:**
- Modify: `tests/test_client.py:576-630`
- Verify: `tests/test_gateway.py:452-469`
- Modify `pyproject.toml` only after the suite is warning-free, to make ResourceWarning a test failure.

**Interfaces:**
- No production interface changes.
- Unit tests must not accidentally invoke real auth/browser recovery or leave `APIClient` resources open.

- [ ] **Step 1: Make upload tests deterministic async tests and close clients**

Convert the three `TestUploadFile` methods to `@pytest.mark.asyncio`. For the auth test, inject a fake auth manager so a unit test cannot enter real network/browser recovery:

```python
class TestUploadFile:
    @pytest.mark.asyncio
    async def test_returns_file_id_from_biz_data(self):
        client = APIClient(_make_config())

        async def fake_pow(_path):
            return ""

        client._generate_pow_token_for_path = fake_pow  # type: ignore[assignment]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"data": {"biz_data": {"id": "f_123"}}})

        try:
            with patch.object(client.client, "post", AsyncMock(return_value=mock_resp)):
                assert await client.upload_file(b"hello", "x.txt", "text/plain") == "f_123"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_raises_auth_on_401(self):
        auth = MagicMock()
        auth.invalidate = MagicMock()
        auth.recover = AsyncMock(return_value=False)
        auth.ensure_valid = AsyncMock(return_value=False)
        client = APIClient(_make_config(), auth_manager=auth)

        async def fake_pow(_path):
            return ""

        client._generate_pow_token_for_path = fake_pow  # type: ignore[assignment]
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"

        try:
            with patch.object(client.client, "post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(AuthExpiredError):
                    await client.upload_file(b"x", "f.txt")
            auth.recover.assert_awaited_once()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_raises_runtime_when_no_file_id(self):
        client = APIClient(_make_config())

        async def fake_pow(_path):
            return ""

        client._generate_pow_token_for_path = fake_pow  # type: ignore[assignment]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"data": {}})

        try:
            with patch.object(client.client, "post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(RuntimeError, match="file_id"):
                    await client.upload_file(b"x", "f.txt")
        finally:
            await client.close()
```

- [ ] **Step 2: Simplify the explicit event-loop lock test to pytest-asyncio**

Replace `TestAccountLockReset.test_lock_can_be_replaced` in `tests/test_gateway.py` with:

```python
class TestAccountLockReset:
    @pytest.mark.asyncio
    async def test_lock_can_be_replaced(self):
        acc = _make_account("stuck")
        await acc.lock.acquire()
        assert acc.lock.locked() is True

        acc.lock = asyncio.Lock()
        assert acc.lock.locked() is False
```

Add `import asyncio` at the top of `tests/test_gateway.py`; do not create a manual event loop.

- [ ] **Step 3: Run the test files with ResourceWarning promoted to an error**

```powershell
python -W error::ResourceWarning -m pytest tests/test_client.py tests/test_gateway.py -q
```

Expected: PASS with no ResourceWarning.

- [ ] **Step 4: Run the full suite with ResourceWarning promoted to an error**

```powershell
python -W error::ResourceWarning -m pytest -q
```

Expected: 0 warnings and exit code 0. If a warning remains, use:

```powershell
python -X tracemalloc=25 -W error::ResourceWarning -m pytest -q
```

and fix the exact test-owned resource reported by the allocation traceback. Do not add another warning filter.

- [ ] **Step 5: Turn ResourceWarning into a permanent CI gate**

In `pyproject.toml`, replace:

```toml
"default::ResourceWarning",
```

with:

```toml
"error::ResourceWarning",
```

Keep the narrowly scoped upstream Starlette/AnyIO deprecation filters already present.

- [ ] **Step 6: Re-run the full suite normally**

```powershell
python -m pytest -q
```

Expected: all tests pass with no warning summary.

- [ ] **Step 7: Commit**

```powershell
git add tests/test_client.py tests/test_gateway.py pyproject.toml
git commit -m "test: close async resources and fail on leaks"
```

---

### Task 8: Make the Project Installable With a `deepseek` Console Command

**Files:**
- Modify: `pyproject.toml:1-16`
- Modify: `deepseek/__main__.py:82-148`
- Modify: `.github/workflows/ci.yml:17-33`
- Package assets: existing `deepseek/assets/pow_solver.js`, `deepseek/assets/sha3_wasm_bg.wasm`, `deepseek/assets/wasm_b64.txt`
- Test: add `tests/test_package_entrypoint.py`

**Interfaces:**
- Produces: `deepseek.__main__.run() -> None`
- Produces installed console script: `deepseek`
- Preserves module execution: `python -m deepseek`

- [ ] **Step 1: Add an entry-point unit test before changing `__main__.py`**

Create `tests/test_package_entrypoint.py`:

```python
from unittest.mock import patch

import deepseek.__main__ as cli_main


def test_run_executes_async_main_once():
    with patch("deepseek.__main__.asyncio.run") as run_async:
        cli_main.run()
    run_async.assert_called_once()
    coroutine = run_async.call_args.args[0]
    assert coroutine.cr_code.co_name == "main"
    coroutine.close()
```

- [ ] **Step 2: Run the new test and verify it fails because `run` does not exist**

```powershell
python -m pytest tests/test_package_entrypoint.py -q
```

Expected: FAIL with missing `run`.

- [ ] **Step 3: Add the synchronous console entry point**

At the bottom of `deepseek/__main__.py`:

```python
def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

Replace the existing direct `asyncio.run(main())` block rather than duplicating it.

- [ ] **Step 4: Expand `pyproject.toml` into package metadata while keeping pytest config**

Use this structure above the existing `[tool.pytest.ini_options]` block:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "deepseek-web-cli"
dynamic = ["version"]
description = "OpenAI-compatible CLI and API gateway for authenticated DeepSeek Web sessions"
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
dependencies = [
    "httpx==0.28.1",
    "playwright==1.50.0",
    "rich==14.0.0",
    "playwright-stealth==1.0.6",
    "wasmtime==28.0.0",
    "fastapi==0.128.8",
    "uvicorn==0.40.0",
    "python-multipart==0.0.20",
]

[project.scripts]
deepseek = "deepseek.__main__:run"

[tool.setuptools.dynamic]
version = { attr = "deepseek.constants.VERSION" }

[tool.setuptools.packages.find]
include = ["deepseek*"]

[tool.setuptools.package-data]
deepseek = ["assets/*.js", "assets/*.wasm", "assets/*.txt"]
```

Keep test-only dependencies in `requirements.txt` for the existing development workflow. Runtime versions in `pyproject.toml` must match the same versions in `requirements.txt`.

- [ ] **Step 5: Install the package through its declared metadata**

```powershell
python -m pip install .
```

Expected: successful wheel/build installation.

- [ ] **Step 6: Smoke-test both entry points**

```powershell
deepseek --version
python -m deepseek --version
```

Expected: both print the same `DeepSeek Web CLI v<VERSION>` value and exit 0.

- [ ] **Step 7: Update CI to prove installability**

After `Install dependencies`, add:

```yaml
      - name: Install project package
        run: python -m pip install .
      - name: Smoke test installed CLI
        run: |
          deepseek --version
          python -m deepseek --version
```

Keep `pip check`, `compileall`, and pytest after installation.

- [ ] **Step 8: Run package and test gates locally**

```powershell
python -m pip check
python -m compileall -q deepseek tests
python -m pytest tests/test_package_entrypoint.py -q
deepseek --version
python -m deepseek --version
```

Expected: all commands exit 0.

- [ ] **Step 9: Commit**

```powershell
git add pyproject.toml deepseek/__main__.py .github/workflows/ci.yml tests/test_package_entrypoint.py
git commit -m "feat(package): add installable deepseek console entry point"
```

---

### Task 9: Align CLI Security Guidance and README With Current Behavior

**Files:**
- Modify: `deepseek/server.py:900-955`
- Test: `tests/test_server_integration.py:1-220`
- Modify: `README.md:57-78`, `README.md:244-335`, `README.md:370-440`, `README.md:641-650`

**Interfaces:**
- No API behavior change.
- Non-loopback bind remains allowed but emits an operator warning.
- Documentation must describe current self-healing auth and manual `--repair` fallback.

- [ ] **Step 1: Add a non-loopback warning route-start test**

Extend `_capture_app()` in `tests/test_server_integration.py` to accept `**serve_kwargs` and call:

```python
    await server_module.serve_mode(**serve_kwargs)
```

Then add:

```python
@pytest.mark.asyncio
async def test_non_loopback_bind_warns_operator(monkeypatch, tmp_path):
    printed = []
    monkeypatch.setattr(server_module.console, "print", lambda value: printed.append(str(value)))

    await _capture_app(monkeypatch, tmp_path, host="0.0.0.0")

    assert any("non-loopback" in value.lower() for value in printed)
```

- [ ] **Step 2: Run the test and confirm it fails before implementation**

```powershell
python -m pytest tests/test_server_integration.py::test_non_loopback_bind_warns_operator -q
```

Expected: FAIL because no warning is currently printed.

- [ ] **Step 3: Add the warning immediately before the server startup panel**

In `serve_mode()` before the existing `console.print(Panel.fit(...))` block:

```python
    if host.strip().lower() not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[yellow]Warning: API server is binding to a non-loopback address; "
            "keep API authentication enabled and restrict network access.[/]"
        )
```

Do not block LAN/container usage and do not expose any key value in the warning.

- [ ] **Step 4: Rewrite Quick Start to use the installed command first**

`README.md:57-78` should show:

```powershell
python -m pip install .
playwright install chromium

deepseek --discover
deepseek --chat
```

Also state that `python -m deepseek ...` remains equivalent for source-tree use.

- [ ] **Step 5: Update API Server and CLIProxyAPI instructions**

In `README.md:244-335`, document the existing API flow with concrete commands:

```powershell
deepseek --serve --gateway
```

Base URL:

```text
http://127.0.0.1:8000/v1
```

Document that CLIProxyAPI should configure this as an OpenAI-compatible upstream and use the locally generated DeepSeek CLI API key from `~/.deepseek_cli/api_key` (or `DEEPSEEK_API_KEY` / `--api-key` override). Do not describe DeepSeek Web credentials as the local API key.

- [ ] **Step 6: Update auth recovery wording**

Replace the stale limitation that says automatic re-auth does not exist with current behavior:

```markdown
* **Interactive recovery can still be required** — saved auth is probed automatically. When it is stale, `AuthManager` attempts serialized browser recovery and persists only validated credentials. Use `--repair` when automatic recovery cannot restore the session.
```

- [ ] **Step 7: Keep real limitations explicit**

Retain and verify wording for:

- heuristic token estimates
- buffered/synthetic tool-call streaming
- no Anthropic `/v1/messages`
- process-local metrics reset
- foreground server/no daemon mode

Do not add historical wording such as "we added", "previously", or "removed".

- [ ] **Step 8: Run security/API tests and search docs for stale wording**

```powershell
python -m pytest tests/test_api_key.py tests/test_server_integration.py tests/test_cli.py -q
rg -n "no automatic re-auth|Re-run --discover|Browser sessions expire" README.md
```

Expected: tests PASS; `rg` returns no stale automatic-auth claim.

- [ ] **Step 9: Commit**

```powershell
git add deepseek/server.py tests/test_server_integration.py README.md
git commit -m "docs: align cli and api guidance with recovery behavior"
```

---

### Task 10: Release Verification and Scope Audit

**Files:**
- No planned production edit.
- Review all files changed by Tasks 1-9.

**Interfaces:**
- Produces release evidence only; no new interface.

- [ ] **Step 1: Verify dependency consistency**

```powershell
python -m pip install -r requirements.txt
python -m pip install .
python -m pip check
```

Expected: exit 0 for all commands.

- [ ] **Step 2: Compile all shipped Python code and tests**

```powershell
python -m compileall -q deepseek tests
```

Expected: exit 0.

- [ ] **Step 3: Run the full test suite with resource leaks treated as failures**

```powershell
python -W error::ResourceWarning -m pytest -q
```

Expected: all tests pass with no warning summary.

- [ ] **Step 4: Smoke-test installed CLI entry points**

```powershell
deepseek --version
python -m deepseek --version
deepseek --help
```

Expected: all commands exit 0 and show the same supported flags.

- [ ] **Step 5: Verify API/config compatibility tests explicitly**

```powershell
python -m pytest tests/test_models.py tests/test_session.py tests/test_api_key.py tests/test_server_integration.py tests/test_gateway.py tests/test_gateway_affinity.py tests/test_file_affinity.py -q
```

Expected: PASS.

- [ ] **Step 6: Review the diff for forbidden scope expansion**

```powershell
git diff 8331d6f..HEAD --stat
git diff 8331d6f..HEAD -- deepseek/_client_legacy.py deepseek/_tools_legacy.py requirements.txt
```

Expected:

- no style rewrite of `_client_legacy.py`
- no style rewrite of `_tools_legacy.py`
- `requirements.txt` unchanged unless version consistency truly required it
- no new service/repository/factory/retry framework

`8331d6f` is the approved-spec baseline for this implementation plan, so use it as the fixed review base.

- [ ] **Step 7: Check committed content for obvious secret leakage patterns**

```powershell
rg -n "Authorization: Bearer|ds_session_id|X-Admin-Token:|sk-ds-[0-9a-f]{16,}" deepseek tests README.md docs
```

Expected: only documentation/test literals that are clearly fake; no real token, cookie, admin token, or generated local API key.

- [ ] **Step 8: Confirm clean working tree**

```powershell
git status --short
```

Expected: no output.

- [ ] **Step 9: Record final verification result in the implementation handoff**

The handoff must state the exact final test count, package smoke results, current HEAD commit, and whether any approved task ended test-only with no production change. Do not create another architectural summary document.
