"""
Discovery mode — open browser, sniff traffic, save config.
"""
import json
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .constants import CONFIG_DIR, COMPLETION_PATH
from .models import APIConfig
from .session import SessionManager
from .sniffer import NetworkSniffer


console = Console()


async def discover_mode(target_url: str, headless: bool = False) -> Optional[APIConfig]:
    """Open browser, user logs in, sniff traffic, save APIConfig.

    Returns the captured APIConfig on success, or None on failure / cancel.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]🔍 Discovery Mode[/]\n"
            f"Target: {target_url}\n\n"
            "A browser window will open.\n"
            "1. [bold]Log in[/] if not already logged in.\n"
            "2. [bold yellow]Send at least one message[/] to the AI.\n"
            "3. Press [bold yellow]Ctrl+C[/] here when done (or close the browser).",
            border_style="cyan",
        )
    )

    import asyncio

    sniffer = NetworkSniffer(headless=headless)
    page = await sniffer.start(target_url)

    try:
        while True:
            await asyncio.sleep(0.5)
            if page.is_closed():
                console.print("\n[dim]Browser closed — capturing traffic...[/]")
                break
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[dim]Capturing traffic...[/]")

    # Save captured URL log
    captured_log = CONFIG_DIR / "captured_urls.txt"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(captured_log, "w", encoding="utf-8") as f:
        for c in sniffer.captured:
            f.write(f"{c.method} {c.url}\n")
            if "pow" in c.url.lower() or "challenge" in c.url.lower():
                f.write(f"  --> [MATCH] ReqBody: {c.request_body[:400] if c.request_body else ''}\n")
                f.write(f"  --> [MATCH] ResBody: {c.response_body[:400]}\n")

    # Detect API paths from captured traffic
    completion_path, session_path, pow_challenge_path = _detect_api_paths(sniffer)

    # Find the best completion endpoint
    completion_reqs = [
        c for c in sniffer.captured
        if ("completion" in c.url.lower() or "chat/complete" in c.url.lower())
        and c.method == "POST"
        and c.response_status
    ]
    if completion_reqs:
        chat_endpoint = completion_reqs[-1]
        console.print(f"[green]✓ Found completion endpoint: {chat_endpoint.url}[/]")
    else:
        chat_endpoint = sniffer.detect_chat_api()
        if chat_endpoint:
            console.print(f"[yellow]⚠ Using heuristic endpoint: {chat_endpoint.url}[/]")
        else:
            console.print("[yellow]⚠ No completion request captured — send a message in the browser next time[/]")

    if not chat_endpoint:
        candidates = sniffer.list_candidates()
        if not candidates:
            console.print("[red]❌ No XHR/fetch requests captured. Try sending a message in the browser first.[/]")
            await sniffer.stop()
            return None
        console.print("\n[bold]Candidate endpoints found:[/]")
        for i, cap in enumerate(candidates[:10]):
            preview = cap.response_body[:120].replace("\n", " ")
            console.print(f"  [{i}] {cap.method} {cap.url}")
            console.print(f"      Status: {cap.response_status} | Body: {preview}…\n")
        choice = console.input("[bold]Select endpoint number (or Enter for auto-detect): [/]")
        try:
            chat_endpoint = candidates[int(choice)] if choice.strip() else candidates[0]
        except (ValueError, IndexError):
            chat_endpoint = candidates[0]

    # Extract auth
    auth_token, auth_type = _extract_auth(chat_endpoint)
    if not auth_token:
        try:
            cookies = await page.context.cookies()
            for ck in cookies:
                if "token" in ck.get("name", "").lower() or "auth" in ck.get("name", "").lower():
                    auth_token = ck.get("value", "")
                    auth_type = "cookie"
                    break
        except Exception:
            pass

    # Build body template
    body_template: dict = {}
    if chat_endpoint.request_body:
        try:
            body_template = json.loads(chat_endpoint.request_body)
            if "messages" in body_template and isinstance(body_template["messages"], list):
                body_template["messages"] = [
                    m for m in body_template["messages"] if m.get("role") != "user"
                ]
        except json.JSONDecodeError:
            body_template = {"_raw": chat_endpoint.request_body}

    # Gather cookies
    try:
        all_cookies = await page.context.cookies()
        cookie_dict = {c["name"]: c["value"] for c in all_cookies}
    except Exception as e:
        console.print(f"[dim]Could not gather cookies (browser may be closed): {e}[/]")
        cookie_dict = {}

    # Extract PoW token
    pow_response = chat_endpoint.request_headers.get("x-ds-pow-response", "")
    if pow_response:
        console.print("[green]✓ PoW token captured from browser[/]")
    else:
        console.print("[yellow]⚠ No PoW token found — re-run --discover and send a message in the browser[/]")

    config = APIConfig(
        target_url=target_url,
        api_endpoint=chat_endpoint.url,
        method=chat_endpoint.method,
        headers={
            k: v
            for k, v in chat_endpoint.request_headers.items()
            if k.lower()
            not in ("authorization", "cookie", "content-length", "host", "x-ds-pow-response")
        },
        body_template=body_template,
        auth_type=auth_type,
        auth_token=auth_token,
        cookies=cookie_dict,
        use_stream=(
            "text/event-stream" in chat_endpoint.response_headers.get("content-type", "")
        ),
        pow_response=pow_response,
        completion_path=completion_path,
        session_path=session_path,
        pow_challenge_path=pow_challenge_path,
    )

    SessionManager.save_config(config)
    await sniffer.stop()

    console.print(
        Panel.fit(
            f"[bold green]✅ Config saved![/]\n"
            f"Endpoint: {config.api_endpoint}\n"
            f"Auth type: {config.auth_type}\n"
            f"Streaming: {config.use_stream}\n"
            f"Completion path: {config.completion_path}\n\n"
            "Run without --discover to start chatting.",
            border_style="green",
        )
    )
    return config


def _detect_api_paths(sniffer: NetworkSniffer) -> tuple[str, str, str]:
    """Detect API paths from captured traffic, falling back to known defaults.

    Prefers endpoints with the most specific keyword match. Tracks first match
    per category to avoid later, less-specific URLs overwriting good detections.
    """
    completion_path = COMPLETION_PATH
    session_path = "/api/v0/chat_session/create"
    pow_challenge_path = "/api/v0/chat/create_pow_challenge"

    completion_found = False
    session_found = False
    pow_found = False

    from urllib.parse import urlparse

    for cap in sniffer.captured:
        if cap.method != "POST":
            continue
        path = urlparse(cap.url).path
        path_l = path.lower()
        if not completion_found and (
            "chat/completion" in path_l or "chat/complete" in path_l
        ):
            completion_path = path
            completion_found = True
        elif not session_found and "chat_session/create" in path_l:
            session_path = path
            session_found = True
        elif not session_found and "session" in path_l and "create" in path_l:
            session_path = path
            # don't mark as found — keep looking for the more specific match
        elif not pow_found and ("pow_challenge" in path_l or "pow/challenge" in path_l):
            pow_challenge_path = path
            pow_found = True

    return completion_path, session_path, pow_challenge_path


def _extract_auth(chat_endpoint) -> tuple[str, str]:
    """Extract auth token and type from captured request headers (case-insensitive)."""
    auth_header = ""
    for k, v in chat_endpoint.request_headers.items():
        if k.lower() == "authorization":
            auth_header = v
            break
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:], "bearer"
    if auth_header:
        return auth_header, "header"
    return "", "bearer"
