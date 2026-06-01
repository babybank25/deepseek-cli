"""
Terminal chat mode using Rich for display.
Simple, reliable, no layout bugs.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .client import APIClient
from .constants import HISTORY_LIMIT, VERSION, DEFAULT_SYSTEM_PROMPT
from .exceptions import AuthExpiredError
from .session import SessionManager

console = Console()

COMMANDS = {
    "/exit":    "quit",
    "/new":     "new conversation",
    "/compact": "summarize → start new session keeping context",
    "/think":   "toggle DeepThink R1  (resets conversation)",
    "/search":  "toggle web search   (resets conversation)",
    "/model":   "switch model        (resets conversation)",
    "/attach":  "attach file  (/attach path/to/file)",
    "/export":  "save chat to file",
    "/status":  "show session info",
    "/help":    "show this help",
}


async def chat_mode(auto_compact_threshold: Optional[int] = None) -> None:
    """Interactive terminal chat.

    ``auto_compact_threshold`` overrides the default rolling-summary threshold
    (``constants.AUTO_COMPACT_THRESHOLD``) for this session. Pass ``0`` to
    disable auto-compact.
    """
    config = SessionManager.load_config()
    if not config:
        console.print("[red]No saved config. Run Discover first (option 3 in menu).[/]")
        return

    # Warn about expiring tokens
    expires_in = config.token_expires_in()
    if expires_in is not None:
        if expires_in < 0:
            console.print("[red]⚠ Auth token expired — run Discover again[/]")
        elif expires_in < 3600:
            console.print(f"[yellow]⚠ Token expires in {int(expires_in/60)} min[/]")

    client = APIClient(config)
    client.system_prompt = DEFAULT_SYSTEM_PROMPT
    # Default preset: Expert ⇒ R1 ON, Search OFF (matches the official web UI).
    client.model_type = "expert"
    client.thinking_enabled = True
    client.search_enabled = False
    if auto_compact_threshold is not None:
        client.auto_compact_threshold = auto_compact_threshold

    console.print(Panel(
        f"[bold cyan]DeepSeek Chat[/]  [dim]v{VERSION}[/]\n"
        f"[dim]Commands: {' | '.join(COMMANDS)}[/]",
        border_style="cyan",
        padding=(0, 1),
    ))

    history: list[dict] = []
    pending_file_ids: list[str] = []  # files attached, sent with next message

    while True:
        # Build prompt tag
        tags = ["Expert" if client.model_type == "expert" else "Fast"]
        if client.thinking_enabled:
            tags.append("R1")
        if client.search_enabled:
            tags.append("Search")
        tag = " · ".join(tags)

        try:
            user_input = console.input(f"\n[bold cyan]You ({tag}) ›[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/]")
            break

        text = user_input.strip()
        if not text:
            continue

        # ── Commands ──────────────────────────────────────────
        cmd = text.lower()

        if cmd == "/exit":
            console.print("[dim]Bye![/]")
            break

        if cmd == "/help":
            for c, desc in COMMANDS.items():
                console.print(f"  [cyan]{c:<10}[/] {desc}")
            continue

        if cmd == "/new":
            await client.reset_session()
            history.clear()
            console.print("[dim]── New conversation ──[/]")
            continue

        if cmd == "/compact":
            if not client.session_id:
                console.print("[dim]No active session to compact.[/]")
                continue
            console.print("[dim]Summarizing previous turns…[/]")
            ok = await client.compact_session()
            if ok:
                history.clear()
                console.print(
                    "[green]✓ Compacted — new session will start with the "
                    "summary as context.[/]"
                )
            else:
                console.print("[yellow]⚠ Compact failed (no session or summary error).[/]")
            continue

        if cmd == "/think":
            client.thinking_enabled = not client.thinking_enabled
            await client.reset_session()
            history.clear()
            console.print(
                f"[blue]DeepThink R1: {'ON' if client.thinking_enabled else 'OFF'}[/]"
                "  [dim](new conversation — mode is bound to session)[/]"
            )
            continue

        if cmd == "/search":
            client.search_enabled = not client.search_enabled
            await client.reset_session()
            history.clear()
            console.print(
                f"[green]Web Search: {'ON' if client.search_enabled else 'OFF'}[/]"
                "  [dim](new conversation — mode is bound to session)[/]"
            )
            continue

        if cmd.startswith("/model"):
            parts = cmd.split()
            if len(parts) > 1 and parts[1] in ("expert", "fast"):
                if parts[1] == "expert":
                    new_type = "expert"
                    new_thinking = True
                    new_search = False
                else:
                    new_type = "default"
                    new_thinking = False
                    new_search = True
                changed = (
                    new_type != client.model_type
                    or new_thinking != client.thinking_enabled
                    or new_search != client.search_enabled
                )
                if changed:
                    client.model_type = new_type
                    client.thinking_enabled = new_thinking
                    client.search_enabled = new_search
                    await client.reset_session()
                    history.clear()
                    label = parts[1].upper()
                    feats = (
                        f"R1={'ON' if new_thinking else 'OFF'}"
                        f"  Search={'ON' if new_search else 'OFF'}"
                    )
                    console.print(
                        f"[magenta]Model: {label}[/]  [dim]{feats}[/]"
                        "  [dim](new conversation)[/]"
                    )
                else:
                    console.print(f"[dim]Already on {parts[1].upper()} preset[/]")
            else:
                console.print("[dim]Usage: /model expert  OR  /model fast[/]")
            continue

        if cmd.startswith("/attach"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[dim]Usage: /attach path/to/file[/]")
                continue
            file_path = Path(parts[1].strip())
            if not file_path.exists():
                console.print(f"[red]File not found: {file_path}[/]")
                continue
            console.print(f"[dim]Uploading {file_path.name}…[/]")
            file_id = await upload_file(client, file_path)
            if file_id:
                pending_file_ids.append(file_id)
                console.print(f"[green]✓ Attached: {file_path.name}  (id: {file_id[:20]}…)[/]")
                console.print(f"[dim]Files queued: {len(pending_file_ids)} — will be sent with next message[/]")
            continue

        if cmd.startswith("/export"):
            parts = text.split(maxsplit=1)
            fname = parts[1] if len(parts) > 1 else None
            path = export_conversation(history, fname)
            console.print(f"[green]✓ Saved to {path}[/]")
            continue

        if cmd == "/status":
            exp = config.token_expires_in()
            exp_str = (
                f"{int(exp/3600)}h remaining" if exp and exp > 0
                else ("expired" if exp is not None else "unknown")
            )
            console.print(
                f"[dim]v{VERSION} | "
                f"session: {client.session_id[:8]+'…' if client.session_id else 'none'} | "
                f"model: {client.model_type} | "
                f"R1: {'on' if client.thinking_enabled else 'off'} | "
                f"search: {'on' if client.search_enabled else 'off'} | "
                f"messages: {len(history)//2} | "
                f"turns: {client._turn_count}/{client.auto_compact_threshold} | "
                f"token: {exp_str}[/]"
            )
            continue

        if text.startswith("/"):
            console.print("[dim]Unknown command. Type /help for list.[/]")
            continue

        # ── Stream response ───────────────────────────────────
        full_response = ""
        try:
            # Inject pending file attachments
            if pending_file_ids:
                client.set_pending_files(pending_file_ids)
                count = len(pending_file_ids)
                pending_file_ids.clear()
                console.print(f"[dim]Sending with {count} attached file(s)[/]")

            console.print("[bold magenta]AI ›[/bold magenta] ", end="")
            async for token_type, token_text in client.send_message_stream(text):
                if token_type == "text":
                    console.print(token_text, end="", highlight=False)
                    full_response += token_text
            console.print()  # newline

            if not full_response:
                console.print("[yellow]⚠ Empty response — try again[/]")

        except AuthExpiredError:
            console.print("\n[yellow]⚠ Token expired — run Discover again[/]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")

        if full_response:
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": full_response})
            if HISTORY_LIMIT > 0 and len(history) > HISTORY_LIMIT * 2:
                history = history[-(HISTORY_LIMIT * 2):]

    await client.close()


async def upload_file(client: APIClient, file_path: Path) -> Optional[str]:
    """Upload a file to DeepSeek and return the file_id, or None on failure."""
    import mimetypes
    content = file_path.read_bytes()
    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        return await client.upload_file(content, filename, content_type)
    except AuthExpiredError:
        console.print("[red]⚠ Auth expired — run Discover again[/]")
        return None
    except Exception as e:
        console.print(f"[red]Upload error: {e}[/]")
        return None


def export_conversation(
    history: list[dict], filename: Optional[str] = None
) -> Path:
    """Write conversation history to a Markdown file. Returns the path."""
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deepseek_chat_{ts}.md"
    path = Path(filename)
    lines = [
        f"# DeepSeek Chat Export\n\n"
        f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n"
    ]
    for msg in history:
        role = "**You**" if msg["role"] == "user" else "**AI**"
        lines.append(f"{role}\n\n{msg['content']}\n\n---\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
