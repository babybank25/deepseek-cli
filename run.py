#!/usr/bin/env python3
"""
DeepSeek CLI — Quick launcher
รันไฟล์นี้แล้วเลือกเมนูได้เลย
"""
import asyncio
import sys
from pathlib import Path

# ── ensure package is importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from deepseek.constants import VERSION
from deepseek.session import SessionManager

console = Console()


def show_menu() -> str:
    has_config = SessionManager.config_exists()
    config_status = "[green]✓ Session saved[/]" if has_config else "[yellow]⚠ No session — run Discover first[/]"

    console.print(Panel.fit(
        f"[bold cyan]DeepSeek Web CLI[/] [dim]v{VERSION}[/]\n"
        f"Status: {config_status}\n\n"
        "[bold]1[/]  💬  Chat  (terminal chat)\n"
        "[bold]2[/]  🌐  Serve (OpenAI-compatible API server)\n"
        "[bold]3[/]  🔍  Discover (capture session from browser)\n"
        "[bold]4[/]  �  Account Pool (manage multiple accounts)\n"
        "[bold]5[/]  �🚪  Exit",
        border_style="cyan",
        title="[bold]Main Menu[/]",
    ))
    return Prompt.ask("[bold cyan]เลือก[/]", choices=["1", "2", "3", "4", "5"], default="1")


async def account_pool_menu() -> None:
    """Sub-menu for managing the multi-account pool."""
    from deepseek.gateway import AccountPool, AccountStore

    while True:
        console.print(Panel.fit(
            "[bold]Account Pool[/]\n\n"
            "  [bold]a)[/] List accounts + status\n"
            "  [bold]b)[/] Add new account (opens browser)\n"
            "  [bold]c)[/] Remove account\n"
            "  [bold]d)[/] Back",
            border_style="cyan",
            title="[bold]Account Pool[/]",
        ))
        sub = Prompt.ask("เลือก", choices=["a", "b", "c", "d"], default="a")

        if sub == "a":
            pool = AccountPool()
            count = pool.load_all()
            if count == 0:
                console.print("[yellow]⚠ No accounts found. Use option b) to add one.[/]")
            else:
                status = pool.status()
                console.print(
                    f"\n[bold]Pool status:[/] "
                    f"{status['available_accounts']}/{status['total_accounts']} available"
                    f"  |  in_use: {status['in_use']}\n"
                )
                for acc in status["accounts"]:
                    avail = "[green]✓ available[/]" if acc["available"] else (
                        f"[red]✗ exhausted ({acc['cooldown_remaining_seconds']}s)[/]"
                    )
                    console.print(
                        f"  [bold]{acc['name']}[/]  {avail}"
                        f"  requests={acc['total_requests']}  errors={acc['total_errors']}"
                        + (f"  last_error={acc['last_error']}" if acc["last_error"] else "")
                    )
                console.print()
            await pool.close_all()
            input("กด Enter เพื่อกลับ...")

        elif sub == "b":
            from deepseek.discover import discover_mode
            target = Prompt.ask("Target URL", default="https://chat.deepseek.com")
            headless = Prompt.ask("Headless?", choices=["y", "n"], default="n") == "y"
            # Run discover to capture config, then save as new account
            config = await discover_mode(target_url=target, headless=headless)
            if config is not None:
                name = AccountStore.next_name()
                AccountStore.save(name, config)
                console.print(f"[green]✓ Saved as {name}[/]")
            else:
                console.print("[yellow]⚠ Discover did not return a config — account not saved.[/]")
            input("\nกด Enter เพื่อกลับ...")

        elif sub == "c":
            names = AccountStore.list_accounts()
            if not names:
                console.print("[yellow]⚠ No accounts to remove.[/]")
            else:
                console.print("Accounts: " + ", ".join(names))
                name = Prompt.ask("Account name to remove")
                if name in names:
                    AccountStore.delete(name)
                    console.print(f"[green]✓ Removed {name}[/]")
                else:
                    console.print(f"[red]Account '{name}' not found.[/]")
            input("กด Enter เพื่อกลับ...")

        elif sub == "d":
            break


async def main():
    while True:
        console.clear()
        choice = show_menu()

        if choice == "1":
            if not SessionManager.config_exists():
                console.print("[yellow]ยังไม่มี session — รัน Discover ก่อนนะ (เลือก 3)[/]")
                input("กด Enter เพื่อกลับ...")
                continue
            from deepseek.cli import chat_mode
            await chat_mode()

        elif choice == "2":
            if not SessionManager.config_exists():
                console.print("[yellow]ยังไม่มี session — รัน Discover ก่อนนะ (เลือก 3)[/]")
                input("กด Enter เพื่อกลับ...")
                continue
            from deepseek.server import serve_mode
            from deepseek.gateway import AccountStore
            port = int(Prompt.ask("Port", default="8000"))
            api_key = Prompt.ask("API Key (Enter = ไม่ใช้)", default="") or None
            n_accounts = len(AccountStore.list_accounts())
            use_gw = False
            if n_accounts > 1:
                ans = Prompt.ask(
                    f"ใช้ Gateway ({n_accounts} accounts)?",
                    choices=["y", "n"],
                    default="y",
                )
                use_gw = ans == "y"
            await serve_mode(
                host="127.0.0.1",
                port=port,
                api_key=api_key,
                use_gateway=use_gw,
            )

        elif choice == "3":
            from deepseek.discover import discover_mode
            target = Prompt.ask("Target URL", default="https://chat.deepseek.com")
            headless = Prompt.ask("Headless? (ไม่มีหน้าต่าง)", choices=["y", "n"], default="n") == "y"
            await discover_mode(target_url=target, headless=headless)
            input("\nกด Enter เพื่อกลับ...")

        elif choice == "4":
            await account_pool_menu()

        elif choice == "5":
            console.print("[dim]Bye![/]")
            break


if __name__ == "__main__":
    asyncio.run(main())
